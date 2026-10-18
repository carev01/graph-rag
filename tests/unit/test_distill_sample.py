"""`select_sample`: deterministic stratified selection with a per-vendor floor, hard
cases oversampled 2:1 and a stratified held-out split -- over a fake read driver."""
from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
from neo4j import RoutingControl

from graph_extract.config import ExtractSettings

from scripts.distill import sample as smp
from scripts.extraction_proof import AFFECTED_ARTICLES_QUERY, PRODUCT_NAMES_QUERY

# vendor, product, article count
TABLE = [("Veeam", "Veeam Backup & Replication", 150), ("Veeam", "Veeam Agent", 100),
         ("Microsoft", "Azure Backup", 80), ("AWS", "AWS Backup", 62),
         ("Cohesity", "Cohesity DataProtect", 8)]


PROSE = "\n".join(f"Prose line {i} about backups." for i in range(20))
SMALL_TABLE = "| a | b |\n|---|---|\n| 1 | 2 |"          # cheap tier: 3 of 24 lines
DENSE_TABLE = "\n".join(["| region | yes |", "|---|---|"]
                        + [f"| r{i} | yes |" for i in range(40)])


def _dense(n: int) -> bool:
    return n % 19 == 0


def _long(n: int) -> bool:
    return n % 23 == 0


def _texts(n: int) -> list[str]:
    if _dense(n):
        return ["Availability matrix.", DENSE_TABLE]
    if n % 13 == 0:
        return [PROSE, SMALL_TABLE]
    return [PROSE]


def _rows() -> list[dict]:
    rows = []
    n = 0
    for vendor, product, count in TABLE:
        for i in range(count):
            n += 1
            texts = _texts(n)
            rows.append({"id": f"art-{n:04d}",
                         "title": "Restoring a VM" if n % 11 == 0 else f"Page {n}",
                         "vendor": vendor, "product": product, "source": f"{product} Guide",
                         "has_table": any("|---" in t for t in texts),
                         "has_pipe": any("|" in t for t in texts),
                         "max_tokens": 1500 if _long(n) else 600,
                         "unnamed_chunk": n % 17 == 0, "_texts": texts})
    return rows


def _strong(n: int) -> bool:
    return _dense(n) or _long(n)


class FakeRead:
    """Answers the three selection queries; refuses anything not routed READ."""

    def __init__(self, rows: list[dict], affected_example: set[str],
                 affected_other: set[str]) -> None:
        self.rows = rows
        self.affected = {True: affected_example, False: affected_other}
        self.queries: list[tuple[str, dict]] = []

    async def execute_query(self, query: str, parameters_: dict | None = None,
                            routing_: Any = None) -> Any:
        assert routing_ == RoutingControl.READ
        params = parameters_ or {}
        self.queries.append((query, params))
        if query == PRODUCT_NAMES_QUERY:
            return SimpleNamespace(records=[{"names": ["veeam agent", "aws backup"]}])
        if query == AFFECTED_ARTICLES_QUERY:
            return SimpleNamespace(records=[{"id": i}
                                            for i in sorted(self.affected[params["example"]])])
        if query == smp.CHUNK_TEXTS_QUERY:
            by_id = {r["id"]: r for r in self.rows}
            return SimpleNamespace(records=[{"id": i, "texts": by_id[i]["_texts"]}
                                            for i in params["ids"]])
        if query == smp.ARTICLES_QUERY:
            wanted = set(params["vendors"])
            # deliberately NOT in id order: the selection must sort for itself
            return SimpleNamespace(records=[r for r in reversed(self.rows)
                                            if r["vendor"].lower() in wanted])
        raise AssertionError(f"unexpected query {query[:60]}")


def _fake() -> FakeRead:
    return FakeRead(_rows(), affected_example={"art-0003", "art-0390"},
                    affected_other={"art-0201"})


async def test_same_output_twice_and_floor_and_heldout() -> None:
    first = await smp.select_sample(_fake(), n_total=40, n_heldout=8)
    second = await smp.select_sample(_fake(), n_total=40, n_heldout=8)
    assert first == second
    assert len(first) == 40
    assert len({a.article_id for a in first}) == 40
    per_vendor = Counter(a.vendor for a in first)
    # proportional share of Cohesity would be ~1 of 40; the floor lifts it to 5
    assert all(per_vendor[v] >= smp.VENDOR_FLOOR for v in smp.DEFAULT_VENDORS)
    held = [a for a in first if a.split == "heldout"]
    train = [a for a in first if a.split == "train"]
    assert len(held) == 8
    assert not {a.article_id for a in held} & {a.article_id for a in train}
    # train first, held out last
    assert [a.split for a in first] == ["train"] * 32 + ["heldout"] * 8


async def test_order_puts_a_mix_of_vendors_first() -> None:
    sample = await smp.select_sample(_fake(), n_total=40, n_heldout=8)
    assert {a.vendor for a in sample[:4]} == set(smp.DEFAULT_VENDORS)


async def test_heldout_is_stratified() -> None:
    sample, counts = await smp.select_sample_with_counts(_fake(), n_total=40, n_heldout=8)
    held = Counter(a.stratum for a in sample if a.split == "heldout")
    assert sum(held.values()) == 8
    assert len(held) >= 3                      # spread over strata, not one
    for stratum, c in counts.items():
        assert c["train"] + c["heldout"] == c["selected"]
        assert c["heldout"] == held.get(stratum, 0)


async def test_hard_cases_flagged_and_oversampled() -> None:
    sample, counts = await smp.select_sample_with_counts(_fake(), n_total=60, n_heldout=10)
    rows = _rows()
    pop_hard = sum(c["hard_population"] for c in counts.values()) / len(rows)
    sample_hard = sum(1 for a in sample if a.hard_cases) / len(sample)
    assert sample_hard > pop_hard


async def test_hard_case_flags() -> None:
    fake = _fake()
    cands = {c.article_id: c for c in await smp.read_candidates(fake, smp.DEFAULT_VENDORS)}
    assert "affected" in cands["art-0003"].hard_cases     # example-subject half
    assert "affected" in cands["art-0201"].hard_cases     # other-subject half
    assert "restore_title" in cands["art-0011"].hard_cases
    assert "table" in cands["art-0013"].hard_cases and not cands["art-0013"].strong_tier
    assert "unnamed_chunk" in cands["art-0017"].hard_cases
    assert cands["art-0001"].hard_cases == ()
    # the affected query was run for both halves, unbounded, excluding nothing
    affected_calls = [p for q, p in fake.queries if q == AFFECTED_ARTICLES_QUERY]
    assert sorted(p["example"] for p in affected_calls) == [False, True]
    assert all(p["exclude"] == [] and p["limit"] >= 100_000 for p in affected_calls)
    # the articles query gets the structural product names for the unnamed check
    (_, params), = [(q, p) for q, p in fake.queries if q == smp.ARTICLES_QUERY]
    assert params["product_names"] == ["aws backup", "veeam agent"]
    assert params["vendors"] == [v.lower() for v in smp.DEFAULT_VENDORS]


async def test_seed_changes_the_sample() -> None:
    a = await smp.select_sample(_fake(), n_total=40, n_heldout=8, seed=1)
    b = await smp.select_sample(_fake(), n_total=40, n_heldout=8, seed=2)
    assert {x.article_id for x in a} != {x.article_id for x in b}


async def test_missing_vendor_raises() -> None:
    with pytest.raises(ValueError, match="Rubrik"):
        await smp.select_sample(_fake(), vendors=("Veeam", "Rubrik"))


def test_apportion() -> None:
    assert smp.apportion(10, {"a": 50, "b": 30, "c": 20}) == {"a": 5, "b": 3, "c": 2}
    floored = smp.apportion(20, {"a": 900, "b": 100, "c": 2}, {"a": 5, "b": 5, "c": 5})
    assert floored["c"] == 2                   # floor capped at the stratum's size
    assert sum(floored.values()) == 20
    assert floored["a"] > floored["b"] >= 5
    assert smp.apportion(100, {"a": 3, "b": 4}) == {"a": 3, "b": 4}  # never above size
    with pytest.raises(ValueError):
        smp.apportion(3, {"a": 10, "b": 10}, {"a": 2, "b": 2})


async def test_write_and_load_roundtrip(tmp_path) -> None:
    sample, counts = await smp.select_sample_with_counts(_fake(), n_total=40, n_heldout=8)
    path = tmp_path / "sample.json"
    smp.write_sample(path, sample, counts, seed=20261017, n_total=40, n_heldout=8,
                     vendors=smp.DEFAULT_VENDORS)
    assert smp.load_sample(path) == sample
    payload = json.loads(path.read_text())
    assert payload["strata"] == counts
    assert payload["train"] == 32 and payload["heldout"] == 8


async def test_strong_tier_articles_are_excluded_and_counted() -> None:
    fake = _fake()
    cands = {c.article_id: c for c in await smp.read_candidates(fake, smp.DEFAULT_VENDORS)}
    assert cands["art-0019"].strong_tier            # dense matrix (production's router)
    assert cands["art-0023"].strong_tier            # a chunk above cheap_max_chunk_tokens
    assert not cands["art-0013"].strong_tier        # a small table stays cheap
    # chunk texts were fetched only for articles that contain a '|'
    (params,) = [p for q, p in fake.queries if q == smp.CHUNK_TEXTS_QUERY]
    rows = {r["id"]: r for r in _rows()}
    assert params["ids"] == sorted(i for i, r in rows.items() if r["has_pipe"])

    sample, counts = await smp.select_sample_with_counts(_fake(), n_total=60, n_heldout=10)
    strong_ids = {r["id"] for r in _rows() if _strong(int(r["id"][4:]))}
    assert not strong_ids & {a.article_id for a in sample}
    per_stratum: Counter = Counter()
    for r in _rows():
        if r["id"] in strong_ids:
            per_stratum[f"{r['vendor']} / {r['product']}"] += 1
    assert {s: c["excluded_strong_tier"] for s, c in counts.items()} == {
        s: per_stratum.get(s, 0) for s in counts}
    assert sum(c["population"] + c["excluded_strong_tier"] for c in counts.values()) == len(rows)


async def test_production_thresholds_are_used() -> None:
    lenient = smp.RoutingThresholds(dense_table_line_ratio=0.99, dense_pipe_count=10_000,
                                    cheap_max_chunk_tokens=10_000)
    cands = await smp.read_candidates(_fake(), smp.DEFAULT_VENDORS, lenient)
    assert not any(c.strong_tier for c in cands)
    s = ExtractSettings(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u",
                        neo4j_password="p", docext_base_url="http://x", docext_read_key="k")
    assert smp.RoutingThresholds.from_settings(s) == smp.RoutingThresholds()
