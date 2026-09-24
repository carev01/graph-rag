import importlib.util
import random
from pathlib import Path

from graph_extract.config import ExtractSettings

_spec = importlib.util.spec_from_file_location(
    "chunk_ab", Path(__file__).parents[2] / "scripts" / "chunk_ab.py")
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

S = ExtractSettings(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                    docext_base_url="https://x", docext_read_key="k")


def _art(i, vendor, n_chunks, tok=300):
    return {"id": f"a{i}", "vendor": vendor, "title": f"T{i}",
            "chunks": [{"t": f"chunk {j} " + "w " * 50, "tok": tok} for j in range(n_chunks)]}


def test_today_episode_count_uses_the_production_builder():
    assert ab.today_episode_count(_art(0, "V", 5), S) == 5


def test_select_prefers_multi_episode_articles_across_vendors():
    sample = [_art(i, v, n) for i, (v, n) in enumerate(
        [("A", 5), ("A", 5), ("A", 1), ("B", 4), ("B", 2), ("C", 6)])]
    picked = ab.select_articles(sample, 3, random.Random(0), S)
    assert {a["vendor"] for a in picked} == {"A", "B", "C"}
    assert all(ab.today_episode_count(a, S) >= 3 for a in picked)


def test_article_cost_uses_the_tier_price():
    assert ab.article_cost(1_000_000, 0, "cheap") == 0.09
    assert ab.article_cost(0, 1_000_000, "strong") == 2.00


def test_summarise_totals_and_ratios():
    rows = [{"episodes": 3, "facts": 10, "entities": 6, "seconds": 30.0, "cost": 0.02},
            {"episodes": 1, "facts": 5, "entities": 4, "seconds": 10.0, "cost": 0.01}]
    s = ab.summarise(rows)
    assert s["episodes"] == 4 and s["facts"] == 15 and s["entities"] == 10
    assert abs(s["cost"] - 0.03) < 1e-9 and s["seconds"] == 40.0
