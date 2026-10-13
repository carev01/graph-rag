# Facet-Balanced Answers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every named side of a comparison is answered, every vendor matching an enumeration is listed, and narrow scopes no longer starve local search — without regressing single-vendor questions.

**Architecture:** A deterministic *question shape* (single / comparison / enumeration) and its *facets* (one `Scope` per named side, or per content vendor) are computed once per request from scope detection. Local, global and DRIFT retrieve per facet when there are two or more facets; enumeration gets its own per-vendor gather → cheap check → synthesis path. Single-facet behaviour stays byte-identical. A free retrieval check gates any paid eval.

**Tech Stack:** Python 3.12, FastAPI, neo4j async driver, graphiti-core 0.30.1, OpenAI-compatible chat clients (`bounded_llm_client`), pytest + pytest-asyncio, `uv`.

**Spec:** `docs/superpowers/specs/2026-10-12-facet-balanced-answers-design.md`

## Global Constraints

- Citations resolve deterministically from fact ids; the LLM never writes a URL (`_finalize_answer` stays on every answer path).
- A failure never reads as absence: a facet whose retrieval raises, or a vendor whose check is unusable after one retry, is reported ("Could not check X."), listed in `degraded_facets`, and logged — never folded into "not documented".
- Never coerce an empty LLM reply: use `usable_content` / `_complete_or_none`, never `content or ""`.
- Single-facet (and unscoped) behaviour and prompts are byte-identical to today's.
- An explicit scope is the caller's instruction: one facet, shape `single`. `scope=none` gives `single` with an empty facet.
- A *detected* scope that grounds nothing is still relaxed and re-run unscoped; an explicit scope never is.
- Settings (in `ExtractSettings`): `local_min_per_facet = 6`, `local_scoped_fetch_cap = 450`, `global_facet_top_n = 3`, `enum_communities_per_vendor = 2`, `enum_facts_per_vendor = 8`.
- Lint gate is `uv run ruff check src tests` (E702: no semicolons in test fakes; E402: imports at top). Types: `uv run mypy src`. Tests: `uv run --extra dev pytest -m "not live"`.
- No paid run (LLM calls against real endpoints) inside any task. Tasks 1–9 are hermetic; Task 10 is the controller's, with user confirmation.

## File Structure

- Create `src/answer_api/shape.py` — `Facet`, `QuestionShape`, `classify_shape`, `group_by_facet`, `could_not_check_note`. Pure; depends on `scope` and `attribution`.
- Modify `src/answer_api/scope.py` — `ScopeResolver.detect_mentions`, `vendor_of_source`, `content_vendors` (loaded in `load`).
- Modify `src/answer_api/search.py` — adaptive over-fetch for scoped searches.
- Modify `src/answer_api/synthesize.py` — per-facet local answer.
- Modify `src/answer_api/global_search.py` — `top_n` on `shortlist_communities`; per-facet global search and grouped reduce.
- Modify `src/answer_api/drift.py` — per-facet primer, facet-scoped follow-ups, grouped synthesis.
- Create `src/answer_api/enumeration.py` — the enumeration path.
- Modify `src/answer_api/router.py`, `src/answer_api/app.py`, `src/answer_api/eval_router.py` — shape wiring and envelope.
- Modify `src/answer_api/eval_router.py` (metrics) and `tests/unit/test_router_golden_schema.py` — facet coverage, enumeration completeness, `expected_vendors`.
- Create `src/answer_api/eval_retrieval.py` — free retrieval check CLI.
- Modify `src/graph_extract/config.py` — the five settings.
- Modify `CLAUDE.md` — module layout lines for the new modules.

---

### Task 1: Question shape and facets

**Files:**
- Create: `src/answer_api/shape.py`
- Modify: `src/answer_api/scope.py` (class `ScopeResolver`: `__init__`, new `detect_mentions`, `vendor_of_source`, `load`)
- Modify: `src/graph_extract/config.py` (after `global_scope_min_share`)
- Test: `tests/unit/test_shape.py`

**Interfaces:**
- Consumes: `Scope`, `ScopeResolver`, `_CROSS_VENDOR` from `answer_api.scope`; `in_scope` from `answer_api.attribution`.
- Produces:
  - `ScopeResolver(vendors, products, aliases, sources=None, content_vendors=())`; attribute `content_vendors: tuple[str, ...]`
  - `ScopeResolver.detect_mentions(q: str) -> list[tuple[str, Scope]]` — one `(label, Scope(source="detected"))` per regex mention, in question order; ignores `_CROSS_VENDOR`.
  - `ScopeResolver.vendor_of_source(name: str) -> str | None`
  - `@dataclass(frozen=True) class Facet: label: str; scope: Scope`
  - `@dataclass(frozen=True) class QuestionShape: kind: Literal["single","comparison","enumeration"]; facets: tuple[Facet, ...]`
  - `classify_shape(q: str, resolver: ScopeResolver, scope: Scope, *, scope_disabled: bool = False) -> QuestionShape`
  - `group_by_facet(items: list[dict], facets: Sequence[Facet]) -> list[tuple[Facet, list[dict]]]` — each item (has `"sources"`) goes to the FIRST facet whose scope it is in; unmatched items are dropped from the groups (callers keep their own full list).
  - `could_not_check_note(labels: Sequence[str]) -> str` — `""` for no labels, else `"Could not check: A, B."`
  - Settings: `local_min_per_facet: int = 6`, `local_scoped_fetch_cap: int = 450`, `global_facet_top_n: int = 3`, `enum_communities_per_vendor: int = 2`, `enum_facts_per_vendor: int = 8`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_shape.py
"""Question shape and facets (spec §2)."""
from answer_api.scope import Scope, ScopeResolver
from answer_api.shape import (Facet, classify_shape, could_not_check_note,
                              group_by_facet)

_PRODUCTS = [("Veeam Backup & Replication", "Veeam"), ("Veeam ONE", "Veeam"),
             ("Veeam Backup for Microsoft 365", "Veeam"), ("AWS Backup", "AWS"),
             ("Azure Backup", "Microsoft"), ("Cohesity DataProtect", "Cohesity")]
_SOURCES = [("Veeam Agent for Linux", "Veeam Backup & Replication"),
            ("Veeam Agent for Mac", "Veeam Backup & Replication"),
            ("User Guide", "Veeam ONE")]
_ALIASES = {"Veeam Agent": ["Veeam Agent for Linux", "Veeam Agent for Mac"]}


def _resolver(content=("AWS", "Cohesity", "Microsoft", "Veeam")):
    return ScopeResolver(["AWS", "Cohesity", "Microsoft", "Veeam"], _PRODUCTS,
                         _ALIASES, _SOURCES, content_vendors=content)


def _shape(q, **kw):
    r = _resolver()
    return classify_shape(q, r, r.resolve(q), **kw)


def test_single_product_question_is_single_with_the_detected_scope():
    q = "How does Veeam Backup & Replication detect malware?"
    s = _shape(q)
    assert s.kind == "single"
    assert s.facets == (Facet("Veeam Backup & Replication", _resolver().resolve(q)),)


def test_unscoped_question_is_single_with_an_empty_facet():
    s = _shape("How do I restore a file?")
    assert s.kind == "single"
    assert s.facets == (Facet("", Scope()),)


def test_two_named_sides_make_a_comparison_in_question_order():
    s = _shape("Compare how Veeam hardened repositories and AWS Backup Vault Lock "
               "protect backups.")
    assert s.kind == "comparison"
    assert [f.label for f in s.facets] == ["Veeam", "AWS Backup"]
    assert s.facets[0].scope == Scope(vendors=("Veeam",), source="detected")
    assert s.facets[1].scope == Scope(products=("AWS Backup",), source="detected")


def test_a_multi_target_alias_is_one_facet():
    s = _shape("What should I consider for Veeam Agent backups?")
    assert s.kind == "single"
    assert s.facets[0].label == "Veeam Agent"
    assert set(s.facets[0].scope.sources) == {"Veeam Agent for Linux",
                                              "Veeam Agent for Mac"}


def test_a_vendor_mention_is_subsumed_by_its_own_product():
    s = _shape("In Veeam's Veeam ONE, how are alarms approved?")
    assert s.kind == "single"
    assert s.facets[0].label == "Veeam ONE"


def test_two_products_of_one_vendor_are_a_comparison():
    s = _shape("Compare Veeam ONE with Veeam Backup & Replication reporting.")
    assert s.kind == "comparison"
    assert [f.label for f in s.facets] == ["Veeam ONE", "Veeam Backup & Replication"]


def test_cross_vendor_wording_is_an_enumeration_over_content_vendors():
    s = _shape("Which backup vendors support immutable backups?")
    assert s.kind == "enumeration"
    assert [f.label for f in s.facets] == ["AWS", "Cohesity", "Microsoft", "Veeam"]
    assert s.facets[2].scope == Scope(vendors=("Microsoft",), source="detected")


def test_cross_vendor_wording_naming_a_workload_is_still_an_enumeration():
    s = _shape("Which vendors can protect Azure Backup vaults?")
    assert s.kind == "enumeration"


def test_explicit_scope_is_single_whatever_the_wording():
    r = _resolver()
    explicit = r.resolve("Which vendors support immutability?", vendors=["AWS"])
    s = classify_shape("Which vendors support immutability?", r, explicit)
    assert s.kind == "single"
    assert s.facets == (Facet("AWS", explicit),)


def test_scope_disabled_is_single_unscoped():
    r = _resolver()
    s = classify_shape("Compare Veeam and AWS Backup", r, Scope(), scope_disabled=True)
    assert s == classify_shape("x", r, Scope(), scope_disabled=True)
    assert s.kind == "single" and s.facets == (Facet("", Scope()),)


def test_enumeration_without_content_vendors_falls_back_to_single():
    r = _resolver(content=())
    s = classify_shape("Which vendors support immutability?", r, Scope())
    assert s.kind == "single"


def test_vendor_of_source():
    r = _resolver()
    assert r.vendor_of_source("Veeam Agent for Linux") == "Veeam"
    assert r.vendor_of_source("nope") is None


def test_group_by_facet_assigns_each_item_to_its_first_matching_facet():
    facets = [Facet("Veeam", Scope(vendors=("Veeam",))),
              Facet("AWS Backup", Scope(products=("AWS Backup",)))]
    items = [{"id": 1, "sources": [{"vendor": "AWS", "product": "AWS Backup"}]},
             {"id": 2, "sources": [{"vendor": "Veeam", "product": "Veeam ONE"}]},
             {"id": 3, "sources": [{"vendor": "Cohesity", "product": "X"}]}]
    groups = group_by_facet(items, facets)
    assert [(f.label, [i["id"] for i in xs]) for f, xs in groups] == [
        ("Veeam", [2]), ("AWS Backup", [1])]


def test_could_not_check_note():
    assert could_not_check_note([]) == ""
    assert could_not_check_note(["Cohesity", "AWS"]) == "Could not check: Cohesity, AWS."
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_shape.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'answer_api.shape'`.

- [ ] **Step 3: Add the settings**

In `src/graph_extract/config.py`, directly after the line `global_scope_min_share: float = 0.5`, insert:

```python
    # --- facet-balanced answers (spec 2026-10-12) ---
    # Comparison: each facet's local search gets max(this, ceil(k / facets)) facts.
    local_min_per_facet: int = 6
    # Scoped local search re-fetches at k*3, k*10, k*30 until k edges survive the
    # scope filter, never above this (the Veeam Agent scope kept 7 of 45, 2026-10-12).
    local_scoped_fetch_cap: int = 450
    # Comparison: communities per facet reaching the map step (single keeps rerank_top_n).
    global_facet_top_n: int = 3
    # Enumeration: evidence gathered per content vendor before its cheap check.
    enum_communities_per_vendor: int = 2
    enum_facts_per_vendor: int = 8
```

- [ ] **Step 4: Extend `ScopeResolver`**

In `src/answer_api/scope.py`:

(a) Add this query after `_EPISODES_AMONG`:

```python
# Vendors whose documentation has extracted content (at least one episode): the set
# an enumeration question checks (spec 2026-10-12 §2). EXISTS short-circuits per
# source, so this stays cheap at bootstrap scale.
_CONTENT_VENDORS = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "AND EXISTS { MATCH (s)-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(:Episodic) } "
    "RETURN DISTINCT v.name AS vendor ORDER BY vendor")
```

(b) Change the `__init__` signature and store the new state. Replace

```python
    def __init__(self, vendors: list[str], products: list[tuple[str, str]],
                 aliases: dict[str, str | list[str]],
                 sources: list[tuple[str, str]] | None = None) -> None:
```

with

```python
    def __init__(self, vendors: list[str], products: list[tuple[str, str]],
                 aliases: dict[str, str | list[str]],
                 sources: list[tuple[str, str]] | None = None,
                 content_vendors: tuple[str, ...] | list[str] = ()) -> None:
```

and directly after `self._vendor_of = {p: v for p, v in products}` add:

```python
        self._product_of_source = {n: p for n, p in sources or []}
        self.content_vendors: tuple[str, ...] = tuple(content_vendors)
```

(c) Add these two methods directly after `vendor_of`:

```python
    def vendor_of_source(self, name: str) -> str | None:
        product = self._product_of_source.get(name)
        return self._vendor_of.get(product) if product else None

    def detect_mentions(self, q: str) -> list[tuple[str, Scope]]:
        """One (label, Scope) per mention, in question order -- the grouping
        `detect` flattens. An alias with several targets is ONE mention ("Veeam
        Agent" -> its guides). Ignores the cross-vendor rule: shape.classify_shape
        decides enumeration before it asks for mentions."""
        if self._re is None:
            return []
        out: list[tuple[str, Scope]] = []
        for m in self._re.finditer(q):
            hits = self._terms[m.group(1).lower()]
            found: dict[str, list[str]] = {"vendor": [], "product": [], "source": []}
            for kind, name in hits:
                if name not in found[kind]:
                    found[kind].append(name)
            label = hits[0][1] if len(hits) == 1 else m.group(1)
            out.append((label, Scope(tuple(found["vendor"]), tuple(found["product"]),
                                     "detected", tuple(found["source"]))))
        return out
```

(d) In `load`, replace `return cls(vendors, pairs, aliases, sources)` with:

```python
        content, _, _ = await driver.execute_query(_CONTENT_VENDORS)
        return cls(vendors, pairs, aliases, sources,
                   content_vendors=tuple(r["vendor"] for r in content))
```

- [ ] **Step 5: Create `src/answer_api/shape.py`**

```python
"""Question shape and facets (spec 2026-10-12 §2).

A question is SINGLE (zero or one named side), a COMPARISON (two or more named
sides), or an ENUMERATION (cross-vendor wording with no explicit scope: every vendor
with content is checked). Each side is a FACET -- a label plus its own Scope -- and
every retrieval mode balances across facets. Deterministic: no LLM.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from answer_api.attribution import in_scope
from answer_api.scope import _CROSS_VENDOR, Scope, ScopeResolver


@dataclass(frozen=True)
class Facet:
    label: str
    scope: Scope


@dataclass(frozen=True)
class QuestionShape:
    kind: Literal["single", "comparison", "enumeration"]
    facets: tuple[Facet, ...]


def _label_of(scope: Scope) -> str:
    return ", ".join((*scope.vendors, *scope.products, *scope.sources))


def _vendors_covered(scope: Scope, resolver: ScopeResolver) -> set[str]:
    out = {resolver.vendor_of(p) for p in scope.products}
    out |= {resolver.vendor_of_source(s) for s in scope.sources}
    return {v for v in out if v}


def _facets_from_mentions(resolver: ScopeResolver, q: str) -> list[Facet]:
    facets: list[Facet] = []
    for label, scope in resolver.detect_mentions(q):
        if all(f.scope != scope for f in facets):
            facets.append(Facet(label, scope))
    # A bare vendor facet is subsumed by a product/source facet of that vendor:
    # "Veeam's Veeam ONE" is one side, not two.
    covered = set().union(*(_vendors_covered(f.scope, resolver) for f in facets)) \
        if facets else set()
    return [f for f in facets
            if not (f.scope.vendors and not f.scope.products and not f.scope.sources
                    and set(f.scope.vendors) <= covered)]


def classify_shape(q: str, resolver: ScopeResolver, scope: Scope, *,
                   scope_disabled: bool = False) -> QuestionShape:
    if scope_disabled:
        return QuestionShape("single", (Facet("", Scope()),))
    if scope.source == "explicit":
        return QuestionShape("single", (Facet(_label_of(scope), scope),))
    if _CROSS_VENDOR.search(q) and resolver.content_vendors:
        return QuestionShape("enumeration", tuple(
            Facet(v, Scope(vendors=(v,), source="detected"))
            for v in resolver.content_vendors))
    facets = _facets_from_mentions(resolver, q)
    if len(facets) >= 2:
        return QuestionShape("comparison", tuple(facets))
    if len(facets) == 1:
        # The single facet carries the caller's (detected) scope unchanged, so the
        # single path is exactly today's.
        return QuestionShape("single", (Facet(facets[0].label, scope),))
    return QuestionShape("single", (Facet("", scope),))


def group_by_facet(items: list[dict], facets: Sequence[Facet]
                   ) -> list[tuple[Facet, list[dict]]]:
    groups: list[tuple[Facet, list[dict]]] = [(f, []) for f in facets]
    for item in items:
        for f, bucket in groups:
            if in_scope(item.get("sources", []), f.scope):
                bucket.append(item)
                break
    return groups


def could_not_check_note(labels: Sequence[str]) -> str:
    return f"Could not check: {', '.join(labels)}." if labels else ""
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_shape.py tests/unit/test_scope.py -q`
Expected: PASS (the existing scope tests still pass: `content_vendors` defaults to `()`).

- [ ] **Step 7: Lint, types, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/shape.py src/answer_api/scope.py src/graph_extract/config.py tests/unit/test_shape.py
git commit -m "feat(answer): question shape and facets -- single, comparison, enumeration"
```

---

### Task 2: Adaptive over-fetch for scoped local search

**Files:**
- Modify: `src/answer_api/search.py` (`search_local`)
- Test: `tests/unit/test_search_overfetch.py`

**Interfaces:**
- Consumes: `scope_episode_uuids`, `_retrieve_edges` (same module).
- Produces: `search_local(graphiti, driver, *, q, k=10, scope=None, include_invalid=False, group_id, center_node_uuid=None, fetch_cap: int = 450) -> dict` — unscoped searches fetch once at `max(k*3, k)`; scoped searches walk the limits `(k*3, k*10, k*30)` (each capped at `fetch_cap`, duplicates dropped) and stop at the first limit that leaves `>= k` edges after filtering, or at the last limit.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_search_overfetch.py
"""Scoped local search re-fetches until k edges survive the scope filter."""
from types import SimpleNamespace

import pytest

import answer_api.search as search_mod
from answer_api.scope import Scope

pytestmark = pytest.mark.asyncio


def _edge(i, in_scope):
    return SimpleNamespace(uuid=f"e{i}", fact=f"f{i}", invalid_at=None, valid_at=None,
                           episodes=[f"in{i}" if in_scope else f"out{i}"])


class _Graph:
    """Every 5th edge is in scope."""

    def __init__(self):
        self.limits = []

    async def _search(self, q, config, group_ids, center_node_uuid=None):
        self.limits.append(config.limit)
        return SimpleNamespace(edges=[_edge(i, i % 5 == 0) for i in range(config.limit)])


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    async def _allowed(driver, scope, candidates=None):
        return {c for c in candidates if c.startswith("in")}

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {}

    monkeypatch.setattr(search_mod, "scope_episode_uuids", _allowed)
    monkeypatch.setattr(search_mod, "Provenance", _Prov)


async def test_scoped_search_refetches_until_k_survive():
    g = _Graph()
    res = await search_mod.search_local(g, None, q="q", k=15, group_id="g",
                                        scope=Scope(vendors=("Veeam",)))
    assert g.limits == [45, 150]          # 9 survive at 45, 30 at 150 -> stop
    assert res["count"] == 15


async def test_scoped_search_stops_at_the_cap():
    g = _Graph()
    res = await search_mod.search_local(g, None, q="q", k=15, group_id="g",
                                        scope=Scope(vendors=("Veeam",)), fetch_cap=100)
    assert g.limits == [45, 100]
    assert res["count"] == 15             # 20 survive at 100, trimmed to k


async def test_unscoped_search_fetches_once():
    g = _Graph()
    await search_mod.search_local(g, None, q="q", k=15, group_id="g")
    assert g.limits == [45]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_search_overfetch.py -q`
Expected: FAIL — `test_scoped_search_refetches_until_k_survive` sees `[45]`; `fetch_cap` is an unexpected keyword.

- [ ] **Step 3: Implement**

Replace the body of `search_local` in `src/answer_api/search.py` up to (not including) `edges = edges[:k]` with:

```python
async def search_local(graphiti, driver, *, q, k=10, scope: Scope | None = None,
                       include_invalid=False, group_id, center_node_uuid=None,
                       fetch_cap: int = 450):
    scoped = scope is not None and not scope.is_empty()
    # Over-fetch so post-filters (validity, scope) still leave ~k results. A narrow
    # scope can filter out most of one fetch (the six Veeam Agent guides kept 7 of
    # 45, 2026-10-12), so a scoped search re-fetches wider until k survive.
    first = max(k * 3, k)
    limits = [first] if not scoped else list(dict.fromkeys(
        min(n, fetch_cap) for n in (first, k * 10, k * 30)))
    edges: list = []
    for limit in limits:
        edges = await _retrieve_edges(graphiti, q, fetch_limit=limit,
                                      group_id=group_id,
                                      center_node_uuid=center_node_uuid)
        if not include_invalid:
            edges = [e for e in edges if is_current(getattr(e, "invalid_at", None))]
        if scoped:
            assert scope is not None
            allowed = await scope_episode_uuids(
                driver, scope, candidates=[u for e in edges for u in (e.episodes or [])])
            edges = [e for e in edges if allowed.intersection(e.episodes or [])]
        if len(edges) >= k:
            break
```

(The lines from `edges = edges[:k]` onward are unchanged.)

- [ ] **Step 4: Run to verify pass, including the existing local tests**

Run: `uv run --extra dev pytest tests/unit/test_search_overfetch.py tests/unit/test_search_local_filtering.py tests/unit/test_scoped_local.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/search.py tests/unit/test_search_overfetch.py
git commit -m "fix(search): scoped local search re-fetches until k edges survive the scope filter"
```

---

### Task 3: Per-facet local answers

**Files:**
- Modify: `src/answer_api/synthesize.py` (`answer_local`; new `_FACET_PROMPT`, `_facet_facts_block`)
- Test: `tests/unit/test_local_facets.py`

**Interfaces:**
- Consumes: `Facet` (Task 1), `search_local(..., fetch_cap=...)` (Task 2), `could_not_check_note` (Task 1).
- Produces: `answer_local(graphiti, driver, synth_client, synth_model, *, q, k=15, scope=None, group_id, facets: Sequence[Facet] | None = None, min_per_facet: int = 6, fetch_cap: int = 450) -> dict`. With `facets` of length >= 2: one `search_local` per facet at `k_facet = max(min_per_facet, ceil(k / n))`, merged in facet order and de-duplicated by `fact_uuid`; facet-grouped prompt; result dict additionally carries `"degraded_facets": [labels]`. With `facets` None or length < 2: exactly today's code path and prompt.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_local_facets.py
"""Local answers retrieve per facet and group the facts (spec §3.1)."""
from types import SimpleNamespace

import pytest

import answer_api.synthesize as synth
from answer_api.scope import Scope
from answer_api.shape import Facet

pytestmark = pytest.mark.asyncio

_V = Facet("Veeam", Scope(vendors=("Veeam",)))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",)))


def _fact(u, vendor, product):
    return {"fact": f"fact {u}", "fact_uuid": u, "valid_at": None, "invalid_at": None,
            "sources": [{"vendor": vendor, "product": product, "article_id": u}]}


class _Synth:
    def __init__(self, reply="Veeam does X [1]. AWS does Y [2]."):
        self.prompts = []
        self.reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.reply), finish_reason="stop")])


@pytest.fixture
def searches(monkeypatch):
    calls = []

    async def _search(graphiti, driver, *, q, k, scope=None, group_id,
                      fetch_cap=450, **kw):
        calls.append((scope, k))
        if scope == _V.scope:
            rows = [_fact("v1", "Veeam", "Veeam ONE"), _fact("shared", "Veeam", "Veeam ONE")]
        elif scope == _A.scope:
            rows = [_fact("a1", "AWS", "AWS Backup"), _fact("shared", "Veeam", "Veeam ONE")]
        else:
            rows = [_fact("u1", "Veeam", "Veeam ONE")]
        return {"query": q, "count": len(rows), "results": rows}

    monkeypatch.setattr(synth.search_mod, "search_local", _search)
    return calls


async def test_each_facet_searches_with_its_own_scope_and_share_of_k(searches):
    await synth.answer_local(None, None, _Synth(), "m", q="q", k=15, group_id="g",
                             facets=[_V, _A], min_per_facet=6)
    assert searches == [(_V.scope, 8), (_A.scope, 8)]


async def test_facts_are_grouped_under_facet_headings_and_deduplicated(searches):
    s = _Synth()
    res = await synth.answer_local(None, None, s, "m", q="q", k=15, group_id="g",
                                   facets=[_V, _A])
    p = s.prompts[0]
    assert p.index("=== Veeam ===") < p.index("[1]") < p.index("=== AWS Backup ===")
    assert p.count("fact shared") == 1
    assert "Address EVERY side" in p and "Veeam; AWS Backup" in p
    assert res["retrieved"] == 3 and res["degraded_facets"] == []


async def test_a_facet_with_no_facts_is_shown_empty_not_dropped(searches, monkeypatch):
    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450, **kw):
        rows = [_fact("v1", "Veeam", "Veeam ONE")] if scope == _V.scope else []
        return {"query": q, "count": len(rows), "results": rows}
    monkeypatch.setattr(synth.search_mod, "search_local", _search)
    s = _Synth("Veeam does X [1].")
    await synth.answer_local(None, None, s, "m", q="q", group_id="g", facets=[_V, _A])
    assert "=== AWS Backup ===\n(no facts retrieved)" in s.prompts[0]


async def test_a_failed_facet_is_reported_never_read_as_absence(monkeypatch):
    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450, **kw):
        if scope == _A.scope:
            raise RuntimeError("neo4j down")
        return {"query": q, "count": 1, "results": [_fact("v1", "Veeam", "Veeam ONE")]}
    monkeypatch.setattr(synth.search_mod, "search_local", _search)
    s = _Synth("Veeam does X [1].")
    res = await synth.answer_local(None, None, s, "m", q="q", group_id="g",
                                   facets=[_V, _A])
    assert res["degraded_facets"] == ["AWS Backup"]
    assert res["answer"].endswith("Could not check: AWS Backup.")
    assert "=== AWS Backup ===" not in s.prompts[0]


async def test_single_facet_path_is_unchanged(searches):
    s = _Synth("Answer [1].")
    res = await synth.answer_local(None, None, s, "m", q="q", group_id="g",
                                   facets=[_V])
    assert searches == [(None, 15)]
    assert "===" not in s.prompts[0]
    assert "degraded_facets" not in res
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_local_facets.py -q`
Expected: FAIL — `answer_local() got an unexpected keyword argument 'facets'`.

- [ ] **Step 3: Implement**

In `src/answer_api/synthesize.py`:

(a) Add imports at the top (with the other `answer_api` imports):

```python
import asyncio
import math
from collections.abc import Sequence

from answer_api.shape import Facet, could_not_check_note
```

(b) After `_REFUSAL = ...`, add:

```python
# Comparison prompt (spec 2026-10-12 §3.1): facts arrive grouped under one heading
# per facet, and every facet must be answered. Saying a facet is NOT covered is
# required here -- unlike the global reducer's "absence of evidence" rule -- because
# a comparison that silently drops one side reads as a complete answer.
_FACET_PROMPT = (
    "You are answering a question about backup products using ONLY the numbered "
    "facts below, grouped under a heading for each side of the question. "
    + ATTRIBUTION_RULES +
    "The question has these sides: {labels}. Address EVERY side, in that order. If a "
    "side shows \"(no facts retrieved)\", say in one sentence that the retrieved "
    "documentation does not cover it, and do not guess. Cite every claim inline with "
    "its [N] marker. Do NOT use outside knowledge. Do NOT write any URL or link. If no "
    "side has facts that answer the question, reply exactly: "
    "\"I don't have enough information to answer that from the available sources.\"\n\n"
    "FACTS:\n{facts}\n\nQUESTION: {q}\n\nAnswer:"
)


def _facet_facts_block(groups: list[tuple[str, list[int]]],
                       marker_map: dict[int, dict]) -> str:
    parts = []
    for label, markers in groups:
        lines = [fact_line(i, marker_map[i]["fact"], marker_map[i]["sources"])
                 for i in markers] or ["(no facts retrieved)"]
        parts.append(f"=== {label} ===\n" + "\n".join(lines))
    return "\n\n".join(parts)
```

(c) Change the `answer_local` signature to

```python
async def answer_local(graphiti, driver, synth_client, synth_model, *,
                       q, k=15, scope: Scope | None = None, group_id,
                       facets: Sequence[Facet] | None = None, min_per_facet: int = 6,
                       fetch_cap: int = 450) -> dict:
```

and insert, as the first statements of the body (after the docstring):

```python
    if facets is not None and len(facets) >= 2:
        return await _answer_local_facets(
            graphiti, driver, synth_client, synth_model, q=q, k=k, group_id=group_id,
            facets=facets, min_per_facet=min_per_facet, fetch_cap=fetch_cap)
```

In the existing single-facet `search_mod.search_local(...)` call inside `answer_local`, add `fetch_cap=fetch_cap` (the default 450 equals the setting, so behaviour is unchanged unless the setting changes).

(d) Add this function directly after `answer_local`:

```python
async def _answer_local_facets(graphiti, driver, synth_client, synth_model, *, q, k,
                               group_id, facets: Sequence[Facet], min_per_facet: int,
                               fetch_cap: int) -> dict:
    k_facet = max(min_per_facet, math.ceil(k / len(facets)))
    found = await asyncio.gather(*[
        search_mod.search_local(graphiti, driver, q=q, k=k_facet, scope=f.scope,
                                group_id=group_id, fetch_cap=fetch_cap)
        for f in facets], return_exceptions=True)
    marker_map: dict[int, dict] = {}
    seen: set[str] = set()
    groups: list[tuple[str, list[int]]] = []
    degraded: list[str] = []
    for f, res in zip(facets, found):
        if isinstance(res, BaseException):
            # Never read as absence: the facet is reported, not shown as empty.
            logger.warning("local facet %r retrieval failed: %s", f.label, res)
            degraded.append(f.label)
            continue
        markers = []
        for r in res["results"]:
            if r["fact_uuid"] in seen:
                continue
            seen.add(r["fact_uuid"])
            n = len(marker_map) + 1
            marker_map[n] = r
            markers.append(n)
        groups.append((f.label, markers))
    note = could_not_check_note(degraded)
    if not marker_map:
        return {"query": q, "answer": (_REFUSAL + (" " + note if note else "")),
                "citations": [], "retrieved": 0, "cited": 0, "applies_to": [],
                "degraded_facets": degraded}
    prompt = _FACET_PROMPT.format(labels="; ".join(lab for lab, _ in groups),
                                  facts=_facet_facts_block(groups, marker_map), q=q)
    raw = await _complete_or_none(synth_client, synth_model, prompt, max_tokens=3000)
    if raw is None:
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "retrieved": len(marker_map), "cited": 0, "applies_to": [],
                "degraded_facets": degraded}
    answer, cited = _finalize_answer(raw, marker_map)
    if note:
        answer = f"{answer}\n\n{note}"
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": marker_map[m].get("valid_at"),
                  "invalid_at": marker_map[m].get("invalid_at"),
                  "sources": marker_map[m]["sources"]} for m in cited]
    return {"query": q, "answer": answer, "citations": citations,
            "retrieved": len(marker_map), "cited": len(cited),
            "applies_to": applies_to(citations), "degraded_facets": degraded}
```

Note: the refusal-with-no-facts case above returns `retrieved: 0`, so the router's local→drift fallback still fires exactly as it does for a single facet.

- [ ] **Step 4: Run to verify pass**

Run: `uv run --extra dev pytest tests/unit/test_local_facets.py tests/unit/test_scoped_local.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/synthesize.py tests/unit/test_local_facets.py
git commit -m "feat(answer): local answers retrieve per facet and address every side"
```

---

### Task 4: Per-facet global search

**Files:**
- Modify: `src/answer_api/global_search.py` (`shortlist_communities`, `global_search`; new `_REDUCE_FACETS_PROMPT`, `_render_facet_blocks`)
- Test: `tests/unit/test_global_facets.py`

**Interfaces:**
- Consumes: `Facet`, `could_not_check_note` (Task 1); existing `shortlist_communities`, `map_report`, `_render_blocks`, `_fact_texts`, `_in_scope_ci`.
- Produces:
  - `shortlist_communities(..., top_n: int | None = None)` — when set, replaces `settings.rerank_top_n` in `_apply_rerank` and `k` in the unreranked path.
  - `global_search(..., facets: Sequence[Facet] | None = None)` — with >= 2 facets: one shortlist per facet (`scope=f.scope`, `top_n=settings.global_facet_top_n`, `k=settings.global_facet_top_n`); a community already taken by an earlier facet is skipped for later ones; each map result's fact ids are filtered by its OWN facet's scope; reduce gets facet-grouped blocks; result carries `"degraded_facets"`. Fewer than 2 facets: unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_global_facets.py
"""Global search shortlists per facet and groups the reduce input (spec §3.2)."""
from types import SimpleNamespace

import pytest

import answer_api.global_search as gs
from answer_api.scope import Scope
from answer_api.shape import Facet
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_V = Facet("Veeam", Scope(vendors=("Veeam",)))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",)))
_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _hit(cid, facts):
    return gs.CommunityHit(community_id=cid, title=f"T{cid}", summary="s", level=1,
                           rating=5.0, cited_fact_uuids=facts, full_report="[]",
                           similarity=0.9)


_SRC = {"v1": [{"vendor": "Veeam", "product": "Veeam ONE", "article_id": "a"}],
        "a1": [{"vendor": "AWS", "product": "AWS Backup", "article_id": "b"}]}


class _Reducer:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Veeam X [1]. AWS Y [2]."), finish_reason="stop")])


@pytest.fixture
def stubs(monkeypatch):
    calls = []

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        calls.append((scope, top_n, k))
        if scope == _V.scope:
            return [_hit("c1", ["v1", "a1"]), _hit("shared", ["v1"])]
        return [_hit("shared", ["v1"]), _hit("c2", ["a1"])]

    async def _map(client, model, q, hit):
        return gs.MapResult(community_id=hit.community_id, title=hit.title,
                            relevance=None, key_points=["p"],
                            fact_ids=list(hit.cited_fact_uuids))

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {u: {"sources": _SRC[u]} for u in uuids if u in _SRC}

    async def _texts(driver, group_id, uuids):
        return {u: f"text {u}" for u in uuids}

    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    monkeypatch.setattr(gs, "map_report", _map)
    monkeypatch.setattr(gs, "Provenance", _Prov)
    monkeypatch.setattr(gs, "_fact_texts", _texts)
    return calls


async def test_each_facet_gets_its_own_shortlist(stubs):
    await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1, k=10,
                           group_id="g", settings=_S, facets=[_V, _A])
    assert stubs == [(_V.scope, 3, 3), (_A.scope, 3, 3)]


async def test_map_output_is_filtered_by_its_own_facet_and_communities_dedup(stubs):
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A])
    p = r.prompts[0]
    # c1 (Veeam facet) keeps only its Veeam fact; "shared" is taken by Veeam first.
    assert p.index("=== Veeam ===") < p.index("=== AWS Backup ===")
    veeam_part, aws_part = p.split("=== AWS Backup ===")
    assert "text v1" in veeam_part and "text a1" not in veeam_part
    assert "text a1" in aws_part
    assert [c["community_id"] for c in res["communities_used"]] == ["c1", "shared", "c2"]
    assert "Address EVERY side" in p
    assert res["degraded_facets"] == []


async def test_a_facet_whose_shortlist_fails_is_reported(stubs, monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        if scope == _A.scope:
            raise RuntimeError("rerank down")
        return [_hit("c1", ["v1"])]
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    res = await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1,
                                 k=10, group_id="g", settings=_S, facets=[_V, _A])
    assert res["degraded_facets"] == ["AWS Backup"]
    assert res["answer"].endswith("Could not check: AWS Backup.")


async def test_single_facet_is_unchanged(stubs):
    await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1, k=10,
                           group_id="g", settings=_S, facets=[_V])
    assert stubs == [(None, None, 10)]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_global_facets.py -q`
Expected: FAIL — `global_search() got an unexpected keyword argument 'facets'`.

- [ ] **Step 3: Add `top_n` to `shortlist_communities`**

Change its signature to end with `scope: Scope | None = None, top_n: int | None = None) -> list[CommunityHit]:`. In the unreranked branch replace both `k` uses with `k_eff`, defined as the first line of the function body:

```python
    k_eff = top_n if top_n is not None else k
```

(so `pool_k = max(k_eff, settings.global_scope_candidates) if (scoped and settings) else k_eff` and `return candidates[:k_eff]`). In the reranked branch replace `top_n=settings.rerank_top_n` in the `_apply_rerank` call with `top_n=top_n if top_n is not None else settings.rerank_top_n`.

- [ ] **Step 4: Add the facet reduce prompt and renderer**

Add imports `from collections.abc import Sequence` and `from answer_api.shape import Facet, could_not_check_note`. After `_REDUCE_PROMPT`, add:

```python
# Comparison reducer (spec 2026-10-12 §3.2). Same rules as _REDUCE_PROMPT, except a
# side with no findings MUST be named as uncovered: a comparison that silently drops
# one side reads as complete.
_REDUCE_FACETS_PROMPT = (
    "Answer the QUESTION by synthesizing these community findings, which are grouped "
    "under a heading for each side of the question. Each finding is shown as a marker "
    "[N] followed by the fact it refers to.\n"
    + ATTRIBUTION_RULES +
    "Rules:\n"
    "- The question has these sides: {labels}. Address EVERY side, in that order. If a "
    "side shows \"(no findings)\", say in one sentence that the retrieved "
    "documentation does not cover it, and do not guess.\n"
    "- Cite every claim with the [N] marker of the fact it rests on. A sentence with "
    "no marker is not allowed.\n"
    "- Write each marker individually: [1] [2] [3]. Never write a range or span such "
    "as [1]-[3]; only the markers you write out are cited.\n"
    "- Use ONLY these findings. Do NOT use outside knowledge.\n"
    "- Do NOT write any URL.\n"
    "- Let the evidence set the length. Say what the findings support and then stop.\n"
    "- If no side has findings that support an answer, reply exactly: "
    "\"" + _REFUSAL + "\"\n\nQUESTION: {q}\n\nFINDINGS:\n{blocks}\n\nAnswer:"
)


def _render_facet_blocks(facet_results: list[tuple[str, list[MapResult]]],
                         fact_to_marker: dict[str, int], texts: dict[str, str],
                         sources: dict[str, list[dict]]) -> str:
    parts = []
    for label, results in facet_results:
        blocks = _render_blocks(results, fact_to_marker, texts, sources)
        parts.append(f"=== {label} ===\n" + ("\n\n".join(blocks) or "(no findings)"))
    return "\n\n".join(parts)
```

- [ ] **Step 5: Add the facet branch to `global_search`**

Change the signature to add `facets: Sequence[Facet] | None = None` after `scope`. As the first statement of the body, add:

```python
    if facets is not None and len(facets) >= 2:
        return await _global_search_facets(
            driver, embedder, map_client, map_model, synth_client, synth_model, q=q,
            level=level, group_id=group_id, settings=settings, facets=facets)
```

Add this function directly after `global_search`:

```python
async def _global_search_facets(driver, embedder, map_client, map_model, synth_client,
                                synth_model, *, q, level, group_id, settings,
                                facets: Sequence[Facet]) -> dict:
    stats = RerankStats()
    n = settings.global_facet_top_n
    shortlists = await asyncio.gather(*[
        shortlist_communities(driver, embedder, q, level=level, k=n, group_id=group_id,
                              settings=settings, stats=stats, scope=f.scope, top_n=n)
        for f in facets], return_exceptions=True)
    degraded: list[str] = []
    taken: set[str] = set()
    per_facet: list[tuple[Facet, list[CommunityHit]]] = []
    for f, hits in zip(facets, shortlists):
        if isinstance(hits, BaseException):
            logger.warning("global facet %r shortlist failed: %s", f.label, hits)
            degraded.append(f.label)
            continue
        mine = [h for h in hits if h.community_id not in taken]
        taken.update(h.community_id for h in mine)
        per_facet.append((f, mine))
    flat = [(f, h) for f, hs in per_facet for h in hs]
    maps = await asyncio.gather(*[map_report(map_client, map_model, q, h)
                                  for _, h in flat], return_exceptions=True)
    by_facet: dict[str, list[MapResult]] = {f.label: [] for f, _ in per_facet}
    for (f, _h), m in zip(flat, maps):
        if isinstance(m, BaseException):
            logger.warning("global map failed: %s", m)
        elif m is not None:
            by_facet[f.label].append(m)
    all_ids = sorted({fid for ms in by_facet.values() for m in ms for fid in m.fact_ids})
    resolved = await Provenance(driver).resolve_citations(all_ids) if all_ids else {}
    facet_results: list[tuple[str, list[MapResult]]] = []
    for f, _ in per_facet:
        kept = [replace(m, fact_ids=[
            x for x in m.fact_ids
            if _in_scope_ci(resolved.get(x, {}).get("sources", []), f.scope)])
            for m in by_facet[f.label]]
        facet_results.append((f.label, [m for m in kept if m.fact_ids]))
    communities_used = [{"community_id": m.community_id, "title": m.title,
                         "relevance": m.relevance}
                        for _, ms in facet_results for m in ms]
    note = could_not_check_note(degraded)
    marker_map: dict[int, dict] = {}
    fact_to_marker: dict[str, int] = {}
    for _, ms in facet_results:
        for m in ms:
            for fid in m.fact_ids:
                if fid not in fact_to_marker:
                    fact_to_marker[fid] = len(fact_to_marker) + 1
                    marker_map[fact_to_marker[fid]] = {"fact_uuid": fid}
    texts = await _fact_texts(driver, group_id, list(fact_to_marker))
    marker_map = {k: v for k, v in marker_map.items() if v["fact_uuid"] in texts}
    refusal = {"query": q, "answer": _REFUSAL + (" " + note if note else ""),
               "citations": [], "communities_used": communities_used,
               "degraded": stats.degraded, "applies_to": [], "degraded_facets": degraded}
    if not marker_map:
        return refusal
    sources_by_fact = {fid: resolved.get(fid, {}).get("sources", []) for fid in fact_to_marker}
    blocks = _render_facet_blocks(facet_results, fact_to_marker, texts, sources_by_fact)
    raw = await _complete_or_none(
        synth_client, synth_model,
        _REDUCE_FACETS_PROMPT.format(labels="; ".join(lab for lab, _ in facet_results),
                                     q=q, blocks=blocks), max_tokens=3000)
    if raw is None:
        return refusal
    answer, cited = _finalize_answer(raw, marker_map)
    citations = _build_citations(cited, marker_map, resolved)
    answer = _with_disclaimer(answer, stats.degraded)
    if note:
        answer = f"{answer}\n\n{note}"
    return {"query": q, "answer": answer, "citations": citations,
            "degraded": stats.degraded, "communities_used": communities_used,
            "applies_to": applies_to(citations), "degraded_facets": degraded}
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run --extra dev pytest tests/unit/test_global_facets.py tests/unit/test_global_scope.py tests/unit/test_global_rank.py tests/unit/test_global_reduce_binding.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/global_search.py tests/unit/test_global_facets.py
git commit -m "feat(answer): global search shortlists per facet and addresses every side"
```

---

### Task 5: Per-facet DRIFT

**Files:**
- Modify: `src/answer_api/drift.py` (`FollowUp`, `_primer`, `drift_search`, `_synthesize`; new `_SYNTH_FACETS_PROMPT`)
- Test: `tests/unit/test_drift_facets.py`

**Interfaces:**
- Consumes: `Facet`, `group_by_facet`, `could_not_check_note` (Task 1); `shortlist_communities(..., top_n=)` (Task 4).
- Produces:
  - `FollowUp` gains `scope: Scope | None = None` (the facet scope that runs it; None = the request scope).
  - `drift_search(..., facets: Sequence[Facet] | None = None)` — with >= 2 facets: the primer shortlists per facet (`top_n=settings.global_facet_top_n`) and records community→facet; each follow-up runs with its community's facet scope (or the request scope if untagged); any facet left with no follow-up gets one: the original question scoped to that facet; synthesis groups facts with `group_by_facet` and uses `_SYNTH_FACETS_PROMPT`; result carries `"degraded_facets"`. Fewer than 2: unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_drift_facets.py
"""DRIFT primes per facet and keeps each follow-up in its facet (spec §3.3)."""
from types import SimpleNamespace

import pytest

import answer_api.drift as drift
import answer_api.global_search as gs
from answer_api.scope import Scope
from answer_api.shape import Facet
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_V = Facet("Veeam", Scope(vendors=("Veeam",)))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",)))
_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _hit(cid):
    return gs.CommunityHit(community_id=cid, title=cid, summary="s", level=1, rating=5,
                           cited_fact_uuids=[], full_report="[]", similarity=0.9)


class _Synth:
    """Primer reply: only a Veeam-community follow-up. Final synthesis: fixed text."""

    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        p = kw["messages"][0]["content"]
        self.prompts.append(p)
        if "follow-up questions" in p:
            text = ('{"preliminary_answer": "d", "follow_ups": [{"query": "veeam detail",'
                    ' "community_id": "cv", "relevance": 9}]}')
        else:
            text = "Veeam X [1]. AWS Y [2]."
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason="stop")])


@pytest.fixture
def stubs(monkeypatch):
    ran = []

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("cv")] if scope == _V.scope else [_hit("ca")]

    async def _followup(graphiti, driver, fu, *, k, group_id, scope=None):
        ran.append((fu.query, scope))
        vendor, product = (("Veeam", "Veeam ONE") if scope == _V.scope
                           else ("AWS", "AWS Backup"))
        return [{"fact": fu.query, "fact_uuid": fu.query, "valid_at": None,
                 "invalid_at": None,
                 "sources": [{"vendor": vendor, "product": product, "article_id": "x"}]}]

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {}

    monkeypatch.setattr(drift, "shortlist_communities", _shortlist)
    monkeypatch.setattr(drift, "_run_followup", _followup)
    monkeypatch.setattr(drift, "Provenance", _Prov)
    return ran


async def test_every_facet_gets_a_follow_up_in_its_own_scope(stubs):
    await drift.drift_search(None, None, None, _Synth(), "m", q="compare", level=1,
                             iterations=1, primer_k=5, max_followups=4, followup_k=8,
                             group_id="g", settings=_S, facets=[_V, _A])
    assert ("veeam detail", _V.scope) in stubs
    assert ("compare", _A.scope) in stubs        # AWS had none -> scoped original question


async def test_final_synthesis_groups_facts_by_facet(stubs):
    s = _Synth()
    res = await drift.drift_search(None, None, None, s, "m", q="compare", level=1,
                                   iterations=1, primer_k=5, max_followups=4,
                                   followup_k=8, group_id="g", settings=_S,
                                   facets=[_V, _A])
    final = s.prompts[-1]
    assert final.index("=== Veeam ===") < final.index("=== AWS Backup ===")
    assert "Address EVERY side" in final
    assert res["degraded_facets"] == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_drift_facets.py -q`
Expected: FAIL — `drift_search() got an unexpected keyword argument 'facets'`.

- [ ] **Step 3: Implement**

In `src/answer_api/drift.py`:

(a) Imports: add `import asyncio`, `from collections.abc import Sequence`, and `from answer_api.shape import Facet, could_not_check_note, group_by_facet`.

(b) `FollowUp`: add the field `scope: Scope | None = None` after `iteration`.

(c) After `_SYNTH_PROMPT`, add:

```python
_SYNTH_FACETS_PROMPT = (
    "Answer the QUESTION using ONLY the numbered FACTS, which are grouped under a "
    "heading for each side of the question, refining the DRAFT where the facts support "
    "it. The question has these sides: {labels}. Address EVERY side, in that order. If "
    "a side shows \"(no facts retrieved)\", say in one sentence that the retrieved "
    "documentation does not cover it, and do not guess. Cite every claim inline with "
    "its [N] marker. Do NOT use outside knowledge. Do NOT write any URL. If no side has "
    "facts that answer the question, reply exactly: \"{refusal}\"\n\n"
    + ATTRIBUTION_RULES +
    "\nQUESTION: {q}\n\nDRAFT: {draft}\n\nFACTS:\n{facts}\n\nAnswer:"
)
```

(d) Change `_synthesize` to accept `facets: Sequence[Facet] | None = None` and, when it has >= 2 entries, build the grouped block. Replace its body with:

```python
    facts = _dedup_facts(facts)
    if facets is not None and len(facets) >= 2:
        groups = group_by_facet(facts, facets)
        ordered = [f for _, xs in groups for f in xs]
        ordered += [f for f in facts if f not in ordered]   # unmatched facts last
        marker_map = {i: f for i, f in enumerate(ordered, 1)}
        index = {f["fact_uuid"]: i for i, f in marker_map.items()}
        parts = []
        for facet, xs in groups:
            lines = [fact_line(index[f["fact_uuid"]], f["fact"], f.get("sources", []))
                     for f in xs] or ["(no facts retrieved)"]
            parts.append(f"=== {facet.label} ===\n" + "\n".join(lines))
        rest = [f for f in ordered if all(f not in xs for _, xs in groups)]
        if rest:
            parts.append("=== Other ===\n" + "\n".join(
                fact_line(index[f["fact_uuid"]], f["fact"], f.get("sources", []))
                for f in rest))
        prompt = _SYNTH_FACETS_PROMPT.format(
            labels="; ".join(f.label for f in facets), refusal=_REFUSAL, q=q,
            draft=preliminary_answer or "(none)", facts="\n\n".join(parts))
    else:
        marker_map = {i: f for i, f in enumerate(facts, 1)}
        facts_block = "\n".join(
            fact_line(i, f["fact"], f.get("sources", [])) for i, f in marker_map.items())
        prompt = _SYNTH_PROMPT.format(refusal=_REFUSAL, q=q,
                                      draft=preliminary_answer or "(none)",
                                      facts=facts_block)
    raw = await _complete_or_none(synth_client, synth_model, prompt, max_tokens=3000)
    if raw is None:
        return _REFUSAL, []
    answer, cited = _finalize_answer(raw, marker_map)
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = _build_citations(cited, marker_map, resolved)
    return answer, citations
```

and change its signature to `async def _synthesize(synth_client, synth_model, driver, *, q, preliminary_answer, facts, facets: Sequence[Facet] | None = None) -> tuple[str, list[dict]]:`.

(e) Change `_primer` to accept `facets: Sequence[Facet] | None = None` and `top_n: int | None = None`. Replace its first statement (the `hits = await shortlist_communities(...)` call) with:

```python
    facet_of: dict[str, Facet] = {}
    degraded: list[str] = []
    if facets is not None and len(facets) >= 2:
        found = await asyncio.gather(*[
            shortlist_communities(driver, embedder, q, level=level, k=k,
                                  group_id=group_id, settings=settings, stats=stats,
                                  scope=f.scope, top_n=top_n)
            for f in facets], return_exceptions=True)
        hits = []
        for f, fh in zip(facets, found):
            if isinstance(fh, BaseException):
                logger.warning("drift facet %r primer failed: %s", f.label, fh)
                degraded.append(f.label)
                continue
            for h in fh:
                if h.community_id not in facet_of:
                    facet_of[h.community_id] = f
                    hits.append(h)
    else:
        hits = await shortlist_communities(driver, embedder, q, level=level, k=k,
                                           group_id=group_id, settings=settings,
                                           stats=stats, scope=scope)
```

At its end, before every `return preliminary, fus, hits` / `return "", [...], hits`, tag follow-ups and add coverage follow-ups. Replace the final two returns with:

```python
    if obj is None:
        fus = [FollowUp(query=q, community_id=None, iteration=1)]
        preliminary = ""
    else:
        preliminary = str(obj.get("preliminary_answer", "")).strip()
        fus = _parse_followups(obj, hit_ids, max_followups, iteration=1) \
            or [FollowUp(query=q, community_id=None, iteration=1)]
    if facet_of:
        fus = [FollowUp(query=fu.query, community_id=fu.community_id,
                        iteration=fu.iteration,
                        scope=facet_of[fu.community_id].scope
                        if fu.community_id in facet_of else None) for fu in fus]
        covered = {fu.scope for fu in fus if fu.scope is not None}
        for f in facets or []:
            if f.label not in degraded and f.scope not in covered:
                fus.append(FollowUp(query=q, community_id=None, iteration=1,
                                    scope=f.scope))
    return preliminary, fus, hits, degraded
```

(The existing early `if obj is None: return ...` statement is removed; `_primer` now always returns a 4-tuple `(preliminary, follow_ups, hits, degraded)` or `None`.)

(f) In `drift_search`: add the parameter `facets: Sequence[Facet] | None = None` after `scope`. Pass `facets=facets, top_n=settings.global_facet_top_n if facets and len(facets) >= 2 else None` to `_primer`; unpack `preliminary, followups, hits, degraded = primed`. In BOTH follow-up loops replace `scope=scope` in the `_run_followup` call with `scope=fu.scope if fu.scope is not None else scope`. Pass `facets=facets` to `_synthesize`. After the existing `_with_disclaimer` step, add:

```python
    note = could_not_check_note(degraded)
    if note:
        answer = f"{answer}\n\n{note}"
```

and add `"degraded_facets": degraded` to the returned dict (and to the no-facts refusal dict). In the `primed is None` branch, keep calling `answer_local` but forward `facets=facets`.

- [ ] **Step 4: Run to verify pass, including existing DRIFT tests**

Run: `uv run --extra dev pytest tests/unit/test_drift_facets.py tests/unit/test_drift_parse.py tests/unit/test_drift_scope.py -q`
Expected: PASS. If an existing test unpacks `_primer`'s old 3-tuple, update it to the 4-tuple `(preliminary, follow_ups, hits, degraded)` with `degraded == []`.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/drift.py tests/unit/test_drift_facets.py tests/unit/test_drift_*.py
git commit -m "feat(answer): DRIFT primes per facet and keeps every follow-up in its facet"
```

---

### Task 6: Enumeration path

**Files:**
- Create: `src/answer_api/enumeration.py`
- Test: `tests/unit/test_enumeration.py`

**Interfaces:**
- Consumes: `Facet`, `could_not_check_note` (Task 1); `search_local` (Task 2); `shortlist_communities(..., top_n=)`, `_fact_texts`, `_extract_json` (global_search); `_complete_or_none`, `_finalize_answer`, `_build_citations`, `_usable_content` (synthesize); `Provenance`; `fact_line`, `applies_to`, `ATTRIBUTION_RULES` (attribution).
- Produces: `enumerate_vendors(graphiti, driver, embedder, map_client, map_model, synth_client, synth_model, *, q: str, facets: Sequence[Facet], group_id: str, settings) -> dict` with keys `query, answer, citations, applies_to, checked_vendors, documented_vendors, degraded_facets, communities_used`. Constant `NONE_DOCUMENTED` template.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_enumeration.py
"""Enumeration checks every content vendor (spec §3.4)."""
from types import SimpleNamespace

import pytest

import answer_api.enumeration as en
import answer_api.global_search as gs
from answer_api.scope import Scope
from answer_api.shape import Facet
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_VENDORS = ["AWS", "Cohesity", "Microsoft", "Veeam"]
_FACETS = [Facet(v, Scope(vendors=(v,), source="detected")) for v in _VENDORS]


def _src(v):
    return [{"vendor": v, "product": f"{v} P", "article_id": v}]


class _Client:
    """Check replies by vendor; synthesis returns one cited line per documented vendor."""

    def __init__(self, checks):
        self.checks = checks
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        p = kw["messages"][0]["content"]
        self.prompts.append(p)
        if p.startswith("Decide whether"):
            vendor = next(v for v in _VENDORS if f"from {v}'s documentation" in p)
            text = self.checks[vendor]
        else:
            text = "AWS locks vaults [1]. Veeam hardens repos [2]."
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason="stop")])


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        v = scope.vendors[0]
        return [gs.CommunityHit(community_id=f"c{v}", title=f"{v} T", summary="s",
                                level=1, rating=5, cited_fact_uuids=[f"cf-{v}"],
                                full_report='[{"finding": "f", "fact_ids": ["cf-%s"]}]' % v,
                                similarity=0.9)]

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450):
        v = scope.vendors[0]
        return {"results": [{"fact": f"{v} fact", "fact_uuid": f"lf-{v}",
                             "valid_at": None, "invalid_at": None, "sources": _src(v)}]}

    async def _texts(driver, group_id, uuids):
        return {u: f"text {u}" for u in uuids}

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {u: {"sources": _src(u.split("-")[1])} for u in uuids}

    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    monkeypatch.setattr(en, "search_local", _search)
    monkeypatch.setattr(en, "_fact_texts", _texts)
    monkeypatch.setattr(en, "Provenance", _Prov)


def _checks(**over):
    base = {v: '{"documented": false}' for v in _VENDORS}
    base.update(over)
    return base


async def _run(client):
    return await en.enumerate_vendors(None, None, None, client, "map", client, "syn",
                                      q="Which vendors lock backups?", facets=_FACETS,
                                      group_id="g", settings=_S)


async def test_every_vendor_is_checked_and_documented_ones_are_listed():
    c = _Client(_checks(AWS='{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}',
                        Veeam='{"documented": true, "points": ["p"], "fact_ids": ["cf-Veeam"]}'))
    res = await _run(c)
    assert res["checked_vendors"] == _VENDORS
    assert res["documented_vendors"] == ["AWS", "Veeam"]
    assert res["answer"].endswith(
        "Checked: AWS, Cohesity, Microsoft, Veeam. Documented for: AWS, Veeam.")
    syn = c.prompts[-1]
    assert syn.index("=== AWS ===") < syn.index("=== Veeam ===")
    assert "=== Cohesity ===" not in syn
    assert {x["fact_uuid"] for x in res["citations"]} == {"lf-AWS", "cf-Veeam"}


async def test_fact_ids_outside_the_evidence_are_dropped_and_unverifiable_is_degraded():
    c = _Client(_checks(AWS='{"documented": true, "points": ["p"], "fact_ids": ["made-up"]}'))
    res = await _run(c)
    assert res["documented_vendors"] == []
    assert res["degraded_facets"] == ["AWS"]


async def test_an_unusable_check_after_retry_is_could_not_check_not_absence():
    c = _Client(_checks(Cohesity="not json"))
    res = await _run(c)
    assert "Cohesity" in res["degraded_facets"]
    assert "Could not check: Cohesity." in res["answer"]


async def test_none_documented_names_the_checked_vendors():
    res = await _run(_Client(_checks()))
    assert res["documented_vendors"] == []
    assert res["answer"].startswith(
        "None of the checked vendors (AWS, Cohesity, Microsoft, Veeam) document this")
    assert res["citations"] == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_enumeration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'answer_api.enumeration'`.

- [ ] **Step 3: Create `src/answer_api/enumeration.py`**

```python
"""Enumeration answers (spec 2026-10-12 §3.4): "Which vendors ...?" checks EVERY
vendor with extracted content -- no ceiling. Per vendor: gather scoped evidence,
one cheap-tier check (documented or not, with fact ids drawn from that evidence),
then one synthesis over the documented vendors. The closing "Checked / Documented
for" line is built here, deterministically -- the LLM never decides who was checked.
A vendor whose check fails is reported as "Could not check", never as absent.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence

from answer_api.attribution import ATTRIBUTION_RULES, applies_to, fact_line
from answer_api.global_search import _extract_json, _fact_texts, shortlist_communities
from answer_api.search import search_local
from answer_api.shape import Facet, could_not_check_note
from answer_api.synthesize import (_REFUSAL, _build_citations, _complete_or_none,
                                   _finalize_answer, _usable_content)
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

NONE_DOCUMENTED = ("None of the checked vendors ({vendors}) document this in the "
                   "available sources.")

_CHECK_PROMPT = (
    "Decide whether the EVIDENCE below, all from {vendor}'s documentation, shows what "
    "the QUESTION asks about for {vendor}. Use ONLY the evidence. Absence of evidence "
    "is not a finding. Respond with ONLY JSON: {{\"documented\": true or false, "
    "\"points\": [short strings stating what the evidence shows], \"fact_ids\": [ids "
    "from the evidence that support those points]}}. If documented is false, points and "
    "fact_ids must be empty. Do NOT write URLs.\n\nQUESTION: {q}\n\nEVIDENCE:\n{evidence}"
)

_ENUM_PROMPT = (
    "Answer the QUESTION by covering EVERY vendor below, in the order given, with one "
    "short paragraph each, using ONLY that vendor's numbered facts. "
    + ATTRIBUTION_RULES +
    "Cite every claim with the [N] marker of the fact it rests on, writing each marker "
    "individually. Do NOT use outside knowledge. Do NOT write any URL. Do NOT mention "
    "any vendor that is not listed below.\n\nQUESTION: {q}\n\n{blocks}\n\nAnswer:"
)


async def _gather(graphiti, driver, embedder, *, q, facet: Facet, group_id, settings):
    n = settings.enum_communities_per_vendor
    communities, facts = await asyncio.gather(
        shortlist_communities(driver, embedder, q, level=settings.global_default_level,
                              k=n, group_id=group_id, settings=settings,
                              scope=facet.scope, top_n=n),
        search_local(graphiti, driver, q=q, k=settings.enum_facts_per_vendor,
                     scope=facet.scope, group_id=group_id,
                     fetch_cap=settings.local_scoped_fetch_cap))
    return communities, facts["results"]


def _evidence(communities, facts) -> tuple[str, set[str]]:
    lines, valid = [], set()
    for f in facts:
        lines.append(f"[{f['fact_uuid']}] {f['fact']}")
        valid.add(f["fact_uuid"])
    for c in communities:
        try:
            findings = json.loads(c.full_report or "[]")
        except ValueError:
            findings = []
        for fd in findings if isinstance(findings, list) else []:
            ids = [i for i in fd.get("fact_ids", []) if i in c.cited_fact_uuids]
            if ids:
                lines.append(f"FINDING: {fd.get('finding', '')} (facts: {', '.join(ids)})")
                valid.update(ids)
    return "\n".join(lines), valid


async def _check(client, model, *, q, vendor, evidence, valid) -> list[str] | None:
    """Supporting fact ids (possibly empty: not documented), or None: unusable."""
    prompt = _CHECK_PROMPT.format(vendor=vendor, q=q, evidence=evidence)
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(_usable_content(resp) or "")
        if obj is not None and isinstance(obj.get("documented"), bool):
            if not obj["documented"]:
                return []
            ids = [i for i in obj.get("fact_ids") or [] if i in valid]
            if not ids:
                logger.warning("enumeration check for %s claimed documented with no "
                               "fact id from its evidence", vendor)
                return None
            return ids
    logger.warning("enumeration check for %s returned no usable verdict", vendor)
    return None


async def enumerate_vendors(graphiti, driver, embedder, map_client, map_model,
                            synth_client, synth_model, *, q: str,
                            facets: Sequence[Facet], group_id: str, settings) -> dict:
    checked = [f.label for f in facets]
    gathered = await asyncio.gather(*[
        _gather(graphiti, driver, embedder, q=q, facet=f, group_id=group_id,
                settings=settings) for f in facets], return_exceptions=True)
    degraded: list[str] = []
    supported: dict[str, list[str]] = {}
    communities_used: list[dict] = []
    local_texts: dict[str, str] = {}

    async def _one(f: Facet, got):
        if isinstance(got, BaseException):
            logger.warning("enumeration gather for %s failed: %s", f.label, got)
            return f, None
        communities, facts = got
        communities_used.extend({"community_id": c.community_id, "title": c.title,
                                 "relevance": c.relevance} for c in communities)
        local_texts.update({x["fact_uuid"]: x["fact"] for x in facts})
        evidence, valid = _evidence(communities, facts)
        if not evidence:
            return f, []
        return f, await _check(map_client, map_model, q=q, vendor=f.label,
                               evidence=evidence, valid=valid)

    for f, ids in await asyncio.gather(*[_one(f, g) for f, g in zip(facets, gathered)]):
        if ids is None:
            degraded.append(f.label)
        elif ids:
            supported[f.label] = ids
    documented = [v for v in checked if v in supported]
    note = could_not_check_note(degraded)
    base = {"query": q, "checked_vendors": checked, "documented_vendors": documented,
            "degraded_facets": degraded, "communities_used": communities_used}
    if not documented:
        answer = NONE_DOCUMENTED.format(vendors=", ".join(checked))
        return {**base, "answer": answer + (f" {note}" if note else ""),
                "citations": [], "applies_to": []}
    all_ids = [i for v in documented for i in supported[v]]
    texts = {**await _fact_texts(driver, group_id,
                                 [i for i in all_ids if i not in local_texts]),
             **{i: t for i, t in local_texts.items() if i in all_ids}}
    resolved = await Provenance(driver).resolve_citations(all_ids)
    marker_map: dict[int, dict] = {}
    blocks = []
    for v in documented:
        lines = []
        for fid in supported[v]:
            if fid not in texts or any(m["fact_uuid"] == fid for m in marker_map.values()):
                continue
            n = len(marker_map) + 1
            marker_map[n] = {"fact_uuid": fid}
            lines.append(fact_line(n, texts[fid], resolved.get(fid, {}).get("sources", [])))
        blocks.append(f"=== {v} ===\n" + "\n".join(lines))
    closing = f"Checked: {', '.join(checked)}. Documented for: {', '.join(documented)}."
    raw = await _complete_or_none(synth_client, synth_model,
                                  _ENUM_PROMPT.format(q=q, blocks="\n\n".join(blocks)),
                                  max_tokens=3000)
    if raw is None:
        return {**base, "answer": f"{_REFUSAL}\n\n{closing}", "citations": [],
                "applies_to": [], "degraded": "synthesis-failed"}
    answer, cited = _finalize_answer(raw, marker_map)
    citations = _build_citations(cited, marker_map, resolved)
    tail = f"{note}\n\n{closing}" if note else closing
    return {**base, "answer": f"{answer}\n\n{tail}", "citations": citations,
            "applies_to": applies_to(citations)}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --extra dev pytest tests/unit/test_enumeration.py -q`
Expected: PASS. (`test_an_unusable_check...`: the closing line follows the note, so the answer contains "Could not check: Cohesity." and ends with the Checked line only when another vendor is documented; with none documented it ends with the note — both satisfy the assertion.)

- [ ] **Step 5: Commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/enumeration.py tests/unit/test_enumeration.py
git commit -m "feat(answer): enumeration checks every content vendor and lists all documented"
```

---

### Task 7: Wire shape through router, app and the eval harness

**Files:**
- Modify: `src/answer_api/router.py` (`answer_router`, `_run_mode`, `_dispatch`, `_normalize`)
- Modify: `src/answer_api/app.py` (`/answer`, `/search/global`, `/search/drift`, `/search/local` handlers; `_resolve_scope` callers)
- Modify: `src/answer_api/eval_router.py` (`_score_one`)
- Test: `tests/unit/test_router_shape.py`

**Interfaces:**
- Consumes: `QuestionShape`, `classify_shape` (Task 1); `answer_local(facets=..)` (Task 3); `global_search(facets=..)` (Task 4); `drift_search(facets=..)` (Task 5); `enumerate_vendors` (Task 6).
- Produces:
  - `answer_router(..., q, mode_override, scope, settings, shape: QuestionShape | None = None) -> dict`. `shape` None → treated as single.
  - Enumeration: classifier still runs (routing label), then `enumerate_vendors` runs instead of the mode; envelope `routing.path = "enumeration"`.
  - Comparison: `_run_mode`/`_dispatch` forward `facets=shape.facets` to local, global and drift; timeline ignores facets.
  - Envelope gains `shape` (kind), `facets` (`[{"label", "scope": Scope.as_dict()}]`), and when present `degraded_facets`, `checked_vendors`, `documented_vendors`.
  - Relaxation: a comparison or single with a *detected* scope that grounds nothing re-runs as single unscoped (`shape=None`), exactly as today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_router_shape.py
"""The router passes the question shape to every mode (spec §3)."""
import pytest

import answer_api.router as router
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_V = Facet("Veeam", Scope(vendors=("Veeam",), source="detected"))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",), source="detected"))
_CITE = [{"marker": 1, "fact_uuid": "f", "sources": []}]


@pytest.fixture
def seen(monkeypatch):
    calls = []

    async def _classify(q, **kw):
        return "global", "heuristic"

    async def _global(*a, **kw):
        calls.append(("global", kw.get("facets")))
        return {"answer": "A [1]", "citations": _CITE, "communities_used": [{"x": 1}],
                "applies_to": [], "degraded_facets": []}

    async def _enum(*a, **kw):
        calls.append(("enumerate", kw["facets"]))
        return {"answer": "E", "citations": _CITE, "applies_to": [],
                "checked_vendors": ["AWS"], "documented_vendors": ["AWS"],
                "degraded_facets": []}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.global_mod, "global_search", _global)
    monkeypatch.setattr(router.enum_mod, "enumerate_vendors", _enum)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    return calls


async def _answer(shape, scope=Scope()):
    return await router.answer_router(None, None, None, None, "s", None, "m", None, "c",
                                      q="q", mode_override=None, scope=scope,
                                      settings=_S, shape=shape)


async def test_a_comparison_passes_its_facets_to_the_mode(seen):
    shape = QuestionShape("comparison", (_V, _A))
    env = await _answer(shape, Scope(vendors=("Veeam",), products=("AWS Backup",),
                                     source="detected"))
    assert seen == [("global", (_V, _A))]
    assert env["shape"] == "comparison"
    assert [f["label"] for f in env["facets"]] == ["Veeam", "AWS Backup"]


async def test_an_enumeration_runs_the_enumeration_path(seen):
    shape = QuestionShape("enumeration", (Facet("AWS", Scope(vendors=("AWS",))),))
    env = await _answer(shape)
    assert seen[0][0] == "enumerate"
    assert env["routing"]["path"] == "enumeration"
    assert env["mode"] == "global"
    assert env["checked_vendors"] == ["AWS"] and env["documented_vendors"] == ["AWS"]


async def test_no_shape_is_single_and_passes_no_facets(seen):
    env = await _answer(None)
    assert seen == [("global", None)]
    assert env["shape"] == "single"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_router_shape.py -q`
Expected: FAIL — `answer_router() got an unexpected keyword argument 'shape'` / no attribute `enum_mod`.

- [ ] **Step 3: Implement the router changes**

In `src/answer_api/router.py`:

(a) Imports: add `from answer_api import enumeration as enum_mod` and `from answer_api.shape import QuestionShape`.

(b) `_dispatch`: add the keyword parameter `facets=None`; pass `facets=facets` to `answer_local` together with `min_per_facet=settings.local_min_per_facet, fetch_cap=settings.local_scoped_fetch_cap`; pass `facets=facets` to `global_search` and to `drift_search`. Timeline unchanged.

(c) `_run_mode`: add the keyword parameter `facets=None` and forward it to all three `_dispatch` calls.

(d) `answer_router`: add the parameter `shape: QuestionShape | None = None`. After `classify(...)`, insert:

```python
    if shape is not None and shape.kind == "enumeration":
        raw = await enum_mod.enumerate_vendors(
            graphiti, driver, embedder, map_client, map_model, synth_client,
            synth_model, q=q, facets=shape.facets, group_id=settings.group_id,
            settings=settings)
        env = _normalize(mode, via, None, raw, q, scope)
        env["routing"]["path"] = "enumeration"
        _shape_fields(env, shape, raw)
        env["freshness"] = await freshness_mod.freshness(
            driver, settings.group_id, reports=True)
        return env
    facets = shape.facets if shape is not None and shape.kind == "comparison" else None
```

Pass `facets=facets` to the first `_run_mode` call. The relaxation re-run keeps `scope=Scope()` and passes no facets (single, unscoped). After `env = _normalize(...)` and the relaxed-scope line, add `_shape_fields(env, shape if not relaxed else None, raw)`.

(e) Add this helper above `answer_router`:

```python
def _shape_fields(env: dict, shape: QuestionShape | None, raw: dict) -> None:
    env["shape"] = shape.kind if shape is not None else "single"
    env["facets"] = [{"label": f.label, "scope": f.scope.as_dict()}
                     for f in (shape.facets if shape is not None else ())]
    for key in ("degraded_facets", "checked_vendors", "documented_vendors"):
        if key in raw:
            env[key] = raw[key]
```

- [ ] **Step 4: Compute the shape in the app and the eval harness**

In `src/answer_api/app.py`, add `from answer_api.shape import classify_shape`. In the `/answer` handler, after `resolved = await _resolve_scope(...)`, add:

```python
        shape = classify_shape(q, st.scope_resolver, resolved,
                               scope_disabled=(scope == "none"))
```

and pass `shape=shape` to `router_mod.answer_router(...)`. In `/search/global` and `/search/drift`, compute `shape` the same way and pass `facets=shape.facets if shape.kind == "comparison" else None`; for an enumeration shape these endpoints keep their mode (they are mode-specific endpoints) and pass no facets. `/search/local` and `/timeline` are unchanged.

In `src/answer_api/eval_router.py` `_score_one`, after `scope = resolver.resolve(q["question"])`, add `shape = classify_shape(q["question"], resolver, scope)` (import `classify_shape` from `answer_api.shape`) and pass `shape=shape` to `answer_router`.

- [ ] **Step 5: Run the router and app tests**

Run: `uv run --extra dev pytest tests/unit/test_router_shape.py tests/unit/test_router_dispatch.py tests/unit/test_router_scope.py tests/unit/test_router_normalize.py tests/unit/test_eval_router_harness.py -q`
Expected: PASS. If an existing fake of `_dispatch`, `global_search`, `drift_search` or `answer_local` rejects the new `facets` / `min_per_facet` / `fetch_cap` keywords, give the fake `**kw` (do not change production signatures back).

- [ ] **Step 6: Commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/router.py src/answer_api/app.py src/answer_api/eval_router.py tests/unit/test_router_shape.py tests/unit/test_router_*.py tests/unit/test_eval_router_harness.py
git commit -m "feat(answer): route by question shape -- facets to every mode, enumeration path"
```

---

### Task 8: Facet coverage and enumeration completeness in the eval

**Files:**
- Modify: `src/answer_api/eval_router.py` (new `facet_coverage_hit`, `enumeration_hit`; `run_eval` record and summary; `format_report`)
- Modify: `tests/unit/test_router_golden_schema.py` (allow optional `expected_vendors`)
- Test: `tests/unit/test_eval_facets.py`

**Interfaces:**
- Consumes: envelope `facets`, `shape`, `documented_vendors`, `citations` (Task 7); `in_scope` (attribution).
- Produces:
  - `facet_coverage_hit(env: dict) -> bool | None` — None unless `env["shape"] == "comparison"`; True iff every facet has at least one citation whose `sources` are `in_scope` of that facet's scope.
  - `enumeration_hit(env: dict, expected_vendors: list[str] | None) -> bool | None` — None unless `env["shape"] == "enumeration"` and `expected_vendors` is given; True iff `set(documented_vendors) == set(expected_vendors)` AND every documented vendor has at least one citation from that vendor.
  - Per-question record keys `facet_coverage`, `enumeration_complete`; summary keys `facet_coverage` (mean over non-None, or None), `facet_coverage_n`, `enumeration_completeness`, `enumeration_n`; two report lines.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_eval_facets.py
"""Facet coverage and enumeration completeness are scored from citations."""
from answer_api.eval_router import enumeration_hit, facet_coverage_hit


def _cite(vendor, product):
    return {"sources": [{"vendor": vendor, "product": product}]}


_FACETS = [{"label": "Veeam", "scope": {"vendors": ["Veeam"], "products": [],
                                        "sources": [], "source": "detected"}},
           {"label": "AWS Backup", "scope": {"vendors": [], "products": ["AWS Backup"],
                                             "sources": [], "source": "detected"}}]


def test_comparison_covered_only_when_every_facet_is_cited():
    env = {"shape": "comparison", "facets": _FACETS,
           "citations": [_cite("Veeam", "Veeam ONE"), _cite("AWS", "AWS Backup")]}
    assert facet_coverage_hit(env) is True
    env["citations"] = [_cite("Veeam", "Veeam ONE")]
    assert facet_coverage_hit(env) is False


def test_facet_coverage_is_not_scored_for_other_shapes():
    assert facet_coverage_hit({"shape": "single", "facets": [], "citations": []}) is None


def test_enumeration_complete_needs_the_exact_set_and_a_citation_each():
    env = {"shape": "enumeration", "documented_vendors": ["AWS", "Veeam"],
           "citations": [_cite("AWS", "AWS Backup"), _cite("Veeam", "Veeam ONE")]}
    assert enumeration_hit(env, ["Veeam", "AWS"]) is True
    assert enumeration_hit(env, ["Veeam", "AWS", "Cohesity"]) is False
    env["citations"] = [_cite("AWS", "AWS Backup")]
    assert enumeration_hit(env, ["Veeam", "AWS"]) is False


def test_enumeration_unscored_without_ground_truth():
    assert enumeration_hit({"shape": "enumeration", "documented_vendors": []}, None) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_eval_facets.py -q`
Expected: FAIL — `ImportError: cannot import name 'enumeration_hit'`.

- [ ] **Step 3: Implement the metric functions**

In `src/answer_api/eval_router.py`, add `from answer_api.attribution import in_scope` and `from answer_api.scope import Scope` (if not already imported), then:

```python
def _scope_of(d: dict) -> Scope:
    return Scope(tuple(d.get("vendors", [])), tuple(d.get("products", [])),
                 d.get("source", "none"), tuple(d.get("sources", [])))


def facet_coverage_hit(env: dict) -> bool | None:
    """Comparison only: every facet has a citation from its own side."""
    if env.get("shape") != "comparison":
        return None
    cites = env.get("citations", [])
    return all(any(in_scope(c.get("sources", []), _scope_of(f["scope"])) for c in cites)
               for f in env.get("facets", []))


def enumeration_hit(env: dict, expected_vendors: list[str] | None) -> bool | None:
    """Enumeration only: exactly the ground-truth vendors, each cited."""
    if env.get("shape") != "enumeration" or expected_vendors is None:
        return None
    documented = env.get("documented_vendors", [])
    if set(documented) != set(expected_vendors):
        return False
    cited = {s.get("vendor") for c in env.get("citations", []) for s in c.get("sources", [])}
    return all(v in cited for v in documented)
```

- [ ] **Step 4: Record, aggregate, report**

In `run_eval`, add to `rec`: `"facet_coverage": facet_coverage_hit(env)` and `"enumeration_complete": enumeration_hit(env, q.get("expected_vendors"))`; also add `"facet_coverage": None, "enumeration_complete": None` to the failed-question record in the `except` branch. `aggregate` lives in `router_eval.py` and returns a dict literal, so leave it alone: in `run_eval`, directly after `summary = aggregate(per_question)`, add:

```python
    fc = [r["facet_coverage"] for r in per_question if r.get("facet_coverage") is not None]
    ec = [r["enumeration_complete"] for r in per_question
          if r.get("enumeration_complete") is not None]
    summary["facet_coverage"] = round(sum(fc) / len(fc), 2) if fc else None
    summary["facet_coverage_n"] = len(fc)
    summary["enumeration_completeness"] = round(sum(ec) / len(ec), 2) if ec else None
    summary["enumeration_n"] = len(ec)
```

In `format_report`, after the misattributed line, add:

```python
             f"**Facet coverage (comparisons): {summary.get('facet_coverage')} "
             f"(n={summary.get('facet_coverage_n')})** | **Enumeration completeness: "
             f"{summary.get('enumeration_completeness')} "
             f"(n={summary.get('enumeration_n')})**",
```

In `tests/unit/test_router_golden_schema.py`, allow the optional key: wherever the test asserts the exact key set of a golden entry, accept `expected_vendors` as an optional list of strings (e.g. `assert set(q) - {"expected_vendors"} == REQUIRED_KEYS` and, if present, `assert all(isinstance(v, str) for v in q["expected_vendors"])`).

- [ ] **Step 5: Run to verify pass**

Run: `uv run --extra dev pytest tests/unit/test_eval_facets.py tests/unit/test_router_golden_schema.py tests/unit/test_eval_router_harness.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/eval_router.py tests/unit/test_eval_facets.py tests/unit/test_router_golden_schema.py
git commit -m "feat(eval): score facet coverage and enumeration completeness from citations"
```

---

### Task 9: Free retrieval check (`eval_retrieval`)

**Files:**
- Create: `src/answer_api/eval_retrieval.py`
- Modify: `CLAUDE.md` (answer_api module list: `shape.py`, `enumeration.py`, `eval_retrieval.py`)
- Test: `tests/unit/test_eval_retrieval.py`

**Interfaces:**
- Consumes: `classify_shape`, `Facet` (Task 1); `search_local` (Task 2); `shortlist_communities(top_n=)` (Task 4); `Provenance`.
- Produces:
  - `check_question(q: dict, facet_evidence: list[dict]) -> dict` — pure. `facet_evidence` items: `{"label", "articles": set[str], "vendors": set[str], "communities": int, "facts": int}`. Returns `{"question", "shape", "expected_hit": bool | None, "facets_with_evidence": int, "facets": int, "vendors_found": sorted list, "vendors_expected": list | None, "pass": bool}`. `expected_hit` = any expected article among all facets' articles (None if no expected ids). `pass` = every facet has `communities + facts > 0` AND `expected_hit is not False` AND (for enumeration with `expected_vendors`) every expected vendor is in `vendors_found`.
  - CLI `python -m answer_api.eval_retrieval --questions <file>`: per question runs per-facet `search_local` (k per facet as in Task 3, or `enum_facts_per_vendor` for enumeration) and per-facet `shortlist_communities` (`top_n` as in Tasks 4/6), builds `facet_evidence` from the LOCAL facts' sources (articles, vendors) plus community counts, prints one line per question and a final `PASS n/N`, exits non-zero unless all pass. No LLM calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_eval_retrieval.py
"""The free retrieval gate's pass/fail logic."""
from answer_api.eval_retrieval import check_question


def _ev(label, articles=(), vendors=(), communities=1, facts=1):
    return {"label": label, "articles": set(articles), "vendors": set(vendors),
            "communities": communities, "facts": facts}


def test_passes_when_every_facet_has_evidence_and_an_expected_article_is_found():
    q = {"question": "q", "expected_article_ids": ["a1"]}
    r = check_question(q, [_ev("Veeam", ["a1"], ["Veeam"]), _ev("AWS", ["b"], ["AWS"])])
    assert r["pass"] and r["expected_hit"] is True and r["facets_with_evidence"] == 2


def test_fails_when_a_facet_has_no_evidence():
    q = {"question": "q", "expected_article_ids": ["a1"]}
    r = check_question(q, [_ev("Veeam", ["a1"]), _ev("AWS", communities=0, facts=0)])
    assert not r["pass"]


def test_fails_when_no_expected_article_is_retrieved():
    q = {"question": "q", "expected_article_ids": ["zz"]}
    assert not check_question(q, [_ev("Veeam", ["a1"])])["pass"]


def test_enumeration_needs_every_expected_vendor():
    q = {"question": "q", "expected_article_ids": [], "expected_vendors": ["AWS", "Veeam"]}
    ok = check_question(q, [_ev("AWS", vendors=["AWS"]), _ev("Veeam", vendors=["Veeam"])])
    assert ok["pass"] and ok["expected_hit"] is None
    bad = check_question(q, [_ev("AWS", vendors=["AWS"]), _ev("Veeam", facts=0,
                                                               communities=0)])
    assert not bad["pass"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_eval_retrieval.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/answer_api/eval_retrieval.py`**

```python
"""Free retrieval gate (spec 2026-10-12 §6): per golden question, does per-facet
retrieval find evidence for every facet, an expected article, and every expected
vendor? No LLM calls -- the local embedder, Neo4j and the reranker only. Run before
any paid eval:

    uv run python -m answer_api.eval_retrieval --questions src/answer_api/router_golden_tier1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

from answer_api.global_search import shortlist_communities
from answer_api.scope import ScopeResolver
from answer_api.search import search_local
from answer_api.shape import classify_shape
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti


def check_question(q: dict, facet_evidence: list[dict]) -> dict:
    expected = set(q.get("expected_article_ids") or [])
    articles = set().union(*(e["articles"] for e in facet_evidence)) if facet_evidence else set()
    vendors = set().union(*(e["vendors"] for e in facet_evidence)) if facet_evidence else set()
    expected_hit = bool(expected & articles) if expected else None
    with_ev = sum(1 for e in facet_evidence if e["communities"] + e["facts"] > 0)
    want = q.get("expected_vendors")
    vendors_ok = want is None or all(
        any(e["label"] == v and e["communities"] + e["facts"] > 0 for e in facet_evidence)
        for v in want)
    return {"question": q["question"], "shape": q.get("_shape"),
            "expected_hit": expected_hit, "facets_with_evidence": with_ev,
            "facets": len(facet_evidence), "vendors_found": sorted(vendors),
            "vendors_expected": want,
            "pass": with_ev == len(facet_evidence) and expected_hit is not False
            and vendors_ok}


async def _evidence(graphiti, driver, embedder, settings, q: str, shape) -> list[dict]:
    n = len(shape.facets)
    if shape.kind == "enumeration":
        k, top_n = settings.enum_facts_per_vendor, settings.enum_communities_per_vendor
    elif shape.kind == "comparison":
        k = max(settings.local_min_per_facet, math.ceil(15 / n))
        top_n = settings.global_facet_top_n
    else:
        k, top_n = 15, None
    out = []
    for f in shape.facets:
        scope = f.scope if not f.scope.is_empty() else None
        loc = await search_local(graphiti, driver, q=q, k=k, scope=scope,
                                 group_id=settings.group_id,
                                 fetch_cap=settings.local_scoped_fetch_cap)
        hits = await shortlist_communities(
            driver, embedder, q, level=settings.global_default_level,
            k=top_n or settings.global_shortlist_k, group_id=settings.group_id,
            settings=settings, scope=scope, top_n=top_n)
        srcs = [s for r in loc["results"] for s in r["sources"]]
        out.append({"label": f.label,
                    "articles": {s.get("article_id") for s in srcs},
                    "vendors": {s.get("vendor") for s in srcs if s.get("vendor")},
                    "communities": len(hits), "facts": loc["count"]})
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    args = ap.parse_args()
    settings = get_extract_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    graphiti = build_graphiti(settings)
    embedder = build_embedder(settings)
    try:
        resolver = await ScopeResolver.load(driver)
        results = []
        for q in json.loads(args.questions.read_text()):
            shape = classify_shape(q["question"], resolver, resolver.resolve(q["question"]))
            ev = await _evidence(graphiti, driver, embedder, settings, q["question"], shape)
            r = check_question({**q, "_shape": shape.kind}, ev)
            results.append(r)
            print(f"{'PASS' if r['pass'] else 'FAIL'} {shape.kind:11} "
                  f"facets {r['facets_with_evidence']}/{r['facets']} "
                  f"expected={r['expected_hit']} vendors={r['vendors_found']} "
                  f"| {q['question']}", flush=True)
        passed = sum(r["pass"] for r in results)
        print(f"PASS {passed}/{len(results)}")
        return 0 if passed == len(results) else 1
    finally:
        await graphiti.close()
        await driver.close()
        await embedder.client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: Update CLAUDE.md**

In the `answer_api/` bullet of `CLAUDE.md`, after `` `drift.py`, `timeline.py`, eval harness. `` add:

```
**Question shape (2026-10-12):** `shape.py` classifies every question as single /
comparison (two or more named facets) / enumeration (cross-vendor wording, no explicit
scope); local, global and DRIFT retrieve per facet, and `enumeration.py` checks EVERY
vendor with content (no ceiling) and closes with a deterministic "Checked / Documented
for" line. `eval_retrieval.py` is the free retrieval gate run before any paid eval.
```

- [ ] **Step 5: Run to verify pass, full suite, commit**

```bash
uv run --extra dev pytest tests/unit/test_eval_retrieval.py -q
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live" -q
git add src/answer_api/eval_retrieval.py tests/unit/test_eval_retrieval.py CLAUDE.md
git commit -m "feat(eval): free retrieval gate -- per-facet evidence, expected articles, vendors"
```

Expected: full suite PASS.

---

### Task 10: Ground truth, golden additions, free gate, one paid eval (controller)

**Not for a subagent.** Requires user confirmation at two points and spends money once.

- [ ] **Step 1: Build enumeration ground truth (free).** For each enumeration question in `router_golden_tier1.json` ("Which backup vendors support immutable backups…", "What approaches do backup vendors take to verify…"), query Neo4j per content vendor (AWS, Cohesity, Microsoft, Veeam) for current facts matching the capability (immutability / WORM / vault lock / object lock; restore verification / restore testing / SureBackup), and list 2–3 example facts with article titles per vendor. Show the user; on confirmation, add `"expected_vendors": [...]` to those entries.
- [ ] **Step 2: Propose ~4 new questions** (two enumerations, e.g. "Which vendors support cross-region backup copies?"; two comparisons in the other direction, AWS vs Veeam and Cohesity vs Veeam) with expected articles and, for enumerations, expected vendors gathered the same way. Add them only after user approval.
- [ ] **Step 3: Review narrow expected-article lists.** For each golden question whose `eval_retrieval` hit is False, show the user the retrieved articles next to the expected ones; the user decides whether to widen the list.
- [ ] **Step 4: Free gate.** `uv run python -m answer_api.eval_retrieval --questions src/answer_api/router_golden_tier1.json` must print `PASS N/N`. Fix and repeat until it does; no paid run before.
- [ ] **Step 5: One paid Tier 1 eval** (~$0.20): `uv run python -m answer_api.eval_router --questions src/answer_api/router_golden_tier1.json`. Save runs 1–5 first (the harness overwrites the report), record the result as run 6. Targets: routing 1.00, faithfulness 5.0, misattributed 0, scoped grounding 1.00, facet coverage 1.00, enumeration completeness 1.00, no Veeam-only regression. On a miss: diagnose with `eval_retrieval` and the saved raw answers; at most one more paid run, then report to the user.

---

## Amendment (2026-10-13, approved by the user after the first free-gate run)

The free gate passed 8/12. Diagnosis on the four misses: (1) every facet searches with
the WHOLE question, so the AWS side of "Compare Veeam backup copy jobs with AWS Backup
cross-Region copy…" and every vendor of a "Which backup vendors…" question retrieve
generic pages; (2) the scoped post-filter over a corpus-wide top-450 found zero
Cohesity immutability facts although 29 exist — a large vendor fills the 450. Tasks 11
and 12 fix these; Task 13 adds the confirmed golden ground truth and questions. Task 10
(free gate, one paid eval) runs after them.

### Task 11: Scoped search falls back to ranking the scope's own facts

**Files:**
- Modify: `src/answer_api/search.py` (`search_local`; new `_scoped_scan`)
- Test: `tests/unit/test_search_scoped_scan.py`

**Interfaces:**
- Produces: `search_local(...)` unchanged signature. When `scoped` and the adaptive limits
  are exhausted with fewer than `k` edges, it calls `_scoped_scan(graphiti, driver, *, q,
  scope, k, group_id, exclude: set[str]) -> list` and appends its edges (deduplicated by
  uuid, current facts only unless `include_invalid`) until `k`.

Requirements:
1. `_scoped_scan` embeds `q` with the same embedder graphiti uses for edge search
   (`graphiti.embedder.create(input_data=[q])` — confirm the call shape against
   graphiti-core 0.30.1 and how `vector_search.py` embeds queries; use the same) and ranks
   the scope's RELATES_TO edges by `vector.similarity.cosine(f.fact_embedding, $vec)`
   (Neo4j normalises to (1+cos)/2; ranking is unaffected), returning the top `k` not in
   `exclude`.
2. Scope membership is the structural chain used by `scope._EPISODES` (vendor / product /
   source names, case-insensitive) → the scope's Episodic uuids → edges whose
   `episodes` list intersects them. Write the Cypher so it does not scan every
   RELATES_TO edge in the corpus: go from the scope's episodes through
   `(:Episodic)-[:MENTIONS]->(:Entity)` to candidate entities and their RELATES_TO edges,
   then keep edges whose `episodes` intersect the scope's episodes. Measure it against the
   live graph read-only (`.env` is configured in the main repo checkout
   `/home/openclaw/Documents/Shared/projects/graph-rag/.env`; source it with
   `set -a; . <path>; set +a` but NEVER print it) for three scopes: vendor Cohesity,
   the six Veeam Agent guides (sources "Veeam Agent for Microsoft Windows", "Veeam Agent
   for Linux", …), and product "AWS Backup". Each must finish in under 3 s; report the
   timings. If a scope has more than `local_scoped_scan_max_episodes` episodes (new
   setting, default 20000), skip the scan and log at INFO (large scopes are served by the
   post-filter).
3. Returned objects expose what `search_local` reads: `uuid`, `fact`, `episodes`,
   `valid_at`, `invalid_at` (a small dataclass is fine).
4. Unscoped searches and scoped searches that fill `k` from the post-filter never call
   the scan (byte-identical path).
5. A scan failure is logged (warning) and the post-filter results are returned as is —
   never an exception that empties the facet.

Tests (hermetic, fake graphiti/driver): scan not called when post-filter fills k; called
when short, results appended after post-filter results and deduplicated; not called for
unscoped; over-limit scope skips the scan; scan exception returns the post-filter
results. Also add an `@live`-marked test (excluded from the default run) asserting a
Cohesity-scoped search for "Which backup vendors support immutable backups?" returns at
least one fact mentioning DataLock or WORM.

### Task 12: Per-facet questions

**Files:**
- Modify: `src/answer_api/shape.py` (`Facet.query`; new `async def facet_queries(...)`)
- Modify: `src/answer_api/router.py` (compute once per comparison/enumeration request)
- Modify: `src/answer_api/synthesize.py`, `global_search.py`, `drift.py`,
  `enumeration.py`, `eval_retrieval.py` (use `f.query or q` for retrieval)
- Test: `tests/unit/test_facet_queries.py` plus updates to mode tests

**Interfaces:**
- `Facet` gains `query: str = ""` (frozen; default keeps every existing construction valid).
- `async def facet_queries(client, model, q: str, facets: Sequence[Facet]) -> tuple[Facet, ...]`
  — ONE cheap-tier call returning JSON `{"<label>": "<standalone question about that side
  only>", ...}`; returns the facets with `query` set (`dataclasses.replace`). Prompt: rewrite
  the QUESTION into one standalone question per SIDE, each asking only about that side,
  keeping the question's intent and its specific terms (feature names, workloads), never
  adding facts or other vendors. Use `max_tokens` with headroom (≥ 1500) and
  `usable_content`; on an unusable reply or a missing/empty label, fall back to `q` for that
  facet and log a warning (a fallback is not an absence: retrieval still runs).
- Router: after classifying, for `comparison` and `enumeration` shapes (not under a mode
  override for enumerations, consistent with the existing rule), call `facet_queries` with
  the MAP client/model (`map_client`, `map_model`) and pass the rewritten facets on.
- Retrieval uses the facet question; prompts that ANSWER keep the original question:
  - local: per-facet `search_local(q=f.query or q)`; the answer prompt keeps `q`.
  - global: per-facet `shortlist_communities(q=f.query or q)` and `map_report` with the
    facet question; the reduce prompt keeps `q`.
  - DRIFT: per-facet primer shortlist and the coverage follow-ups use `f.query or q`.
  - enumeration: `_gather` uses `f.query or q`; `_check` keeps `q` (it names the vendor).
  - eval_retrieval: compute facet queries the same way (it then makes one cheap call per
    comparison/enumeration question — update its docstring: "no generative calls except
    the per-facet question rewrite").
- Single shape: no call, unchanged.

Tests: facet_queries parses the JSON and sets each query; falls back per facet on a
missing label and on an unusable reply (with a warning); router calls it once for a
comparison and for an enumeration, never for single or an overridden enumeration; each
mode's per-facet retrieval receives `f.query` while its answering prompt still contains
the original question.

### Task 13: Golden ground truth and new questions (data)

**Files:**
- Modify: `src/answer_api/router_golden_tier1.json`

Add `"expected_vendors"` to two existing entries:
- "Which backup vendors support immutable backups, and how do their approaches differ?" →
  `["AWS", "Cohesity", "Microsoft", "Veeam"]`
- "What approaches do backup vendors take to verify that backups can actually be
  restored?" → `["AWS", "Microsoft", "Veeam"]`

Append four entries (same key order and style as the existing ones):
1. `{"question": "Which backup vendors can keep a copy of backups in another region or at
   another site?", "intent": "global", "expected_modes": ["global", "drift"], "kind":
   "cross_vendor", "expected_article_ids": ["2c92266f-f84c-49c2-9e89-3b4376ec9043",
   "3f4fec00-cd76-41b4-ac19-bec8894d09bb", "41cb167c-fffd-4a26-acdb-8db2e2992b2a",
   "78e2b9e5-bf0c-4e76-8b83-fc54e76f52cd", "8f869443-b48d-4a45-a9fb-7d0df772e632",
   "e53dfaee-916c-4947-9224-43ba63f6a14f", "ecc02e55-394d-4e8d-8020-33c62a938c19"],
   "expected_vendors": ["AWS", "Microsoft", "Veeam"]}`
2. `{"question": "Compare AWS Backup Vault Lock with Cohesity DataLock.", "intent": "global",
   "expected_modes": ["global", "local", "drift"], "kind": "comparison",
   "expected_article_ids": ["dad69d7c-189b-4d53-8d70-40697eae685a",
   "3b8017a4-63ab-4fb7-b7dd-42dd6c574935", "0f3cb305-4373-49e3-baeb-e02daa49cda7",
   "6f3b9166-974d-43cb-891a-db656254d314", "1e48210b-9421-404a-886f-3794a60408d8"]}`
3. `{"question": "How does Cohesity FortKnox compare with Veeam hardened repositories for
   ransomware protection?", "intent": "global", "expected_modes": ["global", "local",
   "drift"], "kind": "comparison", "expected_article_ids":
   ["fc409999-9c70-47cd-9944-dc829cf5824c", "bc0bdcbc-b80d-4498-bf90-1cd52c4e8150",
   "689a50cd-5ae3-48dd-8c65-12a765199cb5", "1eaa2ba2-ea95-4feb-ad13-4f51ce86bd68",
   "572832b9-67d7-40cc-b228-ea4c4139c8c9", "f3894faa-9a9a-414a-b9bf-43464ad2114a",
   "e586b6b1-cb89-47fa-b572-80363237ff7e", "f6e78d99-29ad-46c2-8f7a-579d6d80b21c"]}`
4. `{"question": "Which vendors does Veeam integrate with?", "intent": "global",
   "expected_modes": ["global", "drift"], "kind": "cross_vendor", "expected_article_ids": []}`
   (a probe: routing/faithfulness only; no ground truth).

Run `uv run --extra dev pytest tests/unit/test_router_golden_schema.py -q`; if the schema
test rejects an empty `expected_article_ids`, allow it only for `kind == "cross_vendor"`
entries and say so.

### Task 14: Better side questions and reranked facet searches (amendment 2)

Gate run 2 (12/16) showed: the cheap model rewrote enumeration sides as "Which backup
vendors support immutable backups on AWS…" (vendor treated as a platform, cross-vendor
wording kept); and large scopes (Veeam, Microsoft) fill k from the corpus-wide hybrid
search with poor ranking ("PostgreSQL WAL Files Backup" for a ransomware question).
Measured: a vendor-subject rewrite prompt fixes the rewrites on the cheap model; a
cross-encoder rerank of a larger candidate pool puts relevant facts on top.

**Files:** `src/answer_api/shape.py`, `src/answer_api/search.py`,
`src/graph_extract/config.py`, facet callers (`synthesize._answer_local_facets`,
`drift` facet-scoped follow-ups, `enumeration._gather`, `eval_retrieval._evidence`),
tests.

1. Replace `_FACET_QUERIES_PROMPT` with exactly:
```python
_FACET_QUERIES_PROMPT = (
    "Each SIDE below is a vendor or product whose OWN documentation will be searched "
    "separately. Rewrite the QUESTION into one standalone question per SIDE, asked "
    "directly about that side alone, as you would ask its documentation: start with "
    "the side's name as the subject (e.g. \"How does <side> ...\", \"Does <side> ...\"). "
    "Keep the QUESTION's intent and its specific terms (feature names, workloads, "
    "settings). Drop cross-vendor wording such as \"which vendors\", \"compare\", "
    "\"differ\" or \"each\". Do not answer, do not add facts, and do not mention any other "
    "side or vendor. Respond with ONLY JSON mapping each side's label, exactly as "
    "written below, to its question: {{\"<label>\": \"<question>\", ...}}.\n\n"
    "SIDES:\n{sides}\n\nQUESTION: {q}\nJSON:")
```
2. `search_local` gains `rerank_pool: int = 0`. When `rerank_pool > 0`, `settings` is
   given and `rerank_configured(settings)`: collect up to `max(k, rerank_pool)` filtered
   edges (same adaptive limits / scan fallback, with the target raised to the pool size),
   rerank them with `answer_api.rerank.rerank(q, [e.fact ...], top_k=k, settings=settings)`
   and return that order, padding a short/empty rerank result from the original order;
   on reranker failure (None/raise) log a warning and return the original top k. With
   `rerank_pool == 0` (default) behaviour is byte-identical to today.
3. New setting `local_facet_rerank_pool: int = 40` (comment: gate run 2 measurement).
4. Facet callers pass `rerank_pool=settings.local_facet_rerank_pool` and
   `settings=settings`: `_answer_local_facets` (per facet), DRIFT follow-ups that run in a
   facet scope (`fu.scope is not None`) and the facet coverage follow-ups, enumeration
   `_gather`'s local search, and `eval_retrieval._evidence` for comparison/enumeration
   shapes. Single-shape searches never pass it.
5. Tests: prompt text (snapshot of the new constant's key phrases); rerank_pool reorders
   via a fake rerank and respects k; pads a short result; failure falls back with a
   warning; pool 0 leaves the call path unchanged (no rerank call); each facet caller
   passes the pool, single callers do not.

### Task 15: Vendor checks on the vendor's own question; global sides with no community findings use their own facts (run 6 fixes)

Tier 1 run 6 (paid, $0.15): enumeration completeness 0/3, facet coverage 5/6.
1. "Which backup vendors support immutable backups…" answered "None of the checked vendors
   … document this": each vendor's `_check` received the ORIGINAL cross-vendor question
   ("…and how do their approaches differ?"), which a single vendor's evidence cannot show,
   so the cheap check model said `documented: false` for all four. (Task 12 had ruled that
   `_check` keeps `q`; that ruling is reversed.)
2. "How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in
   Azure Backup?" was routed global; the Veeam Backup for Microsoft 365 facet's scoped
   community shortlist found nothing usable, so the reducer said the documentation does
   not cover it — although the free gate's local search found that side's facts. A false
   absence.

**Files:** `src/answer_api/enumeration.py`, `src/answer_api/global_search.py`, tests.

1. `enumeration._check` receives the facet's question (`f.query or q`) and the
   `_CHECK_PROMPT` asks whether the evidence answers THAT question for the vendor
   ("Decide whether the EVIDENCE below, all from {vendor}'s documentation, answers the
   QUESTION about {vendor}…"); keep every other guard (evidence-only, absence is not a
   finding, ids from the evidence, JSON shape, retry). The final synthesis prompt keeps the
   original question.
2. `global_search._global_search_facets`: a facet whose shortlist succeeded but yields no
   usable in-scope findings (no communities, or every community's facts fall outside the
   facet scope) gets a LOCAL fill: `search_local(q=f.query or q, k=settings.global_facet_local_k
   (new setting, default 8), scope=f.scope, settings=settings,
   fetch_cap=settings.local_scoped_fetch_cap, rerank_pool=settings.local_facet_rerank_pool)`.
   Its facts are rendered under that facet's heading in the reduce prompt as `[N] (Vendor ·
   Product) fact` lines (numbered in the same marker map, resolved through the same
   `resolve_citations` path as community facts, so citations stay fact-id based). Only if
   the local fill also returns nothing does the facet render "(no findings)". A local-fill
   exception degrades the facet (Could not check), never "(no findings)". Communities_used is
   unchanged (local facts are not communities). Shortlist failures and all-maps-failed
   facets keep their existing degraded handling.
3. Tests: `_check` gets the facet question and the synthesis still gets the original
   question; a facet with an empty shortlist gets local facts under its heading, cited
   through the marker map with resolved sources; a facet with communities whose facts are
   all out of scope also gets the local fill; local fill returning nothing → "(no findings)";
   local fill raising → degraded + note; facets with in-scope findings do NOT call the local
   fill; single path untouched.

### Task 16: Product-named side questions, enumeration marker rule, cross-site ground truth (run 7 fixes)

Run 7: restore-verification enumeration documented only Veeam. Measured: "How does
Microsoft verify that backups can actually be restored?" never surfaces Microsoft's
"Azure Backup test restores can be used to verify … RPO" fact; "How does Azure Backup
verify …" puts it first (documentation is organised by product, not vendor). Run 7 also:
the enumeration answer carried a 14-marker sentence (bag share 0.64); one closing
comparison sentence cited both vendors together (judge: 1 misattribution). User decision
2026-10-13: Cohesity does cross-site copies → counts for the cross-region question.

**Files:** `src/answer_api/scope.py`, `src/answer_api/shape.py`, `src/answer_api/enumeration.py`,
`src/answer_api/global_search.py`, `src/answer_api/synthesize.py`, `src/answer_api/drift.py`,
`src/answer_api/router_golden_tier1.json`, tests.

1. `ScopeResolver.products_of(vendor: str) -> tuple[str, ...]` (structural products of that
   vendor, sorted). `Facet` gains `products: tuple[str, ...] = ()`; `classify_shape` fills it for
   ENUMERATION facets (one per content vendor). Comparison facets keep `()`.
2. `facet_queries`: in the SIDES list, a facet with `products` renders as
   `- <label> (its documented products: P1, P2, …)`; add to `_FACET_QUERIES_PROMPT` (keep
   everything else verbatim): "When a side lists its documented products, make the product
   the question concerns the subject (e.g. \"How does Azure Backup …\"); if several apply, name
   the most relevant one or two." The label used as the JSON key stays the facet label.
3. `_ENUM_PROMPT`: add "Cite each claim with the one or two markers of the facts it rests on;
   never attach a long list of markers to one sentence."
4. Every facet-answer prompt (`_FACET_PROMPT` in synthesize, `_REDUCE_FACETS_PROMPT`,
   `_SYNTH_FACETS_PROMPT` in drift, `_ENUM_PROMPT`): add "If you compare the sides, give each
   side's claim in its own clause with only that side's markers."
5. Golden: the entry "Which backup vendors can keep a copy of backups in another region or at
   another site?" → `"expected_vendors": ["AWS", "Cohesity", "Microsoft", "Veeam"]`.
6. Tests: `products_of`; enumeration facets carry products, comparison facets do not;
   `facet_queries` renders products in the SIDES list and keeps label keys; the new prompt
   rules are present; golden schema still passes.
