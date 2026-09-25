# Vendor and Product End to End — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every answer names the vendors/products each claim applies to, and a question
that names vendors/products is answered from their documentation only — across local,
timeline, global, DRIFT and the `/answer` router.

**Architecture:** a `ScopeResolver` (structural layer + alias file) turns a question or
explicit API params into a `Scope`; every retrieval mode filters by it (local/timeline:
episode set; global/DRIFT: per-community in-scope share at query time, plus a fact filter
before reduce). Attribution labels come from the provenance chain that
`Provenance.resolve_citations` already walks (`sources[].vendor/product`), rendered by one
module into every fact line an LLM sees and aggregated into an `applies_to` envelope field.

**Tech Stack:** Python 3.12, FastAPI, neo4j async driver, graphiti-core 0.30.1 (pinned,
never patched here), pytest + testcontainers.

**Spec:** `docs/superpowers/specs/2026-09-25-vendor-product-end-to-end-design.md`

## Global Constraints

- Vendor/product labels come ONLY from graph traversal; the LLM never authors a label or URL (invariant #2).
- Read ONLY structural Vendor/Product nodes: `(:Vendor)-[:HAS_PRODUCT]->(:Product)` with a non-null `id`. graphiti entities also carry `:Vendor`/`:Product` labels — never match by label alone.
- Label format, exactly: `(Vendor · Product)`; several pairs joined by `; ` → `(Veeam · Veeam Backup & Replication; Commvault · Commvault Cloud)`. Always `Vendor · Product`, even when the product repeats the vendor.
- `global_scope_candidates` default **24**; `global_scope_min_share` default **0.5**.
- No new LLM calls in the answer path.
- An empty `Scope` must reproduce today's behaviour exactly (every existing test stays green unmodified except where a signature changes).
- Tests: `UV_OFFLINE=1 uv run --frozen --extra dev pytest ...`; lint `uv run ruff check src tests` (tests too; no semicolons in fakes — E702); types `uv run mypy src`.
- Commit messages end with the two attribution lines:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv`.

## Plan rulings (decided while planning; the spec is the authority)

1. **Spec §4.3 step 4 "drop fact lines outside the scope in the map input"** — the map
   step's input is the community report's prose (`full_report`), not fact lines. The
   equivalent is applied to the map step's *output*: each `MapResult.fact_ids` is
   filtered to in-scope facts before numbering and reduce. Same effect on the answer; the
   LLM-written report text is untouched.
2. **Scope matches vendor OR product:** an article is in scope if its vendor is in
   `scope.vendors` (named directly) or its product is in `scope.products`. A product
   named alone does not widen to its whole vendor.
3. **The structural layer only lists ingested vendors** (6 today). A vendor that is not
   ingested cannot be detected, so such a question stays unscoped — correct, since
   nothing of theirs exists to scope to.
4. **Resolver reload:** lazily on use when older than `scope_reload_seconds` (default
   3600), instead of the spec's "freshness interval" — the freshness module has no
   interval.
5. **Label source for global reduce:** `Provenance.resolve_citations` over the selected
   facts, called once before reduce; the same result is reused to build citations (one
   query, not two).

## File structure

| file | responsibility |
|---|---|
| `src/answer_api/scope.py` (new) | `Scope`, `ScopeResolver` (catalog load, aliases, detection, explicit resolution), `scope_episode_uuids` |
| `src/answer_api/scope_aliases.json` (new) | alias → canonical vendor/product name |
| `src/answer_api/attribution.py` (new) | `pairs_of`, `label`, `fact_line`, `applies_to`, `ATTRIBUTION_RULES` |
| `src/answer_api/search.py` | `search_local(scope=)` replaces `vendor=` |
| `src/answer_api/timeline.py` | `timeline_local(scope=)` replaces `vendor=` |
| `src/answer_api/synthesize.py` | labelled lines + rules in `_PROMPT`; `applies_to` |
| `src/answer_api/global_search.py` | scoped shortlist (`_scope_shares`), map-output fact filter, labelled reduce lines |
| `src/answer_api/drift.py` | scoped primer + follow-ups, labelled synthesis lines |
| `src/answer_api/router.py` | scope to every mode; envelope `scope` + `applies_to`; labelled timeline lines |
| `src/answer_api/app.py` | resolver in lifespan; `vendor`/`product`/`scope` params on every route |
| `src/graph_extract/config.py` | `global_scope_candidates`, `global_scope_min_share`, `scope_reload_seconds` |
| `src/answer_api/eval_router.py` | attribution judge + report fields |

---

### Task 1: Scope and ScopeResolver

**Files:**
- Create: `src/answer_api/scope.py`, `src/answer_api/scope_aliases.json`
- Test: `tests/unit/test_scope.py`, `tests/integration/test_scope_catalog.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Scope:
      vendors: tuple[str, ...] = ()
      products: tuple[str, ...] = ()
      source: str = "none"                  # "explicit" | "detected" | "none"
      def is_empty(self) -> bool
      def as_dict(self) -> dict             # {"vendors": [...], "products": [...], "source": str}

  class UnknownScopeName(ValueError): ...   # carries .names: list[str]

  class ScopeResolver:
      def __init__(self, vendors: list[str], products: list[tuple[str, str]],
                   aliases: dict[str, str]) -> None       # products: (product, vendor)
      @classmethod
      async def load(cls, driver, aliases_path: Path | None = None) -> "ScopeResolver"
      def detect(self, q: str) -> Scope                     # source "detected" or "none"
      def resolve(self, q: str, *, vendors: list[str] | None = None,
                  products: list[str] | None = None, disabled: bool = False) -> Scope
      def vendor_of(self, product: str) -> str | None

  async def scope_episode_uuids(driver, scope: Scope) -> set[str]
  ```

- [ ] **Step 1: alias seed** — `src/answer_api/scope_aliases.json`:
  ```json
  {"VBR": "Veeam Backup & Replication", "VB365": "Veeam Backup for Microsoft 365",
   "VSPC": "Veeam Service Provider Console", "Kasten": "Veeam Kasten for Kubernetes",
   "Azure": "Microsoft", "Amazon": "AWS"}
  ```
  Add it to the wheel: `pyproject.toml` already packages `src/answer_api` (JSON beside
  `router_golden.json` ships the same way — confirm with `tests/unit/test_packaging.py`).

- [ ] **Step 2: failing unit tests** — `tests/unit/test_scope.py`:
  ```python
  import pytest
  from answer_api.scope import Scope, ScopeResolver, UnknownScopeName

  VENDORS = ["AWS", "Microsoft", "Veeam", "Cohesity"]
  PRODUCTS = [("AWS Backup", "AWS"), ("Azure Backup", "Microsoft"),
              ("Microsoft 365 Backup", "Microsoft"),
              ("Veeam Backup & Replication", "Veeam"),
              ("Veeam Backup for Microsoft 365", "Veeam"), ("FortKnox", "Cohesity")]
  ALIASES = {"VBR": "Veeam Backup & Replication", "Azure": "Microsoft",
             "Ghost": "No Such Product"}

  def _r():
      return ScopeResolver(VENDORS, PRODUCTS, ALIASES)

  def test_products_and_vendors_are_detected_case_insensitively():
      s = _r().detect("compare aws backup and Azure Backup encryption")
      assert s == Scope((), ("AWS Backup", "Azure Backup"), "detected")

  def test_longest_match_wins_so_a_product_is_not_split_into_vendors():
      s = _r().detect("How does Veeam Backup for Microsoft 365 restore mail?")
      assert s.products == ("Veeam Backup for Microsoft 365",) and s.vendors == ()

  def test_a_bare_vendor_name_scopes_to_the_vendor():
      assert _r().detect("What does Cohesity offer for ransomware?").vendors == ("Cohesity",)

  def test_aliases_resolve_to_canonical_names_and_dangling_aliases_are_ignored():
      assert _r().detect("VBR hardened repository").products == ("Veeam Backup & Replication",)
      assert _r().detect("Azure soft delete").vendors == ("Microsoft",)
      assert _r().detect("Ghost feature").is_empty()

  def test_whole_words_only():
      assert _r().detect("aws-style awsome tooling").vendors == ("AWS",)   # "aws" matches, "awsome" does not
      assert _r().detect("backupsome FortKnoxes").is_empty()

  def test_cross_vendor_phrasing_naming_nobody_is_unscoped():
      s = _r().detect("How should I plan retention across all vendors?")
      assert s.is_empty() and s.source == "none"

  def test_explicit_names_override_detection_and_are_canonicalised():
      s = _r().resolve("Compare AWS Backup and Azure Backup", vendors=["veeam"])
      assert s == Scope(("Veeam",), (), "explicit")

  def test_unknown_explicit_name_raises():
      with pytest.raises(UnknownScopeName) as e:
          _r().resolve("q", products=["Nope"])
      assert e.value.names == ["Nope"]

  def test_disabled_returns_an_empty_scope():
      assert _r().resolve("Compare AWS Backup and Azure Backup", disabled=True).is_empty()

  def test_as_dict_and_vendor_of():
      r = _r()
      assert Scope(("AWS",), ("FortKnox",), "detected").as_dict() == {
          "vendors": ["AWS"], "products": ["FortKnox"], "source": "detected"}
      assert r.vendor_of("FortKnox") == "Cohesity" and r.vendor_of("x") is None
  ```
  Run: `UV_OFFLINE=1 uv run --frozen --extra dev pytest tests/unit/test_scope.py -q` — FAIL (module missing).

- [ ] **Step 3: implement `src/answer_api/scope.py`**
  ```python
  """Which vendors/products a question is about (spec §4.1).

  Built from the STRUCTURAL layer only: graphiti's extracted entities also carry
  :Vendor/:Product labels, so nodes are selected by the HAS_PRODUCT edge and a
  non-null `id`, never by label alone. Detection is deterministic: whole-word,
  case-insensitive, longest match first, so a product name is never split into
  the vendor names it contains ("Veeam Backup for Microsoft 365").
  """
  from __future__ import annotations

  import json
  import logging
  import re
  from dataclasses import dataclass
  from pathlib import Path

  logger = logging.getLogger(__name__)
  _ALIASES = Path(__file__).with_name("scope_aliases.json")

  _CATALOG = (
      "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product) "
      "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
      "RETURN v.name AS vendor, p.name AS product")
  _EPISODES = (
      "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(:Source)"
      "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
      "WHERE v.id IS NOT NULL AND (v.name IN $vendors OR p.name IN $products) "
      "RETURN collect(DISTINCT e.uuid) AS u")


  @dataclass(frozen=True)
  class Scope:
      vendors: tuple[str, ...] = ()
      products: tuple[str, ...] = ()
      source: str = "none"

      def is_empty(self) -> bool:
          return not self.vendors and not self.products

      def as_dict(self) -> dict:
          return {"vendors": list(self.vendors), "products": list(self.products),
                  "source": self.source}


  class UnknownScopeName(ValueError):
      def __init__(self, names: list[str]) -> None:
          super().__init__(f"unknown vendor/product: {', '.join(names)}")
          self.names = names


  class ScopeResolver:
      def __init__(self, vendors: list[str], products: list[tuple[str, str]],
                   aliases: dict[str, str]) -> None:
          self._vendors = {v.lower(): v for v in vendors}
          self._products = {p.lower(): p for p, _ in products}
          self._vendor_of = {p: v for p, v in products}
          terms: dict[str, tuple[str, str]] = {}          # lower term -> (kind, canonical)
          for v in vendors:
              terms[v.lower()] = ("vendor", v)
          for p, _ in products:
              terms[p.lower()] = ("product", p)
          for alias, target in aliases.items():
              hit = self._canonical(target)
              if hit is None:
                  logger.warning("scope alias %r -> %r: target is not an ingested "
                                 "vendor/product; ignored", alias, target)
                  continue
              terms[alias.lower()] = hit
          self._terms = terms
          alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
          self._re = re.compile(rf"(?<![\w-])({alternation})(?![\w])", re.IGNORECASE) \
              if terms else None

      def _canonical(self, name: str) -> tuple[str, str] | None:
          n = name.strip().lower()
          if n in self._products:
              return ("product", self._products[n])
          if n in self._vendors:
              return ("vendor", self._vendors[n])
          return None

      def vendor_of(self, product: str) -> str | None:
          return self._vendor_of.get(product)

      def detect(self, q: str) -> Scope:
          if self._re is None:
              return Scope()
          vendors: list[str] = []
          products: list[str] = []
          for m in self._re.finditer(q):
              kind, name = self._terms[m.group(1).lower()]
              bucket = products if kind == "product" else vendors
              if name not in bucket:
                  bucket.append(name)
          if not vendors and not products:
              return Scope()
          return Scope(tuple(vendors), tuple(products), "detected")

      def resolve(self, q: str, *, vendors: list[str] | None = None,
                  products: list[str] | None = None, disabled: bool = False) -> Scope:
          if disabled:
              return Scope()
          if vendors or products:
              unknown: list[str] = []
              vs: list[str] = []
              ps: list[str] = []
              for name in vendors or []:
                  hit = self._canonical(name)
                  if hit is None or hit[0] != "vendor":
                      unknown.append(name)
                  elif hit[1] not in vs:
                      vs.append(hit[1])
              for name in products or []:
                  hit = self._canonical(name)
                  if hit is None or hit[0] != "product":
                      unknown.append(name)
                  elif hit[1] not in ps:
                      ps.append(hit[1])
              if unknown:
                  raise UnknownScopeName(unknown)
              return Scope(tuple(vs), tuple(ps), "explicit")
          return self.detect(q)

      @classmethod
      async def load(cls, driver, aliases_path: Path | None = None) -> "ScopeResolver":
          records, _, _ = await driver.execute_query(_CATALOG)
          pairs = [(r["product"], r["vendor"]) for r in records]
          vendors = sorted({v for _, v in pairs})
          aliases = json.loads((aliases_path or _ALIASES).read_text())
          return cls(vendors, pairs, aliases)


  async def scope_episode_uuids(driver, scope: Scope) -> set[str]:
      """Episode uuids of every article in scope (vendor named directly OR product
      named). Empty scope -> empty set; callers skip filtering on an empty scope."""
      if scope.is_empty():
          return set()
      records, _, _ = await driver.execute_query(
          _EPISODES, vendors=list(scope.vendors), products=list(scope.products))
      return set(records[0]["u"]) if records else set()
  ```
  Note `(?<![\w-])` keeps "aws-style" matching "aws" at its start while `(?![\w])`
  rejects "awsome". Verify the whole-word test passes as written; adjust the lookarounds,
  never the test's intent.

- [ ] **Step 4: run** the unit tests — PASS.

- [ ] **Step 5: integration test** — `tests/integration/test_scope_catalog.py` uses the
  `extract_neo4j` fixture (see `tests/integration/test_state_crosscheck_read.py` for the
  pattern): create `(:Vendor {id:'v1',name:'Veeam'})-[:HAS_PRODUCT]->(:Product {id:'p1',
  name:'Veeam ONE'})-[:HAS_SOURCE]->(:Source {id:'s1'})-[:HAS_ARTICLE]->(:Article {id:'a1'})
  -[:HAS_EPISODE]->(:Episodic {uuid:'e1'})`, plus a graphiti-style
  `(:Entity:Vendor {uuid:'x', name:'Fake Vendor'})` and `(:Entity:Product {uuid:'y',
  name:'Cohesity Copilot'})` with no `id`. Assert `ScopeResolver.load` detects
  "Veeam ONE" and does NOT detect "Fake Vendor" or "Cohesity Copilot";
  `scope_episode_uuids(driver, Scope(("Veeam",),(),"x")) == {"e1"}` and with
  `Scope((), ("Veeam ONE",), "x")` also `{"e1"}`. Wipe with `MATCH (n) DETACH DELETE n`
  before and after.

- [ ] **Step 6: run, lint, commit** — `feat(answer): Scope + ScopeResolver (vendor/product detection from the structural layer)`.

---

### Task 2: Attribution helpers

**Files:**
- Create: `src/answer_api/attribution.py`
- Test: `tests/unit/test_attribution.py`

**Interfaces:**
- Produces:
  ```python
  def pairs_of(sources: list[dict]) -> list[tuple[str, str]]   # ordered unique (vendor, product), skipping missing
  def label(sources: list[dict]) -> str                         # "(V · P; V2 · P2)" or ""
  def fact_line(n: int, fact: str, sources: list[dict]) -> str  # "[n] (V · P) fact" / "[n] fact"
  def applies_to(citations: list[dict]) -> list[dict]           # [{"vendor","products","facts"}]
  def in_scope(sources: list[dict], scope: Scope) -> bool
  ATTRIBUTION_RULES: str
  ```

- [ ] **Step 1: failing tests** — `tests/unit/test_attribution.py`:
  ```python
  from answer_api.attribution import applies_to, fact_line, in_scope, label, pairs_of
  from answer_api.scope import Scope

  V = {"vendor": "Veeam", "product": "Veeam Backup & Replication", "url": "u1"}
  V2 = {"vendor": "Veeam", "product": "Veeam Backup & Replication", "url": "u2"}
  C = {"vendor": "Commvault", "product": "Commvault Cloud", "url": "u3"}
  NONE = {"vendor": None, "product": None, "url": "u4"}

  def test_pairs_are_ordered_unique_and_skip_unlabelled_sources():
      assert pairs_of([V, V2, C, NONE]) == [("Veeam", "Veeam Backup & Replication"),
                                             ("Commvault", "Commvault Cloud")]

  def test_label_format_is_exact():
      assert label([V, C]) == "(Veeam · Veeam Backup & Replication; Commvault · Commvault Cloud)"
      assert label([NONE]) == "" and label([]) == ""

  def test_fact_line_with_and_without_label():
      assert fact_line(3, "Immutability lasts 7 days.", [V]) == \
          "[3] (Veeam · Veeam Backup & Replication) Immutability lasts 7 days."
      assert fact_line(4, "x", []) == "[4] x"

  def test_applies_to_counts_facts_per_vendor_not_sources():
      cits = [{"sources": [V, V2]}, {"sources": [V, C]}, {"sources": [NONE]}]
      assert applies_to(cits) == [
          {"vendor": "Veeam", "products": ["Veeam Backup & Replication"], "facts": 2},
          {"vendor": "Commvault", "products": ["Commvault Cloud"], "facts": 1}]

  def test_in_scope_matches_vendor_or_product_and_empty_scope_matches_all():
      assert in_scope([V], Scope(("Veeam",), (), "x"))
      assert in_scope([C], Scope((), ("Commvault Cloud",), "x"))
      assert not in_scope([C], Scope(("Veeam",), (), "x"))
      assert in_scope([C], Scope())
  ```
- [ ] **Step 2: run** — FAIL.
- [ ] **Step 3: implement**
  ```python
  """Vendor/product attribution for fact lines and answers (spec §3).

  Every label is derived from provenance sources (Provenance.resolve_citations:
  fact -> episode -> article -> source -> product -> vendor); the LLM only reads
  them (invariant #2 extended to labels)."""
  from __future__ import annotations

  from answer_api.scope import Scope

  ATTRIBUTION_RULES = (
      "Each fact is prefixed with the vendor and product whose documentation states "
      "it, as (Vendor · Product). Attribute every claim to the vendor/product shown "
      "on the facts it cites. Never apply a fact labelled with one vendor or product "
      "to another. When comparing vendors, keep each vendor's claims separate.\n")


  def pairs_of(sources: list[dict]) -> list[tuple[str, str]]:
      out: list[tuple[str, str]] = []
      for s in sources:
          v, p = s.get("vendor"), s.get("product")
          if v and p and (v, p) not in out:
              out.append((v, p))
      return out


  def label(sources: list[dict]) -> str:
      pairs = pairs_of(sources)
      return "(" + "; ".join(f"{v} · {p}" for v, p in pairs) + ")" if pairs else ""


  def fact_line(n: int, fact: str, sources: list[dict]) -> str:
      lab = label(sources)
      return f"[{n}] {lab} {fact}" if lab else f"[{n}] {fact}"


  def applies_to(citations: list[dict]) -> list[dict]:
      facts: dict[str, int] = {}
      products: dict[str, list[str]] = {}
      for c in citations:
          seen: set[str] = set()
          for v, p in pairs_of(c.get("sources", [])):
              products.setdefault(v, [])
              if p not in products[v]:
                  products[v].append(p)
              if v not in seen:
                  facts[v] = facts.get(v, 0) + 1
                  seen.add(v)
      order = sorted(facts, key=lambda v: (-facts[v], list(facts).index(v)))
      return [{"vendor": v, "products": products[v], "facts": facts[v]} for v in order]


  def in_scope(sources: list[dict], scope: Scope) -> bool:
      if scope.is_empty():
          return True
      return any(v in scope.vendors or p in scope.products for v, p in pairs_of(sources))
  ```
- [ ] **Step 4: run** — PASS. **Step 5: commit** — `feat(answer): attribution labels and applies_to`.

---

### Task 3: Local search, local synthesis and timeline — scope + labels

**Files:**
- Modify: `src/answer_api/search.py` (replace `_vendor_episode_uuids` use; `search_local`)
- Modify: `src/answer_api/timeline.py` (`timeline_local`)
- Modify: `src/answer_api/synthesize.py` (`_PROMPT`, `answer_local`)
- Test: `tests/unit/test_scoped_local.py`; update existing tests that pass `vendor=` to these functions (`grep -rn "vendor=" tests/`)

**Interfaces:**
- Consumes: `Scope`, `scope_episode_uuids` (Task 1); `fact_line`, `applies_to`, `ATTRIBUTION_RULES` (Task 2).
- Produces:
  - `search_local(graphiti, driver, *, q, k=10, scope: Scope | None = None, include_invalid=False, group_id, center_node_uuid=None) -> dict` (the `vendor` parameter is removed)
  - `timeline_local(graphiti, driver, *, q, limit=30, scope: Scope | None = None, group_id) -> dict`
  - `answer_local(graphiti, driver, synth_client, synth_model, *, q, k=15, scope: Scope | None = None, group_id) -> dict` — result gains `"applies_to": list[dict]`

- [ ] **Step 1: failing tests** — `tests/unit/test_scoped_local.py` with fakes (copy the
  fake-graphiti pattern of `tests/unit/test_answer_local.py` / `test_search_local*.py`;
  read them first). Cases:
  1. `search_local(scope=Scope(("Veeam",),(),"detected"))` keeps only edges whose
     `episodes` intersect the set returned by a monkeypatched
     `answer_api.search.scope_episode_uuids`; `scope=None` and `scope=Scope()` never call
     it (assert a call counter stays 0).
  2. `answer_local` sends the synthesis client a prompt containing
     `"[1] (Veeam · Veeam Backup & Replication) <fact>"` and `ATTRIBUTION_RULES`, and
     returns `applies_to == [{"vendor": "Veeam", "products": [...], "facts": 1}]` built
     from the cited facts only (an uncited fact's vendor must not appear).
  3. `timeline_local(scope=...)` filters the same way as search_local.
- [ ] **Step 2: run** — FAIL.
- [ ] **Step 3: implement.** In `search.py` delete `_vendor_episode_uuids` and use:
  ```python
  if scope is not None and not scope.is_empty():
      allowed = await scope_episode_uuids(driver, scope)
      edges = [e for e in edges if allowed.intersection(e.episodes or [])]
  ```
  Same block in `timeline.py`. In `synthesize.py`, `_PROMPT` gains
  `ATTRIBUTION_RULES` after its first sentence (keep every existing rule and the exact
  refusal string), and `answer_local` builds
  `facts_block = "\n".join(fact_line(i, r["fact"], r["sources"]) for i, r in marker_map.items())`
  and adds `"applies_to": applies_to(citations)` to the success return and
  `"applies_to": []` to both refusal returns.
- [ ] **Step 4: run** the new tests plus every existing answer-api unit test
  (`tests/unit/test_answer*.py tests/unit/test_search*.py tests/unit/test_timeline*.py tests/unit/test_finalize_answer.py`) — PASS.
- [ ] **Step 5: commit** — `feat(answer): scope + vendor/product labels in local, timeline and local synthesis`.

---

### Task 4: Global search — scoped shortlist, fact filter, labelled reduce

**Files:**
- Modify: `src/graph_extract/config.py` (three settings)
- Modify: `src/answer_api/global_search.py`
- Test: `tests/unit/test_global_scope.py`, `tests/integration/test_global_scope_shares.py`

**Interfaces:**
- Consumes: `Scope`; `fact_line`, `applies_to`, `in_scope`, `ATTRIBUTION_RULES`.
- Produces:
  - settings: `global_scope_candidates: int = 24`, `global_scope_min_share: float = 0.5`, `scope_reload_seconds: int = 3600`
  - `async def _scope_shares(driver, group_id: str, hits: list[CommunityHit], scope: Scope) -> dict[str, float]` — community_id → in-scope share of its `cited_fact_uuids` (0.0 when it has none)
  - `shortlist_communities(..., scope: Scope | None = None)`
  - `global_search(..., scope: Scope | None = None)` — result gains `"applies_to"`

- [ ] **Step 1: integration test for `_scope_shares`** — fixture graph: two vendors, a
  community citing 3 Veeam facts + 1 Cohesity fact (facts are `RELATES_TO {uuid, group_id,
  episodes:[...]}` between two Entity nodes; episodes linked to Articles under each
  vendor's Product/Source). Assert share for `Scope(("Veeam",),(),"x")` is 0.75 and for
  `Scope((), ("FortKnox",), "x")` is 0.25; a community with no cited facts → 0.0. The
  Cypher:
  ```cypher
  UNWIND $hits AS h
  UNWIND h.facts AS fu
  OPTIONAL MATCH ()-[f:RELATES_TO {uuid: fu, group_id: $g}]->()
  OPTIONAL MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(:Source)
                 -[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic)
  WHERE e.uuid IN f.episodes AND v.id IS NOT NULL
  WITH h.cid AS cid, fu,
       any(x IN collect(DISTINCT [v.name, p.name])
           WHERE x[0] IN $vendors OR x[1] IN $products) AS hit
  RETURN cid, toFloat(sum(CASE WHEN hit THEN 1 ELSE 0 END)) / count(fu) AS share
  ```
  with `hits=[{"cid": h.community_id, "facts": list(h.cited_fact_uuids)} ...]`; add
  `0.0` for any hit absent from the result.
- [ ] **Step 2: unit tests** (`tests/unit/test_global_scope.py`, monkeypatch
  `_scope_shares`, `rerank`, embedder as the existing `tests/unit/test_global*.py` do):
  1. empty scope → `_scope_shares` never called, shortlist identical to today;
  2. scoped, unreranked: candidates ranked with `k=settings.global_scope_candidates`,
     those with share < `global_scope_min_share` dropped, then cut to `k`;
  3. scoped, reranked: the pool is `max(rerank_candidates, global_scope_candidates)`,
     filtered before `rerank` is called (assert the docs handed to `rerank` exclude the
     dropped community);
  4. `global_search(scope=...)` removes out-of-scope fact ids from every `MapResult`
     before numbering (use `resolve_citations` sources for the check, via `in_scope`),
     the reduce prompt contains `"[1] (Veeam · ...) <fact>"` lines and
     `ATTRIBUTION_RULES`, and `applies_to` is returned;
  5. all communities filtered out → returns `communities_used: []` (so the router's
     existing global→local fallback fires).
- [ ] **Step 3: run** — FAIL.
- [ ] **Step 4: implement.** In `shortlist_communities`, after `rows` are read:
  ```python
  scoped = scope is not None and not scope.is_empty()
  if settings is None or not rerank_configured(settings):
      pool_k = max(k, settings.global_scope_candidates) if (scoped and settings) else k
      candidates = _rank_hits(query_vec, rows, k=pool_k, rating_boost=rating_boost)
      if scoped:
          candidates = await _keep_in_scope(driver, group_id, candidates, scope, settings)
      return candidates[:k]
  pool_k = settings.rerank_candidates
  if scoped:
      pool_k = max(pool_k, settings.global_scope_candidates)
  candidates = _rank_hits(query_vec, rows, k=pool_k, rating_boost=rating_boost)
  if scoped:
      candidates = await _keep_in_scope(driver, group_id, candidates, scope, settings)
  ...  # unchanged rerank path
  ```
  where `_keep_in_scope` calls `_scope_shares` and keeps `share >= settings.global_scope_min_share`
  (when `settings` is None, use 0.5 — callers always pass settings in production), logging
  one INFO line `global scope %s: kept %d of %d candidates`.
  In `global_search`: after `results` exist and when scoped, one
  `resolved = await Provenance(driver).resolve_citations(<all map fact ids>)`, then
  `m.fact_ids = [f for f in m.fact_ids if in_scope(resolved.get(f, {}).get("sources", []), scope)]`;
  when not scoped, resolve the same union anyway (labels need it). `_render_blocks`
  gains a `sources: dict[str, list[dict]]` argument and renders
  `fact_line(fact_to_marker[f], texts[f], sources.get(f, []))`. `_REDUCE_PROMPT`
  gains `ATTRIBUTION_RULES` before `"Rules:"`. Citations are built from the same
  `resolved` (no second `resolve_citations`). Add `"applies_to": applies_to(citations)` to
  the success return, `[]` to every refusal return.
- [ ] **Step 5: run** new + existing global tests — PASS. **Step 6: commit** — `feat(global): vendor/product-scoped shortlist and labelled reduce (BACKLOG 52)`.

---

### Task 5: DRIFT — scoped primer and follow-ups, labelled synthesis

**Files:**
- Modify: `src/answer_api/drift.py`
- Test: `tests/unit/test_drift_scope.py`

**Interfaces:**
- Consumes: Tasks 1–4 (`shortlist_communities(scope=)`, `search_local(scope=)`, `fact_line`, `applies_to`, `ATTRIBUTION_RULES`).
- Produces: `drift_search(..., scope: Scope | None = None)` — result gains `"applies_to"`.

- [ ] **Step 1: failing tests** — the primer's `shortlist_communities` receives the scope;
  every `_run_followup` calls `search_local(..., scope=scope)`; the no-primer fallback
  calls `answer_local(..., scope=scope)`; the synthesis prompt carries labelled lines and
  `ATTRIBUTION_RULES`; `applies_to` returned (and `[]` on refusal). Fakes as in the
  existing `tests/unit/test_drift*.py`.
- [ ] **Step 2: run** — FAIL. **Step 3: implement** — thread `scope` through `_primer`,
  `_run_followup`, the fallback; `_synthesize` uses
  `fact_line(i, f["fact"], f.get("sources", []))` (follow-up facts come from
  `search_local` and already carry `sources`); `_SYNTH_PROMPT` gains `ATTRIBUTION_RULES`.
- [ ] **Step 4: run** new + existing DRIFT tests — PASS. **Step 5: commit** — `feat(drift): vendor/product scope and labels`.

---

### Task 6: Router and API — resolve once, pass everywhere, envelope fields

**Files:**
- Modify: `src/answer_api/router.py`, `src/answer_api/app.py`
- Test: `tests/unit/test_router_scope.py`; update `tests/unit/test_answer_api_app.py` and existing router tests for the new signatures

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `answer_router(..., *, q, mode_override, scope: Scope, settings) -> dict` — replaces `vendor`; `_dispatch(..., scope=...)` passes it to all four modes.
  - Envelope gains `"scope": scope.as_dict()` and `"applies_to"`: the mode's own `applies_to` when present, else `attribution.applies_to(env["citations"])` (timeline).
  - `_render_timeline` lines become `- **{fact}** {label} — {span} ...` using `attribution.label(e["sources"])` (omit the label when empty).
  - `app.state.scope_resolver: ScopeResolver` loaded in the lifespan (after the driver; on failure close what was built, as the existing guards do) plus `app.state.scope_loaded_at: float`; a helper `_scope(app, q, vendor, product, scope_mode) -> Scope` reloads when older than `settings.scope_reload_seconds`, then calls `resolver.resolve(...)`; `UnknownScopeName` → `HTTPException(422, detail={"unknown": e.names})`.
  - Every route (`/answer`, `/search/local`, `/search/global`, `/search/drift`, `/timeline`) accepts `vendor: list[str] | None = Query(None)`, `product: list[str] | None = Query(None)`, `scope: Literal["auto", "none"] = "auto"` and passes the resolved `Scope`. `/search/*` and `/timeline` responses also carry `"scope"`.
- [ ] **Step 1: failing tests** — router: scope reaches every mode (spy fakes per mode),
  envelope has `scope` and `applies_to`, timeline lines carry labels; app: `vendor=Veeam`
  repeated/`product=` params resolve and reach the router, `scope=none` yields an empty
  scope, an unknown name → 422 with `{"unknown": [...]}`, the resolver reloads after
  `scope_reload_seconds` (monkeypatch `time.monotonic`). Use the stubbed lifespan pattern
  already in `tests/unit/test_answer_api_app.py` (stub `ScopeResolver.load`).
- [ ] **Step 2: run** — FAIL. **Step 3: implement.** **Step 4: run** the whole unit suite — PASS.
- [ ] **Step 5: commit** — `feat(answer): resolve vendor/product scope once and pass it to every mode; applies_to + scope in the envelope`.

---

### Task 7: Eval — attribution judge

**Files:**
- Modify: `src/answer_api/eval_router.py`, `src/answer_api/router_eval.py` (aggregation)
- Test: `tests/unit/test_router_eval.py` (extend), `tests/unit/test_eval_attribution.py`

**Interfaces:**
- Produces: `async def judge_attribution(client, model, q, answer, labelled_facts: str) -> int | None` — number of claims attributed to a vendor/product not on their cited facts' labels; `None` when the reply is unusable (never coerced to 0 — see memory "LLM empty reply coerced to a value"). Per-question record gains `"misattributed"`; the report header gains `Misattributed claims: <total> across <n> answers (unscored: <u>)`.

- [ ] **Step 1: failing tests** — `_parse_attribution("2")==2`, `"none"`→`None`, empty →
  `None`; `aggregate` sums `misattributed` over scored answers only and counts unscored;
  the judge prompt receives labelled fact lines built with `attribution.fact_line` from the
  citations' sources.
- [ ] **Step 2: run** — FAIL. **Step 3: implement** with a prompt:
  `"Count the claims in the ANSWER that attribute something to a vendor or product that is NOT among the (Vendor · Product) labels of the facts that claim cites. Reply with ONLY the integer (0 if none).\n\nQUESTION: {q}\n\nANSWER:\n{answer}\n\nCITED FACTS:\n{facts}"`,
  same client, bounded `max_tokens` retry pattern as the faithfulness judge (`_attempt`).
- [ ] **Step 4: run** — PASS. **Step 5: commit** — `feat(eval): attribution check`.

---

### Task 8: Acceptance run and docs

**Files:**
- Modify: `docs/superpowers/BACKLOG.md` (52 → DONE with numbers), `CLAUDE.md` (module layout: `scope.py`, `attribution.py`; one line on scope/attribution), `docs/superpowers/router-eval-report.md` (regenerated)
- Create: `docs/superpowers/vendor-product-end-to-end-report.md`

- [ ] **Step 1: full gate** — `uv run ruff check src tests && uv run mypy src && UV_OFFLINE=1 uv run --frozen --extra dev pytest -m "not live" -q`.
- [ ] **Step 2: paid acceptance (≈ the 2026-09-25 router eval, ~26 min)** —
  `UV_OFFLINE=1 uv run --frozen --extra dev python -m answer_api.eval_router`. Before
  launching, prove the variable is live (CLAUDE.md cost rule): run one global golden
  question through `answer_router` in a scratch script and assert the envelope's
  `scope.source == "detected"` with vendors/products set AND the INFO line
  `global scope ...: kept N of M candidates` was logged with `N < M`. Refuse to launch
  the eval otherwise.
- [ ] **Step 3: acceptance criteria** — global grounding ≥ 0.8 (was 0.29), routing ≥ 0.97,
  faithfulness ≥ 4.8, misattributed reported; if global grounding < 0.8, stop and report
  with the per-question shortlists, do not tune thresholds blindly.
- [ ] **Step 4: write the report and update docs; commit** — `docs: vendor/product end to end — report, BACKLOG 52 done`.
- [ ] **Follow-up, not in this plan:** Tier 1 golden questions (spec §6) once Veeam and
  Commvault are ingested.
