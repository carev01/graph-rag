# Exhaustive List Answers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer single-vendor "list" questions with every documented item (grouped, cited, limitations separate) and pick better facts for every other single question.

**Architecture:** A deterministic list detector routes single list questions to a new `list_mode` path inside the `/answer` router: cheap-tier query expansion → per-query reranked local search → near-duplicate collapse (keeping all sources) → final cross-encoder rerank → supported-first budgets → a list-answer prompt. Other single questions get a reranked pool and the same collapse. The eval gains five golden list questions with user-confirmed expected items and an item-coverage metric.

**Tech Stack:** Python 3.12, FastAPI answer-api, Neo4j (graphiti RELATES_TO edges, `fact_embedding`), Voyage cross-encoder (`answer_api.rerank.rerank`), OpenAI-compatible cheap tier (`map_client`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-list-answers-design.md`

## Global Constraints

- List mode applies only when the question shape is `single`, the router was NOT given an explicit `mode=` (via != "override"), `settings.list_mode_enabled` is true, and `is_list_question(q)` matches. Comparisons and enumerations are untouched.
- Settings (in `ExtractSettings`, exact names/defaults): `list_mode_enabled = True`, `list_expansion_queries_max = 10`, `list_per_query_pool = 40`, `list_per_query_k = 20`, `list_supported_k = 50`, `list_limitations_k = 12`, `list_dedup_cosine = 0.92`, `local_single_rerank_pool = 80`.
- The LLM never writes a URL; citations resolve from the marker map exactly as `answer_local` does (`_finalize_answer`, `fact_line`, `applies_to`).
- A failure never reads as absence: a failed expansion falls back to the question alone with a WARNING; a failed per-query retrieval is dropped and recorded in `raw["degraded"]`; if every retrieval fails, raise (the router surfaces it as an error). Never write `content or ""` coercions — use `usable_content` / `_complete_or_none`.
- Near-duplicate collapse keeps EVERY source of the merged facts (deduplicated by url) — no citation is lost.
- Cosine for collapse is RAW cosine of `fact_embedding` vectors computed in Python (not Neo4j's normalized `(1+cos)/2`).
- Lint gate `uv run ruff check src tests` (lints tests: no semicolons E702, imports at top E402); `uv run mypy src`; unit tests hermetic (`ExtractSettings(_env_file=None, ...)`, settings never built at collection time); no paid calls in tests.

---

## File Structure

- Modify `src/graph_extract/config.py` — the eight settings.
- Create `src/answer_api/list_mode.py` — `is_list_question`, `is_limitation`, `expand_queries`, `retrieve_list`, `answer_list`, `_LIST_PROMPT`.
- Create `src/answer_api/fact_dedup.py` — `normalize_fact`, `collapse_near_duplicates` (+ embedding fetch).
- Modify `src/answer_api/synthesize.py` — single path: reranked pool + collapse.
- Modify `src/answer_api/router.py` — list-mode branch.
- Create `src/answer_api/item_coverage.py` — `match_items`.
- Modify `src/answer_api/eval_router.py`, `src/answer_api/eval_retrieval.py`, `src/answer_api/router_golden_tier1.json`.
- Tests: `tests/unit/test_list_mode.py`, `tests/unit/test_fact_dedup.py`, `tests/unit/test_list_router.py`, `tests/unit/test_item_coverage.py`; amend existing tests that pin the single path.
- Docs: `CLAUDE.md`.

---

### Task 1: Settings, list detection, limitation classifier, query expansion

**Files:**
- Modify: `src/graph_extract/config.py` (next to `local_facet_rerank_pool`)
- Create: `src/answer_api/list_mode.py`
- Test: `tests/unit/test_list_mode.py`, `tests/unit/test_config.py`

**Interfaces:**
- Produces: the eight settings; `is_list_question(q: str) -> bool`; `is_limitation(fact: str) -> bool`; `async expand_queries(client, model: str, q: str, max_n: int) -> list[str]` (never raises; `[]` on failure with a WARNING).

- [ ] **Step 1: Failing tests** — `tests/unit/test_list_mode.py`:

```python
"""List-mode detection, limitation classifier and expansion (spec 2026-10-15 §3-§4)."""
import logging
from types import SimpleNamespace

import pytest

import answer_api.list_mode as lm

LISTS = [
    "Which workloads does Veeam support immutability for?",
    "Where can I restore a Windows agent backup?",
    "Which hypervisors does Veeam Backup & Replication support?",
    "Which resource types can AWS Backup protect?",
    "What recovery options does Azure Backup offer for Azure VMs?",
    "What are the ways to restore a Veeam Agent backup?",
    "List the supported databases for Veeam plug-ins",
    "What destinations can AWS Backup copy recovery points to?",
]
NOT_LISTS = [
    "What is a hardened repository?",
    "How does Veeam SureBackup verify backups?",
    "How do you halt and resume plan testing in Veeam Recovery Orchestrator?",
    "Does AWS Backup support Amazon S3?",
    "What should I consider when enabling backup immutability for Veeam Agent backups?",
    "Why does a backup copy job fail?",
]


@pytest.mark.parametrize("q", LISTS)
def test_list_questions_are_detected(q):
    assert lm.is_list_question(q)


@pytest.mark.parametrize("q", NOT_LISTS)
def test_other_questions_are_not_lists(q):
    assert not lm.is_list_question(q)


@pytest.mark.parametrize("fact,lim", [
    ("Veeam Agent does not support the backup immutability feature for Google Cloud.", True),
    ("A Windows host is not supported as a hardened repository.", True),
    ("Immutability cannot be disabled once set.", True),
    ("Selective disk restore is only available for managed disks.", True),
    ("Veeam Agent provides instant restore capability to Hyper-V VMs", False),
    ("AWS Backup supports Amazon EFS file systems.", False),
])
def test_limitation_classifier(fact, lim):
    assert lm.is_limitation(fact) is lim


class _Client:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.prompts = content, exc, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        if self.exc:
            raise self.exc
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


async def test_expansion_returns_item_kind_queries_capped():
    c = _Client('["restore Veeam Agent backup to VMware vSphere", "to Hyper-V", "to Amazon EC2", '
                '"x", "a", "b", "c", "d", "e", "f", "g", "h"]')
    out = await lm.expand_queries(c, "m", "Where can I restore a Windows agent backup?", 10)
    assert out[0] == "restore Veeam Agent backup to VMware vSphere"
    assert len(out) == 10
    assert "kinds of item" in c.prompts[0] and "Where can I restore" in c.prompts[0]


@pytest.mark.parametrize("client", [_Client(None), _Client("not json"), _Client("{}"),
                                    _Client(exc=RuntimeError("down"))])
async def test_failed_expansion_returns_empty_and_warns(client, caplog):
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        assert await lm.expand_queries(client, "m", "Which X?", 10) == []
    assert "list expansion" in caplog.text


async def test_expansion_drops_blank_and_duplicate_queries():
    c = _Client('["a", " ", "a", "b"]')
    assert await lm.expand_queries(c, "m", "Which X?", 10) == ["a", "b"]
```

Append to `tests/unit/test_config.py`:

```python
def test_list_mode_defaults():
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.list_mode_enabled is True
    assert (s.list_expansion_queries_max, s.list_per_query_pool, s.list_per_query_k) == (10, 40, 20)
    assert (s.list_supported_k, s.list_limitations_k) == (50, 12)
    assert s.list_dedup_cosine == 0.92 and s.local_single_rerank_pool == 80
```

- [ ] **Step 2: Run** `uv run --extra dev pytest tests/unit/test_list_mode.py tests/unit/test_config.py -q` — expect FAIL (module/attributes missing).

- [ ] **Step 3: Settings** in `src/graph_extract/config.py`, after `local_facet_rerank_pool`:

```python
    # --- list mode (spec 2026-10-15) ---
    # Single-vendor "list" questions ("Which workloads ...", "Where can I restore ...")
    # are answered exhaustively: cheap-tier expansion into item-kind queries, a reranked
    # local search per query, near-duplicate collapse, a final rerank, then up to
    # list_supported_k supported facts plus list_limitations_k limitation facts.
    list_mode_enabled: bool = True
    list_expansion_queries_max: int = 10
    list_per_query_pool: int = 40
    list_per_query_k: int = 20
    list_supported_k: int = 50
    list_limitations_k: int = 12
    # Raw cosine of fact embeddings at or above which two facts are one (sources merged).
    list_dedup_cosine: float = 0.92
    # Every other single question reranks this many candidates down to k (was raw
    # hybrid order: Q1/Q2 of 2026-10-15 retrieved mostly limitations and duplicates).
    local_single_rerank_pool: int = 80
```

- [ ] **Step 4: Implement** `src/answer_api/list_mode.py` (this task's part):

```python
"""List mode (spec 2026-10-15): exhaustive answers to single-vendor list questions.

Detection is deterministic. Expansion asks the cheap tier for retrieval queries that
name likely KINDS of item (hypervisors, clouds, ...): a paraphrase expansion did not
find the missing items (measured 2026-10-15), item-kind queries did. The model's
domain knowledge only steers which facts are searched for; the answer states only
what retrieved facts say."""
from __future__ import annotations

import json
import logging
import re

from graph_extract.usage import usable_content

logger = logging.getLogger(__name__)

_LIST_RE = re.compile(
    r"^\s*(which|what)\s+(\w+\s+){0,3}?\w+s\b(?!\s+(is|was)\b)"     # which/what ... <plural>
    r"|^\s*where\s+can\s+(i|you|we)\b"                             # where can I ...
    r"|^\s*what\s+are\s+the\s+(\w+\s+)?(options|ways|methods|types|kinds|destinations|targets)\b"
    r"|^\s*list\b"
    r"|\bsupported\s+\w+s\b",
    re.IGNORECASE)
# "What should I consider ..." / "What is ..." start with what + a non-plural; the
# pattern needs a plural noun within the first words. Questions that are clearly
# single-answer are excluded explicitly.
_NOT_LIST_RE = re.compile(r"^\s*(what|which)\s+(is|was|should|does|do|did|can)\b",
                          re.IGNORECASE)

_LIMIT_RE = re.compile(
    r"\b(not supported|does not|doesn't|do not|don't|cannot|can't|can not|is not|are not|"
    r"unsupported|only (available|supported|for|when|if|with|on|in)|not available)\b",
    re.IGNORECASE)

_EXPAND_PROMPT = (
    "You help search vendor backup documentation. The QUESTION asks for a list. Write "
    "up to {n} short search queries that together would find EVERY item the "
    "documentation might list as an answer. Name the likely kinds of item explicitly "
    "(for example, for restore destinations: specific hypervisors, clouds, physical "
    "machines, file and application level; for workloads: VMs, physical machines, "
    "databases, SaaS, Kubernetes). Keep the vendor and product names from the "
    "question in every query. Reply with ONLY a JSON list of strings.\n\n"
    "QUESTION: {q}\nJSON:")


def is_list_question(q: str) -> bool:
    return bool(_LIST_RE.search(q)) and not _NOT_LIST_RE.search(q)


def is_limitation(fact: str) -> bool:
    return bool(_LIMIT_RE.search(fact))


async def expand_queries(client, model: str, q: str, max_n: int) -> list[str]:
    """Up to `max_n` item-kind retrieval queries; [] (logged) when unusable."""
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=800,
            messages=[{"role": "user", "content": _EXPAND_PROMPT.format(n=max_n, q=q)}])
        content = usable_content(resp)
        if content is None:
            logger.warning("list expansion returned no usable content; using the question alone")
            return []
        start, end = content.find("["), content.rfind("]")
        parsed = json.loads(content[start:end + 1]) if start >= 0 and end > start else None
    except Exception as exc:
        logger.warning("list expansion failed (%s); using the question alone", exc)
        return []
    if not isinstance(parsed, list):
        logger.warning("list expansion reply was not a JSON list; using the question alone")
        return []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
    return out[:max_n]
```

If a detection test fails, adjust `_LIST_RE` / `_NOT_LIST_RE` until all listed matches and non-matches pass — do not change the test lists.

- [ ] **Step 5: Run** the two test files — expect PASS. Then `uv run ruff check src tests && uv run mypy src`.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/config.py src/answer_api/list_mode.py tests/unit/test_list_mode.py tests/unit/test_config.py
git commit -m "feat(answer): list-question detection, limitation classifier, item-kind expansion"
```

---

### Task 2: Near-duplicate collapse and the improved single path

**Files:**
- Create: `src/answer_api/fact_dedup.py`
- Modify: `src/answer_api/synthesize.py` (`answer_local`, single branch only)
- Modify: `src/answer_api/eval_retrieval.py` (`_evidence`, single branch mirrors /answer)
- Test: `tests/unit/test_fact_dedup.py`; adjust any existing test that pins the single path's `search_local` kwargs

**Interfaces:**
- Consumes: `settings.local_single_rerank_pool`, `settings.list_dedup_cosine` (Task 1).
- Produces: `normalize_fact(text: str) -> str`; `async collapse_near_duplicates(driver, results: list[dict], *, cosine: float) -> list[dict]` — input/output items are search-result dicts `{"fact", "fact_uuid", "valid_at", "invalid_at", "sources"}`; order preserved (first occurrence wins); merged items carry the union of sources (deduped by `url`, falling back to `article_id`).

- [ ] **Step 1: Failing tests** — `tests/unit/test_fact_dedup.py`:

```python
"""Near-duplicate collapse keeps every source (spec 2026-10-15 §4.3, §6)."""
import pytest

import answer_api.fact_dedup as fd


def _r(uuid, fact, url):
    return {"fact": fact, "fact_uuid": uuid, "valid_at": None, "invalid_at": None,
            "sources": [{"url": url, "title": url, "vendor": "Veeam", "product": "VBR"}]}


class _Driver:
    def __init__(self, vectors):
        self.vectors, self.calls = vectors, 0

    async def execute_query(self, query, **params):
        self.calls += 1
        assert query is fd.EMBEDDINGS_QUERY
        return ([{"uuid": u, "embedding": self.vectors.get(u)} for u in params["uuids"]],
                None, None)


def test_normalize_fact_ignores_case_punctuation_and_spacing():
    assert fd.normalize_fact("Veeam  Agent, provides X.") == fd.normalize_fact("veeam agent provides x")


async def test_identical_text_collapses_and_merges_sources():
    rs = [_r("a", "Veeam Agent provides instant restore to Hyper-V VMs.", "u1"),
          _r("b", "veeam agent provides instant restore to hyper-v vms", "u2")]
    out = await fd.collapse_near_duplicates(_Driver({}), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a"]
    assert [s["url"] for s in out[0]["sources"]] == ["u1", "u2"]


async def test_near_identical_embeddings_collapse_distinct_ones_stay():
    rs = [_r("a", "Instant Recovery restores an agent computer as a Hyper-V VM.", "u1"),
          _r("b", "Agent computers can be restored by Instant Recovery to a Hyper-V VM.", "u2"),
          _r("c", "Restore to Amazon EC2 is supported.", "u3")]
    vec = {"a": [1.0, 0.0, 0.0], "b": [0.99, 0.05, 0.0], "c": [0.0, 1.0, 0.0]}
    out = await fd.collapse_near_duplicates(_Driver(vec), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "c"]
    assert [s["url"] for s in out[0]["sources"]] == ["u1", "u2"]


async def test_missing_embeddings_keep_the_fact():
    rs = [_r("a", "fact one", "u1"), _r("b", "fact two", "u2")]
    out = await fd.collapse_near_duplicates(_Driver({"a": None}), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "b"]


async def test_duplicate_source_urls_are_not_repeated():
    rs = [_r("a", "same fact", "u1"), _r("b", "same fact", "u1")]
    out = await fd.collapse_near_duplicates(_Driver({}), rs, cosine=0.92)
    assert [s["url"] for s in out[0]["sources"]] == ["u1"]


async def test_an_embedding_fetch_failure_falls_back_to_text_collapse(caplog):
    class _Broken:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")
    rs = [_r("a", "same", "u1"), _r("b", "same", "u2"), _r("c", "other", "u3")]
    out = await fd.collapse_near_duplicates(_Broken(), rs, cosine=0.92)
    assert [o["fact_uuid"] for o in out] == ["a", "c"]
    assert "embedding" in caplog.text
```

- [ ] **Step 2: Run** `uv run --extra dev pytest tests/unit/test_fact_dedup.py -q` — expect FAIL.

- [ ] **Step 3: Implement** `src/answer_api/fact_dedup.py`:

```python
"""Collapse near-duplicate facts (spec 2026-10-15 §4.3): the same statement extracted
from several articles in different wordings filled answer slots (four variants of
"Instant Recovery ... as a Hyper-V VM"). The first occurrence is kept; every source
of the merged facts is kept on it, so no citation is lost."""
from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)

EMBEDDINGS_QUERY = (
    "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
    "RETURN f.uuid AS uuid, f.fact_embedding AS embedding")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_fact(text: str) -> str:
    return _NON_WORD.sub(" ", (text or "").lower()).strip()


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _merge_sources(into: dict, other: dict) -> None:
    seen = {s.get("url") or s.get("article_id") for s in into["sources"]}
    for s in other.get("sources") or []:
        key = s.get("url") or s.get("article_id")
        if key not in seen:
            seen.add(key)
            into["sources"].append(s)


async def collapse_near_duplicates(driver, results: list[dict], *, cosine: float) -> list[dict]:
    if not results:
        return []
    vectors: dict[str, list[float]] = {}
    try:
        records, _, _ = await driver.execute_query(
            EMBEDDINGS_QUERY, uuids=[r["fact_uuid"] for r in results])
        vectors = {r["uuid"]: list(r["embedding"]) for r in records if r["embedding"]}
    except Exception as exc:
        logger.warning("fact embedding fetch failed (%s); collapsing identical text only", exc)
    kept: list[dict] = []
    for r in results:
        norm = normalize_fact(r["fact"])
        vec = vectors.get(r["fact_uuid"])
        twin = None
        for k in kept:
            if normalize_fact(k["fact"]) == norm:
                twin = k
                break
            kv = vectors.get(k["fact_uuid"])
            if vec is not None and kv is not None and _cos(vec, kv) >= cosine:
                twin = k
                break
        if twin is None:
            kept.append({**r, "sources": list(r.get("sources") or [])})
        else:
            _merge_sources(twin, r)
    return kept
```

- [ ] **Step 4: Single path in `answer_local`** (`src/answer_api/synthesize.py`), replace the single-branch `search_mod.search_local(...)` call and `results = res["results"]` with:

```python
    pool = settings.local_single_rerank_pool if settings is not None else 0
    res = await search_mod.search_local(
        graphiti, driver, q=q, k=2 * k if pool else k, scope=scope, group_id=group_id,
        fetch_cap=fetch_cap, scan_max_episodes=scan_max_episodes, settings=settings,
        rerank_pool=pool)
    results = res["results"]
    if settings is not None:
        # Reranked pool + near-duplicate collapse (spec 2026-10-15 §6): over-fetch 2k
        # so k distinct facts survive the collapse.
        results = (await collapse_near_duplicates(
            driver, results, cosine=settings.list_dedup_cosine))[:k]
```

with `from answer_api.fact_dedup import collapse_near_duplicates` at the top. Then mirror it in `eval_retrieval._evidence` for `shape.kind == "single"`: pass `rerank_pool=settings.local_single_rerank_pool` and `k=30`, collapse, keep 15 (import the same function). Update any existing unit test that asserts the single path's exact `search_local` kwargs to the new values (search the tests for `answer_local` fakes that record kwargs) — keep their intent.

- [ ] **Step 5: Run** `uv run --extra dev pytest tests/unit -q` (fix only pinned-kwargs expectations), then ruff + mypy.

- [ ] **Step 6: Commit**

```bash
git add src/answer_api/fact_dedup.py src/answer_api/synthesize.py src/answer_api/eval_retrieval.py tests/unit
git commit -m "feat(answer): near-duplicate collapse; single questions rerank a pool of 80"
```

---

### Task 3: List retrieval and the list answer

**Files:**
- Modify: `src/answer_api/list_mode.py`
- Test: `tests/unit/test_list_mode.py` (append)

**Interfaces:**
- Consumes: Task 1 functions/settings; `collapse_near_duplicates` (Task 2); `answer_api.search.search_local`; `answer_api.search._rerank_top(q, cands, *, k, settings, what, order_name)` (cands need a `.fact` attribute); `answer_api.synthesize._complete_or_none`, `_finalize_answer`, `_REFUSAL`; `answer_api.attribution.fact_line`, `applies_to`, `ATTRIBUTION_RULES`, `CLAIM_RULES`; `answer_api.shape.raise_if_cancelled`.
- Produces:
  - `@dataclass ListRetrieval: results: list[dict]; limitations: list[dict]; expansion_queries: list[str]; degraded: list[str]`
  - `async retrieve_list(graphiti, driver, map_client, map_model, *, q, scope, group_id, settings) -> ListRetrieval` (raises `RuntimeError` if every retrieval failed)
  - `async answer_list(graphiti, driver, map_client, map_model, synth_client, synth_model, *, q, scope, group_id, settings) -> dict` returning `{"query","answer","citations","retrieved","cited","applies_to","list_mode": True,"expansion_queries": [...], "degraded": [...] (only when non-empty)}`

- [ ] **Step 1: Failing tests** (append to `tests/unit/test_list_mode.py`; add imports `import answer_api.search as search_mod`, `import answer_api.fact_dedup as fd`, `from graph_extract.config import ExtractSettings`, `from answer_api.scope import Scope` at the top):

```python
def _settings(**kw):
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p", **kw)


def _fact(uuid, text):
    return {"fact": text, "fact_uuid": uuid, "valid_at": None, "invalid_at": None,
            "sources": [{"url": f"https://d/{uuid}", "title": uuid,
                         "vendor": "Veeam", "product": "Veeam Backup & Replication"}]}


@pytest.fixture
def patched(monkeypatch):
    calls = []

    def install(per_query, expansion='["q-hyperv", "q-ec2"]', fail=()):
        async def _search(graphiti, driver, *, q, k, scope=None, group_id, **kw):
            calls.append({"q": q, "k": k, "pool": kw.get("rerank_pool")})
            if q in fail:
                raise RuntimeError(f"search failed for {q}")
            return {"results": per_query.get(q, []), "count": 0}

        async def _collapse(driver, results, *, cosine):
            seen, out = set(), []
            for r in results:
                if r["fact_uuid"] not in seen:
                    seen.add(r["fact_uuid"])
                    out.append(r)
            return out

        async def _rerank_top(q, cands, *, k, settings, what, order_name):
            return list(cands)[:k]

        monkeypatch.setattr(lm, "search_local", _search)
        monkeypatch.setattr(lm, "collapse_near_duplicates", _collapse)
        monkeypatch.setattr(lm, "_rerank_top", _rerank_top)
        return _Client(expansion)
    return install, calls


async def test_retrieve_list_merges_every_query_and_budgets_limitations(patched):
    install, calls = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({
        q: [_fact("az", "Restore to Microsoft Azure is supported."),
            _fact("lim", "Restore to Google Cloud is not supported for encrypted disks.")],
        "q-hyperv": [_fact("hv", "Instant Recovery to a Hyper-V VM is supported.")],
        "q-ec2": [_fact("ec2", "Restore to Amazon EC2 is supported."),
                  _fact("az", "Restore to Microsoft Azure is supported.")]})
    s = _settings(list_supported_k=50, list_limitations_k=12)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=s)
    assert [c["q"] for c in calls] == [q, "q-hyperv", "q-ec2"]
    assert all(c["k"] == 20 and c["pool"] == 40 for c in calls)
    assert [r["fact_uuid"] for r in got.results] == ["az", "hv", "ec2"]
    assert [r["fact_uuid"] for r in got.limitations] == ["lim"]
    assert got.expansion_queries == ["q-hyperv", "q-ec2"] and got.degraded == []


async def test_supported_budget_is_never_taken_by_limitations(patched):
    install, _ = patched
    q = "Which workloads does Veeam support immutability for?"
    facts = [_fact(f"l{i}", f"Immutability is not supported for case {i}.") for i in range(20)]
    facts += [_fact(f"s{i}", f"Immutability is supported for workload {i}.") for i in range(5)]
    client = install({q: facts}, expansion="[]")
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings(list_supported_k=50, list_limitations_k=12))
    assert len(got.results) == 5 and len(got.limitations) == 12


async def test_a_failed_query_degrades_and_all_failed_raises(patched):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, fail=("q-ec2",))
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert got.degraded == ["retrieval failed for: q-ec2"]
    client = install({}, fail=(q, "q-hyperv", "q-ec2"))
    with pytest.raises(RuntimeError, match="every list retrieval failed"):
        await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                               settings=_settings())


async def test_answer_list_groups_supported_then_limitations_and_cites_by_marker(patched, monkeypatch):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    map_client = install({q: [_fact("az", "Restore to Microsoft Azure is supported."),
                              _fact("lim", "Restore is not supported to Xen.")]}, expansion="[]")
    synth = _Client("**Clouds**\n- Microsoft Azure [1]\n\n**Not supported / limitations**\n- Xen [2]")
    out = await lm.answer_list(None, None, map_client, "m", synth, "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings())
    prompt = synth.prompts[0]
    assert "SUPPORTED FACTS:\n[1] " in prompt and "LIMITATION FACTS:\n[2] " in prompt
    assert "every item" in prompt.lower() and "Not supported / limitations" in prompt
    assert out["list_mode"] is True and out["expansion_queries"] == []
    assert [c["fact_uuid"] for c in out["citations"]] == ["az", "lim"]
    assert out["citations"][0]["sources"][0]["url"] == "https://d/az"
    assert "degraded" not in out


async def test_answer_list_with_no_facts_refuses_without_an_llm_call(patched):
    install, _ = patched
    map_client = install({}, expansion="[]")
    synth = _Client("unused")
    out = await lm.answer_list(None, None, map_client, "m", synth, "s", q="Which X?",
                               scope=Scope(), group_id="g", settings=_settings())
    assert out["citations"] == [] and synth.prompts == []
```

- [ ] **Step 2: Run** — expect FAIL (functions missing).

- [ ] **Step 3: Implement** (append to `src/answer_api/list_mode.py`; add imports at the top of the file):

```python
import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

from answer_api.attribution import ATTRIBUTION_RULES, CLAIM_RULES, applies_to, fact_line
from answer_api.fact_dedup import collapse_near_duplicates
from answer_api.search import _rerank_top, search_local
from answer_api.shape import raise_if_cancelled
from answer_api.synthesize import _REFUSAL, _complete_or_none, _finalize_answer

_LIST_PROMPT = (
    "You are answering a question that asks for a LIST, about backup products, using "
    "ONLY the numbered facts below. " + ATTRIBUTION_RULES + CLAIM_RULES +
    "List EVERY item the SUPPORTED FACTS state as an answer -- no item limit -- grouped "
    "under a few short headings that the facts suggest. Give each item one short line "
    "with its [N] marker; when several facts state the same item, put it on one line "
    "with their markers. After the list, add a section titled \"Not supported / "
    "limitations\" summarising the LIMITATION FACTS briefly with their markers; omit "
    "that section when there are none. Do not pad, do not repeat items, and do not "
    "claim anything is undocumented beyond what the facts show. Do NOT use outside "
    "knowledge. Do NOT write any URL or link. If no fact answers the question, reply "
    "exactly: \"" + _REFUSAL + "\"\n\n"
    "SUPPORTED FACTS:\n{supported}\n\nLIMITATION FACTS:\n{limitations}\n\n"
    "QUESTION: {q}\n\nAnswer:")


@dataclass
class ListRetrieval:
    results: list[dict]
    limitations: list[dict]
    expansion_queries: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


async def retrieve_list(graphiti, driver, map_client, map_model, *, q, scope, group_id,
                        settings) -> ListRetrieval:
    expansions = await expand_queries(map_client, map_model, q,
                                      settings.list_expansion_queries_max)
    queries = [q] + [x for x in expansions if x != q]
    found = await asyncio.gather(*[
        search_local(graphiti, driver, q=sq, k=settings.list_per_query_k, scope=scope,
                     group_id=group_id, fetch_cap=settings.local_scoped_fetch_cap,
                     scan_max_episodes=settings.local_scoped_scan_max_episodes,
                     settings=settings, rerank_pool=settings.list_per_query_pool)
        for sq in queries], return_exceptions=True)
    raise_if_cancelled(found)
    merged: list[dict] = []
    degraded: list[str] = []
    for sq, res in zip(queries, found):
        if isinstance(res, BaseException):
            logger.warning("list retrieval failed for %r: %s", sq, res)
            degraded.append(f"retrieval failed for: {sq}")
            continue
        merged.extend(res["results"])
    if len(degraded) == len(queries):
        raise RuntimeError("every list retrieval failed")
    distinct = await collapse_near_duplicates(driver, merged,
                                              cosine=settings.list_dedup_cosine)
    cands = [SimpleNamespace(fact=r["fact"], item=r) for r in distinct]
    ranked = await _rerank_top(q, cands, k=len(cands), settings=settings,
                               what="list answer", order_name="merge order")
    supported, limits = [], []
    for c in ranked:
        (limits if is_limitation(c.fact) else supported).append(c.item)
    return ListRetrieval(results=supported[:settings.list_supported_k],
                         limitations=limits[:settings.list_limitations_k],
                         expansion_queries=expansions, degraded=degraded)


async def answer_list(graphiti, driver, map_client, map_model, synth_client, synth_model,
                      *, q, scope, group_id, settings) -> dict:
    got = await retrieve_list(graphiti, driver, map_client, map_model, q=q, scope=scope,
                              group_id=group_id, settings=settings)
    facts = got.results + got.limitations
    base: dict = {"query": q, "list_mode": True, "expansion_queries": got.expansion_queries}
    if got.degraded:
        base["degraded"] = got.degraded
    if not facts:
        return {**base, "answer": _REFUSAL, "citations": [], "retrieved": 0, "cited": 0,
                "applies_to": []}
    marker_map = {i: r for i, r in enumerate(facts, 1)}
    n_sup = len(got.results)
    sup = "\n".join(fact_line(i, r["fact"], r["sources"])
                    for i, r in marker_map.items() if i <= n_sup) or "(none)"
    lim = "\n".join(fact_line(i, r["fact"], r["sources"])
                    for i, r in marker_map.items() if i > n_sup) or "(none)"
    raw = await _complete_or_none(synth_client, synth_model,
                                  _LIST_PROMPT.format(supported=sup, limitations=lim, q=q),
                                  max_tokens=6000)
    if raw is None:
        return {**base, "answer": _REFUSAL, "citations": [], "retrieved": len(facts),
                "cited": 0, "applies_to": []}
    answer, cited = _finalize_answer(raw, marker_map)
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": marker_map[m].get("valid_at"),
                  "invalid_at": marker_map[m].get("invalid_at"),
                  "sources": marker_map[m]["sources"]} for m in cited]
    return {**base, "answer": answer, "citations": citations, "retrieved": len(facts),
            "cited": len(cited), "applies_to": applies_to(citations)}
```

If importing `answer_api.synthesize` from `list_mode` creates an import cycle, move the import inside the functions and say so in the report.

- [ ] **Step 4: Run** `uv run --extra dev pytest tests/unit/test_list_mode.py -q` — PASS; then ruff + mypy.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/list_mode.py tests/unit/test_list_mode.py
git commit -m "feat(answer): list retrieval (expansion, merge, collapse, rerank, budgets) and list answer"
```

---

### Task 4: Router — the list-mode branch

**Files:**
- Modify: `src/answer_api/router.py` (`answer_router`)
- Test: `tests/unit/test_list_router.py`

**Interfaces:**
- Consumes: `list_mode.is_list_question`, `list_mode.answer_list` (Tasks 1, 3); existing `_normalize`, `_shape_fields`, `freshness_mod.freshness`.
- Produces: envelope additions for list answers: `env["routing"]["path"] == "list"`, `env["list_mode"] is True`, `env["expansion_queries"]`, `env["routing"]["degraded"]` (via `_normalize` from `raw["degraded"]`); `env["mode"] == "local"`.

- [ ] **Step 1: Failing tests** — `tests/unit/test_list_router.py` (model the fakes on the existing router tests: read `tests/unit/test_answer_router.py` or the file that tests `answer_router`, and reuse its fake clients/settings helpers):

```python
"""The router sends single list questions to list mode (spec 2026-10-15 §2, §5)."""
import pytest

import answer_api.router as router_mod
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape

Q = "Where can I restore a Windows agent backup?"
_ENV_RAW = {"query": Q, "answer": "- Azure [1]", "list_mode": True,
            "expansion_queries": ["q1"], "citations": [{"marker": 1, "fact_uuid": "az",
            "sources": [{"url": "u", "vendor": "Veeam", "product": "VBR"}]}],
            "retrieved": 1, "cited": 1, "applies_to": []}


@pytest.fixture
def wired(monkeypatch):
    seen = {"list": 0, "modes": []}

    async def _answer_list(*a, **kw):
        seen["list"] += 1
        return dict(seen.get("raw") or _ENV_RAW)

    async def _run_mode(mode, *a, **kw):
        seen["modes"].append(mode)
        return mode, None, {"answer": "normal", "citations": [{"marker": 1, "fact_uuid": "x",
                                                               "sources": []}]}

    async def _classify(q, **kw):
        return ("local", "override") if kw.get("mode_override") else ("drift", "llm")

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router_mod.list_mod, "answer_list", _answer_list)
    monkeypatch.setattr(router_mod, "_run_mode", _run_mode)
    monkeypatch.setattr(router_mod, "classify", _classify)
    monkeypatch.setattr(router_mod.freshness_mod, "freshness", _fresh)
    return seen


def _settings():
    from graph_extract.config import ExtractSettings
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


async def _route(q=Q, override=None, shape=None):
    return await router_mod.answer_router(
        None, None, None, None, "s", None, "m", None, "c", q=q, mode_override=override,
        scope=Scope(), settings=_settings(),
        shape=shape or QuestionShape("single", (Facet("", Scope()),)))


async def test_a_single_list_question_takes_list_mode(wired):
    env = await _route()
    assert wired["list"] == 1 and wired["modes"] == []
    assert env["routing"]["path"] == "list" and env["list_mode"] is True
    assert env["mode"] == "local" and env["expansion_queries"] == ["q1"]


async def test_an_explicit_mode_skips_list_mode(wired):
    await _route(override="global")
    assert wired["list"] == 0 and wired["modes"] == ["local"]


async def test_a_non_list_question_takes_the_normal_route(wired):
    await _route(q="How does SureBackup verify backups?")
    assert wired["list"] == 0 and wired["modes"] == ["drift"]


async def test_an_ungrounded_list_answer_falls_back_to_the_normal_route(wired):
    wired["raw"] = {**_ENV_RAW, "citations": [], "answer": "I don't have enough information"}
    env = await _route()
    assert wired["list"] == 1 and wired["modes"] == ["drift"]
    assert env["routing"].get("path") != "list"


async def test_list_degradation_reaches_the_envelope(wired):
    wired["raw"] = {**_ENV_RAW, "degraded": ["retrieval failed for: q1"]}
    env = await _route()
    assert env["routing"]["degraded"] == ["retrieval failed for: q1"]


async def test_comparisons_never_take_list_mode(wired, monkeypatch):
    async def _fq(client, model, q, facets):
        return facets
    monkeypatch.setattr(router_mod, "facet_queries", _fq)
    shape = QuestionShape("comparison", (Facet("A", Scope(vendors=("A",))),
                                         Facet("B", Scope(vendors=("B",)))))
    await _route(q="Which features differ between A and B?", shape=shape)
    assert wired["list"] == 0
```

Adapt `_route`'s positional args to `answer_router`'s real signature (`graphiti, driver, embedder, synth_client, synth_model, map_client, map_model, cheap_client, cheap_model`) and the classify/freshness fakes to their real call shapes; keep every assertion.

- [ ] **Step 2: Run** — expect FAIL.

- [ ] **Step 3: Implement** in `src/answer_api/router.py`: import `from answer_api import list_mode as list_mod`; in `answer_router`, right after the `classify(...)` call and the `requested = ...` line, before the comparison/enumeration rewrite block:

```python
    if (requested == "single" and via != "override" and settings.list_mode_enabled
            and list_mod.is_list_question(q)):
        # Single-vendor list questions are answered exhaustively (spec 2026-10-15).
        # An ungrounded list answer falls through to the normal route, whose detected-
        # scope relaxation and local/drift fallbacks still apply.
        raw = await list_mod.answer_list(
            graphiti, driver, map_client, map_model, synth_client, synth_model,
            q=q, scope=scope, group_id=settings.group_id, settings=settings)
        if raw.get("citations"):
            env = _normalize("local", via, None, raw, q, scope)
            env["routing"]["path"] = "list"
            env["list_mode"] = True
            env["expansion_queries"] = raw.get("expansion_queries", [])
            _shape_fields(env, shape, raw, requested)
            env["freshness"] = await freshness_mod.freshness(
                driver, settings.group_id, reports=False)
            return env
        logger.info("list mode grounded nothing for %r; taking the normal route", q)
```

(`logger` already exists in router.py; if not, add `logger = logging.getLogger(__name__)`.) An exception from `answer_list` (every retrieval failed) propagates, like any mode failure.

- [ ] **Step 4: Run** `uv run --extra dev pytest tests/unit/test_list_router.py tests/unit -q`, ruff, mypy.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/router.py tests/unit/test_list_router.py
git commit -m "feat(answer): route single list questions to list mode, fall back when ungrounded"
```

---

### Task 5: Eval — golden list questions, item coverage, free gate; docs

**Files:**
- Create: `src/answer_api/item_coverage.py`
- Modify: `src/answer_api/router_golden_tier1.json` (append five questions)
- Modify: `src/answer_api/eval_router.py` (per-question `item_coverage`, summary)
- Modify: `src/answer_api/eval_retrieval.py` (list questions: items in retrieved facts)
- Modify: `CLAUDE.md` (answer_api bullet: list mode)
- Test: `tests/unit/test_item_coverage.py`

**Interfaces:**
- Consumes: `list_mode.is_list_question`, `list_mode.retrieve_list` (Tasks 1, 3).
- Produces: `match_items(text: str, items: list[dict]) -> tuple[list[str], list[str]]` (found names, missing names); golden entries with `"kind": "single_list"`, `"expected_items": [{"item": str, "aliases": [str, ...]}]`, optional `"bonus_items"`.

- [ ] **Step 1: Failing tests** — `tests/unit/test_item_coverage.py`:

```python
"""Item coverage for list questions (spec 2026-10-15 §7)."""
import json
from pathlib import Path

from answer_api.item_coverage import match_items

ITEMS = [{"item": "Hyper-V", "aliases": ["hyper-v"]},
         {"item": "Disk export", "aliases": ["vhd", "vhdx", "vmdk"]},
         {"item": "Amazon S3", "aliases": ["s3"]}]


def test_aliases_match_case_insensitively_on_word_boundaries():
    found, missing = match_items("Restore to Microsoft HYPER-V, or export as VHDX.", ITEMS)
    assert found == ["Hyper-V", "Disk export"] and missing == ["Amazon S3"]


def test_an_alias_inside_a_longer_word_does_not_match():
    _, missing = match_items("The s3cure option and vhdtool.", ITEMS)
    assert missing == ["Hyper-V", "Disk export", "Amazon S3"]


def test_golden_list_questions_are_well_formed():
    gold = json.loads((Path(__file__).resolve().parents[2] /
                       "src/answer_api/router_golden_tier1.json").read_text())
    lists = [q for q in gold if q.get("kind") == "single_list"]
    assert len(lists) == 5
    counts = {q["question"]: len(q["expected_items"]) for q in lists}
    assert counts == {
        "Which workloads does Veeam support immutability for?": 9,
        "Where can I restore a Windows agent backup?": 17,
        "Which hypervisors does Veeam Backup & Replication support?": 14,
        "Which resource types can AWS Backup protect?": 17,
        "What recovery options does Azure Backup offer for Azure VMs?": 10}
    for q in lists:
        assert q["expected_modes"] == ["local"] and q["expected_article_ids"] == []
        assert all(i["item"] and i["aliases"] for i in q["expected_items"])
```

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement** `src/answer_api/item_coverage.py`:

```python
"""Deterministic item matching for list-question ground truth (spec 2026-10-15 §7)."""
from __future__ import annotations

import re


def match_items(text: str, items: list[dict]) -> tuple[list[str], list[str]]:
    low = (text or "").lower()
    found, missing = [], []
    for it in items:
        hit = any(re.search(rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])", low)
                  for a in it["aliases"])
        (found if hit else missing).append(it["item"])
    return found, missing
```

- [ ] **Step 4: Golden data** — append these five entries to `src/answer_api/router_golden_tier1.json` (keep the file's JSON style; the item lists were confirmed by the user on 2026-10-15):

```json
{"question": "Which workloads does Veeam support immutability for?", "intent": "local",
 "expected_modes": ["local"], "kind": "single_list", "expected_article_ids": [],
 "expected_items": [
  {"item": "Virtual machines", "aliases": ["virtual machine", "virtual machines", "vms", "vmware", "vsphere", "hyper-v"]},
  {"item": "Physical machines (Veeam Agent)", "aliases": ["veeam agent", "agent", "physical", "computers"]},
  {"item": "Unstructured data / NAS", "aliases": ["unstructured", "nas", "file share", "file shares"]},
  {"item": "Enterprise applications / databases", "aliases": ["sap", "oracle", "sql", "plug-in", "plug-ins", "database", "databases", "intersystems"]},
  {"item": "Microsoft 365", "aliases": ["microsoft 365", "m365", "office 365"]},
  {"item": "Kubernetes", "aliases": ["kubernetes", "kasten"]},
  {"item": "Cloud VMs", "aliases": ["azure vm", "azure vms", "ec2", "google cloud", "veeam backup for aws", "veeam backup for microsoft azure"]},
  {"item": "Salesforce", "aliases": ["salesforce"]},
  {"item": "Entra ID", "aliases": ["entra"]}],
 "bonus_items": [
  {"item": "Hardened repository", "aliases": ["hardened"]},
  {"item": "Object storage", "aliases": ["object storage", "s3", "blob", "wasabi", "data cloud vault"]},
  {"item": "Deduplicating appliances", "aliases": ["data domain", "storeonce", "deduplicating"]}]},
{"question": "Where can I restore a Windows agent backup?", "intent": "local",
 "expected_modes": ["local"], "kind": "single_list", "expected_article_ids": [],
 "expected_items": [
  {"item": "VMware vSphere", "aliases": ["vmware", "vsphere"]},
  {"item": "Microsoft Hyper-V", "aliases": ["hyper-v"]},
  {"item": "Nutanix AHV", "aliases": ["nutanix", "ahv"]},
  {"item": "Proxmox VE", "aliases": ["proxmox"]},
  {"item": "oVirt KVM (OLVM / RHV)", "aliases": ["ovirt", "olvm", "red hat virtualization", "rhv"]},
  {"item": "Scale Computing HyperCore", "aliases": ["scale computing", "hypercore"]},
  {"item": "Sangfor aSV", "aliases": ["sangfor"]},
  {"item": "HPE Morpheus VM Essentials", "aliases": ["morpheus"]},
  {"item": "Microsoft Azure", "aliases": ["azure"]},
  {"item": "Amazon EC2", "aliases": ["ec2"]},
  {"item": "Google Compute Engine", "aliases": ["google compute engine", "google cloud", "google"]},
  {"item": "Bare metal", "aliases": ["bare metal", "bare-metal", "recovery media"]},
  {"item": "Volumes", "aliases": ["volume", "volumes"]},
  {"item": "Files and folders", "aliases": ["files and folders", "file-level", "individual files", "guest os file"]},
  {"item": "Application items", "aliases": ["application item", "application items", "veeam explorer", "veeam explorers"]},
  {"item": "Disk export (VMDK/VHD/VHDX)", "aliases": ["vmdk", "vhd", "vhdx", "export disks", "disk export"]},
  {"item": "Publish disks", "aliases": ["publish", "publishing"]}]},
{"question": "Which hypervisors does Veeam Backup & Replication support?", "intent": "local",
 "expected_modes": ["local"], "kind": "single_list", "expected_article_ids": [],
 "expected_items": [
  {"item": "VMware vSphere", "aliases": ["vmware vsphere", "vsphere"]},
  {"item": "VMware Cloud Director", "aliases": ["cloud director", "vcloud director"]},
  {"item": "Microsoft Hyper-V", "aliases": ["hyper-v"]},
  {"item": "Nutanix AHV", "aliases": ["nutanix", "ahv"]},
  {"item": "Proxmox VE", "aliases": ["proxmox"]},
  {"item": "oVirt KVM", "aliases": ["ovirt"]},
  {"item": "Oracle Linux Virtualization Manager", "aliases": ["olvm", "oracle linux virtualization"]},
  {"item": "Red Hat Virtualization", "aliases": ["red hat virtualization", "rhv"]},
  {"item": "XCP-ng / Citrix XenServer", "aliases": ["xcp-ng", "xenserver"]},
  {"item": "Scale Computing HyperCore", "aliases": ["scale computing", "hypercore"]},
  {"item": "Sangfor aSV", "aliases": ["sangfor"]},
  {"item": "HPE Morpheus VM Essentials", "aliases": ["morpheus"]},
  {"item": "Universal Hypervisors", "aliases": ["universal hypervisor", "universal hypervisors", "platform9"]},
  {"item": "Red Hat OpenShift Virtualization", "aliases": ["openshift", "kubevirt"]}]},
{"question": "Which resource types can AWS Backup protect?", "intent": "local",
 "expected_modes": ["local"], "kind": "single_list", "expected_article_ids": [],
 "expected_items": [
  {"item": "Amazon EC2", "aliases": ["ec2"]},
  {"item": "Amazon EBS", "aliases": ["ebs", "elastic block store"]},
  {"item": "Amazon S3", "aliases": ["s3"]},
  {"item": "Amazon RDS", "aliases": ["rds"]},
  {"item": "Amazon Aurora", "aliases": ["aurora"]},
  {"item": "Amazon DynamoDB", "aliases": ["dynamodb"]},
  {"item": "Amazon EFS", "aliases": ["efs", "elastic file system"]},
  {"item": "Amazon FSx", "aliases": ["fsx"]},
  {"item": "AWS Storage Gateway", "aliases": ["storage gateway"]},
  {"item": "Amazon DocumentDB", "aliases": ["documentdb"]},
  {"item": "Amazon Neptune", "aliases": ["neptune"]},
  {"item": "Amazon Redshift", "aliases": ["redshift"]},
  {"item": "Amazon Timestream", "aliases": ["timestream"]},
  {"item": "Amazon EKS", "aliases": ["eks"]},
  {"item": "VMware VMs", "aliases": ["vmware"]},
  {"item": "SAP HANA on EC2", "aliases": ["sap hana"]},
  {"item": "AWS CloudFormation stacks", "aliases": ["cloudformation"]}]},
{"question": "What recovery options does Azure Backup offer for Azure VMs?", "intent": "local",
 "expected_modes": ["local"], "kind": "single_list", "expected_article_ids": [],
 "expected_items": [
  {"item": "Create a new VM", "aliases": ["create a new vm", "new vm", "create vm", "alternate location"]},
  {"item": "Restore disks", "aliases": ["restore disks", "restore disk", "restore the disks"]},
  {"item": "Replace existing VM (in-place)", "aliases": ["replace existing", "in-place", "in place"]},
  {"item": "Instant restore", "aliases": ["instant restore"]},
  {"item": "File / folder-level recovery", "aliases": ["file-level", "file recovery", "files and folders", "item-level", "file or folder"]},
  {"item": "Selective disk restore", "aliases": ["selective disk"]},
  {"item": "Encrypted VM restore", "aliases": ["encrypted"]},
  {"item": "Cross-region restore", "aliases": ["cross-region", "cross region", "paired region", "secondary region"]},
  {"item": "Cross-subscription restore", "aliases": ["cross-subscription", "cross subscription", "different subscription"]},
  {"item": "Cross-zonal restore", "aliases": ["cross-zonal", "cross zonal", "availability zone", "zone"]}]}
```

Also change the existing probe entry "Which vendors does Veeam integrate with?" (shape single by the named-subject rule, and a list question): set its `expected_modes` to `["global", "drift", "local"]` — list mode answers it, labelled `local`. Keep every other existing entry unchanged.

- [ ] **Step 5: `eval_router.py`** — per question, when `q.get("expected_items")`: `found, missing = match_items(env["answer"], q["expected_items"])`; record `item_coverage = round(len(found) / len(q["expected_items"]), 2)` and `items_missing = missing` (bonus items: record `bonus_found` count, not scored); add `item_coverage` to the progress line and a summary `item_coverage` mean over list questions (n), printed next to enumeration completeness and written to the report table (follow how `enumeration_complete` is threaded through record, failed-record, summary and the markdown table). Non-list questions: `None`.

- [ ] **Step 6: `eval_retrieval.py`** — when a golden question has `expected_items`, run `list_mode.retrieve_list(...)` (the map client is already built in `main`; pass `scope` resolved as for other questions) and check `match_items(" || ".join(fact texts of results + limitations), expected_items)`; `pass` requires no missing item; print `items {found}/{total} missing={missing}`. Other questions unchanged.

- [ ] **Step 7: `CLAUDE.md`** — in the `answer_api/` module bullet add: "`list_mode.py` (single-vendor list questions: deterministic detection, cheap-tier item-kind expansion, per-query reranked search, `fact_dedup.collapse_near_duplicates`, supported-first budgets, list-answer prompt; routed before the normal modes unless `mode=` is explicit), `item_coverage.py` (golden `expected_items` matching)."

- [ ] **Step 8: Run** `uv run --extra dev pytest tests/unit -q`, ruff, mypy.

- [ ] **Step 9: Commit**

```bash
git add src/answer_api/item_coverage.py src/answer_api/router_golden_tier1.json src/answer_api/eval_router.py src/answer_api/eval_retrieval.py CLAUDE.md tests/unit/test_item_coverage.py
git commit -m "feat(eval): five golden list questions, item coverage, list retrieval gate"
```

---

## Controller steps after the final review (each needs the user's approval where noted)

1. Free gate: `uv run python -m answer_api.eval_retrieval --questions src/answer_api/router_golden_tier1.json` — every list question must find all expected items in retrieval (cheap expansion calls only, fractions of a cent). Iterate on detection/expansion/budgets (no paid run) until it passes or a miss is diagnosed and shown to the user.
2. Full CI suite; one paid Tier 1 eval (21 questions, ~$0.25, user approval): item coverage 1.00 on list questions, no regression on the existing 16.
3. Merge/push/deploy (user approval).

---

## Amendment tasks (spec §10, approved 2026-10-15)

### Task 6: Tighter collapse, vendorless rule, fixed-wording degradation note, expansion prompt

**Files:**
- Modify: `src/graph_extract/config.py` (`list_dedup_cosine` 0.92 → 0.96), `src/answer_api/fact_dedup.py` (`_same_vendor`), `src/answer_api/list_mode.py` (`degraded_note`, `_EXPAND_PROMPT`)
- Test: `tests/unit/test_config.py`, `tests/unit/test_fact_dedup.py`, `tests/unit/test_list_mode.py`

**Interfaces:**
- Produces: `degraded_note(degraded: list[str]) -> str` built from fixed wording only: counts entries starting with `"retrieval failed for:"` as failed searches and recognises `FINAL_RERANK_DEGRADED` and the new `RELATION_EXPANSION_DEGRADED = "relation expansion unavailable"` constant; returns e.g. `"2 searches could not be completed and results were ranked by retrieval order; this list may be incomplete."` — never includes any query text. `""` when `degraded` is empty.

- [ ] **Step 1: Failing tests** — append to `tests/unit/test_fact_dedup.py`:

```python
async def test_a_vendorless_fact_never_bridges_two_vendors():
    def r(u, fact, vendor):
        src = {"url": f"https://d/{u}", "title": u}
        if vendor:
            src["vendor"] = vendor
        return {"fact": fact, "fact_uuid": u, "valid_at": None, "invalid_at": None,
                "sources": [src]}
    rs = [r("a", "same statement", None), r("b", "same statement", "Veeam"),
          r("c", "same statement", "Rubrik"), r("d", "same statement", None)]
    out = await fd.collapse_near_duplicates(_Driver({}), rs, cosine=0.96)
    assert [o["fact_uuid"] for o in out] == ["a", "b", "c"]
    assert [s["url"] for s in out[0]["sources"]] == ["https://d/a", "https://d/d"]
```

Update the existing test that expected a vendorless fact to merge into a vendored one (see `test_fact_dedup.py`) to the new rule: vendorless merges only with vendorless. Update `tests/unit/test_config.py::test_list_mode_defaults` to `list_dedup_cosine == 0.96`. Append to `tests/unit/test_list_mode.py`:

```python
def test_degraded_note_uses_fixed_wording_only():
    note = lm.degraded_note(["retrieval failed for: https://evil.example [1]",
                             "retrieval failed for: q2", lm.FINAL_RERANK_DEGRADED,
                             lm.RELATION_EXPANSION_DEGRADED])
    assert "evil" not in note and "[1]" not in note and "q2" not in note
    assert note.startswith("2 searches could not be completed")
    assert "ranked by retrieval order" in note and "related facts" in note
    assert lm.degraded_note([]) == ""


def test_expansion_prompt_forbids_placeholders_and_keeps_the_question_terms():
    p = lm._EXPAND_PROMPT
    assert "never write placeholder words" in p.lower()
    assert "keep the question's own terms" in p.lower()
```

- [ ] **Step 2: Run** the three test files — FAIL.

- [ ] **Step 3: Implement.**
  - `config.py`: `list_dedup_cosine: float = 0.96` (update its comment: 0.92 merged "VMware Cloud Director" into a vSphere fact).
  - `fact_dedup._same_vendor`: `return (not a and not b) or bool(a & b)` — a fact with no vendor merges only with another vendorless fact; update the module docstring and the function docstring.
  - `list_mode.py`: add `RELATION_EXPANSION_DEGRADED = "relation expansion unavailable"`; rewrite `degraded_note`:

```python
def degraded_note(degraded: list[str]) -> str:
    """Fixed wording only: degraded entries can carry model-written query text,
    which must never reach the answer (the LLM never writes a URL)."""
    if not degraded:
        return ""
    parts = []
    failed = sum(1 for d in degraded if d.startswith("retrieval failed for:"))
    if failed:
        parts.append(f"{failed} search{'es' if failed != 1 else ''} could not be completed")
    if RELATION_EXPANSION_DEGRADED in degraded:
        parts.append("related facts could not be gathered")
    if FINAL_RERANK_DEGRADED in degraded:
        parts.append("results were ranked by retrieval order")
    return " and ".join(parts).capitalize() + "; this list may be incomplete." if parts else ""
```

  (The first letter is capitalised; the test checks `startswith("2 searches")`, which a leading digit satisfies.)
  - `_EXPAND_PROMPT`: replace "Keep the vendor and product names from the question in every query." with "Keep the vendor and product names from the question in every query; when the question names no vendor or product, keep the question's own terms (for example 'Windows agent'). Never write placeholder words such as 'vendor', 'product' or 'software' in place of a name."

- [ ] **Step 4: Run** the files, then the unit suite, ruff, mypy.

- [ ] **Step 5: Commit** — `git commit -m "fix(answer): collapse at 0.96, no vendorless bridge, fixed-wording degradation note, expansion keeps question terms"` (add the trailer lines).

---

### Task 7: Relation expansion, batched final rerank, MMR selection

**Files:**
- Modify: `src/graph_extract/config.py` (new settings; `list_supported_k` 50 → 80)
- Modify: `src/answer_api/list_mode.py` (`relation_siblings`, `_final_order` batching with scores, `select_diverse`, `retrieve_list`)
- Test: `tests/unit/test_list_mode.py`, `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `collapse_near_duplicates`, `fact_dedup.EMBEDDINGS_QUERY`, `answer_api.scope.scope_episode_uuids(driver, scope, candidates=...)`, `graph_extract.provenance.Provenance(driver).resolve_citations(uuids) -> {uuid: {"sources", "valid_at", "invalid_at"}}`, `answer_api.rerank.rerank`, `rerank_configured`.
- Produces:
  - Settings: `list_relation_seed_facts = 40`, `list_relation_groups = 6`, `list_relation_group_cap = 300`, `list_mmr_lambda = 0.5`, `list_rerank_batch = 900`; `list_supported_k = 80`.
  - `async relation_siblings(driver, seeds: list[dict], *, scope, settings, exclude: set[str]) -> list[dict]` — result dicts like search results.
  - `async _final_order(q, facts, *, settings, degraded) -> list[tuple[dict, float | None]]` — every fact with its rerank score (None when unranked), best first; batched by `list_rerank_batch`.
  - `select_diverse(scored: list[tuple[dict, float | None]], vectors: dict[str, list[float]], *, k: int, lam: float) -> list[dict]`.

- [ ] **Step 1: Failing tests** — append to `tests/unit/test_list_mode.py` (reuse `_fact`, `_settings`, `patched`):

```python
class _GraphDriver:
    """Answers the relation-expansion queries; records calls."""
    def __init__(self, meta, siblings):
        self.meta, self.siblings, self.calls = meta, siblings, []

    async def execute_query(self, query, **p):
        self.calls.append(query)
        if query is lm.SEED_EDGES_QUERY:
            return ([m for m in self.meta if m["uuid"] in p["uuids"]], None, None)
        if query is lm.SIBLINGS_QUERY:
            return (self.siblings.get((p["src"], p["rel"]), [])[: p["cap"]], None, None)
        return ([], None, None)


async def test_relation_siblings_follow_the_top_subject_relation_groups(monkeypatch):
    seeds = [_fact("s1", "VBR supports Hyper-V"), _fact("s2", "VBR supports Nutanix AHV")]
    drv = _GraphDriver(
        meta=[{"uuid": "s1", "src": "vbr", "rel": "Supports"},
              {"uuid": "s2", "src": "vbr", "rel": "Supports"}],
        siblings={("vbr", "Supports"): [
            {"uuid": "s1", "fact": "VBR supports Hyper-V", "episodes": ["e1"]},
            {"uuid": "x1", "fact": "VBR supports Sangfor aSV", "episodes": ["e2"]},
            {"uuid": "x2", "fact": "VBR supports Scale Computing HyperCore", "episodes": ["e3"]}]})

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {u: {"sources": [{"url": f"https://d/{u}", "vendor": "Veeam"}],
                        "valid_at": None, "invalid_at": None} for u in uuids}

    async def _allowed(driver, scope, candidates=None):
        return {"e2"}                      # e3 is out of scope

    monkeypatch.setattr(lm, "Provenance", _Prov)
    monkeypatch.setattr(lm, "scope_episode_uuids", _allowed)
    out = await lm.relation_siblings(drv, seeds, scope=Scope(vendors=("Veeam",)),
                                     settings=_settings(), exclude={"s1", "s2"})
    assert [r["fact_uuid"] for r in out] == ["x1"]
    assert out[0]["sources"][0]["url"] == "https://d/x1"


def test_mmr_prefers_a_new_item_over_a_near_copy():
    a, a2, b = _fact("a", "A item"), _fact("a2", "A item again"), _fact("b", "B item")
    vecs = {"a": [1.0, 0.0], "a2": [0.99, 0.01], "b": [0.0, 1.0]}
    got = lm.select_diverse([(a, 0.9), (a2, 0.89), (b, 0.6)], vecs, k=2, lam=0.5)
    assert [r["fact_uuid"] for r in got] == ["a", "b"]


def test_mmr_without_vectors_keeps_rerank_order():
    facts = [(_fact(u, u), s) for u, s in (("a", 0.9), ("b", 0.8), ("c", 0.7))]
    assert [r["fact_uuid"] for r in lm.select_diverse(facts, {}, k=2, lam=0.5)] == ["a", "b"]


async def test_final_order_batches_and_merges_by_score(monkeypatch):
    calls = []

    async def _rerank(q, docs, *, top_k, settings):
        calls.append(len(docs))
        return [(i, float(int(d.split()[-1]))) for i, d in enumerate(docs)]

    monkeypatch.setattr(lm, "rerank", _rerank)
    monkeypatch.setattr(lm, "rerank_configured", lambda s: True)
    facts = [_fact(f"f{i}", f"fact {i}") for i in range(5)]
    out = await lm._final_order("q", facts, settings=_settings(list_rerank_batch=2),
                                degraded=[])
    assert calls == [2, 2, 1]
    assert [f["fact_uuid"] for f, _ in out] == ["f4", "f3", "f2", "f1", "f0"]


async def test_a_relation_expansion_failure_degrades_and_keeps_round_one(patched, monkeypatch):
    install, _ = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")

    async def _boom(*a, **k):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(lm, "relation_siblings", _boom)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert [r["fact_uuid"] for r in got.results] == ["a"]
    assert lm.RELATION_EXPANSION_DEGRADED in got.degraded
```

Update the existing `patched` fixture's fakes so `_final_order`/rerank and `relation_siblings` are faked consistently (relation_siblings returning `[]` by default) and existing assertions keep passing; update `test_list_mode_defaults` for `list_supported_k == 80` and the five new settings.

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement** in `list_mode.py` (module-level imports so tests can monkeypatch: `from answer_api.scope import scope_episode_uuids`, `from graph_extract.provenance import Provenance`, `from answer_api.fact_dedup import EMBEDDINGS_QUERY, collapse_near_duplicates`, `import numpy as np`, `import collections`):

```python
SEED_EDGES_QUERY = (
    "MATCH (a:Entity)-[f:RELATES_TO]->(:Entity) WHERE f.uuid IN $uuids "
    "RETURN f.uuid AS uuid, a.uuid AS src, f.name AS rel")
SIBLINGS_QUERY = (
    "MATCH (a:Entity {uuid: $src})-[f:RELATES_TO]->(:Entity) WHERE f.name = $rel "
    "AND (f.invalid_at IS NULL OR f.invalid_at > datetime()) "
    "RETURN f.uuid AS uuid, f.fact AS fact, f.episodes AS episodes LIMIT $cap")


async def relation_siblings(driver, seeds, *, scope, settings, exclude):
    """Same subject entity + same relation name as the top round-one facts (spec §10.1):
    grounded recall for long-tail items the expansion never named."""
    top = seeds[:settings.list_relation_seed_facts]
    meta, _, _ = await driver.execute_query(
        SEED_EDGES_QUERY, uuids=[r["fact_uuid"] for r in top])
    groups = collections.Counter((m["src"], m["rel"]) for m in meta if m["src"] and m["rel"])
    rows: dict[str, dict] = {}
    for (src, rel), _n in groups.most_common(settings.list_relation_groups):
        found, _, _ = await driver.execute_query(
            SIBLINGS_QUERY, src=src, rel=rel, cap=settings.list_relation_group_cap)
        for r in found:
            if r["uuid"] not in exclude and r["uuid"] not in rows:
                rows[r["uuid"]] = r
    if scope is not None and not scope.is_empty() and rows:
        allowed = await scope_episode_uuids(
            driver, scope, candidates=[e for r in rows.values() for e in (r["episodes"] or [])])
        rows = {u: r for u, r in rows.items() if allowed.intersection(r["episodes"] or [])}
    if not rows:
        return []
    resolved = await Provenance(driver).resolve_citations(list(rows))
    return [{"fact": r["fact"], "fact_uuid": u,
             "valid_at": resolved.get(u, {}).get("valid_at"),
             "invalid_at": resolved.get(u, {}).get("invalid_at"),
             "sources": resolved.get(u, {}).get("sources", [])} for u, r in rows.items()]


def select_diverse(scored, vectors, *, k, lam):
    """Maximal marginal relevance: score minus lam x highest raw cosine to a chosen fact."""
    if not scored:
        return []
    if not vectors:
        return [f for f, _ in scored[:k]]
    dim = len(next(iter(vectors.values())))
    mat = np.zeros((len(scored), dim))
    has = []
    for i, (f, _) in enumerate(scored):
        v = vectors.get(f["fact_uuid"])
        ok = v is not None and len(v) == dim
        has.append(ok)
        if ok:
            mat[i] = v
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.divide(mat, norms, out=mat, where=norms > 0)
    rel = np.array([s if s is not None else 0.0 for _, s in scored])
    chosen: list[int] = []
    best_sim = np.zeros(len(scored))
    remaining = set(range(len(scored)))
    while remaining and len(chosen) < k:
        i = max(remaining, key=lambda j: (rel[j] - lam * best_sim[j], -j))
        chosen.append(i)
        remaining.discard(i)
        if has[i]:
            best_sim = np.maximum(best_sim, np.where(has, mat @ mat[i], 0.0))
    return [scored[i][0] for i in chosen]
```

`_final_order` returns `list[tuple[dict, float | None]]`: when unconfigured or failed, `[(f, None) for f in facts]` plus the degraded note (as today); otherwise rerank in batches of `settings.list_rerank_batch` (`top_k=len(batch)` each), collect `(fact, score)`, append any fact a batch reply omitted with score `None`, and sort by score descending (None last, stable).

`retrieve_list` becomes: round one (unchanged up to `distinct`) → `ranked1 = await _final_order(q, distinct, ...)` → try `siblings = await relation_siblings(driver, [f for f, _ in ranked1], scope=scope, settings=settings, exclude={f["fact_uuid"] for f in distinct})` except `Exception` (log WARNING, `degraded.append(RELATION_EXPANSION_DEGRADED)`, `siblings = []`) → if siblings: `distinct2 = await collapse_near_duplicates(driver, [f for f, _ in ranked1] + siblings, cosine=...)`, `ranked = await _final_order(q, distinct2, ...)` else `ranked = ranked1` → split into supported/limitations by `is_limitation` keeping scores → fetch vectors for supported uuids with `EMBEDDINGS_QUERY` (on failure, `{}` with a WARNING) → `results = select_diverse(supported_scored, vectors, k=settings.list_supported_k, lam=settings.list_mmr_lambda)`; `limitations = [f for f, _ in limits_scored][:settings.list_limitations_k]`. Record each degraded note only once.

- [ ] **Step 4: Run** `tests/unit/test_list_mode.py`, the unit suite, ruff, mypy.

- [ ] **Step 5: Commit** — `git commit -m "feat(answer): list relation expansion, batched final rerank, MMR selection"` (add the trailer lines).

---

### Task 8: List-mode latency (user-approved option 1, 2026-10-16)

Measured stage times per list question (6 relation groups): searches ~13-20 s wall (11 concurrent), `relation_siblings` 6-9 s unscoped/small scope but **434 s** for "Which hypervisors does Veeam Backup & Replication support?" (scoped `EXISTS` chain walk on every edge of a big group before ORDER BY/LIMIT), `collapse_near_duplicates` 35-39 s over 2 calls, `_fact_vectors` 18-33 s — the same `fact_embedding` vectors (~2000 x 768) are fetched three times. Target: < 45 s per list question with unchanged item coverage.

**Files:** `src/answer_api/list_mode.py`, `src/answer_api/fact_dedup.py`, `src/graph_extract/config.py`; tests `tests/unit/test_list_mode.py`, `tests/unit/test_fact_dedup.py`, `tests/integration/test_list_siblings.py`.

**Changes:**
1. **Sibling fetch:** per group, run an UNORDERED bounded fetch `LIMIT $cap * 4` (no `ORDER BY`, no in-query scope `EXISTS`), then — when scoped — filter in Python with `scope_episode_uuids(driver, scope, candidates=<episodes of the fetched edges>)` (bounded by the fetched set), then sort the survivors by `uuid` and keep `list_relation_group_cap`. Keep groups concurrent. Add `list_relation_total_cap: int = 600` (siblings kept across all groups, in group order then uuid). Update the integration test to the new queries (deterministic output for a static graph, scope respected, caps respected).
2. **Embeddings once:** add `async fetch_vectors(driver, uuids, cache: dict[str, list[float]]) -> dict[str, list[float]]` in `fact_dedup` that queries only uuids missing from `cache` (one `EMBEDDINGS_QUERY`), updates the cache and returns it; `collapse_near_duplicates(driver, results, *, cosine, vectors: dict | None = None)` uses a supplied cache instead of fetching (behaviour otherwise identical; a failed fetch still degrades to text-only). `retrieve_list` creates one cache per question and passes it to both collapses and to MMR selection (`_fact_vectors` reuses the cache; only uuids not yet cached are fetched). The single-question path keeps calling `collapse_near_duplicates` without a cache (unchanged).
3. **Timing log:** `retrieve_list` logs one INFO line with per-stage durations (expansion, searches, collapse, rerank, relations, selection) so latency is visible in production logs.

Tests: fetch_vectors fetches only missing uuids and reuses the cache; collapse with a supplied cache issues no embeddings query; retrieve_list issues at most one embeddings query per distinct uuid across the whole call (count via the fake driver); sibling fetch with the unordered cap*4 + Python scope filter keeps only in-scope siblings and respects both caps; the stage-timing log line is emitted. Keep every existing assertion that still describes intended behaviour.

---

## Amendment 2 tasks (spec §12, approved 2026-10-16)

### Task 9: Keyword sweep

**Files:** create `src/answer_api/list_sweep.py`; modify `src/answer_api/list_mode.py`, `src/graph_extract/config.py`; tests: create `tests/unit/test_list_sweep.py`, modify `tests/unit/test_list_mode.py`, `tests/unit/test_config.py`.

**Interfaces produced (Task 10 relies on them):**
- `list_sweep.STOPWORDS: frozenset[str]`, `list_sweep.stem(word: str) -> str`, `list_sweep.question_terms(q: str, scope: Scope | None) -> list[str]` (lower-case words, question order, deduplicated).
- `list_sweep._PLURAL` and `list_sweep._ITEM_NOUNS` — MOVED here from `list_mode.py` (list_mode imports them; list_sweep must never import list_mode).
- `list_sweep.lucene_query(terms: list[str]) -> str`, `list_sweep.SWEEP_QUERY`, `list_sweep.FULLTEXT_INDEX = "edge_name_and_fact"`.
- `async list_sweep.keyword_sweep(driver, q: str, *, scope: Scope | None, group_id: str, settings) -> list[dict]` — fact dicts shaped exactly like `relation_siblings` output (`fact, fact_uuid, valid_at, invalid_at, sources`).
- `list_mode.SWEEP_DEGRADED = "keyword search unavailable"`.

**Settings** (in the list-mode block of `ExtractSettings`, with a comment citing spec §12.1): `list_sweep_cap: int = 2000`, `list_sweep_scoped_overfetch: int = 4`.

**`list_sweep.py` content:**

```python
"""Keyword sweep for list mode (spec 2026-10-15 §12.1): a full-text search of fact text
for the question's own distinctive words. Every word comes from the question and every
hit is a stored fact -- nothing is inferred. It reaches long-tail items whose facts hang
off subjects the relation expansion never visits (2026-10-16: every item still missing
at 250 facts had a fact naming it, e.g. "... backups created by Veeam Agents to the HPE
Morpheus VM Essentials environment")."""
from __future__ import annotations

import re

from answer_api.scope import Scope, scope_episode_uuids
from graph_extract.provenance import Provenance

# A plural noun: ends in -s but not -ss/-us/-is ("process", "status", "analysis").
_PLURAL = r"\w+s(?<!ss)(?<!us)(?<!is)"
_ITEM_NOUNS = (r"(options|ways|methods|types|kinds|destinations|targets|choices|"
               r"alternatives|modes|tiers|levels|formats|platforms|locations)")

STOPWORDS = frozenset((
    "a an the and or not of for to in on at by with from into via as is are was were be "
    "been can could do does did will would should must may might shall i you we they it "
    "its my our your me us which what where when how who whom whose why list all any "
    "every each some there their this that these those than then also about between "
    "other support supports supported supporting provide provides offer offers use uses "
    "using allow allows enable enables available possible").split())
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
# The list's own item noun ("Which workloads ...", "What recovery options ...").
_ITEM_NOUN_RE = re.compile(r"^\s*(?:which|what)\s+(?:[\w\-]+\s+){0,3}?(" + _PLURAL + r")\b",
                           re.IGNORECASE)
_ITEM_NOUN_WORDS = frozenset(_ITEM_NOUNS.strip("()").split("|"))
# Longest first; a stem keeps at least 4 characters.
_SUFFIXES = ("ility", "ations", "ation", "ions", "ion", "ings", "ing", "ies", "es", "ed",
             "er", "s", "e", "y")

FULLTEXT_INDEX = "edge_name_and_fact"
SWEEP_QUERY = (
    "CALL db.index.fulltext.queryRelationships($index, $query, {limit: $fetch}) "
    "YIELD relationship AS f, score "
    "WHERE f.group_id = $group_id AND (f.invalid_at IS NULL OR f.invalid_at > datetime()) "
    "RETURN f.uuid AS uuid, f.fact AS fact, f.episodes AS episodes, score")


def stem(word: str) -> str:
    """Crude suffix strip for prefix matching: immutability -> immutab, restore ->
    restor, recovery -> recover, databases -> databas; agent stays agent."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def question_terms(q: str, scope: Scope | None) -> list[str]:
    """The question's distinctive words: no stopwords or generic verbs, no list item
    noun, no word of the scope's vendor/product/source names (the scope already
    restricts to them)."""
    drop = set(STOPWORDS) | _ITEM_NOUN_WORDS
    m = _ITEM_NOUN_RE.search(q)
    if m:
        drop.add(m.group(1).lower())
    if scope is not None:
        for name in (*scope.vendors, *scope.products, *scope.sources):
            drop.update(_WORD_RE.findall(name.lower()))
    out: list[str] = []
    for w in _WORD_RE.findall(q.lower()):
        if w not in drop and w not in out:
            out.append(w)
    return out


def lucene_query(terms: list[str]) -> str:
    """OR of every term, plus its stem as a prefix when the stem is a plain word of at
    least 4 characters; '-' is escaped (Lucene syntax) and never wildcarded."""
    parts: list[str] = []
    for t in terms:
        if "-" in t:
            parts.append(t.replace("-", "\\-"))
            continue
        parts.append(t)
        s = stem(t)
        if len(s) >= 4:
            parts.append(f"{s}*")
    return " ".join(parts)


async def keyword_sweep(driver, q: str, *, scope: Scope | None, group_id: str,
                        settings) -> list[dict]:
    """Up to `list_sweep_cap` current facts matching the question's words, best BM25
    first; scoped questions over-fetch `list_sweep_scoped_overfetch` x the cap and keep
    only facts with an episode in scope. Sources resolve through the provenance chain."""
    terms = question_terms(q, scope)
    cap = settings.list_sweep_cap
    if not terms or cap <= 0:
        return []
    scoped = scope is not None and not scope.is_empty()
    fetch = cap * (settings.list_sweep_scoped_overfetch if scoped else 1)
    rows, _, _ = await driver.execute_query(
        SWEEP_QUERY, index=FULLTEXT_INDEX, query=lucene_query(terms), fetch=fetch,
        group_id=group_id)
    rows = sorted(rows, key=lambda r: -r["score"])        # stable: BM25 order
    if scoped:
        assert scope is not None
        allowed = await scope_episode_uuids(
            driver, scope, candidates=[e for r in rows for e in r["episodes"] or ()])
        rows = [r for r in rows if allowed.intersection(r["episodes"] or ())]
    kept: dict[str, dict] = {}
    for r in rows:
        if len(kept) >= cap:
            break
        kept.setdefault(r["uuid"], r)
    if not kept:
        return []
    resolved = await Provenance(driver).resolve_citations(list(kept))
    return [{"fact": r["fact"], "fact_uuid": u,
             "valid_at": resolved.get(u, {}).get("valid_at"),
             "invalid_at": resolved.get(u, {}).get("invalid_at"),
             "sources": resolved.get(u, {}).get("sources", [])} for u, r in kept.items()]
```

**`list_mode.py` changes:**
- Delete the `_PLURAL` and `_ITEM_NOUNS` definitions; `from answer_api.list_sweep import _ITEM_NOUNS, _PLURAL, keyword_sweep`.
- Add `SWEEP_DEGRADED = "keyword search unavailable"` next to the other degraded constants; `degraded_note` adds the part `"the keyword search could not be completed"` when it is present.
- In `retrieve_list`, run the sweep in the SAME `asyncio.gather` as the per-query searches (append `keyword_sweep(driver, q, scope=scope, group_id=group_id, settings=settings)` as the last awaitable; `return_exceptions=True` already set; `raise_if_cancelled` already called). Split its result off before the per-query loop: an exception → WARNING `"list keyword sweep failed (%s); continuing without it"` + `_degrade(degraded, SWEEP_DEGRADED)`; otherwise extend `merged` with the swept facts AFTER the per-query results. The "every list retrieval failed" check still counts only the per-query searches. Add `swept=%d` to the stage-timing INFO line.

**Tests (`tests/unit/test_list_sweep.py`, hermetic):**

```python
"""Keyword sweep (spec 2026-10-15 §12.1)."""
import pytest

import answer_api.list_sweep as ls
from answer_api.scope import _EPISODES_AMONG, Scope
from graph_extract.config import ExtractSettings


def _settings(**kw):
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p", **kw)


@pytest.mark.parametrize("q,scope,terms", [
    ("Which workloads does Veeam support immutability for?", Scope(vendors=("Veeam",)),
     ["immutability"]),
    ("Where can I restore a Windows agent backup?", Scope(),
     ["restore", "windows", "agent", "backup"]),
    ("Which hypervisors does Veeam Backup & Replication support?",
     Scope(products=("Veeam Backup & Replication",)), []),
    ("Which resource types can AWS Backup protect?", Scope(products=("AWS Backup",)),
     ["resource", "protect"]),
    ("What recovery options does Azure Backup offer for Azure VMs?",
     Scope(vendors=("Microsoft",), products=("Azure Backup",)), ["recovery", "vms"]),
])
def test_question_terms_keep_only_distinctive_words(q, scope, terms):
    assert ls.question_terms(q, scope) == terms


@pytest.mark.parametrize("word,stemmed", [
    ("immutability", "immutab"), ("restore", "restor"), ("recovery", "recover"),
    ("databases", "databas"), ("restores", "restor"), ("agent", "agent"), ("vms", "vms")])
def test_stem(word, stemmed):
    assert ls.stem(word) == stemmed


def test_lucene_query_adds_stem_prefixes_and_escapes_hyphens():
    assert (ls.lucene_query(["restore", "windows", "agent", "vms", "hyper-v"])
            == "restore restor* windows window* agent agent* vms hyper\\-v")


class _Driver:
    def __init__(self, rows, in_scope=()):
        self.rows, self.in_scope, self.calls = rows, set(in_scope), []

    async def execute_query(self, query, **p):
        self.calls.append((query, p))
        if query is ls.SWEEP_QUERY:
            return (self.rows[: p["fetch"]], None, None)
        if query is _EPISODES_AMONG:
            return ([{"u": [e for e in p["eps"] if e in self.in_scope]}], None, None)
        raise AssertionError(query)


class _Prov:
    def __init__(self, driver):
        pass

    async def resolve_citations(self, uuids):
        return {u: {"sources": [{"url": f"https://d/{u}", "vendor": "Veeam"}],
                    "valid_at": None, "invalid_at": None} for u in uuids}


def _row(u, score, eps):
    return {"uuid": u, "fact": f"fact {u}", "episodes": eps, "score": score}


async def test_scoped_sweep_overfetches_filters_by_scope_and_caps(monkeypatch):
    monkeypatch.setattr(ls, "Provenance", _Prov)
    rows = [_row("a", 1.0, ["e1"]), _row("b", 3.0, ["e2"]), _row("c", 2.0, ["e3"]),
            _row("d", 0.5, ["e4"])]
    drv = _Driver(rows, in_scope={"e2", "e3", "e4"})
    got = await ls.keyword_sweep(drv, "Which workloads does Veeam support immutability for?",
                                 scope=Scope(vendors=("Veeam",)), group_id="g",
                                 settings=_settings(list_sweep_cap=2,
                                                    list_sweep_scoped_overfetch=4))
    query, p = drv.calls[0]
    assert p["index"] == "edge_name_and_fact" and p["fetch"] == 8 and p["group_id"] == "g"
    assert p["query"] == "immutability immutab*"
    assert [f["fact_uuid"] for f in got] == ["b", "c"]          # BM25 order, in scope, cap 2
    assert got[0]["sources"][0]["url"] == "https://d/b"


async def test_unscoped_sweep_fetches_the_cap_and_skips_the_scope_query(monkeypatch):
    monkeypatch.setattr(ls, "Provenance", _Prov)
    drv = _Driver([_row("a", 1.0, ["e1"]), _row("b", 2.0, [])])
    got = await ls.keyword_sweep(drv, "Where can I restore a Windows agent backup?",
                                 scope=None, group_id="g", settings=_settings())
    assert [c[0] for c in drv.calls] == [ls.SWEEP_QUERY]
    assert drv.calls[0][1]["fetch"] == 2000
    assert [f["fact_uuid"] for f in got] == ["b", "a"]


async def test_a_question_with_no_distinctive_words_runs_no_query():
    drv = _Driver([])
    got = await ls.keyword_sweep(drv, "Which hypervisors does Veeam Backup & Replication support?",
                                 scope=Scope(products=("Veeam Backup & Replication",)),
                                 group_id="g", settings=_settings())
    assert got == [] and drv.calls == []
```

**`tests/unit/test_list_mode.py` changes:** the `patched` fixture and `test_retrieve_list_fetches_each_embedding_at_most_once` stub `lm.keyword_sweep` (an `async def _no_sweep(driver, q, *, scope, group_id, settings): return []`), so existing assertions keep holding. New tests: (1) swept facts join round one — stub the sweep to return `[_fact("sw", "Restore to Sangfor aSV is supported.")]`; assert `"sw"` is among `got.results` and that the first rerank call's documents include its text; (2) a sweep that raises records `lm.SWEEP_DEGRADED` exactly once, the per-query results are still returned, and `lm.degraded_note([lm.SWEEP_DEGRADED])` contains `"keyword search could not be completed"`; (3) a sweep failure alone never triggers the "every list retrieval failed" error. `tests/unit/test_config.py`: assert `(s.list_sweep_cap, s.list_sweep_scoped_overfetch) == (2000, 4)`.

- [ ] **Step 1:** Write the tests above; run them; they fail (no module).
- [ ] **Step 2:** Create `list_sweep.py`, move `_PLURAL`/`_ITEM_NOUNS`, wire `retrieve_list`, add the settings.
- [ ] **Step 3:** Run `uv run --extra dev pytest tests/unit -q`, `uv run ruff check src tests`, `uv run mypy src` — all green.
- [ ] **Step 4:** Commit — `feat(answer): list keyword sweep over the question's own words` (with the trailer lines).

---

### Task 10: Per-item evidence grades

**Files:** create `src/answer_api/item_grades.py`; modify `src/answer_api/list_mode.py`, `src/answer_api/router.py`, `src/answer_api/mcp_server.py`, `src/answer_api/eval_router.py`, `src/graph_extract/config.py`; tests: create `tests/unit/test_item_grades.py`, modify `tests/unit/test_list_mode.py`, `tests/unit/test_list_router.py`, `tests/unit/test_config.py` (and the MCP test file that pins the structured keys, if one does).

**Interfaces:**
- Consumes (Task 9): `list_sweep.question_terms(q, scope)`, `list_sweep.stem`, `list_sweep.STOPWORDS`.
- Produces: `item_grades.CONFIRMED = "confirmed"`, `NEEDS_VALIDATION = "needs_validation"`, `DROPPED = "dropped"`; `@dataclass ItemGrade(item: str, grade: str, missing_terms: list[str], markers: list[int])` with `as_dict(marker_map) -> dict` returning `{"item", "grade", "missing_terms", "markers", "fact_uuids"}`; `grade_items(raw: str, marker_map: dict[int, dict], *, q: str, terms: list[str], scores: dict[str, float | None], floor: float) -> tuple[str, list[ItemGrade]]`.
- `ListRetrieval` gains `scores: dict[str, float | None] = field(default_factory=dict)` — the final rerank score per fact_uuid (`{f["fact_uuid"]: sc for f, sc in ranked}`), set in `retrieve_list`.
- `answer_list` return dict gains `"item_grades": [g.as_dict(marker_map) for g in grades]` (also `[]` on the refusal paths).
- Setting: `list_item_floor: float = 0.1` (comment: "best-supporting fact's rerank score below which a listed item is dropped; calibrated on the free gate, spec §12.3").

**`item_grades.py` rules:**
- An item line matches `^(\s*[-*]\s+)\*\*(?P<item>[^*]+)\*\*(?P<rest>.*)$`. Only lines BEFORE the first line matching `not supported\s*/\s*limitations` (case-insensitive) are graded; everything from that line on is untouched. A bullet line (`^\s*[-*]\s+`) in the graded region that does not match the item form is left untouched and counted; log one WARNING per answer `"list answer: %d bullet line(s) not in item form; left ungraded"` when the count > 0.
- Markers on a line: every integer inside `\[(\d+(?:\s*,\s*\d+)*)\]`, in order, deduplicated, kept only when in `marker_map`.
- Evidence text of a fact: `fact` + " " + every source `title`, lower-cased.
- `_present(term, text)`: true when `re.search(r"(?<![a-z0-9])" + re.escape(stem(term)), text)` matches, OR any member of the term's synonym group does. Synonym groups (by stem): `({"restor", "recover"}, {"protect", "backup"}, {"vms", "vm", "virtual machine"})`.
- Item's distinctive words: `_WORD_RE` words of the item (lower-cased) minus `STOPWORDS`, minus words whose stem equals the stem of a question term, minus `{"vm", "vms", "machine", "machines", "instance", "instances", "environment", "backup", "backups", "restore", "recovery"}`. The fact NAMES the item when at least `ceil(n / 2)` of its n distinctive words are `_present` in the evidence text; n == 0 counts as named.
- Supporting facts: cited facts that name the item and whose score is `None` or `>= floor`. None → DROPPED. Otherwise compute, per supporting fact (in marker order), the question terms NOT `_present` in its evidence; a fact with none missing → CONFIRMED; else NEEDS_VALIDATION with the first supporting fact's missing terms.
- A line with no usable marker → DROPPED.
- Output text: DROPPED lines removed; NEEDS_VALIDATION lines get ` *(needs validation: the cited source does not mention {terms})*` appended, where `{terms}` joins with ", " each missing term spelled as in the question (`re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", q, re.I).group(0)`, falling back to the term). A heading line (`^\s*(#{1,6}\s+\S.*|\*\*[^*]+\*\*:?)\s*$`) in the graded region left with no non-blank line before the next heading (or the limitations line, or the end) is removed too. Log INFO `"list items: %d confirmed, %d needs validation, %d dropped"`.

**`list_mode.py` changes:** in `_LIST_PROMPT`, replace `"Give each item one short line with its [N] marker; "` with `"Write each item as one line in exactly this form: - **<item name>** — <short description> [N]. Include an item even when its fact states it more generally than the question asks (for example for every agent rather than one); such items are checked and flagged automatically. "` (keep the rest). In `answer_list`, after `raw` is obtained and BEFORE `_finalize_answer`: `raw, grades = grade_items(raw, marker_map, q=q, terms=question_terms(q, scope), scores=got.scores, floor=settings.list_item_floor)`.

**Router / MCP / eval:** router list branch sets `env["item_grades"] = raw.get("item_grades", [])`; MCP `answer_question` adds `"item_grades"` to the structured keys copied when present; `eval_router` per-question record adds `"items_flagged"` (count of `needs_validation`) and `"items_dropped"` (count of `dropped`), `None` when the envelope has no `item_grades`; the summary adds their totals and the markdown table gains a `flagged/dropped` column.

**Tests (`tests/unit/test_item_grades.py`)** — build `marker_map` with `_fact`-style dicts (fact, fact_uuid, sources with title); cover:
1. Confirmed: q "Where can I restore a Windows agent backup?", terms `["restore","windows","agent","backup"]`, fact 1 "VBR supports backups created by Veeam Agent for Microsoft Windows for restore to Sangfor aSV." scored 0.6; raw `"**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]"` → text unchanged, grade confirmed, missing [].
2. Needs validation: fact "VBR can restore physical machines from backups created by Veeam Agents to the HPE Morpheus VM Essentials environment." → item `HPE Morpheus VM Essentials` flagged, missing `["windows"]`, line ends with `*(needs validation: the cited source does not mention Windows)*` (capital W, as in the question).
3. Article title counts as evidence: fact lacks "agent" but source title "Restoring Veeam Agent Backup to Proxmox VM" supplies it → confirmed.
4. Dropped — not named: item `Oracle Linux KVM` citing the Sangfor fact → line removed, grade dropped.
5. Dropped — below floor: named fact with score 0.05 at floor 0.1 → dropped; the same with score `None` → kept.
6. No usable marker (`[99]` not in map, or none) → dropped.
7. Synonym: q "What recovery options does Azure Backup offer for Azure VMs?", terms `["recovery","vms"]`, fact "Azure Backup can restore an Azure VM to an alternate location." for item `Alternate location restore` → confirmed (restore ~ recover, VM ~ vms).
8. Lines in and after "Not supported / limitations" are untouched even when they would drop; a heading whose only item was dropped is removed, other headings stay.
9. Comma marker list `[1, 2]` reads both markers; a non-item bullet line is left untouched and the WARNING is logged.

`tests/unit/test_list_mode.py`: the `answer_list` test with a bold-item synth reply returns `item_grades`; `ListRetrieval.scores` carries the rerank scores; the prompt contains `- **<item name>** —`. `tests/unit/test_list_router.py`: the list branch copies `item_grades` into the envelope. `tests/unit/test_config.py`: `list_item_floor == 0.1`.

- [ ] **Step 1:** Write the tests; run; they fail.
- [ ] **Step 2:** Implement `item_grades.py`, wire `answer_list`, router, MCP, eval_router, the setting.
- [ ] **Step 3:** Full unit suite, ruff (src AND tests), mypy — green.
- [ ] **Step 4:** Commit — `feat(answer): per-item evidence grades for list answers` (with the trailer lines).

---

### Task 11: Free-gate calibration output

**Files:** modify `src/answer_api/eval_retrieval.py`; create `tests/unit/test_eval_retrieval_list.py`.

`check_list_items` adds `"item_scores": {item: best_score_or_None}` — for each expected item, the highest `got.scores[fact_uuid]` over retrieved facts (`got.results + got.limitations`) whose text matches the item via `match_items(fact_text, [item])`; `None` when no retrieved fact matches or none is scored. The printed per-question line appends `min_item_score=<lowest non-None score, 3 decimals>`. It also adds `"item_grades": {item: grade}` — a simulation through the production grader: for each expected item, take up to 3 retrieved facts matching it (best score first), build `marker_map = {1: f1, 2: f2, ...}` and `raw = f"- **{item['item']}** — x [1][2][3]"`, and call `item_grades.grade_items(raw, marker_map, q=question, terms=list_sweep.question_terms(question, scope), scores=got.scores, floor=settings.list_item_floor)`; record the one grade (`"unretrieved"` when no fact matches). The printed line appends `flagged=<items graded needs_validation>` and `would_drop=<items graded dropped>` by name, so the calibration sees exactly which expected items the grader would flag or drop (e.g. naming mismatches such as "AWS" vs "Amazon Web Services"). Hermetic tests (monkeypatch `list_mode.retrieve_list` to return a `ListRetrieval` with scores): item scores pick the best matching fact; an unmatched item is `None`; a raising `retrieve_list` still returns FAIL with the error (closes the Task 5 deferred minor).

- [ ] **Step 1:** Tests; fail. **Step 2:** Implement. **Step 3:** Unit suite, ruff, mypy. **Step 4:** Commit — `feat(eval): list gate reports each expected item's best rerank score`.

**Controller after Task 11:** run the free gate; set `list_item_floor` just below the lowest expected-item score (ledger ruling with the numbers), keeping it above the score of clearly irrelevant facts; record coverage (target 67/67) and per-question latency.

---

### Task 12: Calibration adjustments (spec §12.4)

**Files:** `src/answer_api/list_sweep.py`, `src/answer_api/item_grades.py`, `src/answer_api/list_mode.py` (prompt), `src/graph_extract/config.py`, `src/answer_api/router_golden_tier1.json`; tests `tests/unit/test_list_sweep.py`, `tests/unit/test_item_grades.py`, `tests/unit/test_list_mode.py`, `tests/unit/test_config.py`.

1. **Inflected exact forms.** Add `list_sweep.word_forms(term) -> list[str]`: the term itself; its plural (`term + "s"`, or `term + "es"` when it ends in s/x/z/ch/sh — but when the term already ends in "s" and is longer than 4 characters, its singular `term[:-1]` instead); and when `stem(term) != term`, also `stem + "ed"` and `stem + "ing"` (`restore` -> restore, restores, restored, restoring). Deduplicate, keep order. `lucene_query` emits, per term without '-', `word_forms(term)` followed by the existing `stem_prefixes` wildcards; hyphenated terms unchanged. Update the expected strings in `test_lucene_query_*` and the scoped-sweep test (`p["query"]`) to the new output; add `word_forms` tests (restore; agent -> agent, agents; windows -> windows, window; backup -> backup, backups; box -> box, boxes).
2. **Two floors.** Settings: `list_item_floor: float = 0.4` (was 0.1; comment: named support; calibrated 2026-10-16: expected items' best scores >= 0.689, off-topic named facts 0.33–0.48) and new `list_item_unnamed_floor: float = 0.65` (comment: a relevant fact that does not name the item flags it instead of dropping it). Grading: supporting facts are non-limitation cited facts; NAMED support = names the item and (score None or >= `list_item_floor`) → confirmed / needs-validation-missing-terms exactly as today. If there is no named support but some non-limitation cited fact has a score >= `list_item_unnamed_floor` → NEEDS_VALIDATION with `missing_terms=[]` and the fixed note ` *(needs validation: the cited source does not name this item)*`. Otherwise DROPPED (unscored unnamed facts do NOT keep an item). `grade_items` gains the keyword `unnamed_floor: float`; `answer_list` and `eval_retrieval.check_list_items` pass `settings.list_item_unnamed_floor`. `ItemGrade` gains `reason: str` — one of `"named"`, `"missing_terms"`, `"unnamed"`, `"below_floor"`, `"no_marker"` — included in `as_dict`. Tests: unnamed fact at 0.7 → needs validation with the unnamed note; at 0.5 → dropped; unnamed unscored → dropped; named at 0.45 with floor 0.4 → kept; named at 0.35 → falls to the unnamed rule (dropped at 0.35).
3. **Prompt.** In `_LIST_PROMPT`, after the item-form sentence add: `"Name each item the way the facts name it (the product, platform or feature the fact names), not by a broader category. "`; the prompt test asserts this sentence.
4. **Golden alias.** In `router_golden_tier1.json`, the expected item `Publish disks` gains the alias `"publishes"` (the retrieved fact says "publishes disks"; `match_items` is whole-word).

- [ ] Tests first; implement; full unit suite, ruff src tests, mypy src green; commit `fix(answer): list calibration -- inflected sweep terms, unnamed-but-relevant flagged` (trailer lines).

---

### Task 13: Compact embedding transfer (latency, 2026-10-16)

**Measured (free gate after the fix wave):** the collapse stage costs 44–71 s per swept list question. The cause is the transfer of `fact_embedding`: Neo4j reads 1000 vectors in 0.6 s, but sending them as float lists takes 20 s from the answer-api pod (RTT ~0.4 s; this machine 179 ms), and the cost scales with bytes. The neo4j-rust-ext driver extension does not help. Quantized in Cypher to `vector(..., INTEGER8)`, the same 1000 take 3.3 s. Accuracy on 44,850 real pairs: mean |Δcos| 0.0008, at most 0.0018 for pairs above 0.9, and 1 flip at 0.96 / 0 at 0.92.

**Files:** `src/answer_api/fact_dedup.py` (and any type hints that consume the vectors in `src/answer_api/list_mode.py`); tests `tests/unit/test_fact_dedup.py`, `tests/unit/test_list_mode.py`, plus an integration test in `tests/integration/test_list_siblings.py`.

1. `EMBEDDINGS_QUERY` becomes a per-vector max-abs-scaled INTEGER8 vector (cosine is scale-invariant, so per-vector scaling is model-agnostic):
   `MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids WITH f, f.fact_embedding AS e WHERE e IS NOT NULL WITH f, e, reduce(m = 0.0, x IN e | CASE WHEN abs(x) > m THEN abs(x) ELSE m END) AS mx RETURN f.uuid AS uuid, CASE WHEN mx = 0 THEN null ELSE vector([x IN e | toInteger(round(x * 127.0 / mx))], size(e), INTEGER8) END AS embedding`.
   Keep the old float query as `EMBEDDINGS_QUERY_FLOAT`.
2. `fetch_vectors` decodes the returned `neo4j.vector.Vector` to a float numpy array (use the driver's numpy/native conversion; check the installed neo4j 6.2 API). Rows whose value is still a plain list (the float query) are converted the same way. The cache holds `np.ndarray` (float32 is fine); update the type hints (`dict[str, np.ndarray]`) and any consumer (`_collapse`, `select_diverse`, `_fact_vectors`) so arrays work. `select_diverse` already builds a matrix, so make sure it accepts arrays without per-element Python loops.
3. **Fallback:** if the INTEGER8 query raises a Neo4j client/syntax error (servers without `vector()` or INTEGER8, e.g. the Neo4j 5.x testcontainer), log WARNING once per process ("compact embedding transfer unavailable (%s); fetching float vectors") and use `EMBEDDINGS_QUERY_FLOAT` for the rest of the process (a module-level flag). Other exceptions propagate as today; the existing degraded handling stays.
4. Tests:
   - Unit: a fake driver returning a Vector-like value decodes to the right array, and the collapse result on decoded vectors matches the float result.
   - Unit: the fallback switches once and logs once.
   - Unit: no per-question behaviour changes apart from the type (the existing tests stay green).
   - Integration: whichever Neo4j image the fixture runs, `fetch_vectors` returns usable vectors for inserted edges. Assert the cosine of two inserted vectors is within 0.01 of the float cosine.
- [ ] Tests first; implement; unit suite + `tests/integration/test_list_siblings.py`, ruff src tests, mypy src green; commit `perf(answer): fetch fact embeddings as int8 vectors` (trailer lines).
