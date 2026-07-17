# Theme-Builder Community Layer (Slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the theme-builder batch service — GDS hierarchical Leiden community detection over the `backup-docs` entity graph + GLM-5.2 MS-GraphRAG-style community reports (per-finding fact-ID citations, validated) + write-back of a `:Community` subgraph.

**Architecture:** A new `theme_builder` package with focused modules (`detect`, `context`, `report`, `writeback`) orchestrated by a `theme-build` CLI. Full rebuild: detect → per-community assemble context + generate a fact-cited report → delete-and-rewrite the group's `:Community` subgraph. Reuses the Neo4j driver, the shared TEI/Jina embedder, and the synthesis (GLM-5.2) LLM config.

**Tech Stack:** Python 3.12, Neo4j 5.26 + GDS 2.13 (`gds.leiden`), graphiti-core embedder, OpenAI-compatible LLM client (GLM-5.2 via OpenRouter/Ollama judge config), typer CLI, pytest (+ Neo4j testcontainer; GDS-dependent detection is `@live`).

## Global Constraints

- **Design decision #2 — citations are graph traversal, never LLM generation:** the report LLM emits fact UUIDs; a deterministic validator keeps only UUIDs present in the community's real fact set; the LLM must never write a URL (prompt + `_URL_RE` strip). `cited_fact_uuids` are all real.
- **Design decision #4 — one group / one embedding space:** communities are per `group_id`; embed `title+summary` with the SAME shared embedder (TEI/Jina, `embed_model`/`embed_dim` from settings) used for entities/facts/queries.
- **Design decision #5 — Graphiti schema is library-owned:** `:Community` is our label linked via `IN_COMMUNITY`/`PARENT_OF`; never merge into or rename Graphiti nodes.
- **Derived & disposable:** a rebuild `DETACH DELETE`s the group's `:Community` subgraph first; the theme-builder is the only writer.
- Report model defaults to the synthesis/judge tier (GLM-5.2) and is overridable via `report_llm_*` settings.
- Community id is deterministic: `sha1(f"{level}:" + ",".join(sorted(member_uuids)))[:16]`.
- Run everything via `uv` (`uv run --extra dev pytest ...`, `uv run ruff check ...`, `uv run mypy ...`). Tests construct settings hermetically with `ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")`.

---

## File Structure

- **Create** `src/theme_builder/__init__.py` (empty).
- **Create** `src/theme_builder/detect.py` — `Community` dataclass, `_community_id`, `_build_communities_from_rows` (pure), `detect_communities` (GDS I/O).
- **Create** `src/theme_builder/context.py` — `EntityRow`/`FactRow`/`ContextResult` dataclasses, `assemble_context` (pure).
- **Create** `src/theme_builder/report.py` — `CommunityReport` dataclass, `_report_client_and_model`, `_extract_json`, `generate_report` (+ validation).
- **Create** `src/theme_builder/writeback.py` — `write_communities`.
- **Create** `src/theme_builder/cli.py` — `theme-build` typer command + orchestration (member/fact fetch).
- **Modify** `src/graph_extract/config.py` — config fields (Task 1).
- **Modify** `src/graph_extract/graphiti_client.py` — `build_embedder(s)` helper (Task 5).
- **Tests:** `tests/unit/test_theme_context.py`, `tests/unit/test_theme_report.py`, `tests/unit/test_theme_detect.py`, `tests/unit/test_theme_config.py`, `tests/integration/test_theme_writeback.py`, `tests/integration/test_theme_cli.py`, plus an `@live` smoke in `tests/integration/test_theme_detect_live.py`.

---

### Task 1: Config fields for the theme-builder

**Files:**
- Modify: `src/graph_extract/config.py`
- Test: `tests/unit/test_theme_config.py`

**Interfaces:**
- Produces on `ExtractSettings`: `report_llm_base_url: str`, `report_llm_model: str`, `report_llm_api_key: str`, `leiden_min_community_size: int`, `leiden_max_levels: int`, `report_token_budget: int`, `report_top_entities: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_theme_config.py
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def test_theme_defaults():
    s = ExtractSettings(_env_file=None, **_MIN)
    assert s.report_llm_base_url == "" and s.report_llm_model == "" and s.report_llm_api_key == ""
    assert s.leiden_min_community_size == 3
    assert s.leiden_max_levels == 3
    assert s.report_token_budget == 12000
    assert s.report_top_entities == 30
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_theme_config.py -q`
Expected: FAIL (`report_llm_base_url` etc. do not exist).

- [ ] **Step 3: Implement**

In `src/graph_extract/config.py`, inside `class ExtractSettings`, after the `judge_api_key` field, add:

```python
    # --- theme-builder / community layer (design: theme-builder-community-layer) ---
    # Report tier defaults to the synthesis/judge tier (GLM-5.2) when left empty.
    report_llm_base_url: str = ""
    report_llm_model: str = ""
    report_llm_api_key: str = ""
    leiden_min_community_size: int = 3   # drop dust communities smaller than this
    leiden_max_levels: int = 3           # cap on intermediate Leiden levels
    report_token_budget: int = 12000     # per-community context budget (~chars/4)
    report_top_entities: int = 30        # member entities included in a report's context
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_theme_config.py -q`
Expected: PASS. Then `uv run ruff check src/graph_extract/config.py` and `uv run mypy src/graph_extract/config.py` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/config.py tests/unit/test_theme_config.py
git commit -m "feat(config): theme-builder report/leiden settings"
```

---

### Task 2: `context.py` — report context assembly (pure)

**Files:**
- Create: `src/theme_builder/__init__.py` (empty), `src/theme_builder/context.py`
- Test: `tests/unit/test_theme_context.py`

**Interfaces:**
- Produces:
  - `@dataclass EntityRow: uuid: str; name: str; type: str; summary: str; degree: int`
  - `@dataclass FactRow: uuid: str; fact: str; valid_at: str | None; invalid_at: str | None; name: str`
  - `@dataclass ContextResult: text: str; fact_uuids: set[str]`
  - `assemble_context(members: list[EntityRow], facts: list[FactRow], *, top_entities: int, token_budget: int) -> ContextResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_theme_context.py
from theme_builder.context import EntityRow, FactRow, assemble_context


def _ent(u, deg): return EntityRow(uuid=u, name=u, type="Concept", summary="s", degree=deg)
def _fact(u, valid, invalid=None): return FactRow(uuid=u, fact="F"+u, valid_at=valid, invalid_at=invalid, name="Provides")


def test_members_ranked_by_degree_and_capped():
    members = [_ent("aaa", 1), _ent("bbb", 9), _ent("ccc", 5)]
    res = assemble_context(members, [], top_entities=2, token_budget=100000)
    # highest-degree first, capped at 2 -> bbb then ccc ; aaa excluded
    assert res.text.index("- bbb ") < res.text.index("- ccc ")
    assert "- aaa " not in res.text            # lowest degree dropped by the cap
    assert res.text.count("(Concept)") == 2    # only 2 members rendered


def test_facts_current_first_then_recency_and_labelled():
    facts = [_fact("f1", "2020", invalid="2021"), _fact("f2", "2019"), _fact("f3", "2022")]
    res = assemble_context([], facts, top_entities=30, token_budget=100000)
    # current (invalid_at is None) first, recency desc: f3(2022) then f2(2019) then superseded f1
    order = [res.text.index("[f3]"), res.text.index("[f2]"), res.text.index("[f1]")]
    assert order == sorted(order)
    assert res.fact_uuids == {"f1", "f2", "f3"}
    assert "[f1] Ff1" in res.text


def test_token_budget_truncates_facts():
    facts = [_fact(f"f{i}", "2020") for i in range(200)]
    res = assemble_context([], facts, top_entities=30, token_budget=200)  # ~800 chars
    assert len(res.fact_uuids) < 200          # truncated
    assert len(res.text) <= 200 * 4 + 200     # roughly within budget (chars ~= 4*tokens)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_theme_context.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/theme_builder/__init__.py   (empty file)
```

```python
# src/theme_builder/context.py
"""Pure per-community report-context assembly. No I/O — the caller fetches the
member/fact rows from Neo4j; this ranks, orders, budgets, and labels them so the
report LLM can cite fact UUIDs. Kept pure so it is unit-testable without a DB."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntityRow:
    uuid: str
    name: str
    type: str
    summary: str
    degree: int


@dataclass
class FactRow:
    uuid: str
    fact: str
    valid_at: str | None
    invalid_at: str | None
    name: str


@dataclass
class ContextResult:
    text: str
    fact_uuids: set[str]


def assemble_context(members: list[EntityRow], facts: list[FactRow], *,
                     top_entities: int, token_budget: int) -> ContextResult:
    """Render a community's context: top-degree members, then intra-community
    facts current-first / recency-desc, each labelled `[uuid]`, truncated to a
    ~token_budget (approximated as budget*4 chars). Returns the text plus the set
    of fact UUIDs actually included (the citable universe)."""
    lines: list[str] = ["ENTITIES:"]
    for e in sorted(members, key=lambda e: e.degree, reverse=True)[:top_entities]:
        lines.append(f"- {e.name} ({e.type}): {e.summary}")
    lines.append("\nFACTS:")
    char_budget = token_budget * 4
    used = len("\n".join(lines))
    included: set[str] = set()
    # current facts (invalid_at is None) first, then superseded; each group by
    # valid_at descending (recency). reverse=True puts empty/None valid_at last.
    current = [f for f in facts if f.invalid_at is None]
    superseded = [f for f in facts if f.invalid_at is not None]
    current.sort(key=lambda f: f.valid_at or "", reverse=True)
    superseded.sort(key=lambda f: f.valid_at or "", reverse=True)
    for f in current + superseded:
        suffix = (f" (valid {f.valid_at}" + (f", invalid {f.invalid_at})" if f.invalid_at else ")")) \
            if f.valid_at else ""
        line = f"[{f.uuid}] {f.fact}{suffix}"
        if used + len(line) + 1 > char_budget and included:
            break
        lines.append(line)
        included.add(f.uuid)
        used += len(line) + 1
    return ContextResult(text="\n".join(lines), fact_uuids=included)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_theme_context.py -q`
Expected: PASS (3 tests). `ruff` + `mypy src/theme_builder/context.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/__init__.py src/theme_builder/context.py tests/unit/test_theme_context.py
git commit -m "feat(theme): pure per-community report-context assembly"
```

---

### Task 3: `report.py` — LLM report generation + citation validation

**Files:**
- Create: `src/theme_builder/report.py`
- Test: `tests/unit/test_theme_report.py`

**Interfaces:**
- Consumes: `ContextResult` (Task 2), `ExtractSettings`.
- Produces:
  - `@dataclass CommunityReport: title: str; summary: str; full_report: str; rating: float; rating_explanation: str; tags: list[str]; cited_fact_uuids: list[str]`
  - `_report_client_and_model(settings) -> tuple[AsyncOpenAI, str]`
  - `async generate_report(client, model, context: ContextResult) -> CommunityReport | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_theme_report.py
import json
import pytest
from theme_builder.context import ContextResult
from theme_builder.report import generate_report, _report_client_and_model
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


class _FakeClient:
    def __init__(self, contents):
        self._contents = list(contents)
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        c = self._contents.pop(0)
        class _R:
            choices = [type("m", (), {"message": type("mm", (), {"content": c})()})()]
        return _R()


def _ctx():
    return ContextResult(text="FACTS:\n[u1] a\n[u2] b", fact_uuids={"u1", "u2"})


@pytest.mark.asyncio
async def test_valid_report_keeps_only_real_fact_ids():
    payload = json.dumps({"title": "T", "summary": "S",
        "full_report": [{"finding": "f1", "fact_ids": ["u1", "u9-hallucinated"]},
                        {"finding": "f2", "fact_ids": ["u2"]}],
        "rating": 7, "rating_explanation": "why", "tags": ["AWS"]})
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert rep is not None
    assert set(rep.cited_fact_uuids) == {"u1", "u2"}      # u9 dropped
    assert rep.rating == 7.0 and rep.tags == ["AWS"]


@pytest.mark.asyncio
async def test_url_stripped_from_text():
    payload = json.dumps({"title": "T https://evil/x", "summary": "S http://e/y",
        "full_report": [{"finding": "see http://z", "fact_ids": ["u1"]}],
        "rating": 5, "rating_explanation": "", "tags": []})
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert "http" not in rep.title and "http" not in rep.summary and "http" not in rep.full_report


@pytest.mark.asyncio
async def test_bad_json_retries_then_none():
    rep = await generate_report(_FakeClient(["not json", "still not json"]), "m", _ctx())
    assert rep is None                                    # one retry, then give up


@pytest.mark.asyncio
async def test_fenced_json_is_parsed():
    payload = "```json\n" + json.dumps({"title": "T", "summary": "S",
        "full_report": [{"finding": "f", "fact_ids": ["u1"]}], "rating": 3,
        "rating_explanation": "", "tags": []}) + "\n```"
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert rep is not None and rep.cited_fact_uuids == ["u1"]


def test_report_tier_defaults_to_judge():
    s = ExtractSettings(_env_file=None, judge_base_url="http://glm", judge_model="glm-5.2:cloud",
                        judge_api_key="jk", **_MIN)
    client, model = _report_client_and_model(s)
    assert model == "glm-5.2:cloud"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_theme_report.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/theme_builder/report.py
"""Community report generation (strong/synthesis LLM) + deterministic citation
validation. The model references fact UUIDs per finding; we keep only the UUIDs
that are real (present in the community context) and strip any URL — design
decision #2 (the LLM never authors a citation URL)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument
from theme_builder.context import ContextResult

# A URL must never survive into a report (design decision #2). Case-insensitive,
# stops at brackets so it can't swallow an adjacent token.
_URL_RE = re.compile(r"https?://[^\s\[\]]+", re.IGNORECASE)

_PROMPT = (
    "You are analyzing a COMMUNITY of related entities from backup-product "
    "documentation. Using ONLY the numbered FACTS below (each tagged with a "
    "[uuid]), write an analytical community report. Cite the supporting fact "
    "[uuid]s for every finding. Do NOT use outside knowledge. Do NOT write any "
    "URL or link. Respond with ONLY a JSON object of this shape:\n"
    '{"title": str, "summary": str (one paragraph), '
    '"full_report": [{"finding": str, "fact_ids": [uuid, ...]}], '
    '"rating": number 0-10 (thematic importance), "rating_explanation": str, '
    '"tags": [str, ...] (workloads/vendors covered)}\n\n'
    "COMMUNITY CONTEXT:\n{context}"
)


@dataclass
class CommunityReport:
    title: str
    summary: str
    full_report: str          # the findings list, serialized to a JSON string
    rating: float
    rating_explanation: str
    tags: list[str]
    cited_fact_uuids: list[str]


def _report_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    """Resolve the report tier. Uses report_llm_* when set, else falls back to
    the synthesis/judge tier (GLM-5.2)."""
    base = settings.report_llm_base_url or settings.judge_base_url
    model = settings.report_llm_model or settings.judge_model
    key = settings.report_llm_api_key or settings.judge_api_key or "not-needed"
    if not base or not model:
        raise ValueError(
            "No report model configured. Set report_llm_* or the judge_* (GLM-5.2) "
            "config in .env.")
    return instrument(AsyncOpenAI(api_key=key, base_url=base)), model


def _extract_json(raw: str) -> dict | None:
    """Parse a JSON object from possibly fenced / chatty model output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _strip(s: str) -> str:
    return _URL_RE.sub("", s or "").strip()


async def generate_report(client: AsyncOpenAI, model: str,
                          context: ContextResult) -> CommunityReport | None:
    """One LLM call (+ one retry on unparseable JSON). Validates fact_ids against
    the community's real fact UUIDs (drops hallucinations) and strips URLs.
    Returns None if the model never produced valid JSON."""
    prompt = _PROMPT.format(context=context.text)
    obj: dict | None = None
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    if obj is None:
        return None
    findings = obj.get("full_report") or []
    cited: list[str] = []
    for f in findings:
        for fid in (f.get("fact_ids") or []):
            if fid in context.fact_uuids and fid not in cited:
                cited.append(fid)
        f["finding"] = _strip(str(f.get("finding", "")))
    try:
        rating = float(obj.get("rating", 0) or 0)
    except (TypeError, ValueError):
        rating = 0.0
    return CommunityReport(
        title=_strip(str(obj.get("title", ""))),
        summary=_strip(str(obj.get("summary", ""))),
        full_report=json.dumps(findings),
        rating=rating,
        rating_explanation=_strip(str(obj.get("rating_explanation", ""))),
        tags=[str(t) for t in (obj.get("tags") or [])],
        cited_fact_uuids=cited)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_theme_report.py -q`
Expected: PASS (5 tests). `ruff` + `mypy src/theme_builder/report.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/report.py tests/unit/test_theme_report.py
git commit -m "feat(theme): community report generation + citation validation"
```

---

### Task 4: `detect.py` — hierarchical community detection

**Files:**
- Create: `src/theme_builder/detect.py`
- Test: `tests/unit/test_theme_detect.py`

**Interfaces:**
- Produces:
  - `@dataclass Community: community_id: str; level: int; member_uuids: list[str]; parent_id: str | None`
  - `_community_id(level: int, member_uuids: list[str]) -> str`
  - `_build_communities_from_rows(rows: list[dict], *, min_community_size: int, max_levels: int) -> list[Community]` where each row is `{"uuid": str, "levels": list[int]}` (finest→coarsest Leiden labels)
  - `async detect_communities(driver, group_id, *, min_community_size, max_levels) -> list[Community]`

- [ ] **Step 1: Write the failing test** (pure logic only — no GDS)

```python
# tests/unit/test_theme_detect.py
from theme_builder.detect import _community_id, _build_communities_from_rows


def test_community_id_deterministic_order_independent():
    a = _community_id(0, ["u2", "u1", "u3"])
    b = _community_id(0, ["u3", "u2", "u1"])
    assert a == b and len(a) == 16
    assert _community_id(1, ["u1", "u2", "u3"]) != a       # level participates


def test_builds_levels_and_drops_dust():
    # 4 entities; level0 labels: {A:0,B:0,C:1,D:1}; level1: all -> 9
    rows = [
        {"uuid": "A", "levels": [0, 9]},
        {"uuid": "B", "levels": [0, 9]},
        {"uuid": "C", "levels": [1, 9]},
        {"uuid": "D", "levels": [1, 9]},
    ]
    comms = _build_communities_from_rows(rows, min_community_size=2, max_levels=3)
    by_level = {}
    for c in comms:
        by_level.setdefault(c.level, []).append(c)
    assert len(by_level[0]) == 2 and len(by_level[1]) == 1   # two leaves, one parent
    leaf = next(c for c in by_level[0] if "A" in c.member_uuids)
    parent = by_level[1][0]
    assert leaf.parent_id == parent.community_id            # nested parent linked
    assert parent.parent_id is None                          # top level


def test_dust_below_min_size_dropped():
    rows = [{"uuid": "A", "levels": [0]}, {"uuid": "B", "levels": [1]}]  # singletons
    comms = _build_communities_from_rows(rows, min_community_size=2, max_levels=3)
    assert comms == []                                       # both dust


def test_max_levels_caps_hierarchy():
    rows = [{"uuid": "A", "levels": [0, 5, 8]}, {"uuid": "B", "levels": [0, 5, 8]}]
    comms = _build_communities_from_rows(rows, min_community_size=1, max_levels=2)
    assert max(c.level for c in comms) == 1                  # levels 0,1 only
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_theme_detect.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/theme_builder/detect.py
"""GDS hierarchical Leiden community detection over the entity graph. The pure
helpers (`_community_id`, `_build_communities_from_rows`) turn Leiden's per-node
level labels into a hierarchy of communities with deterministic ids and parent
links; `detect_communities` is the thin GDS I/O wrapper (covered by an @live
smoke — the standard Neo4j testcontainer has no GDS plugin)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from neo4j import AsyncDriver


@dataclass
class Community:
    community_id: str
    level: int
    member_uuids: list[str]
    parent_id: str | None


def _community_id(level: int, member_uuids: list[str]) -> str:
    h = hashlib.sha1((f"{level}:" + ",".join(sorted(member_uuids))).encode())
    return h.hexdigest()[:16]


def _build_communities_from_rows(rows: list[dict], *, min_community_size: int,
                                 max_levels: int) -> list[Community]:
    """`rows`: [{"uuid": str, "levels": [finest, ..., coarsest]}]. Build one set of
    communities per level (capped at max_levels), drop those below
    min_community_size, and link each community to the level-above community that
    contains its members (nested Leiden levels)."""
    if not rows:
        return []
    n_levels = min(max(len(r["levels"]) for r in rows), max_levels)
    # level -> {leiden_label -> [uuids]}
    members: list[dict[int, list[str]]] = [{} for _ in range(n_levels)]
    for r in rows:
        for lvl in range(min(len(r["levels"]), n_levels)):
            members[lvl].setdefault(r["levels"][lvl], []).append(r["uuid"])
    # assign ids per surviving community; keep leiden_label -> community_id maps
    label_to_id: list[dict[int, str]] = [{} for _ in range(n_levels)]
    surviving: list[dict[int, list[str]]] = [{} for _ in range(n_levels)]
    for lvl in range(n_levels):
        for label, uuids in members[lvl].items():
            if len(uuids) >= min_community_size:
                cid = _community_id(lvl, uuids)
                label_to_id[lvl][label] = cid
                surviving[lvl][label] = uuids
    out: list[Community] = []
    for lvl in range(n_levels):
        for label, uuids in surviving[lvl].items():
            parent_id = None
            if lvl + 1 < n_levels:
                # every member shares the same level-(lvl+1) leiden label (nested)
                parent_label = next(r["levels"][lvl + 1] for r in rows
                                    if r["uuid"] == uuids[0] and len(r["levels"]) > lvl + 1)
                parent_id = label_to_id[lvl + 1].get(parent_label)  # None if dropped
            out.append(Community(community_id=label_to_id[lvl][label], level=lvl,
                                 member_uuids=uuids, parent_id=parent_id))
    return out


_PROJECT = (
    "MATCH (e:Entity {group_id: $g}) RETURN id(e) AS id",
    "MATCH (a:Entity {group_id: $g})-[r:RELATES_TO {group_id: $g}]->(b:Entity {group_id: $g}) "
    "RETURN id(a) AS source, id(b) AS target, count(r) AS weight",
)


async def detect_communities(driver: AsyncDriver, group_id: str, *,
                             min_community_size: int, max_levels: int) -> list[Community]:
    name = f"theme-{group_id}"
    async with driver.session() as s:
        # pre-drop a stale projection of the same name
        await s.run("CALL gds.graph.exists($n) YIELD exists "
                    "WITH exists WHERE exists CALL gds.graph.drop($n) YIELD graphName "
                    "RETURN graphName", n=name)
        try:
            await s.run(
                "CALL gds.graph.project.cypher($n, $nodeq, $relq, {parameters: {g: $g}}) "
                "YIELD graphName RETURN graphName",
                n=name, nodeq=_PROJECT[0], relq=_PROJECT[1], g=group_id)
            r = await s.run(
                "CALL gds.leiden.stream($n, {relationshipWeightProperty: 'weight', "
                "includeIntermediateCommunities: true, undirectedRelationshipTypes: ['*']}) "
                "YIELD nodeId, intermediateCommunityIds "
                "RETURN gds.util.asNode(nodeId).uuid AS uuid, intermediateCommunityIds AS levels",
                n=name)
            rows = [{"uuid": rec["uuid"], "levels": list(rec["levels"])} async for rec in r]
        finally:
            await s.run("CALL gds.graph.exists($n) YIELD exists "
                        "WITH exists WHERE exists CALL gds.graph.drop($n) YIELD graphName "
                        "RETURN graphName", n=name)
    return _build_communities_from_rows(rows, min_community_size=min_community_size,
                                        max_levels=max_levels)
```

*(GDS note for the implementer: the exact `gds.leiden.stream` config is verified by the `@live` smoke in Task 7 against GDS 2.13. If `undirectedRelationshipTypes: ['*']` is rejected, the working alternative confirmed at that step is to project the relationship undirected; adjust `detect_communities` there — the pure helpers and their tests are unaffected.)*

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_theme_detect.py -q`
Expected: PASS (4 tests). `ruff` + `mypy src/theme_builder/detect.py` clean. (The GDS path is not exercised by unit tests — that's Task 7's `@live` smoke.)

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/detect.py tests/unit/test_theme_detect.py
git commit -m "feat(theme): hierarchical Leiden community detection"
```

---

### Task 5: `build_embedder` helper + `writeback.py`

**Files:**
- Modify: `src/graph_extract/graphiti_client.py`
- Create: `src/theme_builder/writeback.py`
- Test: `tests/integration/test_theme_writeback.py`

**Interfaces:**
- Consumes: `Community` (Task 4), `CommunityReport` (Task 3), an embedder with `async create(input_data: list[str]) -> list[list[float]]`.
- Produces:
  - in `graphiti_client.py`: `build_embedder(s: ExtractSettings) -> OpenAIEmbedder`
  - `async write_communities(driver, embedder, group_id, communities: list[Community], reports: dict[str, CommunityReport], corpus_cursor: str | None) -> dict`

- [ ] **Step 1: Write the failing test** (Neo4j testcontainer, fake embedder — no GDS, no live embed endpoint)

```python
# tests/integration/test_theme_writeback.py
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _FakeEmbedder:
    async def create_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


async def test_writeback_creates_community_subgraph(extract_driver):
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    from theme_builder.writeback import write_communities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {uuid:'e1', group_id:$g, name:'AWS Backup'})", g=g)
        await s.run("CREATE (:Entity {uuid:'e2', group_id:$g, name:'Amazon S3'})", g=g)
    comms = [
        Community("child1", 0, ["e1", "e2"], parent_id="par1"),
        Community("par1", 1, ["e1", "e2"], parent_id=None),
    ]
    reps = {
        "child1": CommunityReport("Leaf", "leaf sum", "[]", 6.0, "why", ["AWS"], ["f1"]),
        "par1": CommunityReport("Top", "top sum", "[]", 8.0, "why2", ["AWS"], ["f1", "f2"]),
    }
    res = await write_communities(extract_driver, _FakeEmbedder(), g, comms, reps, corpus_cursor="cur-1")
    assert res["reports_written"] == 2
    async with extract_driver.session() as s:
        n = (await (await s.run("MATCH (c:Community {group_id:$g}) RETURN count(c) AS c", g=g)).single())["c"]
        assert n == 2
        pf = (await (await s.run(
            "MATCH (:Community {community_id:'par1'})-[:PARENT_OF]->(:Community {community_id:'child1'}) "
            "RETURN count(*) AS c")).single())["c"]
        assert pf == 1
        im = (await (await s.run(
            "MATCH (:Entity {uuid:'e1'})-[:IN_COMMUNITY]->(:Community {community_id:'child1'}) "
            "RETURN count(*) AS c")).single())["c"]
        assert im == 1
        emb = (await (await s.run(
            "MATCH (c:Community {community_id:'par1'}) RETURN c.embedding AS e, c.corpus_cursor AS cur")).single())
        assert emb["e"] == [0.1, 0.2, 0.3] and emb["cur"] == "cur-1"


async def test_writeback_is_full_rebuild(extract_driver):
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    from theme_builder.writeback import write_communities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {uuid:'x1', group_id:$g, name:'X'})", g=g)
        await s.run("CREATE (:Community {community_id:'stale', group_id:$g, level:0})", g=g)
    comms = [Community("fresh", 0, ["x1"], None)]
    reps = {"fresh": CommunityReport("F", "s", "[]", 1.0, "", [], [])}
    await write_communities(extract_driver, _FakeEmbedder(), g, comms, reps, corpus_cursor=None)
    async with extract_driver.session() as s:
        ids = [r["id"] async for r in await s.run(
            "MATCH (c:Community {group_id:$g}) RETURN c.community_id AS id", g=g)]
    assert ids == ["fresh"]                    # stale deleted, only fresh remains
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_theme_writeback.py -q`
Expected: FAIL (`build_embedder`/`write_communities` missing).

- [ ] **Step 3: Implement `build_embedder`**

In `src/graph_extract/graphiti_client.py`, add (near `build_graphiti`; the imports `OpenAIEmbedder`, `OpenAIEmbedderConfig`, `_batch_capped_embeddings`, `AsyncOpenAI` already exist):

```python
def build_embedder(s: ExtractSettings) -> OpenAIEmbedder:
    """The single shared embedder (TEI/Jina) used everywhere — extracted so the
    theme-builder can embed community title+summary in the same space (design
    decision #4)."""
    client = _batch_capped_embeddings(
        AsyncOpenAI(api_key="not-needed", base_url=s.embed_base_url,
                    timeout=90.0, max_retries=4), s.embed_max_batch)
    return OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key="not-needed", embedding_model=s.embed_model,
        embedding_dim=s.embed_dim, base_url=s.embed_base_url), client=client)
```

- [ ] **Step 4: Implement `writeback.py`**

```python
# src/theme_builder/writeback.py
"""Persist the :Community subgraph (full rebuild: delete then write). Derived &
disposable — the theme-builder is the only writer of :Community."""
from __future__ import annotations

from neo4j import AsyncDriver

from theme_builder.detect import Community
from theme_builder.report import CommunityReport


async def write_communities(driver: AsyncDriver, embedder, group_id: str,
                            communities: list[Community],
                            reports: dict[str, CommunityReport],
                            corpus_cursor: str | None) -> dict:
    written = [c for c in communities if c.community_id in reports]
    texts = [f"{reports[c.community_id].title}\n{reports[c.community_id].summary}" for c in written]
    # graphiti's OpenAIEmbedder.create returns ONE vector; create_batch returns
    # one per input (list[list[float]]) — use it for the batch of community texts.
    embeddings = await embedder.create_batch(texts) if texts else []
    by_level: dict[int, int] = {}
    async with driver.session() as s:
        await s.run("MATCH (c:Community {group_id:$g}) DETACH DELETE c", g=group_id)
        for c, emb in zip(written, embeddings):
            r = reports[c.community_id]
            by_level[c.level] = by_level.get(c.level, 0) + 1
            await s.run(
                "MERGE (c:Community {community_id:$cid, group_id:$g}) "
                "SET c += {level:$level, title:$title, summary:$summary, "
                "full_report:$full_report, rating:$rating, rating_explanation:$re, "
                "tags:$tags, cited_fact_uuids:$cited, embedding:$emb, "
                "member_count:$mc, generated_at:datetime(), corpus_cursor:$cur}",
                cid=c.community_id, g=group_id, level=c.level, title=r.title,
                summary=r.summary, full_report=r.full_report, rating=r.rating,
                re=r.rating_explanation, tags=r.tags, cited=r.cited_fact_uuids,
                emb=emb, mc=len(c.member_uuids), cur=corpus_cursor)
            await s.run(
                "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                "UNWIND $members AS mu MATCH (e:Entity {uuid:mu, group_id:$g}) "
                "MERGE (e)-[:IN_COMMUNITY]->(c)",
                cid=c.community_id, g=group_id, members=c.member_uuids)
        # parent edges after all community nodes exist
        for c in written:
            if c.parent_id and c.parent_id in reports:
                await s.run(
                    "MATCH (p:Community {community_id:$pid, group_id:$g}), "
                    "(c:Community {community_id:$cid, group_id:$g}) "
                    "MERGE (p)-[:PARENT_OF]->(c)",
                    pid=c.parent_id, cid=c.community_id, g=group_id)
    return {"reports_written": len(written), "by_level": by_level,
            "facts_cited": sum(len(reports[c.community_id].cited_fact_uuids) for c in written)}
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run --extra dev pytest tests/integration/test_theme_writeback.py -q`
Expected: PASS (2 tests). `ruff` + `mypy src/graph_extract/graphiti_client.py src/theme_builder/writeback.py` clean.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/graphiti_client.py src/theme_builder/writeback.py tests/integration/test_theme_writeback.py
git commit -m "feat(theme): shared build_embedder + :Community writeback (full rebuild)"
```

---

### Task 6: `cli.py` — `theme-build` orchestration

**Files:**
- Create: `src/theme_builder/cli.py`
- Test: `tests/integration/test_theme_cli.py`

**Interfaces:**
- Consumes: `detect_communities`, `assemble_context`/`EntityRow`/`FactRow`, `generate_report`/`_report_client_and_model`, `write_communities`, `build_embedder`, `_build_driver` pattern.
- Produces: `async _run_theme_build(settings) -> dict` (the testable core) and a `theme-build` typer command wrapping it. `_fetch_members`/`_fetch_facts` Cypher helpers.

- [ ] **Step 1: Write the failing test** (testcontainer + monkeypatched detect & report — no GDS, no LLM)

```python
# tests/integration/test_theme_cli.py
import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_run_theme_build_end_to_end(extract_driver, monkeypatch):
    import theme_builder.cli as tc
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    from graph_extract.config import get_extract_settings
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {uuid:'e1', group_id:$g, name:'AWS Backup', summary:'svc'})", g=g)
        await s.run("CREATE (:Entity {uuid:'e2', group_id:$g, name:'Amazon S3', summary:'store'})", g=g)
        await s.run("MATCH (a:Entity {uuid:'e1'}),(b:Entity {uuid:'e2'}) "
                    "CREATE (a)-[:RELATES_TO {group_id:$g, uuid:'f1', fact:'AWS Backup supports Amazon S3', name:'Supports'}]->(b)", g=g)

    async def _fake_detect(driver, group_id, **k):
        return [Community("c1", 0, ["e1", "e2"], None)]
    monkeypatch.setattr(tc, "detect_communities", _fake_detect)

    captured = {}
    async def _fake_report(client, model, context):
        captured["ctx"] = context
        return CommunityReport("T", "S", "[]", 5.0, "", ["AWS"], list(context.fact_uuids))
    monkeypatch.setattr(tc, "generate_report", _fake_report)

    class _Closeable:               # _run_theme_build closes the report client
        async def close(self): pass
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "m"))

    class _FakeEmb:
        async def create_batch(self, texts): return [[0.0] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    s = get_extract_settings.__wrapped__()
    s = s.model_copy(update=dict(group_id=g))
    res = await tc._run_theme_build(s, driver=extract_driver)
    assert res["reports_written"] == 1
    assert "f1" in captured["ctx"].fact_uuids            # the real fact reached the context
    async with extract_driver.session() as s2:
        c = (await (await s2.run(
            "MATCH (:Entity {uuid:'e1'})-[:IN_COMMUNITY]->(c:Community {community_id:'c1'}) "
            "RETURN c.cited_fact_uuids AS cited")).single())
        assert c["cited"] == ["f1"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_theme_cli.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/theme_builder/cli.py
"""theme-build: detect communities -> per community assemble context + generate a
fact-cited report -> write the :Community subgraph. Full rebuild."""
from __future__ import annotations

import asyncio
import json
import logging

import typer
from neo4j import AsyncDriver, AsyncGraphDatabase

from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.graphiti_client import build_embedder
from theme_builder.context import EntityRow, FactRow, assemble_context
from theme_builder.detect import detect_communities
from theme_builder.report import _report_client_and_model, generate_report
from theme_builder.writeback import write_communities

app = typer.Typer()
logger = logging.getLogger(__name__)


async def _fetch_members(driver: AsyncDriver, group_id: str, uuids: list[str]) -> list[EntityRow]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) WHERE e.uuid IN $uuids "
            "OPTIONAL MATCH (e)-[r:RELATES_TO {group_id:$g}]-(m:Entity) WHERE m.uuid IN $uuids "
            "WITH e, count(r) AS degree "
            "RETURN e.uuid AS uuid, e.name AS name, "
            "coalesce([l IN labels(e) WHERE l<>'Entity'][0],'Entity') AS type, "
            "coalesce(e.summary,'') AS summary, degree",
            g=group_id, uuids=uuids)
        return [EntityRow(uuid=x["uuid"], name=x["name"], type=x["type"],
                          summary=x["summary"], degree=x["degree"]) async for x in r]


async def _fetch_facts(driver: AsyncDriver, group_id: str, uuids: list[str]) -> list[FactRow]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->(b:Entity {group_id:$g}) "
            "WHERE a.uuid IN $uuids AND b.uuid IN $uuids "
            "RETURN f.uuid AS uuid, f.fact AS fact, toString(f.valid_at) AS valid_at, "
            "toString(f.invalid_at) AS invalid_at, f.name AS name",
            g=group_id, uuids=uuids)
        return [FactRow(uuid=x["uuid"], fact=x["fact"], valid_at=x["valid_at"],
                        invalid_at=x["invalid_at"], name=x["name"]) async for x in r]


async def _run_theme_build(settings: ExtractSettings, *, driver: AsyncDriver) -> dict:
    communities = await detect_communities(
        driver, settings.group_id,
        min_community_size=settings.leiden_min_community_size,
        max_levels=settings.leiden_max_levels)
    client, model = _report_client_and_model(settings)
    embedder = build_embedder(settings)
    reports: dict = {}
    skipped = 0
    try:
        for c in communities:
            members = await _fetch_members(driver, settings.group_id, c.member_uuids)
            facts = await _fetch_facts(driver, settings.group_id, c.member_uuids)
            ctx = assemble_context(members, facts,
                                   top_entities=settings.report_top_entities,
                                   token_budget=settings.report_token_budget)
            rep = await generate_report(client, model, ctx)
            if rep is None:
                skipped += 1
                continue
            reports[c.community_id] = rep
    finally:
        await client.close()
    res = await write_communities(driver, embedder, settings.group_id,
                                  communities, reports, corpus_cursor=None)
    res["communities_detected"] = len(communities)
    res["reports_skipped"] = skipped
    return res


@app.command("theme-build")
def theme_build() -> None:
    """Full rebuild of the community-report layer for the configured group."""
    async def _main() -> None:
        settings = get_extract_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            res = await _run_theme_build(settings, driver=driver)
            typer.echo(json.dumps(res, indent=2, default=str))
        finally:
            await driver.close()

    asyncio.run(_main())


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/integration/test_theme_cli.py -q`
Expected: PASS. `ruff` + `mypy src/theme_builder/cli.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/cli.py tests/integration/test_theme_cli.py
git commit -m "feat(theme): theme-build CLI orchestration"
```

---

### Task 7: Full-suite gate + `@live` detection smoke + demonstration

**Files:**
- Create: `tests/integration/test_theme_detect_live.py`
- Create: `docs/superpowers/theme-builder-report.md` (demonstration)

- [ ] **Step 1: Full non-live suite**

Run: `uv run --extra dev pytest -m "not live" -q`
Expected: PASS (all green). Fix any fallout before continuing.

- [ ] **Step 2: `@live` detection smoke** (real GDS-enabled Neo4j)

```python
# tests/integration/test_theme_detect_live.py
import pytest

pytestmark = pytest.mark.asyncio


@pytest.mark.live
async def test_detect_communities_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from theme_builder.detect import detect_communities
    s = get_extract_settings()
    comms = await detect_communities(live_extract_driver, s.group_id,
                                     min_community_size=3, max_levels=3)
    assert comms, "expected >=1 community from the live GDS Leiden run"
    assert all(len(c.member_uuids) >= 3 for c in comms)
    # hierarchy present: at least one community has a parent, parents exist
    ids = {c.community_id for c in comms}
    assert all(c.parent_id in ids for c in comms if c.parent_id is not None)
```

Run: `uv run --extra dev pytest -m live tests/integration/test_theme_detect_live.py -q`
Expected: PASS. **If `undirectedRelationshipTypes: ['*']` errors here, this is the point to fix `detect_communities`'s Leiden/projection call against real GDS 2.13** (the pure helpers/tests are unaffected); re-run until green.

- [ ] **Step 3: Controller demonstration**

Run `theme-build` against `backup-docs` (`uv run --extra dev python -m theme_builder.cli theme-build`), then write `docs/superpowers/theme-builder-report.md` showing: per-level community counts; one real community's `title`/`summary`/a finding; and that its `cited_fact_uuids` resolve to real source URLs via `graph_extract.provenance.Provenance(driver).resolve_citations(cited)`. Confirm no report contains a URL authored by the LLM and every cited UUID is a real fact of that community.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_theme_detect_live.py docs/superpowers/theme-builder-report.md
git commit -m "test(theme): live detection smoke + demonstration report"
```

---

## Notes for the implementer

- `_run_theme_build` takes `driver=` so tests inject the testcontainer driver; the `theme-build` command builds its own.
- The report/detect/context modules are pure or fake-client testable; only `detect_communities`' GDS call and the demonstration need live infra.
- Do not add the community label to Graphiti nodes or merge into them (design decision #5). Do not let the report LLM's text carry a URL (design decision #2) — `_URL_RE` strip + fact-id validation are the guardrails.
