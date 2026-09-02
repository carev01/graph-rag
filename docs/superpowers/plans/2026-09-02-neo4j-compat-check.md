# Neo4j / Graphiti Compatibility Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable harness that proves `graphiti-core==0.29.2` and our own Cypher work against Neo4j 2026.07.1 Community before we commit to a bootstrap ingestion there.

**Architecture:** A new `src/compat/` package. Pure dataclasses and the verdict/report renderer have no I/O and are unit-tested hermetically; a runner executes a registry of checks sequentially against a target driver, containing every exception and auto-retrying failed Cypher under `CYPHER 5`; a CLI wires it up and writes a markdown report. Two known Cypher-25 fixes to our own code land alongside it.

**Tech Stack:** Python 3.12, `uv`, `neo4j` async driver, `graphiti-core==0.29.2`, pytest + pytest-asyncio, testcontainers, pydantic-settings.

**Spec:** `docs/superpowers/specs/2026-09-02-neo4j-compat-check-design.md`

## Global Constraints

- Target instance: **Neo4j 2026.07.1 Community**, `db.query.default_language=CYPHER_25`, GDS 2026.07.0 installed, **APOC not installed** — never call `apoc.*`.
- Backward compatibility with Neo4j 5.x is **not** a requirement. The scoped `CALL (x) { … }` form (Neo4j 5.23+) is allowed everywhere.
- Testcontainer image is exactly **`neo4j:2026.07.1-community`**.
- The harness namespace constant is exactly **`COMPAT_GROUP_ID = "compat-check"`**. Every node/edge the harness writes carries `group_id="compat-check"`; every index it creates is named with a `compat_` prefix.
- The harness must be safe against a **populated** instance: never issue an unscoped `DELETE`; every delete is scoped to `group_id='compat-check'`.
- Graphiti's own indexes (from `build_indices_and_constraints()`) are **never** torn down.
- Group 8 (`e2e`) is scoped to **extraction + retrieval only, never synthesis** — it must not require the GLM/judge tier.
- An unreachable LLM or embedder yields `skip`, never `fail`. Absent GDS yields `skip`, never `fail`.
- Embedding dimension is **768** (`settings.embed_dim`).
- Only two graphiti search recipes exist in this codebase: `EDGE_HYBRID_SEARCH_RRF` and `EDGE_HYBRID_SEARCH_NODE_DISTANCE`.
- CI gate, which must be clean at every commit: `uv run ruff check src tests` (E702 forbids semicolons in test fakes; E402 forbids non-top imports), `uv run mypy src`, `uv run pytest -m "not live"`.
- `@live` tests are marked `@pytest.mark.live` and are excluded by the default `addopts`.

---

### Task 1: Pure core — result model and report renderer

**Files:**
- Create: `src/compat/__init__.py`
- Create: `src/compat/model.py`
- Create: `src/compat/report.py`
- Test: `tests/unit/test_compat_report.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Status = Literal["pass","fail","skip"]`; `CheckResult(name, group, status, detail="", cypher5_retry=None, informational=False)` (frozen dataclass); `Verdict = Literal["GO","GO_WITH_CONFIG","NO_GO"]`; `COMPAT_GROUP_ID: str`; `report.verdict(results: list[CheckResult]) -> Verdict`; `report.render(results: list[CheckResult], *, target: dict[str, str], teardown_error: str | None = None) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_compat_report.py`:

```python
from compat.model import CheckResult
from compat.report import render, verdict


def _r(name, status, **kw):
    return CheckResult(name=name, group=kw.pop("group", "g"), status=status, **kw)


def test_verdict_go_when_all_pass():
    assert verdict([_r("a", "pass"), _r("b", "pass")]) == "GO"


def test_verdict_go_with_config_when_every_failure_passes_under_cypher5():
    results = [_r("a", "pass"), _r("b", "fail", cypher5_retry="pass")]
    assert verdict(results) == "GO_WITH_CONFIG"


def test_verdict_no_go_when_a_failure_also_fails_under_cypher5():
    results = [_r("a", "fail", cypher5_retry="pass"), _r("b", "fail", cypher5_retry="fail")]
    assert verdict(results) == "NO_GO"


def test_verdict_no_go_when_a_procedural_check_fails():
    # cypher5_retry is None for CallableChecks -> cannot be excused by config
    assert verdict([_r("a", "fail", cypher5_retry=None)]) == "NO_GO"


def test_verdict_ignores_skips_and_informational():
    results = [
        _r("a", "pass"),
        _r("b", "skip", detail="llm unreachable"),
        _r("c", "fail", cypher5_retry="fail", informational=True),
    ]
    assert verdict(results) == "GO"


def test_verdict_go_on_empty_results():
    assert verdict([]) == "GO"


def test_render_contains_verdict_matrix_and_sections():
    results = [
        _r("kernel version", "pass", group="server", detail="2026.07.1"),
        _r("apoc present", "fail", group="server", detail="not installed", informational=True),
        _r("e2e ingest", "skip", group="e2e", detail="embedder unreachable"),
        _r("sweep", "fail", group="our-cypher", cypher5_retry="pass", detail="boom"),
    ]
    out = render(results, target={"uri": "bolt://h:7687", "kernel": "2026.07.1"})
    assert "GO_WITH_CONFIG" in out
    assert "| kernel version |" in out
    assert "(info)" in out                    # informational marker
    assert "## Not verified" in out
    assert "embedder unreachable" in out
    assert "db.query.default_language=CYPHER_5" in out   # recommended action
    assert "bolt://h:7687" in out


def test_render_reports_teardown_error_prominently():
    out = render([_r("a", "pass")], target={"uri": "u"}, teardown_error="delete failed")
    assert "manual cleanup required" in out.lower()
    assert "compat-check" in out
    assert "delete failed" in out


def test_render_marks_procedural_retry_as_not_applicable():
    out = render([_r("a", "fail", group="e2e", cypher5_retry=None, detail="x")],
                 target={"uri": "u"})
    assert "n/a (procedural)" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_compat_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'compat'`

- [ ] **Step 3: Create the package and the model**

Create `src/compat/__init__.py` as an empty file.

Create `src/compat/model.py`:

```python
"""Pure data model for the Neo4j compatibility harness. No I/O, no heavy imports —
report.py and its unit tests depend only on this."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["pass", "fail", "skip"]
Verdict = Literal["GO", "GO_WITH_CONFIG", "NO_GO"]

#: Every node/edge the harness writes carries this group_id, and teardown deletes
#: exactly this namespace. Not configurable: it is the isolation contract.
COMPAT_GROUP_ID = "compat-check"


@dataclass(frozen=True)
class CheckResult:
    name: str
    group: str
    status: Status
    detail: str = ""
    #: Only set when status == "fail" AND the check was a CypherCheck. None means
    #: a language-version retry was not applicable (a procedural CallableCheck).
    cypher5_retry: Status | None = None
    #: Informational results are rendered but never move the verdict.
    informational: bool = False


@dataclass(frozen=True)
class CypherCheck:
    """A declarative statement. Because the runner holds the text, it can re-run a
    failure as "CYPHER 5 " + cypher and record cypher5_retry."""
    name: str
    group: str
    cypher: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Optional predicate over the returned rows. Absent => "did not raise" passes.
    expect: Callable[[list[dict]], bool] | None = None
    informational: bool = False


@dataclass(frozen=True)
class CallableCheck:
    """An arbitrary async call into graphiti or one of our modules. Returns a short
    detail string on success; raises to fail; raises SkipCheck to skip."""
    name: str
    group: str
    fn: Callable[["CheckContext"], Awaitable[str]]
    informational: bool = False


Check = CypherCheck | CallableCheck


class SkipCheck(Exception):
    """Raised by a CallableCheck to report `skip` rather than `fail` — used when a
    dependency outside Neo4j (LLM, embedder, GDS plugin) is unavailable, which is
    not a Neo4j incompatibility."""


@dataclass
class CheckContext:
    """Shared clients handed to every CallableCheck so checks never build their own."""
    driver: Any                 # neo4j.AsyncDriver
    graphiti: Any               # graphiti_core.Graphiti | None
    settings: Any               # graph_extract.config.ExtractSettings
    embedding: list[float]      # a fabricated 768-d vector, reused everywhere
```

- [ ] **Step 4: Write the report renderer**

Create `src/compat/report.py`:

```python
"""Verdict computation and markdown rendering for the compatibility harness. Pure —
takes CheckResults, returns a string."""
from __future__ import annotations

from compat.model import COMPAT_GROUP_ID, CheckResult, Verdict

_STATUS_ICON = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}


def _scored(results: list[CheckResult]) -> list[CheckResult]:
    """Only non-informational, non-skipped results move the verdict."""
    return [r for r in results if not r.informational and r.status != "skip"]


def verdict(results: list[CheckResult]) -> Verdict:
    failures = [r for r in _scored(results) if r.status == "fail"]
    if not failures:
        return "GO"
    if all(f.cypher5_retry == "pass" for f in failures):
        return "GO_WITH_CONFIG"
    return "NO_GO"


def _retry_cell(r: CheckResult) -> str:
    if r.status != "fail":
        return "-"
    if r.cypher5_retry is None:
        return "n/a (procedural)"
    return _STATUS_ICON[r.cypher5_retry]


def _actions(results: list[CheckResult], v: Verdict) -> list[str]:
    out: list[str] = []
    if v == "GO_WITH_CONFIG":
        out.append(
            "Set `db.query.default_language=CYPHER_5` on the server (every failure "
            "passes under Cypher 5, so no code change is required).")
    for r in _scored(results):
        if r.status == "fail" and r.cypher5_retry != "pass":
            out.append(f"Fix `{r.group}` / **{r.name}** — fails under both language "
                       f"versions: {r.detail}")
    if v == "GO":
        out.append("No action required — the target is compatible.")
    return out


def render(results: list[CheckResult], *, target: dict[str, str],
           teardown_error: str | None = None) -> str:
    v = verdict(results)
    lines = ["# Neo4j / Graphiti Compatibility Report", ""]
    lines.append("## Target")
    lines.append("")
    for key, value in target.items():
        lines.append(f"- **{key}:** {value}")
    lines += ["", f"## Verdict: {v}", ""]

    for action in _actions(results, v):
        lines.append(f"- {action}")
    lines.append("")

    if teardown_error:
        lines += [
            "> **MANUAL CLEANUP REQUIRED.** Harness teardown failed, so test data may "
            f"remain on the target. Delete it with "
            f"`MATCH (n {{group_id:'{COMPAT_GROUP_ID}'}}) DETACH DELETE n` and drop any "
            "`compat_`-prefixed indexes.",
            "",
            f"> Teardown error: {teardown_error}",
            "",
        ]

    lines += ["## Results", "",
              "| group | check | status | Cypher 5 | detail |",
              "|---|---|---|---|---|"]
    for r in results:
        name = f"{r.name} (info)" if r.informational else r.name
        lines.append(f"| {r.group} | {name} | {_STATUS_ICON[r.status]} | "
                     f"{_retry_cell(r)} | {r.detail} |")

    skipped = [r for r in results if r.status == "skip"]
    lines += ["", "## Not verified", ""]
    if skipped:
        for r in skipped:
            lines.append(f"- **{r.name}** ({r.group}) — {r.detail}")
    else:
        lines.append("- Nothing skipped; every check ran.")

    lines += [
        "",
        "## Side effects",
        "",
        "Graphiti's own indexes (created by `build_indices_and_constraints()`) are "
        "left in place deliberately — the call is idempotent and those indexes are "
        "exactly what a real bootstrap needs. All harness *data* "
        f"(`group_id='{COMPAT_GROUP_ID}'`) and all `compat_`-prefixed indexes are "
        "removed in teardown.",
        "",
    ]
    return "\n".join(lines)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_compat_report.py -q`
Expected: PASS (9 passed)

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/compat/__init__.py src/compat/model.py src/compat/report.py tests/unit/test_compat_report.py
git commit -m "feat(compat): result model + verdict/report renderer"
```

---

### Task 2: Runner — execution, Cypher-5 retry, containment, teardown

**Files:**
- Create: `src/compat/runner.py`
- Modify: `src/graph_extract/config.py` (add three settings next to `neo4j_password`)
- Test: `tests/unit/test_compat_runner.py`

**Interfaces:**
- Consumes: `compat.model` (`CheckResult`, `CypherCheck`, `CallableCheck`, `CheckContext`, `SkipCheck`, `COMPAT_GROUP_ID`).
- Produces: `runner.compat_target(settings) -> tuple[str, str, str]`; `async runner.run_cypher_check(driver, check) -> CheckResult`; `async runner.run_callable_check(ctx, check) -> CheckResult`; `async runner.run_all(ctx, checks) -> list[CheckResult]`; `async runner.teardown(driver) -> str | None` (returns an error string, or `None` on success); `runner.fabricate_embedding(dim) -> list[float]`.

**Context the implementer needs:** `ExtractSettings` is a `pydantic_settings.BaseSettings` in `src/graph_extract/config.py` with `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`. It already has `neo4j_uri`, `neo4j_user`, `neo4j_password` as required fields. Hermetic unit tests construct it as `ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_compat_runner.py`:

```python
import pytest

from compat import runner
from compat.model import CallableCheck, CheckContext, CypherCheck, SkipCheck
from graph_extract.config import ExtractSettings


def _settings(**kw):
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://main", neo4j_user="mu", neo4j_password="mp")
    base.update(kw)
    return ExtractSettings(**base)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        async def gen():
            for row in self._rows:
                yield row
        return gen()


class _FakeSession:
    """Records every statement; raises for statements listed in `fail_on`."""

    def __init__(self, calls, fail_on, rows):
        self.calls, self.fail_on, self.rows = calls, fail_on, rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **params):
        self.calls.append(cypher)
        for needle in self.fail_on:
            if cypher.startswith(needle) and not cypher.startswith("CYPHER 5 "):
                raise RuntimeError("SyntaxError: nope")
        return _FakeResult(self.rows)


class _FakeDriver:
    def __init__(self, fail_on=(), rows=()):
        self.calls: list[str] = []
        self.fail_on, self.rows = fail_on, list(rows)

    def session(self):
        return _FakeSession(self.calls, self.fail_on, self.rows)


def test_compat_target_falls_back_to_main_neo4j_settings():
    assert runner.compat_target(_settings()) == ("bolt://main", "mu", "mp")


def test_compat_target_prefers_compat_overrides():
    s = _settings(compat_neo4j_uri="bolt://new", compat_neo4j_user="cu",
                  compat_neo4j_password="cp")
    assert runner.compat_target(s) == ("bolt://new", "cu", "cp")


def test_compat_target_falls_back_per_field():
    s = _settings(compat_neo4j_uri="bolt://new")
    assert runner.compat_target(s) == ("bolt://new", "mu", "mp")


def test_fabricate_embedding_has_requested_dim_and_is_deterministic():
    a, b = runner.fabricate_embedding(768), runner.fabricate_embedding(768)
    assert len(a) == 768
    assert a == b


@pytest.mark.asyncio
async def test_cypher_check_passes_when_statement_runs():
    d = _FakeDriver(rows=[{"v": 1}])
    res = await runner.run_cypher_check(d, CypherCheck("c", "g", "RETURN 1 AS v"))
    assert res.status == "pass"
    assert res.cypher5_retry is None


@pytest.mark.asyncio
async def test_cypher_check_fails_when_expect_predicate_rejects_rows():
    d = _FakeDriver(rows=[{"v": 0}])
    check = CypherCheck("c", "g", "RETURN 0 AS v", expect=lambda rows: rows[0]["v"] == 1)
    res = await runner.run_cypher_check(d, check)
    assert res.status == "fail"
    assert "expect" in res.detail.lower()


@pytest.mark.asyncio
async def test_cypher_check_retries_failure_under_cypher_5():
    d = _FakeDriver(fail_on=["MATCH"], rows=[{"v": 1}])
    res = await runner.run_cypher_check(d, CypherCheck("c", "g", "MATCH (n) RETURN 1 AS v"))
    assert res.status == "fail"
    assert res.cypher5_retry == "pass"
    assert d.calls[-1].startswith("CYPHER 5 MATCH")


@pytest.mark.asyncio
async def test_cypher_check_records_retry_failure():
    d = _FakeDriver(fail_on=["MATCH", "CYPHER 5 MATCH"])
    res = await runner.run_cypher_check(d, CypherCheck("c", "g", "MATCH (n) RETURN 1"))
    assert res.status == "fail"
    assert res.cypher5_retry == "fail"


@pytest.mark.asyncio
async def test_callable_check_skip_and_failure_are_distinguished():
    ctx = CheckContext(driver=None, graphiti=None, settings=_settings(), embedding=[0.1])

    async def skips(_):
        raise SkipCheck("embedder unreachable")

    async def boom(_):
        raise RuntimeError("kaboom")

    async def ok(_):
        return "all good"

    assert (await runner.run_callable_check(ctx, CallableCheck("s", "g", skips))).status == "skip"
    failed = await runner.run_callable_check(ctx, CallableCheck("b", "g", boom))
    assert failed.status == "fail"
    assert failed.cypher5_retry is None
    assert "kaboom" in failed.detail
    passed = await runner.run_callable_check(ctx, CallableCheck("o", "g", ok))
    assert passed.status == "pass"
    assert passed.detail == "all good"


@pytest.mark.asyncio
async def test_run_all_contains_failures_and_keeps_going():
    ctx = CheckContext(driver=_FakeDriver(rows=[{"v": 1}]), graphiti=None,
                       settings=_settings(), embedding=[0.1])

    async def boom(_):
        raise RuntimeError("kaboom")

    checks = [
        CallableCheck("first", "g", boom),
        CypherCheck("second", "g", "RETURN 1 AS v"),
    ]
    results = await runner.run_all(ctx, checks)
    assert [r.status for r in results] == ["fail", "pass"]


@pytest.mark.asyncio
async def test_run_all_preserves_informational_flag():
    ctx = CheckContext(driver=_FakeDriver(rows=[{"v": 1}]), graphiti=None,
                       settings=_settings(), embedding=[0.1])
    results = await runner.run_all(
        ctx, [CypherCheck("i", "g", "RETURN 1 AS v", informational=True)])
    assert results[0].informational is True


@pytest.mark.asyncio
async def test_teardown_scopes_deletes_to_the_compat_group_and_reports_errors():
    d = _FakeDriver()
    assert await runner.teardown(d) is None
    assert any("compat-check" in c for c in d.calls)
    assert not any(c.strip().startswith("MATCH (n) DETACH DELETE") for c in d.calls)

    boom = _FakeDriver(fail_on=["MATCH"])
    err = await runner.teardown(boom)
    assert err is not None and "nope" in err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_compat_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'compat.runner'`

- [ ] **Step 3: Add the compat settings**

In `src/graph_extract/config.py`, immediately after the `neo4j_password: str` line, insert:

```python
    # Optional overrides letting the compatibility harness (src/compat/) target a
    # DIFFERENT Neo4j than the one the rest of the stack uses, without editing the
    # neo4j_* values. Empty means "fall back to the neo4j_* value", field by field.
    compat_neo4j_uri: str = ""
    compat_neo4j_user: str = ""
    compat_neo4j_password: str = ""
```

- [ ] **Step 4: Write the runner**

Create `src/compat/runner.py`:

```python
"""Executes compatibility checks against a target Neo4j.

Two guarantees: (1) one check's failure never aborts the run — every exception is
contained and recorded; (2) a failed CypherCheck is automatically re-run prefixed
`CYPHER 5`, so the report can distinguish "needs a server config change" from
"needs a code change"."""
from __future__ import annotations

import logging

from compat.model import (
    COMPAT_GROUP_ID, CallableCheck, Check, CheckContext, CheckResult, CypherCheck,
    SkipCheck, Status,
)

logger = logging.getLogger(__name__)

_MAX_DETAIL = 300


def compat_target(settings) -> tuple[str, str, str]:
    """(uri, user, password) for the harness target: the compat_* override where
    set, else the main neo4j_* value, resolved field by field."""
    return (settings.compat_neo4j_uri or settings.neo4j_uri,
            settings.compat_neo4j_user or settings.neo4j_user,
            settings.compat_neo4j_password or settings.neo4j_password)


def fabricate_embedding(dim: int) -> list[float]:
    """A deterministic unit-ish vector of `dim` floats. Deterministic so reruns are
    comparable; fabricated so groups 1-7 need no embedder."""
    return [((i % 97) + 1) / 100.0 for i in range(dim)]


def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_MAX_DETAIL]


async def _execute(driver, cypher: str, params: dict) -> list[dict]:
    async with driver.session() as s:
        result = await s.run(cypher, **params)
        return [dict(rec) async for rec in result]


async def run_cypher_check(driver, check: CypherCheck) -> CheckResult:
    try:
        rows = await _execute(driver, check.cypher, check.params)
    except Exception as exc:  # noqa: BLE001 - containment is the point
        detail = _detail(exc)
        retry: Status
        try:
            await _execute(driver, "CYPHER 5 " + check.cypher, check.params)
            retry = "pass"
        except Exception:  # noqa: BLE001
            retry = "fail"
        return CheckResult(name=check.name, group=check.group, status="fail",
                           detail=detail, cypher5_retry=retry,
                           informational=check.informational)
    if check.expect is not None and not check.expect(rows):
        return CheckResult(
            name=check.name, group=check.group, status="fail",
            detail=f"expect predicate rejected rows: {str(rows)[:_MAX_DETAIL]}",
            informational=check.informational)
    return CheckResult(name=check.name, group=check.group, status="pass",
                       detail=str(rows[0])[:_MAX_DETAIL] if rows else "",
                       informational=check.informational)


async def run_callable_check(ctx: CheckContext, check: CallableCheck) -> CheckResult:
    try:
        detail = await check.fn(ctx)
    except SkipCheck as exc:
        return CheckResult(name=check.name, group=check.group, status="skip",
                           detail=str(exc)[:_MAX_DETAIL],
                           informational=check.informational)
    except Exception as exc:  # noqa: BLE001
        # A CallableCheck has no single statement to re-prefix, so a language-version
        # retry is not applicable; cypher5_retry stays None and renders as
        # "n/a (procedural)".
        return CheckResult(name=check.name, group=check.group, status="fail",
                           detail=_detail(exc), informational=check.informational)
    return CheckResult(name=check.name, group=check.group, status="pass",
                       detail=detail[:_MAX_DETAIL], informational=check.informational)


async def run_all(ctx: CheckContext, checks: list[Check]) -> list[CheckResult]:
    """Run every check in registry order. Order matters: group 5 writes the synthetic
    graph that groups 6 and 7 query."""
    results: list[CheckResult] = []
    for check in checks:
        if isinstance(check, CypherCheck):
            res = await run_cypher_check(ctx.driver, check)
        else:
            res = await run_callable_check(ctx, check)
        logger.info("compat check %s/%s -> %s", check.group, check.name, res.status)
        results.append(res)
    return results


async def teardown(driver) -> str | None:
    """Delete every harness artifact. Returns None on success, else an error string
    that the report surfaces as 'manual cleanup required'.

    Deletes are ALWAYS scoped to COMPAT_GROUP_ID — the harness must be safe to run
    against a populated instance."""
    try:
        async with driver.session() as s:
            await s.run(
                "MATCH ()-[r {group_id:$g}]-() DELETE r", g=COMPAT_GROUP_ID)
            await s.run(
                "MATCH (n {group_id:$g}) DETACH DELETE n", g=COMPAT_GROUP_ID)
            result = await s.run(
                "SHOW INDEXES YIELD name WHERE name STARTS WITH 'compat_' "
                "RETURN collect(name) AS names")
            rows = [dict(rec) async for rec in result]
            for name in (rows[0]["names"] if rows else []):
                await s.run(f"DROP INDEX {name} IF EXISTS")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("compat teardown failed", exc_info=True)
        return _detail(exc)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_compat_runner.py -q`
Expected: PASS (12 passed)

- [ ] **Step 6: Confirm the whole non-live suite and gates are clean**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`
Expected: all pass. (Adding optional settings with defaults must not disturb any existing test.)

- [ ] **Step 7: Commit**

```bash
git add src/compat/runner.py src/graph_extract/config.py tests/unit/test_compat_runner.py
git commit -m "feat(compat): check runner with Cypher-5 retry, containment, scoped teardown"
```

---

### Task 3: Checks groups 1-4 — server, bootstrap, vector, fulltext

**Files:**
- Create: `src/compat/checks.py`
- Test: `tests/unit/test_compat_checks.py`

**Interfaces:**
- Consumes: `compat.model` (`CypherCheck`, `CallableCheck`, `CheckContext`, `SkipCheck`, `COMPAT_GROUP_ID`); `graph_extract.graphiti_client.init_indices`; `graph_sync.neo4j_repo.Neo4jRepo`.
- Produces: `checks.server_checks() -> list[Check]`; `checks.bootstrap_checks() -> list[Check]`; `checks.vector_checks() -> list[Check]`; `checks.fulltext_checks() -> list[Check]`. Task 4 adds `graphiti_write_checks()`, `graphiti_search_checks()`, `our_cypher_checks()`; Task 5 adds `e2e_checks()` and `all_checks()`.

**Context the implementer needs:**
- `graph_extract.graphiti_client.init_indices(graphiti)` already wraps `graphiti.build_indices_and_constraints()`.
- `graph_sync.neo4j_repo.Neo4jRepo(uri, user, password)` exposes `async init_schema()` and `async close()`.
- Graphiti 0.29.2 declares exactly four fulltext indexes: `episode_content`, `node_name_and_summary`, `community_name`, `edge_name_and_fact`.
- Graphiti's Lucene sanitiser is `graphiti_core.helpers.lucene_sanitize(query: str) -> str`.
- The target has **no APOC** — the APOC check must be `informational=True` so its failure never moves the verdict.
- GDS may be absent on some targets: the GDS version check is a `CallableCheck` that raises `SkipCheck` when `gds.version()` is unavailable.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_compat_checks.py`:

```python
from compat import checks
from compat.model import CallableCheck, CypherCheck


def _all():
    return (checks.server_checks() + checks.bootstrap_checks()
            + checks.vector_checks() + checks.fulltext_checks())


def test_every_check_has_a_nonempty_name_and_group():
    for c in _all():
        assert c.name and c.group


def test_group_labels_are_the_spec_names():
    groups = {c.group for c in _all()}
    assert groups == {"server", "bootstrap", "vector", "fulltext"}


def test_apoc_check_is_informational():
    apoc = [c for c in checks.server_checks() if "apoc" in c.name.lower()]
    assert len(apoc) == 1
    assert apoc[0].informational is True


def test_gds_check_is_callable_so_it_can_skip():
    gds = [c for c in checks.server_checks() if "gds" in c.name.lower()]
    assert len(gds) == 1
    assert isinstance(gds[0], CallableCheck)


def test_vector_index_ddl_check_is_informational_and_prefixed():
    ddl = [c for c in checks.vector_checks()
           if isinstance(c, CypherCheck) and "CREATE VECTOR INDEX" in c.cypher]
    assert ddl, "expected an informational CREATE VECTOR INDEX probe"
    for c in ddl:
        assert c.informational is True
        assert "compat_" in c.cypher


def test_no_check_calls_apoc_procedures():
    # APOC is not installed on the target and is referenced nowhere in the stack.
    for c in _all():
        if isinstance(c, CypherCheck):
            assert "apoc." not in c.cypher.lower() or c.informational


def test_every_harness_index_is_compat_prefixed():
    for c in _all():
        if isinstance(c, CypherCheck) and "CREATE " in c.cypher and "INDEX" in c.cypher:
            assert "compat_" in c.cypher


def test_cypher_checks_never_issue_an_unscoped_delete():
    for c in _all():
        if isinstance(c, CypherCheck) and "DELETE" in c.cypher.upper():
            assert "compat-check" in c.cypher or "$g" in c.cypher


def test_lucene_escaping_check_covers_the_metacharacters():
    ft = [c for c in checks.fulltext_checks() if "lucene" in c.name.lower()]
    assert len(ft) == 1
    assert isinstance(ft[0], CallableCheck)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'compat.checks'`

- [ ] **Step 3: Write groups 1-4**

Create `src/compat/checks.py`:

```python
"""The compatibility check registry.

Registry ORDER IS SIGNIFICANT: group 5 (`graphiti-write`) creates the synthetic
graph that groups 6 (`graphiti-search`) and 7 (`our-cypher`) query. run_all
executes sequentially in this order."""
from __future__ import annotations

from compat.model import (
    COMPAT_GROUP_ID, CallableCheck, Check, CheckContext, CypherCheck, SkipCheck,
)

# --- synthetic fixture identifiers (stable so checks can reference each other) ---
EP_UUID = "compat-ep-1"
ENT_A, ENT_B, ENT_C = "compat-ent-a", "compat-ent-b", "compat-ent-c"
FACT_AB, FACT_BC = "compat-fact-ab", "compat-fact-bc"

_GRAPHITI_FULLTEXT_INDEXES = [
    "episode_content", "node_name_and_summary", "community_name", "edge_name_and_fact",
]

_LUCENE_METACHARACTERS = r'+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /'


# --------------------------------------------------------------------------- #
# Group 1: server
# --------------------------------------------------------------------------- #

async def _gds_version(ctx: CheckContext) -> str:
    """GDS is a plugin, not a Neo4j capability: absence is a skip, not a failure."""
    try:
        async with ctx.driver.session() as s:
            result = await s.run("CALL gds.version() YIELD gdsVersion RETURN gdsVersion")
            rows = [dict(rec) async for rec in result]
    except Exception as exc:  # noqa: BLE001
        raise SkipCheck(f"GDS not available: {type(exc).__name__}") from exc
    return f"GDS {rows[0]['gdsVersion']}" if rows else "GDS present, no version row"


def server_checks() -> list[Check]:
    return [
        CypherCheck(
            "kernel version and edition", "server",
            "CALL dbms.components() YIELD name, versions, edition "
            "WHERE name = 'Neo4j Kernel' RETURN versions[0] AS version, edition",
            expect=lambda rows: bool(rows)),
        CypherCheck(
            "default Cypher language", "server",
            "SHOW SETTINGS YIELD name, value WHERE name = 'db.query.default_language' "
            "RETURN value",
            expect=lambda rows: bool(rows)),
        CallableCheck("gds version", "server", _gds_version),
        CypherCheck(
            "apoc installed", "server",
            "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' "
            "RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] > 0,
            # Informational: nothing in graphiti-core or src/ calls apoc.*, so its
            # absence on the target is a fact to record, never a blocker.
            informational=True),
        CypherCheck(
            "fulltext procedures available", "server",
            "SHOW PROCEDURES YIELD name "
            "WHERE name IN ['db.index.fulltext.queryNodes', "
            "'db.index.fulltext.queryRelationships'] RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] == 2),
        CypherCheck(
            "vector similarity function available", "server",
            "SHOW FUNCTIONS YIELD name WHERE name = 'vector.similarity.cosine' "
            "RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] == 1),
    ]


# --------------------------------------------------------------------------- #
# Group 2: bootstrap
# --------------------------------------------------------------------------- #

async def _graphiti_bootstrap(ctx: CheckContext) -> str:
    from graph_extract.graphiti_client import init_indices
    await init_indices(ctx.graphiti)
    return "build_indices_and_constraints() completed"


async def _structural_schema(ctx: CheckContext) -> str:
    from compat.runner import compat_target
    from graph_sync.neo4j_repo import Neo4jRepo
    uri, user, password = compat_target(ctx.settings)
    repo = Neo4jRepo(uri, user, password)
    try:
        await repo.init_schema()
    finally:
        await repo.close()
    return "graph_sync init_schema() completed"


def bootstrap_checks() -> list[Check]:
    return [
        CallableCheck("graphiti build_indices_and_constraints", "bootstrap",
                      _graphiti_bootstrap),
        CypherCheck(
            "graphiti fulltext indexes exist", "bootstrap",
            "SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' "
            "RETURN collect(name) AS names",
            expect=lambda rows: bool(rows) and all(
                ix in rows[0]["names"] for ix in _GRAPHITI_FULLTEXT_INDEXES)),
        CypherCheck(
            "graphiti range indexes exist", "bootstrap",
            "SHOW INDEXES YIELD name, type WHERE type = 'RANGE' "
            "RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] > 0),
        CallableCheck("graph_sync structural schema", "bootstrap", _structural_schema),
    ]


# --------------------------------------------------------------------------- #
# Group 3: vector
# --------------------------------------------------------------------------- #
# Narrow by design: graphiti 0.29.2 creates NO vector indexes and scores similarity
# with brute-force `vector.similarity.cosine` in Cypher, so an ANN index is never
# consulted. We verify the function over both a node and a relationship property
# (fact embeddings live on RELATES_TO), plus graphiti's exact min_score query shape.

def vector_checks() -> list[Check]:
    return [
        CypherCheck(
            "seed vector probe nodes", "vector",
            "CREATE (a:CompatVec {uuid:'compat-vec-a', group_id:$g, emb:$v}), "
            "       (b:CompatVec {uuid:'compat-vec-b', group_id:$g, emb:$v}) "
            "CREATE (a)-[:COMPAT_REL {uuid:'compat-vec-rel', group_id:$g, emb:$v}]->(b) "
            "RETURN count(*) AS created",
            params={"g": COMPAT_GROUP_ID, "v": None}),   # v injected by all_checks()
        CypherCheck(
            "cosine similarity over a node property", "vector",
            "MATCH (n:CompatVec {group_id:$g}) "
            "RETURN vector.similarity.cosine(n.emb, $v) AS score LIMIT 1",
            params={"g": COMPAT_GROUP_ID, "v": None},
            expect=lambda rows: bool(rows) and rows[0]["score"] is not None),
        CypherCheck(
            "cosine similarity over a relationship property", "vector",
            "MATCH ()-[r:COMPAT_REL {group_id:$g}]->() "
            "RETURN vector.similarity.cosine(r.emb, $v) AS score LIMIT 1",
            params={"g": COMPAT_GROUP_ID, "v": None},
            expect=lambda rows: bool(rows) and rows[0]["score"] is not None),
        CypherCheck(
            "graphiti min_score query shape", "vector",
            "MATCH (n:CompatVec {group_id:$g}) "
            "WITH n, vector.similarity.cosine(n.emb, $v) AS score "
            "WHERE score > $min_score "
            "RETURN n.uuid AS uuid, score ORDER BY score DESC LIMIT 10",
            params={"g": COMPAT_GROUP_ID, "v": None, "min_score": 0.0},
            expect=lambda rows: len(rows) >= 1),
        CypherCheck(
            "CREATE VECTOR INDEX accepted", "vector",
            "CREATE VECTOR INDEX compat_vec_probe IF NOT EXISTS "
            "FOR (n:CompatVec) ON (n.emb) OPTIONS {indexConfig: "
            "{`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}",
            # Informational: nothing in the stack uses a vector index today. This
            # probe exists to confirm the remediation path for the brute-force-scan
            # follow-up (spec section 9) is open on this server.
            informational=True),
    ]


# --------------------------------------------------------------------------- #
# Group 4: fulltext
# --------------------------------------------------------------------------- #

async def _lucene_escaping(ctx: CheckContext) -> str:
    """Every Lucene metacharacter, pushed through graphiti's own sanitiser and into
    the fulltext procedure. An unsanitised metacharacter raises a Lucene
    ParseException at query time — a break that only shows up on real questions."""
    from graphiti_core.helpers import lucene_sanitize
    sanitized = lucene_sanitize(f"vault {_LUCENE_METACHARACTERS} lock")
    async with ctx.driver.session() as s:
        result = await s.run(
            "CALL db.index.fulltext.queryNodes('compat_ft_probe', $q) "
            "YIELD node, score RETURN count(*) AS n", q=sanitized)
        rows = [dict(rec) async for rec in result]
    return f"sanitised query accepted, {rows[0]['n'] if rows else 0} hits"


def fulltext_checks() -> list[Check]:
    return [
        CypherCheck(
            "CREATE FULLTEXT INDEX accepted", "fulltext",
            "CREATE FULLTEXT INDEX compat_ft_probe IF NOT EXISTS "
            "FOR (n:CompatFt) ON EACH [n.name, n.summary]"),
        CypherCheck(
            "seed fulltext probe node", "fulltext",
            "CREATE (n:CompatFt {uuid:'compat-ft-a', group_id:$g, "
            "name:'vault lock', summary:'immutability and retention'}) "
            "RETURN n.uuid AS uuid",
            params={"g": COMPAT_GROUP_ID}),
        CypherCheck(
            "await fulltext index refresh", "fulltext",
            "CALL db.index.fulltext.awaitEventuallyConsistentIndexRefresh()"),
        CypherCheck(
            "db.index.fulltext.queryNodes returns ranked hits", "fulltext",
            "CALL db.index.fulltext.queryNodes('compat_ft_probe', 'vault') "
            "YIELD node, score RETURN node.uuid AS uuid, score",
            expect=lambda rows: len(rows) >= 1),
        CallableCheck("lucene metacharacter escaping", "fulltext", _lucene_escaping),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/compat/checks.py tests/unit/test_compat_checks.py
git commit -m "feat(compat): server, bootstrap, vector and fulltext checks"
```

---

### Task 4: Checks groups 5-7 — graphiti writes, search recipes, our Cypher

**Files:**
- Modify: `src/compat/checks.py` (append three group functions)
- Test: `tests/unit/test_compat_checks.py` (extend)

**Interfaces:**
- Consumes: everything Task 3 produced.
- Produces: `checks.graphiti_write_checks() -> list[Check]`; `checks.graphiti_search_checks() -> list[Check]`; `checks.our_cypher_checks() -> list[Check]`.

**Context the implementer needs — exact signatures of what group 7 calls:**
- `graph_extract.provenance.Provenance(driver).resolve_citations(fact_uuids: list[str]) -> dict[str, dict]` — an absent uuid stays ABSENT from the dict; callers use `.get(uuid, {})`.
- `answer_api.search._vendor_episode_uuids(driver, vendor: str) -> set[str]`
- `answer_api.freshness.freshness(driver, group_id, *, reports: bool) -> dict` — this one **never raises by design** (it catches and returns `None`), so the check must assert the returned dict shape rather than rely on absence of an exception.
- `answer_api.timeline._sweep_flags(driver, uuids: list[str], group_id) -> dict[str, bool]`
- `answer_api.global_search.shortlist_communities(driver, embedder, q, *, level, k, group_id, min_rating)` — needs an embedder; group 7 must **not** require one, so check the underlying community query via `theme_builder.incremental.load_persisted` instead and leave `shortlist_communities` to group 8's live path.
- `theme_builder.incremental.touched_entities(driver, group_id, prev_cursor) -> set[str] | None`
- `theme_builder.incremental.load_persisted(driver, group_id) -> list[PersistedCommunity]`
- `theme_builder.incremental.prev_corpus_cursor(driver, group_id) -> str | None`
- `theme_builder.cli._corpus_cursor(driver, group_id) -> str | None` — one of the two bare `CALL {}` sites.
- `graph_extract.staleness_sweep.sweep_stale_facts(driver, group_id) -> dict` — the other bare `CALL {}` site.
- `theme_builder.detect.detect_communities(driver, group_id, *, min_community_size, max_levels)` — needs GDS; raise `SkipCheck` if `gds.version()` is unavailable.

**Graphiti model construction (0.29.2), for group 5:**
- `EntityNode(uuid, name, group_id, labels: list[str], created_at: datetime, name_embedding: list[float], summary: str)` — `labels` drives the dynamic-label write.
- `EpisodicNode(uuid, name, group_id, created_at, source: EpisodeType, source_description: str, content: str, valid_at: datetime)`
- `EntityEdge(uuid, group_id, source_node_uuid, target_node_uuid, created_at, name, fact, fact_embedding, episodes: list[str], valid_at, invalid_at)`
- All of them expose `async save(driver)` where `driver` is the **graphiti** driver: pass `ctx.graphiti.driver`, not the raw neo4j driver.
- **The 5.22 fault line is the BULK save, not `.save()`.** `get_entity_node_save_query` (single-node, used by `.save()`) joins labels in Python and interpolates them as literal query text — version-agnostic. Only `get_entity_node_save_bulk_query` (`models/nodes/node_db_queries.py:260`) uses Cypher's native dynamic-label expression `SET n:$(node.labels)`. So `.save()` alone does **not** exercise that construct.
- Do **not** call `graphiti_core.utils.bulk_utils.add_nodes_and_edges_bulk(driver, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder)` here — it takes an `embedder` and this group must stay LLM-free. Instead assert the construct directly with a `CypherCheck` replicating `SET n:$(node.labels)` (see the registry code below), which has the bonus of being Cypher-5-retryable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_compat_checks.py`:

```python
def _later():
    return (checks.graphiti_write_checks() + checks.graphiti_search_checks()
            + checks.our_cypher_checks())


def test_later_group_labels_are_the_spec_names():
    assert {c.group for c in _later()} == {
        "graphiti-write", "graphiti-search", "our-cypher"}


def test_search_checks_cover_exactly_the_two_recipes_used_in_src():
    names = " ".join(c.name.lower() for c in checks.graphiti_search_checks())
    assert "rrf" in names
    assert "node_distance" in names or "node distance" in names


def test_our_cypher_group_covers_the_two_bare_call_subquery_sites():
    names = " ".join(c.name.lower() for c in checks.our_cypher_checks())
    assert "corpus cursor" in names       # theme_builder/cli.py:69
    assert "staleness sweep" in names     # graph_extract/staleness_sweep.py:52


def test_our_cypher_group_covers_every_module_the_spec_lists():
    names = " ".join(c.name.lower() for c in checks.our_cypher_checks())
    for needle in ["resolve_citations", "vendor", "freshness", "timeline",
                   "touched_entities", "load_persisted", "leiden"]:
        assert needle in names, f"missing a check for {needle}"


def test_our_cypher_checks_are_all_callable_so_failures_report_procedural():
    from compat.model import CallableCheck as CC
    for c in checks.our_cypher_checks():
        assert isinstance(c, CC)


def test_graphiti_write_group_exercises_the_bulk_dynamic_label_construct():
    """`.save()` interpolates labels as literal text; only the BULK save uses
    Cypher's native `SET n:$(node.labels)`. That construct is the 5.22 fault line,
    so it must be asserted explicitly."""
    ddl = [c for c in checks.graphiti_write_checks()
           if isinstance(c, CypherCheck) and "SET n:$(" in c.cypher]
    assert len(ddl) == 1


def test_all_synthetic_uuids_are_compat_prefixed():
    for value in [checks.EP_UUID, checks.ENT_A, checks.ENT_B, checks.ENT_C,
                  checks.FACT_AB, checks.FACT_BC]:
        assert value.startswith("compat-")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: FAIL — `AttributeError: module 'compat.checks' has no attribute 'graphiti_write_checks'`

- [ ] **Step 3: Append groups 5-7 to `src/compat/checks.py`**

```python
# --------------------------------------------------------------------------- #
# Group 5: graphiti-write  (dynamic labels + bi-temporal edges, NO LLM)
# --------------------------------------------------------------------------- #
# Builds the synthetic graph that groups 6 and 7 query. Embeddings are fabricated,
# so this group costs nothing and needs no embedder.

async def _write_synthetic_graph(ctx: CheckContext) -> str:
    from datetime import datetime, timezone

    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType

    now = datetime.now(timezone.utc)
    gdriver = ctx.graphiti.driver

    episode = EpisodicNode(
        uuid=EP_UUID, name="compat episode", group_id=COMPAT_GROUP_ID, created_at=now,
        source=EpisodeType.text, source_description="compatibility harness",
        content="AWS Backup Vault Lock enforces immutable retention on recovery points.",
        valid_at=now)
    await episode.save(gdriver)

    entities = [
        EntityNode(uuid=ENT_A, name="AWS Backup", group_id=COMPAT_GROUP_ID,
                   labels=["Entity", "Product"], created_at=now,
                   name_embedding=ctx.embedding, summary="a backup service"),
        EntityNode(uuid=ENT_B, name="Vault Lock", group_id=COMPAT_GROUP_ID,
                   labels=["Entity", "Feature"], created_at=now,
                   name_embedding=ctx.embedding, summary="an immutability control"),
        EntityNode(uuid=ENT_C, name="Recovery Point", group_id=COMPAT_GROUP_ID,
                   labels=["Entity", "Concept"], created_at=now,
                   name_embedding=ctx.embedding, summary="a stored backup"),
    ]
    for entity in entities:
        await entity.save(gdriver)

    edges = [
        EntityEdge(uuid=FACT_AB, group_id=COMPAT_GROUP_ID, source_node_uuid=ENT_A,
                   target_node_uuid=ENT_B, created_at=now, name="HAS_FEATURE",
                   fact="AWS Backup provides Vault Lock.",
                   fact_embedding=ctx.embedding, episodes=[EP_UUID],
                   valid_at=now, invalid_at=None),
        EntityEdge(uuid=FACT_BC, group_id=COMPAT_GROUP_ID, source_node_uuid=ENT_B,
                   target_node_uuid=ENT_C, created_at=now, name="PROTECTS",
                   fact="Vault Lock enforces immutable retention on recovery points.",
                   fact_embedding=ctx.embedding, episodes=[EP_UUID],
                   valid_at=now, invalid_at=None),
    ]
    for edge in edges:
        await edge.save(gdriver)
    return "1 episode, 3 entities, 2 bi-temporal facts written"


async def _dynamic_labels_persisted(ctx: CheckContext) -> str:
    """Graphiti writes extra labels alongside :Entity. On Neo4j 5.22 the bulk form
    `SET n:$(node.labels)` is a syntax error; here we assert the labels landed."""
    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH (n:Entity {uuid:$u}) RETURN labels(n) AS labels", u=ENT_A)
        rows = [dict(rec) async for rec in result]
    if not rows:
        raise RuntimeError(f"entity {ENT_A} was not persisted")
    labels = set(rows[0]["labels"])
    if "Product" not in labels:
        raise RuntimeError(f"dynamic label not applied; got {sorted(labels)}")
    return f"labels persisted: {sorted(labels)}"


async def _bitemporal_properties_persisted(ctx: CheckContext) -> str:
    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:$u}]->() "
            "RETURN f.valid_at IS NOT NULL AS has_valid, "
            "       f.invalid_at IS NULL AS open, "
            "       size(f.fact_embedding) AS dim", u=FACT_AB)
        rows = [dict(rec) async for rec in result]
    if not rows:
        raise RuntimeError(f"fact {FACT_AB} was not persisted")
    row = rows[0]
    if not (row["has_valid"] and row["open"]):
        raise RuntimeError(f"bi-temporal properties wrong: {row}")
    return f"valid_at set, invalid_at null, embedding dim {row['dim']}"


def graphiti_write_checks() -> list[Check]:
    return [
        CallableCheck("write synthetic graph via graphiti models", "graphiti-write",
                      _write_synthetic_graph),
        CallableCheck("dynamic entity labels persisted", "graphiti-write",
                      _dynamic_labels_persisted),
        # The construct from graphiti's BULK node save
        # (models/nodes/node_db_queries.py:260). `.save()` interpolates labels as
        # literal query text, so only this exercises Cypher's native dynamic-label
        # expression -- the exact statement Neo4j 5.22 cannot parse. Declarative so
        # a failure is auto-retried under CYPHER 5.
        CypherCheck(
            "bulk dynamic-label expression SET n:$(node.labels)", "graphiti-write",
            "UNWIND $nodes AS node "
            "MERGE (n:Entity {uuid: node.uuid}) "
            "SET n:$(node.labels) "
            "SET n.group_id = node.group_id "
            "RETURN labels(n) AS labels",
            params={"nodes": [{"uuid": "compat-bulk-1",
                               "labels": ["Product", "Feature"],
                               "group_id": COMPAT_GROUP_ID}]},
            expect=lambda rows: bool(rows) and "Product" in rows[0]["labels"]),
        CallableCheck("bi-temporal fact properties persisted", "graphiti-write",
                      _bitemporal_properties_persisted),
    ]


# --------------------------------------------------------------------------- #
# Group 6: graphiti-search  (the two recipes this codebase actually uses)
# --------------------------------------------------------------------------- #

async def _search_rrf(ctx: CheckContext) -> str:
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = 10
    results = await ctx.graphiti._search(
        "vault lock immutable retention", config, group_ids=[COMPAT_GROUP_ID])
    return f"RRF recipe returned {len(results.edges)} edges"


async def _search_node_distance(ctx: CheckContext) -> str:
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_NODE_DISTANCE)
    config = EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
    config.limit = 10
    results = await ctx.graphiti._search(
        "vault lock immutable retention", config, group_ids=[COMPAT_GROUP_ID],
        center_node_uuid=ENT_A)
    return f"node_distance recipe returned {len(results.edges)} edges"


def graphiti_search_checks() -> list[Check]:
    # Only these two recipes appear anywhere in src/ (answer_api/search.py:3-4):
    # global_search ranks communities with its own Python cosine and drift reuses
    # search_local, so no node or community recipe is exercised by this codebase.
    return [
        CallableCheck("EDGE_HYBRID_SEARCH_RRF recipe", "graphiti-search", _search_rrf),
        CallableCheck("EDGE_HYBRID_SEARCH_NODE_DISTANCE recipe", "graphiti-search",
                      _search_node_distance),
    ]


# --------------------------------------------------------------------------- #
# Group 7: our-cypher  (every query we own, run against the synthetic graph)
# --------------------------------------------------------------------------- #

async def _resolve_citations(ctx: CheckContext) -> str:
    from graph_extract.provenance import Provenance
    resolved = await Provenance(ctx.driver).resolve_citations([FACT_AB, FACT_BC])
    return f"resolve_citations returned {len(resolved)} entries"


async def _vendor_scope(ctx: CheckContext) -> str:
    from answer_api.search import _vendor_episode_uuids
    uuids = await _vendor_episode_uuids(ctx.driver, "AWS")
    return f"vendor episode scope query ran, {len(uuids)} uuids"


async def _freshness(ctx: CheckContext) -> str:
    """freshness() swallows its own errors by design (it must never fail an answer),
    so assert the returned SHAPE instead of relying on an exception."""
    from answer_api.freshness import freshness
    stamps = await freshness(ctx.driver, COMPAT_GROUP_ID, reports=True)
    if set(stamps) != {"graph_cursor_time", "reports_as_of"}:
        raise RuntimeError(f"unexpected freshness shape: {stamps}")
    if stamps["graph_cursor_time"] is None:
        raise RuntimeError("graph_cursor_time is None despite a written episode — "
                           "the freshness query failed and was swallowed")
    return f"freshness stamps resolved: {stamps}"


async def _timeline_sweep_flags(ctx: CheckContext) -> str:
    from answer_api.timeline import _sweep_flags
    flags = await _sweep_flags(ctx.driver, [FACT_AB, FACT_BC], COMPAT_GROUP_ID)
    return f"timeline sweep-flag query ran, {len(flags)} flags"


async def _corpus_cursor_subquery(ctx: CheckContext) -> str:
    """theme_builder/cli.py's watermark — a CALL {} UNION ALL subquery."""
    from theme_builder.cli import _corpus_cursor
    cursor = await _corpus_cursor(ctx.driver, COMPAT_GROUP_ID)
    if cursor is None:
        raise RuntimeError("corpus cursor is None despite a written episode")
    return f"corpus cursor subquery ran: {cursor}"


async def _staleness_sweep(ctx: CheckContext) -> str:
    """graph_extract/staleness_sweep.py — a CALL {} importing-WITH subquery. The
    synthetic facts have a live episode, so nothing should be expired."""
    from graph_extract.staleness_sweep import sweep_stale_facts
    outcome = await sweep_stale_facts(ctx.driver, COMPAT_GROUP_ID)
    return f"staleness sweep subquery ran: expired={outcome.get('expired')}"


async def _touched_entities(ctx: CheckContext) -> str:
    from theme_builder.incremental import touched_entities
    touched = await touched_entities(ctx.driver, COMPAT_GROUP_ID, None)
    count = "all (None sentinel)" if touched is None else len(touched)
    return f"touched_entities ran: {count}"


async def _load_persisted(ctx: CheckContext) -> str:
    from theme_builder.incremental import load_persisted, prev_corpus_cursor
    persisted = await load_persisted(ctx.driver, COMPAT_GROUP_ID)
    cursor = await prev_corpus_cursor(ctx.driver, COMPAT_GROUP_ID)
    return f"load_persisted={len(persisted)} communities, prev_cursor={cursor}"


async def _leiden_detect(ctx: CheckContext) -> str:
    """GDS projection + seeded Leiden. GDS is a plugin: absence is a skip."""
    from theme_builder.detect import detect_communities
    try:
        async with ctx.driver.session() as s:
            await s.run("CALL gds.version() YIELD gdsVersion RETURN gdsVersion")
    except Exception as exc:  # noqa: BLE001
        raise SkipCheck(f"GDS not available: {type(exc).__name__}") from exc
    communities = await detect_communities(
        ctx.driver, COMPAT_GROUP_ID, min_community_size=1, max_levels=2)
    return f"GDS projection + seeded Leiden ran: {len(communities)} communities"


def our_cypher_checks() -> list[Check]:
    return [
        CallableCheck("provenance resolve_citations", "our-cypher", _resolve_citations),
        CallableCheck("vendor episode scope", "our-cypher", _vendor_scope),
        CallableCheck("freshness stamps", "our-cypher", _freshness),
        CallableCheck("timeline sweep flags", "our-cypher", _timeline_sweep_flags),
        CallableCheck("theme-builder corpus cursor (CALL {} UNION ALL)", "our-cypher",
                      _corpus_cursor_subquery),
        CallableCheck("staleness sweep (CALL {} importing WITH)", "our-cypher",
                      _staleness_sweep),
        CallableCheck("incremental touched_entities", "our-cypher", _touched_entities),
        CallableCheck("incremental load_persisted", "our-cypher", _load_persisted),
        CallableCheck("GDS projection + seeded leiden detect", "our-cypher",
                      _leiden_detect),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: PASS (15 passed)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/compat/checks.py tests/unit/test_compat_checks.py
git commit -m "feat(compat): graphiti write/search recipe checks and our-Cypher checks"
```

---

### Task 5: Group 8 (end-to-end) and the CLI

**Files:**
- Modify: `src/compat/checks.py` (append `e2e_checks()` and `all_checks()`)
- Create: `src/compat/cli.py`
- Test: `tests/unit/test_compat_checks.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `checks.e2e_checks() -> list[Check]`; `checks.all_checks(embedding: list[float]) -> list[Check]` (returns every group in order, with the fabricated vector substituted into each `CypherCheck.params` entry whose value is `None` under key `"v"`); `cli.main() -> None`.

**Context the implementer needs:**
- `graph_extract.graphiti_client.build_graphiti(settings) -> Graphiti` and `add_text_episode(graphiti, settings, *, name, body, source_description, reference_time, instructions=...)`.
- `answer_api.search.search_local(graphiti, driver, *, q, k=10, vendor=None, include_invalid=False, group_id, center_node_uuid=None) -> dict` with keys `query`, `count`, `results`.
- `add_text_episode` writes with `group_id=s.group_id`, so the e2e check must build a settings copy pinned to the harness namespace: `settings.model_copy(update={"group_id": COMPAT_GROUP_ID})` (`ExtractSettings` is a pydantic model, so `model_copy` is available — the codebase already uses this pattern in `build_cheap_graphiti`).
- The e2e check must be **extraction + retrieval only** — never call a synthesis function. An unreachable LLM/embedder is a `SkipCheck`.
- `git rev-parse --show-toplevel` is not available inside the module; derive the repo root as `Path(__file__).resolve().parents[2]`, matching `answer_api/eval_router.py:130`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_compat_checks.py`:

```python
def test_all_checks_returns_every_group_in_registry_order():
    groups = [c.group for c in checks.all_checks([0.5] * 768)]
    first_seen = list(dict.fromkeys(groups))
    assert first_seen == ["server", "bootstrap", "vector", "fulltext",
                          "graphiti-write", "graphiti-search", "our-cypher", "e2e"]


def test_all_checks_substitutes_the_fabricated_vector():
    from compat.model import CypherCheck as CypherC
    vector = [0.25] * 768
    for c in checks.all_checks(vector):
        if isinstance(c, CypherC) and "v" in c.params:
            assert c.params["v"] == vector, f"{c.name} kept a None placeholder"


def test_e2e_check_never_references_synthesis():
    """Scoped to the e2e function's own source (not the whole module) so that
    explanatory comments elsewhere naming global_search/drift don't trip it."""
    import inspect

    from compat.checks import _e2e_ingest_and_retrieve
    source = inspect.getsource(_e2e_ingest_and_retrieve)
    for banned in ["answer_local", "synthesize", "_synthesis_client_and_model",
                   "judge", "global_search", "drift_search"]:
        assert banned not in source, f"e2e must not depend on the synthesis tier ({banned})"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: FAIL — `AttributeError: module 'compat.checks' has no attribute 'all_checks'`

- [ ] **Step 3: Append group 8 and the registry assembler to `src/compat/checks.py`**

```python
# --------------------------------------------------------------------------- #
# Group 8: e2e  (ONE real article: extraction + retrieval, never synthesis)
# --------------------------------------------------------------------------- #

_E2E_ARTICLE = """# AWS Backup Vault Lock

AWS Backup Vault Lock enforces a write-once, read-many (WORM) setting on a backup
vault. Once a vault lock is in compliance mode, the retention period of a recovery
point cannot be shortened and recovery points cannot be deleted before they expire.

## Retention

The minimum retention period defines the shortest retention any backup plan may
assign to recovery points in the locked vault.
"""


async def _e2e_ingest_and_retrieve(ctx: CheckContext) -> str:
    """One inlined article through the real extraction pipeline, then retrieval.

    Inlined rather than fetched so the check does not depend on DocExtractor being
    up. Extraction + retrieval only: no synthesis, so this never touches the
    strong/judge tier."""
    from datetime import datetime, timezone

    from answer_api.search import search_local
    from graph_extract.graphiti_client import add_text_episode

    harness_settings = ctx.settings.model_copy(update={"group_id": COMPAT_GROUP_ID})
    try:
        await add_text_episode(
            ctx.graphiti, harness_settings, name="compat-e2e-article",
            body=_E2E_ARTICLE, source_description="compatibility harness",
            reference_time=datetime.now(timezone.utc))
    except Exception as exc:  # noqa: BLE001
        # An unreachable LLM or embedder is not a Neo4j incompatibility.
        raise SkipCheck(
            f"extraction unavailable ({type(exc).__name__}: {exc})") from exc

    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH (e:Episodic {group_id:$g}) "
            "OPTIONAL MATCH (n:Entity {group_id:$g}) "
            "OPTIONAL MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
            "RETURN count(DISTINCT e) AS episodes, count(DISTINCT n) AS entities, "
            "       count(DISTINCT f) AS facts", g=COMPAT_GROUP_ID)
        rows = [dict(rec) async for rec in result]
    counts = rows[0] if rows else {}
    if not counts.get("facts"):
        raise RuntimeError(f"ingest produced no facts: {counts}")

    found = await search_local(
        ctx.graphiti, ctx.driver, q="What does Vault Lock enforce?", k=5,
        group_id=COMPAT_GROUP_ID)
    if found["count"] == 0:
        raise RuntimeError("search_local returned no results after ingest")
    return (f"ingested {counts}; search_local returned {found['count']} results "
            f"with {sum(len(r['sources']) for r in found['results'])} resolved sources")


def e2e_checks() -> list[Check]:
    return [CallableCheck("one article: extract then retrieve", "e2e",
                          _e2e_ingest_and_retrieve)]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def all_checks(embedding: list[float]) -> list[Check]:
    """Every group, in execution order. Group 5 writes the synthetic graph that
    groups 6 and 7 query, so this order is load-bearing.

    CypherChecks declare their vector parameter as `{"v": None}`; the fabricated
    768-d vector is substituted here so the registry stays a pure literal."""
    registry = (server_checks() + bootstrap_checks() + vector_checks()
                + fulltext_checks() + graphiti_write_checks()
                + graphiti_search_checks() + our_cypher_checks() + e2e_checks())
    resolved: list[Check] = []
    for check in registry:
        if isinstance(check, CypherCheck) and check.params.get("v", "") is None:
            params = dict(check.params)
            params["v"] = embedding
            check = CypherCheck(
                name=check.name, group=check.group, cypher=check.cypher,
                params=params, expect=check.expect, informational=check.informational)
        resolved.append(check)
    return resolved
```

- [ ] **Step 4: Write the CLI**

Create `src/compat/cli.py`:

```python
"""Run the Neo4j compatibility harness against the COMPAT_*-resolved target and
write docs/superpowers/neo4j-compat-report.md.

    uv run --extra dev python -m compat.cli
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

from compat import report as report_mod
from compat.checks import all_checks
from compat.model import CheckContext
from compat.runner import compat_target, fabricate_embedding, run_all, teardown
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti

logger = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parents[2] / "docs" / "superpowers" / "neo4j-compat-report.md"


async def _target_facts(driver, uri: str) -> dict[str, str]:
    facts = {"uri": uri}
    probes = {
        "kernel": ("CALL dbms.components() YIELD name, versions, edition "
                   "WHERE name = 'Neo4j Kernel' "
                   "RETURN versions[0] + ' (' + edition + ')' AS v"),
        "default Cypher language": (
            "SHOW SETTINGS YIELD name, value "
            "WHERE name = 'db.query.default_language' RETURN value AS v"),
        "gds": "CALL gds.version() YIELD gdsVersion RETURN gdsVersion AS v",
    }
    for label, cypher in probes.items():
        try:
            async with driver.session() as s:
                result = await s.run(cypher)
                rows = [dict(rec) async for rec in result]
            facts[label] = str(rows[0]["v"]) if rows else "unknown"
        except Exception:  # noqa: BLE001
            facts[label] = "unavailable"
    return facts


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_extract_settings()
    uri, user, password = compat_target(settings)
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        await driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        await driver.close()
        print(f"cannot reach the target Neo4j at {uri}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    compat_settings = settings.model_copy(update={
        "neo4j_uri": uri, "neo4j_user": user, "neo4j_password": password})
    graphiti = build_graphiti(compat_settings)
    ctx = CheckContext(driver=driver, graphiti=graphiti, settings=compat_settings,
                       embedding=fabricate_embedding(settings.embed_dim))
    try:
        results = await run_all(ctx, all_checks(ctx.embedding))
        teardown_error = await teardown(driver)
        facts = await _target_facts(driver, uri)
        rendered = report_mod.render(results, target=facts,
                                     teardown_error=teardown_error)
        _OUT.write_text(rendered)
        print(rendered)
        print(f"\nwrote {_OUT}")
    finally:
        await graphiti.close()
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_compat_checks.py -q`
Expected: PASS (18 passed)

- [ ] **Step 6: Verify the CLI imports and the module is runnable**

Run: `uv run --extra dev python -c "import compat.cli; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Lint, type-check, full non-live suite**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/compat/checks.py src/compat/cli.py tests/unit/test_compat_checks.py
git commit -m "feat(compat): end-to-end ingest check and the harness CLI"
```

---

### Task 6: Cypher-25 modernisation — testcontainer bump and scoped subqueries

**Files:**
- Modify: `tests/integration/conftest.py:18` and `:37` (the two `Neo4jContainer("neo4j:5.22")` calls)
- Modify: `src/graph_extract/staleness_sweep.py:37-62` (docstring note + the `_SWEEP` subquery)
- Modify: `src/theme_builder/cli.py:69-73` (the `_corpus_cursor` subquery)

**Interfaces:**
- Consumes: nothing from earlier tasks — this is independent of the harness.
- Produces: no signature changes. `sweep_stale_facts` and `_corpus_cursor` keep their exact signatures and return shapes.

**Context the implementer needs:** backward compatibility with Neo4j 5.x is explicitly **not** required (spec section 8) — the compose 5.26 instance and its volume no longer exist, so nothing older than 2026.07 is deployed. The scoped `CALL (x) { … }` form requires Neo4j 5.23+, which is satisfied. Expect the image bump to surface pre-existing failures in the integration suite that were latent on 5.22; **fixing those is in scope for this task.**

- [ ] **Step 1: Bump the testcontainer image**

In `tests/integration/conftest.py`, replace both occurrences of `Neo4jContainer("neo4j:5.22")` (lines 18 and 37) with `Neo4jContainer("neo4j:2026.07.1-community")`.

Also update the comment block above the live fixtures (currently beginning `# --- Live fixtures: target the docker-compose Neo4j (5.26), NOT a testcontainer.`) to read:

```python
# --- Live fixtures: target the instance named by the .env neo4j_* settings, NOT a
# testcontainer. The testcontainers above now run the same image as production
# (neo4j:2026.07.1-community), so the historical "graphiti's dynamic-label Cypher
# needs 5.26" caveat no longer applies; these fixtures exist because the live
# tests need a POPULATED graph plus reachable LLM/embedder endpoints.
```

- [ ] **Step 2: Run the integration suite against the new image**

Run: `uv run --extra dev pytest tests/integration -q`
Expected: PASS. If any test fails, it is a latent 5.22-ism — fix the test or the query it exercises, and record what changed in the commit message. Do not revert the image bump.

- [ ] **Step 3: Convert the staleness-sweep subquery to the scoped form**

In `src/graph_extract/staleness_sweep.py`, replace the four docstring lines that begin `Note: the newer `CALL (eps) { ... }` call-scope syntax is rejected by the` and end `including with the outer SET in the same statement.` with:

```
Uses the scoped `CALL (eps) { ... }` call-scope form (Neo4j 5.23+). The target is
Neo4j 2026.07 and nothing older is deployed, so the legacy importing-WITH form is
no longer needed.
```

Then in `_SWEEP`, replace:

```
CALL {
  WITH eps
  UNWIND eps AS epu
```

with:

```
CALL (eps) {
  UNWIND eps AS epu
```

- [ ] **Step 4: Convert the corpus-cursor subquery to the scoped form**

In `src/theme_builder/cli.py`, inside `_corpus_cursor`, change the first line of the Cypher from:

```python
            "CALL { MATCH (e:Episodic {group_id:$g}) RETURN e.created_at AS t "
```

to:

```python
            "CALL () { MATCH (e:Episodic {group_id:$g}) RETURN e.created_at AS t "
```

(The subquery imports nothing, so the empty scope clause `CALL ()` is correct.)

- [ ] **Step 5: Run the tests covering both changed queries**

Run:
```
uv run --extra dev pytest tests/integration/test_staleness_sweep.py tests/integration/test_theme_cli.py tests/integration/test_incremental_cli.py -q
```
Expected: PASS. These exercise `_SWEEP` and `_corpus_cursor` against the bumped testcontainer, which now runs Cypher 25.

- [ ] **Step 6: Run the full gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tests/integration/conftest.py src/graph_extract/staleness_sweep.py src/theme_builder/cli.py
git commit -m "refactor: scoped CALL subqueries and testcontainer bump to 2026.07.1"
```

---

### Task 7: Harness integration test, live run, and the report

**Files:**
- Modify: `tests/integration/conftest.py` (split the `extract_driver` fixture so the container's connection details are also exposed)
- Create: `tests/integration/test_compat_harness.py`
- Create: `tests/integration/test_compat_live.py`
- Create: `docs/superpowers/neo4j-compat-report.md` (generated by the live run)
- Modify: `docs/infrastructure-sizing.md` (amend the vector-search assumption)

**Interfaces:**
- Consumes: `compat.checks.all_checks`, `compat.runner.{run_all, teardown, fabricate_embedding, compat_target}`, `compat.report.{render, verdict}`, `compat.model.CheckContext`.
- Produces: no new source interfaces.

**Context the implementer needs:** the existing `extract_driver` fixture (`tests/integration/conftest.py`) is a module-scoped Neo4j testcontainer, now running `neo4j:2026.07.1-community`. There is no LLM or embedder in that environment, so group 8 must report `skip` — asserting that is how we prove the skip path works. `@live` tests are marked `@pytest.mark.live` and excluded by default `addopts`.

- [ ] **Step 1: Expose the testcontainer's connection details**

The harness's `graphiti-write` and `graphiti-search` groups reach Neo4j through
`ctx.graphiti`, whose internal driver comes from settings — not through `ctx.driver`.
So the test must point graphiti at the *container*, or those groups would silently
target whatever `.env` names (currently a dead host). Split the existing fixture in
`tests/integration/conftest.py` so both the driver and its URL come from one container.

Replace the `extract_driver` fixture (currently lines 36-42) with:

```python
@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def extract_neo4j():
    """The Neo4j testcontainer for graph_extract tests, yielding its connection
    details so callers can point OTHER clients (e.g. a Graphiti instance) at the
    same container rather than at whatever .env names."""
    with Neo4jContainer("neo4j:2026.07.1-community") as neo:
        yield neo.get_connection_url(), "neo4j", neo.password


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def extract_driver(extract_neo4j):
    uri, user, password = extract_neo4j
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    yield driver
    await driver.close()
```

Run: `uv run --extra dev pytest tests/integration -q`
Expected: PASS — `extract_driver` keeps its exact signature and semantics, so every
existing test that consumes it is unaffected.

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_compat_harness.py`:

```python
import pytest

from compat.checks import all_checks
from compat.model import CheckContext
from compat.report import render, verdict
from compat.runner import fabricate_embedding, run_all, teardown
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_harness_runs_end_to_end_against_a_testcontainer(
        extract_driver, extract_neo4j):
    """The harness must produce a complete result set and a rendered report against a
    real Neo4j, and must SKIP (never fail) the checks whose dependencies are absent
    from this environment."""
    uri, user, password = extract_neo4j
    # Point graphiti at the CONTAINER: the graphiti-* groups reach Neo4j through
    # ctx.graphiti, whose driver comes from settings, not through ctx.driver.
    settings = get_extract_settings().model_copy(update={
        "neo4j_uri": uri, "neo4j_user": user, "neo4j_password": password})
    graphiti = build_graphiti(settings)
    embedding = fabricate_embedding(settings.embed_dim)
    ctx = CheckContext(driver=extract_driver, graphiti=graphiti, settings=settings,
                       embedding=embedding)
    try:
        results = await run_all(ctx, all_checks(embedding))
    finally:
        await teardown(extract_driver)
        await graphiti.close()

    groups = {r.group for r in results}
    assert groups == {"server", "bootstrap", "vector", "fulltext", "graphiti-write",
                      "graphiti-search", "our-cypher", "e2e"}

    # No LLM/embedder and no GDS plugin in this container -> those checks SKIP.
    e2e = [r for r in results if r.group == "e2e"]
    assert all(r.status == "skip" for r in e2e), [
        (r.name, r.status, r.detail) for r in e2e]
    gds = [r for r in results if "leiden" in r.name.lower() or "gds" in r.name.lower()]
    assert all(r.status == "skip" for r in gds), [
        (r.name, r.status, r.detail) for r in gds]

    rendered = render(results, target={"uri": "testcontainer"})
    assert "## Verdict:" in rendered
    assert verdict(results) in ("GO", "GO_WITH_CONFIG", "NO_GO")


async def test_teardown_leaves_no_harness_data(extract_driver):
    async with extract_driver.session() as s:
        result = await s.run(
            "MATCH (n {group_id:'compat-check'}) RETURN count(n) AS n")
        rows = [dict(rec) async for rec in result]
    assert rows[0]["n"] == 0
```

- [ ] **Step 3: Run it to verify it fails or reveals real gaps**

Run: `uv run --extra dev pytest tests/integration/test_compat_harness.py -q`
Expected: this is the first true exercise of the registry. Failures here are real bugs in Tasks 3-5 — fix them in `src/compat/checks.py`, not by weakening the assertions. The only assertions that may legitimately be relaxed are the specific groups expected to skip, if the container turns out to provide something unexpectedly.

- [ ] **Step 4: Write the live test**

Create `tests/integration/test_compat_live.py`:

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_compat_harness_live_against_target():
    """Runs the full harness against the COMPAT_*-resolved target and writes
    docs/superpowers/neo4j-compat-report.md."""
    from neo4j import AsyncGraphDatabase

    from compat.checks import all_checks
    from compat.model import CheckContext
    from compat.report import render, verdict
    from compat.runner import compat_target, fabricate_embedding, run_all, teardown
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_graphiti

    settings = get_extract_settings()
    uri, user, password = compat_target(settings)
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    compat_settings = settings.model_copy(update={
        "neo4j_uri": uri, "neo4j_user": user, "neo4j_password": password})
    graphiti = build_graphiti(compat_settings)
    embedding = fabricate_embedding(settings.embed_dim)
    ctx = CheckContext(driver=driver, graphiti=graphiti, settings=compat_settings,
                       embedding=embedding)
    try:
        results = await run_all(ctx, all_checks(embedding))
        teardown_error = await teardown(driver)
        rendered = render(results, target={"uri": uri},
                          teardown_error=teardown_error)
        assert teardown_error is None, teardown_error
        assert verdict(results) != "NO_GO", rendered
    finally:
        await graphiti.close()
        await driver.close()
```

- [ ] **Step 5: Run the live harness via the CLI against the target**

Set the target in `.env` (never committed):
```
COMPAT_NEO4J_URI=bolt://alpcirag01.ai.area51.pp.ua:7687
COMPAT_NEO4J_USER=neo4j
COMPAT_NEO4J_PASSWORD=<the password>
```

Run: `uv run --extra dev python -m compat.cli`
Expected: a rendered report written to `docs/superpowers/neo4j-compat-report.md`, ending in a verdict. Report the verdict verbatim.

- [ ] **Step 6: Fix any of OUR Cypher that the report flags**

Per spec section 8: failures in the `our-cypher` group are fixed in this slice. Failures inside graphiti-core are **not** patched — record them in the report's recommended actions instead. After any fix, re-run step 5 so the committed report reflects the final state.

- [ ] **Step 7: Amend the infrastructure sizing doc**

`docs/infrastructure-sizing.md` assumes index-backed vector search (its RAM estimate is built on "Neo4j wants the vector + full-text indexes resident in page cache"). Graphiti 0.29.2 does not create or use vector indexes. Append to the end of section 4 (`Extrapolation to the full corpus`):

```markdown
> **Correction (2026-09-02).** graphiti-core 0.29.2 creates **no vector indexes** and
> scores similarity with brute-force `vector.similarity.cosine` scans in Cypher
> (`search_ops.py:148-160`). The vector-index line in the disk table above is
> therefore not what the current code produces, and at 3-5 M facts every hybrid
> search would scan every fact embedding. This is a scaling blocker for broad
> ingestion, tracked as its own slice; see
> `docs/superpowers/specs/2026-09-02-neo4j-compat-check-design.md` section 9.
```

- [ ] **Step 8: Run the full gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`
Expected: all pass (the `@live` tests are excluded by default `addopts`).

- [ ] **Step 9: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_compat_harness.py \
        tests/integration/test_compat_live.py \
        docs/superpowers/neo4j-compat-report.md docs/infrastructure-sizing.md
git commit -m "test(compat): harness integration + live run; record compatibility verdict"
```

---

## Verification checklist

After Task 7, confirm every spec acceptance criterion:

1. `uv run --extra dev python -m compat.cli` runs all eight groups against the `COMPAT_*` target and writes `docs/superpowers/neo4j-compat-report.md` with a verdict, matrix, and recommended actions.
2. Harness writes are namespaced to `group_id='compat-check'`, harness indexes are `compat_`-prefixed, teardown is guaranteed and surfaced on failure.
3. A failing Cypher check is retried under `CYPHER 5` and the report distinguishes config-fix from code-fix.
4. Both bare `CALL {}` sites are scoped and the testcontainer is `neo4j:2026.07.1-community`; the full non-live suite passes.
5. The harness reports `GO` against the 2026.07 target.
6. Pure logic unit-tested; the harness loop testcontainer-tested; the `@live` run produced the report.
7. `uv run ruff check src tests` and `uv run mypy src` clean.
