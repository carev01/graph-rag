# Community Report Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A community-report finding is only written if a second model confirms it is supported by the facts that finding cites.

**Architecture:** Verification happens inside `generate_report` in `theme_builder/report.py`, the origin of the leak — every downstream hop (map, reduce, answer) was traced and is faithful, so fixing it here fixes the chain. One verify call per report returns per-finding verdicts; on violation the report is regenerated once with the violations named, then still-unsupported findings are dropped.

**Tech Stack:** Python 3.12, `uv`, pytest, OpenAI-compatible async clients.

**Spec:** `docs/superpowers/specs/2026-09-09-community-report-verification-design.md`

## Global Constraints

- **The verifier MUST NOT be the report model.** Resolve it independently and **raise** if it resolves to the report model — a model grading its own output inflates the result in a way the result cannot reveal. Same shape as `_eval_judge_client_and_model` in `answer_api/eval_router.py`.
- **An unusable verifier reply MUST NEVER read as "supported."** Empty `choices`, `None`/whitespace content, or unparseable JSON after retry ⇒ the report is **skipped**, previous report retained. This exact defect was fixed in the faithfulness judge one day earlier; do not reintroduce it.
- Configured verifier: **`~deepseek/deepseek-v4-flash-latest`** (already `eval_judge_model`). Report tier resolves to `z-ai/glm-5.3-flash` via `judge_model`.
- **Do NOT touch `answer_api/global_search.py`** — the map step is faithful; its prompt is correct. Do not touch `_REDUCE_PROMPT` or `drift.py` either.
- **Do NOT weaken `_PROMPT`'s existing instructions.** "Using ONLY the numbered FACTS" and "Do NOT use outside knowledge" stay exactly as they are; we are adding enforcement, not replacing the instruction.
- A genuinely-true-of-the-world claim absent from the cited facts is **UNSUPPORTED**. The observed leaks (35-day retention, RDS Multi-AZ, 1-second PITR precision) are all real AWS behaviour — that is precisely why they slipped through.
- CI gate, clean at every commit: `uv run ruff check src tests` (lints tests too: E702 no semicolons, E402 imports at top), `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.
- Run every command in the FOREGROUND. Never background the test suite.

## Verified current state (do not re-derive)

```python
# src/theme_builder/context.py:28
@dataclass
class ContextResult:
    text: str
    fact_uuids: set[str]

# src/theme_builder/report.py:114 — 10 existing call sites depend on this signature
async def generate_report(client: AsyncOpenAI, model: str,
                          context: ContextResult,
                          max_tokens: int = 16000) -> CommunityReport | None:

# src/theme_builder/report.py:36
@dataclass
class CommunityReport:
    title: str
    summary: str
    full_report: str          # the findings list, serialized to a JSON string
    rating: float
    rating_explanation: str
    tags: list[str]
    cited_fact_uuids: list[str]
```

`generate_report` is called at `theme_builder/cli.py:97` and `:171`, and faked in
`tests/integration/test_theme_cli.py`, `tests/integration/test_incremental_cli.py`,
`tests/unit/test_theme_report.py`, `tests/unit/test_llm_penalties.py`.

**Its return type must NOT change** — ten call sites depend on it, and stale test doubles
have already cost this project a silently-red branch. Counters travel via an optional
`stats` object instead.

---

### Task 1: Carry fact texts on `ContextResult`

**Files:**
- Modify: `src/theme_builder/context.py:28-30` and the return at the end of `assemble_context`
- Test: `tests/unit/test_theme_context.py`

**Interfaces:**
- Produces: `ContextResult.fact_texts: dict[str, str]` — uuid → fact text, for exactly the facts included in the context.

**Why:** the verifier needs each finding's fact TEXT. `ContextResult` currently carries only `fact_uuids`, and the texts exist only inside the rendered `text` blob.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_theme_context.py`:

```python
def test_fact_texts_maps_uuid_to_text_for_included_facts():
    """The verifier needs each fact's TEXT, not just its uuid."""
    from theme_builder.context import EntityRow, FactRow, assemble_context
    facts = [
        FactRow(uuid="u1", fact="AWS Backup provides continuous backups for Aurora.",
                valid_at=None, invalid_at=None, name="PROVIDES"),
        FactRow(uuid="u2", fact="AWS Backup integrates with AWS KMS.",
                valid_at=None, invalid_at=None, name="INTEGRATES_WITH"),
    ]
    members = [EntityRow(uuid="e1", name="AWS Backup", type="Product", summary="s", degree=2)]
    ctx = assemble_context(members, facts, top_entities=5, token_budget=1000)
    assert ctx.fact_texts == {
        "u1": "AWS Backup provides continuous backups for Aurora.",
        "u2": "AWS Backup integrates with AWS KMS.",
    }
    assert set(ctx.fact_texts) == ctx.fact_uuids


def test_fact_texts_excludes_facts_dropped_by_the_budget():
    """A fact cut for budget is not citable, so it must not appear in fact_texts."""
    from theme_builder.context import EntityRow, FactRow, assemble_context
    facts = [FactRow(uuid=f"u{i}", fact="x" * 200, valid_at=None, invalid_at=None,
                     name="N") for i in range(10)]
    members = [EntityRow(uuid="e1", name="E", type="Product", summary="s", degree=1)]
    ctx = assemble_context(members, facts, top_entities=1, token_budget=100)
    assert set(ctx.fact_texts) == ctx.fact_uuids
    assert len(ctx.fact_texts) < 10
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_theme_context.py -q`
Expected: FAIL — `AttributeError: 'ContextResult' object has no attribute 'fact_texts'`.

- [ ] **Step 3: Add the field**

In `src/theme_builder/context.py`, change the import line to include `field`:

```python
from dataclasses import dataclass, field
```

and the dataclass:

```python
@dataclass
class ContextResult:
    text: str
    fact_uuids: set[str]
    # uuid -> fact text, for exactly the facts included above. The report verifier
    # judges each finding against the text of the facts it cites; `text` renders
    # them into one blob, which is not machine-addressable. Defaulted so existing
    # constructions (several test fakes) keep working.
    fact_texts: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Populate it**

In `assemble_context`, add a `texts` dict beside the existing `included` set. Wherever
`included.add(f.uuid)` appears (there are TWO places — the clipped-fact branch and the
normal branch), add `texts[f.uuid] = f.fact` next to it. Store the FULL fact text even in
the clipped branch: the verifier should judge against the true fact, not a display
truncation. Then return:

```python
    return ContextResult(text="\n".join(lines), fact_uuids=included, fact_texts=texts)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_theme_context.py -q`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/theme_builder/context.py tests/unit/test_theme_context.py
git commit -F - <<'EOF'
feat(theme): carry fact texts on ContextResult

The report verifier judges each finding against the text of the facts it cites.
ContextResult carried only uuids; the texts existed solely inside the rendered
blob.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 2: Move `_usable_content` to a shared module

**Files:**
- Modify: `src/graph_extract/usage.py` (add the function)
- Modify: `src/answer_api/synthesize.py` (import it from there, keep the name importable)
- Test: `tests/unit/test_usable_content_shared.py`

**Interfaces:**
- Produces: `graph_extract.usage.usable_content(resp) -> str | None`
- `answer_api.synthesize._usable_content` remains importable and behaviourally identical.

**Why:** `theme_builder` must not import `answer_api` — that inverts the service layering. Both already depend on `graph_extract.usage` (for `instrument`). This is the fourth site needing the same guard, so it belongs in one shared place.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_usable_content_shared.py`:

```python
"""The empty-reply guard is shared by answer_api and theme_builder, so it lives in
graph_extract.usage. Four separate defects in this codebase came from an LLM reply
carrying no content being coerced into a legitimate value."""
from types import SimpleNamespace

from graph_extract.usage import usable_content


def _resp(content, *, choices=True):
    if not choices:
        return SimpleNamespace(choices=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_returns_text_for_a_normal_reply():
    assert usable_content(_resp("hello")) == "hello"


def test_none_for_empty_choices_list():
    assert usable_content(_resp(None, choices=False)) is None


def test_none_for_none_content():
    assert usable_content(_resp(None)) is None


def test_none_for_whitespace_only_content():
    assert usable_content(_resp("   \n ")) is None


def test_answer_api_still_exposes_it_under_the_old_name():
    from answer_api.synthesize import _usable_content
    assert _usable_content(_resp("hi")) == "hi"
    assert _usable_content(_resp(None, choices=False)) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_usable_content_shared.py -q`
Expected: FAIL — `ImportError: cannot import name 'usable_content' from 'graph_extract.usage'`.

- [ ] **Step 3: Move the function**

Read the current `_usable_content` in `src/answer_api/synthesize.py` and move its body to
`src/graph_extract/usage.py` as a public `usable_content`, preserving its docstring and
behaviour exactly:

```python
def usable_content(resp) -> str | None:
    """The text of a chat completion, or None when the model returned nothing.

    Two ways a HTTP 200 carries no answer: an empty `choices` list, and
    `content=None` because a reasoning model spent its whole max_tokens budget on
    reasoning tokens and emitted no final message. Both must stay distinguishable
    from a real reply — coercing them to "" has caused four separate defects here.
    """
    if not resp.choices:
        return None
    content = resp.choices[0].message.content
    return content if content and content.strip() else None
```

- [ ] **Step 4: Re-export from `synthesize.py`**

In `src/answer_api/synthesize.py`, delete the old definition and import it instead,
keeping the private alias so existing importers (including
`tests/unit/test_empty_answer_guard.py`) keep working:

```python
from graph_extract.usage import instrument, usable_content as _usable_content
```

Match the existing import style in that file — if `instrument` is already imported from
`graph_extract.usage`, extend that line rather than adding a second one.

- [ ] **Step 5: Run the tests**

Run: `uv run --extra dev pytest tests/unit/test_usable_content_shared.py tests/unit/test_empty_answer_guard.py -q`
Expected: PASS — the new file plus all existing empty-answer-guard tests.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/graph_extract/usage.py src/answer_api/synthesize.py tests/unit/test_usable_content_shared.py
git commit -F - <<'EOF'
refactor: move the empty-reply guard to graph_extract.usage

theme_builder needs the same guard and must not import answer_api. Both already
depend on graph_extract.usage. Re-exported as _usable_content so existing
importers are unaffected.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 3: Resolve the verifier tier, refusing the report model

**Files:**
- Modify: `src/graph_extract/config.py` (after the `report_llm_*` block near line 90)
- Modify: `src/theme_builder/report.py` (new function beside `_report_client_and_model`)
- Test: `tests/unit/test_verify_client.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ExtractSettings.verify_llm_base_url` / `verify_llm_model` / `verify_llm_api_key` (all `str = ""`), and `theme_builder.report._verify_client_and_model(settings) -> tuple[AsyncOpenAI, str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_verify_client.py`:

```python
"""The report verifier must never be the report model. A model grading its own
output inflates the result in a way the result cannot reveal — the same guard
answer_api.eval_router applies to the faithfulness judge."""
import pytest

from graph_extract.config import ExtractSettings
from theme_builder.report import _verify_client_and_model


def _settings(**kw):
    base = dict(_env_file=None, judge_base_url="https://judge.example/v1",
                judge_model="z-ai/glm-5.3-flash", judge_api_key="k")
    base.update(kw)
    return ExtractSettings(**base)


def test_uses_verify_llm_when_set():
    s = _settings(verify_llm_base_url="https://verify.example/v1",
                  verify_llm_model="~deepseek/deepseek-v4-flash-latest",
                  verify_llm_api_key="vk")
    _, model = _verify_client_and_model(s)
    assert model == "~deepseek/deepseek-v4-flash-latest"


def test_falls_back_to_eval_judge():
    s = _settings(eval_judge_base_url="https://verify.example/v1",
                  eval_judge_model="~deepseek/deepseek-v4-flash-latest",
                  eval_judge_api_key="vk")
    _, model = _verify_client_and_model(s)
    assert model == "~deepseek/deepseek-v4-flash-latest"


def test_raises_when_it_resolves_to_the_report_model():
    """The whole point of the guard: self-grading is silently worthless."""
    s = _settings(verify_llm_base_url="https://judge.example/v1",
                  verify_llm_model="z-ai/glm-5.3-flash")
    with pytest.raises(ValueError, match="report model"):
        _verify_client_and_model(s)


def test_raises_when_nothing_is_configured():
    with pytest.raises(ValueError, match="verif"):
        _verify_client_and_model(_settings())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_verify_client.py -q`
Expected: FAIL — `ImportError: cannot import name '_verify_client_and_model'`.

- [ ] **Step 3: Add the settings**

In `src/graph_extract/config.py`, after the `report_llm_*` fields:

```python
    # The report verifier. Falls back to eval_judge_*. MUST NOT resolve to the
    # report model -- theme_builder.report._verify_client_and_model raises if it
    # does, because a model grading its own findings inflates the result in a way
    # the result cannot reveal.
    verify_llm_base_url: str = ""
    verify_llm_model: str = ""
    verify_llm_api_key: str = ""
```

- [ ] **Step 4: Add the resolver**

In `src/theme_builder/report.py`, directly below `_report_client_and_model`:

```python
def _verify_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    """Resolve the report VERIFIER, independent of the report writer.

    Uses verify_llm_* when set, else eval_judge_*, and then refuses that result if
    it is the report model. A model checking its own findings for outside-knowledge
    leakage will not find any -- and the report gives no sign that the check was
    vacuous. This mirrors answer_api.eval_router._eval_judge_client_and_model.
    """
    base = settings.verify_llm_base_url or settings.eval_judge_base_url
    model = settings.verify_llm_model or settings.eval_judge_model
    key = (settings.verify_llm_api_key or settings.eval_judge_api_key
           or settings.judge_api_key or "not-needed")
    if not base or not model:
        raise ValueError(
            "No report verifier configured. Set VERIFY_LLM_BASE_URL / VERIFY_LLM_MODEL "
            "(or EVAL_JUDGE_*) in .env.")
    report_base = settings.report_llm_base_url or settings.judge_base_url
    report_model = settings.report_llm_model or settings.judge_model
    if base == report_base and model == report_model:
        raise ValueError(
            f"Report verifier resolves to the report model ({model!r}) -- it would "
            "check its own findings and confirm them. Set VERIFY_LLM_MODEL to a "
            "different model, ideally a different family so the two do not share "
            "failure modes.")
    return instrument(AsyncOpenAI(api_key=key, base_url=base,
                                  timeout=180.0, max_retries=3)), model
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_verify_client.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/graph_extract/config.py src/theme_builder/report.py tests/unit/test_verify_client.py
git commit -F - <<'EOF'
feat(theme): resolve a report verifier that refuses to be the report model

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 4: The verification call

**Files:**
- Modify: `src/theme_builder/report.py`
- Test: `tests/unit/test_verify_report.py`

**Interfaces:**
- Consumes: `graph_extract.usage.usable_content` (Task 2); `ContextResult.fact_texts` (Task 1).
- Produces:
  - `@dataclass class VerifyResult: unsupported: set[int]; summary_supported: bool` — `unsupported` holds **1-based** finding indices.
  - `async def verify_report(client, model: str, findings: list[dict], summary: str, fact_texts: dict[str, str], *, max_tokens: int = 4000) -> VerifyResult | None` — `None` means verification could not be completed.
  - `_VERIFY_PROMPT: str`

`findings` are the parsed dicts `{"finding": str, "fact_ids": [uuid, ...]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_verify_report.py`:

```python
"""The verifier decides whether a finding is supported by the facts it cites."""
import json
from types import SimpleNamespace

import pytest

from theme_builder.report import VerifyResult, verify_report

FACTS = {
    "u1": "AWS Backup provides continuous backups for Amazon Aurora.",
    "u2": "AWS Backup continuously backs up transaction logs for SAP HANA databases.",
}
FINDINGS = [
    {"finding": "AWS Backup provides continuous backups for Aurora.", "fact_ids": ["u1"]},
    {"finding": "PITR has 1-second precision up to 35 days.", "fact_ids": ["u2"]},
]


class _FakeClient:
    """Records the calls it received so tests can assert on retry behaviour."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

        async def _create(**kw):
            self.calls.append(kw)
            reply = self._replies.pop(0)
            if reply is None:                      # empty choices list
                return SimpleNamespace(choices=[])
            return SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=reply))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.mark.asyncio
async def test_parses_unsupported_indices_and_summary_verdict():
    payload = json.dumps({"unsupported": [2], "summary_supported": True})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "sum", FACTS)
    assert got == VerifyResult(unsupported={2}, summary_supported=True)


@pytest.mark.asyncio
async def test_all_supported_is_an_empty_set_not_none():
    """Empty set means 'checked, all fine'. None means 'could not check'. These
    must never be confused."""
    payload = json.dumps({"unsupported": [], "summary_supported": True})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "sum", FACTS)
    assert got is not None and got.unsupported == set()


@pytest.mark.asyncio
async def test_unsupported_summary_is_reported():
    payload = json.dumps({"unsupported": [], "summary_supported": False})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "sum", FACTS)
    assert got is not None and got.summary_supported is False


@pytest.mark.asyncio
async def test_retries_once_with_a_larger_budget_then_succeeds():
    payload = json.dumps({"unsupported": [1], "summary_supported": True})
    c = _FakeClient(["not json at all", payload])
    got = await verify_report(c, "m", FINDINGS, "sum", FACTS, max_tokens=1000)
    assert got == VerifyResult(unsupported={1}, summary_supported=True)
    assert len(c.calls) == 2
    assert c.calls[1]["max_tokens"] > c.calls[0]["max_tokens"]


@pytest.mark.asyncio
async def test_returns_none_when_both_attempts_are_unusable():
    """MUST be None, never a 'supported' verdict. An unusable reply reading as
    'supported' is exactly the defect fixed in the faithfulness judge."""
    got = await verify_report(_FakeClient(["", None]), "m", FINDINGS, "sum", FACTS)
    assert got is None


@pytest.mark.asyncio
async def test_empty_choices_list_does_not_raise():
    got = await verify_report(_FakeClient([None, None]), "m", FINDINGS, "sum", FACTS)
    assert got is None


@pytest.mark.asyncio
async def test_prompt_shows_each_finding_with_the_text_of_its_cited_facts():
    payload = json.dumps({"unsupported": [], "summary_supported": True})
    c = _FakeClient([payload])
    await verify_report(c, "m", FINDINGS, "the summary", FACTS)
    sent = c.calls[0]["messages"][0]["content"]
    assert "AWS Backup provides continuous backups for Amazon Aurora." in sent
    assert "transaction logs for SAP HANA" in sent
    assert "the summary" in sent
    assert "FINDING 1" in sent and "FINDING 2" in sent


@pytest.mark.asyncio
async def test_finding_citing_no_facts_is_shown_as_having_none():
    payload = json.dumps({"unsupported": [1], "summary_supported": True})
    c = _FakeClient([payload])
    await verify_report(c, "m", [{"finding": "x", "fact_ids": []}], "s", FACTS)
    assert "(no facts cited)" in c.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_garbage_indices_are_ignored_not_crashed_on():
    payload = json.dumps({"unsupported": ["2", "nope", None], "summary_supported": True})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "s", FACTS)
    assert got is not None and got.unsupported == {2}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_verify_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'VerifyResult'`.

- [ ] **Step 3: Implement**

In `src/theme_builder/report.py`, add the import at the top (extend the existing
`graph_extract.usage` import line):

```python
from graph_extract.usage import instrument, usable_content
```

Then:

```python
_VERIFY_PROMPT = (
    "You are checking a community report for claims its own evidence does not support.\n"
    "For each FINDING below, decide whether EVERY claim it makes is stated by the FACTS "
    "listed under it. A finding is UNSUPPORTED if it adds anything the facts do not "
    "state -- a number, a limit, a duration, a product name, a mechanism, or a "
    "requirement -- EVEN IF THAT CLAIM IS TRUE IN THE REAL WORLD. Judge only against "
    "the facts shown; outside knowledge is exactly what we are detecting.\n"
    "Also judge whether the SUMMARY is supported by the facts shown anywhere below.\n"
    "Respond with ONLY JSON: {{\"unsupported\": [finding numbers], "
    "\"summary_supported\": true or false}}\n\n"
    "SUMMARY: {summary}\n\n{blocks}"
)


@dataclass
class VerifyResult:
    unsupported: set[int]      # 1-based finding indices
    summary_supported: bool


async def verify_report(client: AsyncOpenAI, model: str, findings: list[dict],
                        summary: str, fact_texts: dict[str, str], *,
                        max_tokens: int = 4000) -> VerifyResult | None:
    """Check each finding against the text of the facts it cites.

    Returns None when verification could NOT be completed (unusable reply after a
    retry). None is not "supported": writing an unverified report because the
    verifier hiccuped is the defect this whole slice exists to prevent.
    """
    blocks = []
    for i, f in enumerate(findings, 1):
        texts = [fact_texts[u] for u in f.get("fact_ids", []) if u in fact_texts]
        listed = "\n".join(f"   - {t}" for t in texts) or "   (no facts cited)"
        blocks.append(f"FINDING {i}: {f.get('finding', '')}\nFACTS:\n{listed}")
    prompt = _VERIFY_PROMPT.format(summary=summary, blocks="\n\n".join(blocks))
    for budget in (max_tokens, max_tokens * 3):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=budget,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(usable_content(resp) or "")
        if obj is not None:
            raw = obj.get("unsupported") or []
            idx = {int(x) for x in raw
                   if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit())}
            return VerifyResult(unsupported=idx,
                                summary_supported=bool(obj.get("summary_supported", False)))
        logger.warning("report verifier returned no usable JSON (budget=%d); retrying",
                       budget)
    logger.warning("report verifier failed twice; report will be skipped")
    return None
```

If `logger` is not already defined in this module, add `logger = logging.getLogger(__name__)`
near the top with the matching `import logging`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_verify_report.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/theme_builder/report.py tests/unit/test_verify_report.py
git commit -F - <<'EOF'
feat(theme): verify each finding against the facts it cites

Returns None when verification could not be completed -- never a "supported"
verdict. An unusable reply reading as a pass is the defect this slice exists to
prevent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 5: Wire verification into `generate_report` (retry, then drop)

**Files:**
- Modify: `src/theme_builder/report.py` (`generate_report` and a small refactor)
- Test: `tests/unit/test_report_verification_flow.py`

**Interfaces:**
- Consumes: `verify_report`, `VerifyResult` (Task 4); `ContextResult.fact_texts` (Task 1).
- Produces:
  - `@dataclass class ReportStats: findings_dropped: int = 0; reverified: bool = False; unverified: bool = False; summary_unsupported: bool = False`
  - `generate_report(client, model, context, max_tokens=16000, *, verifier=None, stats=None) -> CommunityReport | None`
  - `verifier` is an async callable `(findings: list[dict], summary: str, fact_texts: dict[str, str]) -> VerifyResult | None`.

**The return type does not change** — ten call sites depend on `CommunityReport | None`.
`verifier=None` keeps today's behaviour exactly, so every existing test keeps passing
unchanged. Task 6 adds a test that the CLI really does pass one, so the feature cannot be
silently off in production.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_report_verification_flow.py`:

```python
"""generate_report's verify -> retry -> drop flow."""
import json
from types import SimpleNamespace

import pytest

from theme_builder.context import ContextResult
from theme_builder.report import ReportStats, VerifyResult, generate_report

GOOD = {"title": "T", "summary": "S", "rating": 5, "rating_explanation": "r",
        "tags": ["aws"],
        "full_report": [{"finding": "F1", "fact_ids": ["u1"]},
                        {"finding": "F2", "fact_ids": ["u2"]}]}


def _ctx():
    return ContextResult(text="ctx", fact_uuids={"u1", "u2"},
                         fact_texts={"u1": "fact one", "u2": "fact two"})


class _Client:
    def __init__(self, payloads):
        self._payloads = [json.dumps(p) if isinstance(p, dict) else p for p in payloads]
        self.calls = []

        async def _create(**kw):
            self.calls.append(kw)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._payloads.pop(0)))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def _verifier(*results):
    seq = list(results)
    calls = []

    async def _v(findings, summary, fact_texts):
        calls.append((findings, summary, fact_texts))
        return seq.pop(0)

    _v.calls = calls
    return _v


@pytest.mark.asyncio
async def test_supported_report_passes_through_with_one_verify_call():
    v = _verifier(VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None
    assert len(json.loads(rep.full_report)) == 2
    assert len(c.calls) == 1 and len(v.calls) == 1
    assert st == ReportStats()


@pytest.mark.asyncio
async def test_unsupported_finding_triggers_one_regeneration():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD, GOOD])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None and len(json.loads(rep.full_report)) == 2
    assert len(c.calls) == 2, "should regenerate exactly once"
    assert st.reverified is True and st.findings_dropped == 0


@pytest.mark.asyncio
async def test_finding_still_unsupported_after_retry_is_dropped():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported={2}, summary_supported=True))
    c = _Client([GOOD, GOOD])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None
    kept = json.loads(rep.full_report)
    assert [f["finding"] for f in kept] == ["F1"]
    assert st.findings_dropped == 1
    assert rep.cited_fact_uuids == ["u1"], "dropped finding's fact must not stay cited"


@pytest.mark.asyncio
async def test_unsupported_summary_skips_the_whole_report():
    """A one-paragraph summary cannot be partially salvaged, and map_report reads
    it, so an ungrounded summary would leak straight through."""
    v = _verifier(VerifyResult(unsupported=set(), summary_supported=False),
                  VerifyResult(unsupported=set(), summary_supported=False))
    st = ReportStats()
    rep = await generate_report(_Client([GOOD, GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None and st.summary_unsupported is True


@pytest.mark.asyncio
async def test_unverifiable_report_is_skipped_not_written():
    """None from the verifier means 'could not check'. It must NEVER be treated as
    supported."""
    v = _verifier(None)
    st = ReportStats()
    rep = await generate_report(_Client([GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None and st.unverified is True


@pytest.mark.asyncio
async def test_unverifiable_on_the_retry_is_also_skipped():
    v = _verifier(VerifyResult(unsupported={1}, summary_supported=True), None)
    st = ReportStats()
    rep = await generate_report(_Client([GOOD, GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None and st.unverified is True


@pytest.mark.asyncio
async def test_all_findings_dropped_skips_the_report():
    v = _verifier(VerifyResult(unsupported={1, 2}, summary_supported=True),
                  VerifyResult(unsupported={1, 2}, summary_supported=True))
    st = ReportStats()
    rep = await generate_report(_Client([GOOD, GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None, "an empty report must not be written"
    assert st.findings_dropped == 2


@pytest.mark.asyncio
async def test_no_verifier_keeps_todays_behaviour_exactly():
    """Ten existing call sites pass no verifier; they must be unaffected."""
    rep = await generate_report(_Client([GOOD]), "m", _ctx())
    assert rep is not None and len(json.loads(rep.full_report)) == 2


@pytest.mark.asyncio
async def test_report_with_no_findings_does_not_call_the_verifier():
    """Nothing to verify; spending a request on it is waste."""
    empty = dict(GOOD, full_report=[])
    v = _verifier()      # would IndexError if called
    rep = await generate_report(_Client([empty]), "m", _ctx(), verifier=v,
                                stats=ReportStats())
    assert rep is not None and json.loads(rep.full_report) == []
    assert v.calls == []


@pytest.mark.asyncio
async def test_the_retry_names_the_offending_findings():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD, GOOD])
    await generate_report(c, "m", _ctx(), verifier=v, stats=ReportStats())
    retry_prompt = c.calls[1]["messages"][0]["content"]
    assert "F2" in retry_prompt
    assert "F1" not in retry_prompt.split("NOT supported")[-1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_report_verification_flow.py -q`
Expected: FAIL — `ImportError: cannot import name 'ReportStats'`.

- [ ] **Step 3: Refactor generation and building into helpers**

In `src/theme_builder/report.py`, extract the two halves of today's `generate_report` so
the retry can reuse them. Move the existing 2-attempt generation loop into:

```python
async def _generate_once(client: AsyncOpenAI, model: str, prompt: str,
                         max_tokens: int) -> dict | None:
    """One LLM call plus one retry on an unparseable/contentless reply."""
    for _ in range(2):
        resp = await client.chat.completions.create(
            # GLM-5.2 is a reasoning model: a small cap truncates the JSON to empty
            # (finish_reason='length', content='') on larger communities -> parse
            # fail -> skipped report. 8000 gives the reasoning + report headroom
            # (measured: 3000 skipped ~24% of communities, 8000 skipped ~0). Same
            # lesson as answer_api/synthesize + the type_precision judge.
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        # A flaky provider can return HTTP 200 with an error payload and NO
        # choices; indexing [0] then raises and the caller drops the community
        # entirely. Observed 3/29 on the first real theme-build. usable_content
        # folds that in with contentless replies so both take the retry.
        obj = _extract_json(usable_content(resp) or "")
        if obj is not None:
            return obj
    return None
```

Both comments above are moved verbatim from today's code — they record real incidents, so
do not drop or reword them.

Extract the parse/validate/build half into `_build_report`, which is today's logic plus a
`findings` return value:

```python
def _build_report(obj: dict, context: ContextResult) -> tuple[CommunityReport, list[dict]]:
    """Parse one report payload. Returns the report AND its findings list, so the
    caller can verify the findings and rebuild the report with a subset."""
    raw_findings = obj.get("full_report")
    findings: list[dict] = []
    cited: list[str] = []
    if isinstance(raw_findings, list):
        for f in raw_findings:
            if not isinstance(f, dict):
                continue  # structurally-off finding -> skip, don't crash
            fids = [fid for fid in (f.get("fact_ids") or []) if fid in context.fact_uuids]
            for fid in fids:
                if fid not in cited:
                    cited.append(fid)
            # carry ONLY sanitized finding text + validated ids (drop unknown keys,
            # which could smuggle a model-authored URL past the strip)
            findings.append({"finding": _strip(str(f.get("finding", ""))), "fact_ids": fids})
    try:
        rating = float(obj.get("rating", 0) or 0)
    except (TypeError, ValueError):
        rating = 0.0
    report = CommunityReport(
        title=_strip(str(obj.get("title", ""))),
        summary=_strip(str(obj.get("summary", ""))),
        full_report=json.dumps(findings),
        rating=rating,
        rating_explanation=_strip(str(obj.get("rating_explanation", ""))),
        tags=[_strip(str(t)) for t in (obj.get("tags") or [])],
        cited_fact_uuids=cited)
    return report, findings
```

Add a helper to rebuild after dropping:

```python
def _with_findings(report: CommunityReport, findings: list[dict]) -> CommunityReport:
    """Same report carrying only `findings`, with cited_fact_uuids recomputed so a
    dropped finding's facts do not stay listed as cited."""
    cited: list[str] = []
    for f in findings:
        for fid in f["fact_ids"]:
            if fid not in cited:
                cited.append(fid)
    return replace(report, full_report=json.dumps(findings), cited_fact_uuids=cited)
```

Add `from dataclasses import dataclass, replace` to the imports.

- [ ] **Step 4: Add `ReportStats` and the flow**

```python
@dataclass
class ReportStats:
    """Out-parameter for counters the caller surfaces. generate_report's return type
    is load-bearing for ten call sites, so the counts travel separately."""
    findings_dropped: int = 0
    reverified: bool = False
    unverified: bool = False
    summary_unsupported: bool = False


_RETRY_NOTE = (
    "\n\nYour previous answer was REJECTED. These findings state things the facts "
    "you cited do not state, even if they are true in the real world:\n{offenders}\n"
    "Rewrite the report using ONLY what the facts state. Do not restore the "
    "rejected claims."
)


async def generate_report(client: AsyncOpenAI, model: str,
                          context: ContextResult,
                          max_tokens: int = 16000, *,
                          verifier=None,
                          stats: ReportStats | None = None) -> CommunityReport | None:
    """Generate one community report, optionally verified against its own facts.

    With `verifier` set: every finding is checked against the text of the facts it
    cites. On violation the report is regenerated ONCE with the offenders named;
    findings still unsupported are dropped. Returns None -- report skipped -- if the
    summary stays unsupported, if every finding is dropped, or if verification could
    not be completed. Without `verifier` the behaviour is exactly as before.
    """
    prompt = _PROMPT.format(context=context.text)
    obj = await _generate_once(client, model, prompt, max_tokens)
    if obj is None:
        return None
    report, findings = _build_report(obj, context)
    if verifier is None or not findings:
        # Nothing to verify: an empty report is already handled downstream by
        # writeback, and calling the verifier with no findings wastes a request.
        return report

    result = await verifier(findings, report.summary, context.fact_texts)
    if result is None:
        if stats is not None:
            stats.unverified = True
        return None

    if result.unsupported or not result.summary_supported:
        if stats is not None:
            stats.reverified = True
        offenders = "\n".join(
            f"- {findings[i - 1]['finding']}" for i in sorted(result.unsupported)
            if 1 <= i <= len(findings))
        if not result.summary_supported:
            offenders += "\n- The SUMMARY is not supported by the facts."
        retry_obj = await _generate_once(
            client, model, prompt + _RETRY_NOTE.format(offenders=offenders), max_tokens)
        if retry_obj is not None:
            report, findings = _build_report(retry_obj, context)
            result = await verifier(findings, report.summary, context.fact_texts)
            if result is None:
                if stats is not None:
                    stats.unverified = True
                return None

    if not result.summary_supported:
        if stats is not None:
            stats.summary_unsupported = True
        return None

    if result.unsupported:
        kept = [f for i, f in enumerate(findings, 1) if i not in result.unsupported]
        if stats is not None:
            stats.findings_dropped = len(findings) - len(kept)
        if not kept:
            return None
        report = _with_findings(report, kept)
    return report
```

- [ ] **Step 5: Run the new tests**

Run: `uv run --extra dev pytest tests/unit/test_report_verification_flow.py -q`
Expected: PASS (10 passed).

- [ ] **Step 6: Confirm the refactor broke nothing**

Run: `uv run --extra dev pytest tests/unit/test_theme_report.py tests/unit/test_llm_penalties.py -q`
Expected: PASS. These exercise `generate_report` with no verifier and must be unchanged.
If one fails, the refactor changed behaviour — fix the source, do not edit the test.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/theme_builder/report.py tests/unit/test_report_verification_flow.py
git commit -F - <<'EOF'
feat(theme): verify findings, retry once, then drop the unsupported ones

Return type unchanged -- ten call sites depend on CommunityReport | None, so the
counters travel on an optional ReportStats. verifier=None keeps today's behaviour
for every existing caller.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 6: Wire the CLI and surface the counters

**Files:**
- Modify: `src/theme_builder/cli.py` (both `generate_report` call sites, ~line 97 and ~line 171, and both result dicts, ~line 110 and ~line 189)
- Test: `tests/unit/test_theme_cli_verification.py`

**Interfaces:**
- Consumes: `_verify_client_and_model` (Task 3), `verify_report` (Task 4), `ReportStats` (Task 5).
- Produces: `findings_dropped`, `reports_reverified`, `reports_unverified` in both `theme-build` result dicts.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_theme_cli_verification.py`:

```python
"""The CLI must actually pass a verifier -- otherwise verification is silently off
in production while every unit test still passes."""
import inspect

from theme_builder import cli


def test_cli_passes_a_verifier_to_generate_report():
    src = inspect.getsource(cli)
    assert src.count("verifier=") >= 2, "both generate_report call sites must verify"
    assert "_verify_client_and_model" in src


def test_cli_reports_the_new_counters():
    src = inspect.getsource(cli)
    for key in ("findings_dropped", "reports_reverified", "reports_unverified"):
        assert key in src, f"{key} must reach the run summary"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_theme_cli_verification.py -q`
Expected: FAIL — `assert 0 >= 2`.

- [ ] **Step 3: Wire the CLI**

In `src/theme_builder/cli.py`:

1. Extend the import:
```python
from theme_builder.report import (ReportStats, _report_client_and_model,
                                  _verify_client_and_model, generate_report,
                                  verify_report)
```

2. Where the report client is built, also build the verifier client, and define a small
adapter that matches the `verifier` callable shape:

```python
    vclient, vmodel = _verify_client_and_model(settings)

    async def _verifier(findings, summary, fact_texts):
        return await verify_report(vclient, vmodel, findings, summary, fact_texts)
```

3. At BOTH `generate_report` call sites, pass a fresh `ReportStats` and accumulate:

```python
                st = ReportStats()
                rep = await generate_report(client, model, ctx,
                                            settings.report_max_tokens,
                                            verifier=_verifier, stats=st)
                findings_dropped += st.findings_dropped
                reverified += 1 if st.reverified else 0
                unverified += 1 if st.unverified else 0
```

Initialise `findings_dropped = reverified = unverified = 0` beside the existing counters
(`skipped`, `regenerated`, `reused`) in each function.

4. Add the counters to BOTH result dicts:

```python
                    "findings_dropped": findings_dropped,
                    "reports_reverified": reverified,
                    "reports_unverified": unverified,
```

Close `vclient` wherever the existing report `client` is closed, in the same `finally`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_theme_cli_verification.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Update the integration fakes honestly**

`tests/integration/test_theme_cli.py` and `tests/integration/test_incremental_cli.py`
fake `generate_report`. Their fakes must accept the new keyword arguments
(`verifier=None, stats=None`) or the calls will raise `TypeError`. Update the fake
signatures only — do NOT weaken any assertion. These same fakes went stale once already
and left the branch silently red.

Also monkeypatch `_verify_client_and_model` in those tests so they do not try to build a
real client from test settings.

Run: `uv run --extra dev pytest tests/integration/test_theme_cli.py tests/integration/test_incremental_cli.py -q`
Expected: PASS (7 passed).

- [ ] **Step 6: Full suite, lint, type-check**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live" -q`
Expected: all pass. Run the suite in the FOREGROUND (~11 minutes) and report the real numbers.

- [ ] **Step 7: Commit**

```bash
git add src/theme_builder/cli.py tests/
git commit -F - <<'EOF'
feat(theme): wire verification into theme-build and surface its counters

findings_dropped / reports_reverified / reports_unverified are part of the
deliverable: a per-item handler that hides its failure count has already cost
this project weeks.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 7: Regenerate the reports and validate against the traced leaks

**Files:**
- Modify: `docs/superpowers/map-step-trace-2026-09-09.md` (append a "post-fix" section)

**Interfaces:**
- Consumes: everything above.

**Context:** the 41 reports currently in the graph were written before verification and
are contaminated. Verification runs only where a report is *generated*, so a full rebuild
is REQUIRED to close this slice — it is not optional cleanup.

- [ ] **Step 1: Confirm the graph is the one the trace was taken against**

Run:

```
uv run --extra dev python -c "
import asyncio
from neo4j import AsyncGraphDatabase
from graph_extract.config import get_extract_settings
async def m():
    s = get_extract_settings()
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    async with d.session() as sess:
        for lbl, cy in [('episodes','MATCH (n:Episodic) RETURN count(n) AS n'),
                        ('facts','MATCH ()-[f:RELATES_TO]->() RETURN count(f) AS n'),
                        ('communities','MATCH (c:Community) RETURN count(c) AS n')]:
            r = await sess.run(cy)
            print(' ', lbl, [dict(x) async for x in r][0]['n'])
    await d.close()
asyncio.run(m())
"
```

Expected: `episodes 385`, `facts 2173`, `communities 41`. If they differ, say so — the
before/after comparison is then not like-for-like and must be labelled as such.

- [ ] **Step 2: Regenerate every report**

Run: `uv run --extra dev python -m theme_builder.cli theme-build --full`

Run it in the FOREGROUND. Record the full summary dict, especially `findings_dropped`,
`reports_reverified`, `reports_unverified`, `reports_written` and `reports_skipped`.

- [ ] **Step 3: Check the four traced inventions are gone**

Run:

```
uv run --extra dev python -c "
import asyncio
from neo4j import AsyncGraphDatabase
from graph_extract.config import get_extract_settings
PROBES = ['1-second','35 day','1 to 35','Multi-AZ','same organization','from incremental backups']
async def m():
    s = get_extract_settings()
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    async with d.session() as sess:
        r = await sess.run('MATCH (c:Community {group_id:\$g}) RETURN c.title AS t, '
                           'coalesce(c.summary,\"\") + coalesce(c.full_report,\"\") AS body', g=s.group_id)
        rows = [dict(x) async for x in r]
    await d.close()
    print('communities:', len(rows))
    for p in PROBES:
        hits = [x['t'][:60] for x in rows if p.lower() in x['body'].lower()]
        print(f'  {p!r:26} {len(hits)} report(s)', hits[:2])
asyncio.run(m())
"
```

Expected: **0 reports** for the invented probes. Any surviving hit must be checked against
that community's facts by hand before being called a false alarm — the claim may be
legitimately present in a fact.

- [ ] **Step 4: Re-run the eval**

Run: `uv run --extra dev python -m answer_api.eval_router`

Foreground; ~20 minutes of live calls. Record routing, grounding, faithfulness by mode,
and `unscored`.

- [ ] **Step 5: Append the post-fix section to the trace document**

Append a `## Post-fix (2026-09-09)` section to
`docs/superpowers/map-step-trace-2026-09-09.md` recording, honestly:

- the rebuild counters, with `findings_dropped` as the headline number;
- whether the four traced inventions are gone;
- global faithfulness against the **1.6** baseline, stated plainly whichever way it moved;
- **if reports thinned out sharply, say so and name the implication**: the community layer
  was largely embellishment and the real problem is upstream extraction coverage, which is
  a different slice. Per spec §10, honest-but-thinner reports are the correct outcome even
  if answers get less useful. Do not present a drop in usefulness as a failure of this
  slice, and do not present a flat faithfulness number as a success.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/map-step-trace-2026-09-09.md
git commit -F - <<'EOF'
docs: post-fix validation of community report verification

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

## Verification checklist

1. No finding survives that the verifier judged unsupported by its own cited facts (Tasks 4, 5).
2. The four traced inventions are absent from the regenerated reports (Task 7 Step 3).
3. An unusable verifier reply never results in a written report (Tasks 4, 5).
4. The verifier cannot be the report model (Task 3).
5. `findings_dropped`, `reports_reverified`, `reports_unverified` visible in the summary (Task 6).
6. The eval is re-run and global faithfulness reported against 1.6, whatever it shows (Task 7).
7. Full non-live suite, ruff and mypy clean (Task 6 Step 6).
