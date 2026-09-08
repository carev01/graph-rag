# Answer Citation Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A citation marker visible in an answer always corresponds to a real citation in that answer's envelope — and the global reduce step stops writing prose that outruns its evidence.

**Architecture:** `_finalize_answer` in `synthesize.py` is shared by all three answering modes (local, global reduce, DRIFT), so the marker fix lands in one function and repairs all three. It gains one responsibility: remove unresolvable markers from the answer *text*, not just from the `cited` list. Separately, `_REDUCE_PROMPT` gains constraints binding the answer to its evidence.

**Tech Stack:** Python 3.12, `uv`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-answer-citation-integrity-design.md`

## Global Constraints

- **A visible `[N]` must always mean a real, graph-derived citation.** This extends design decision #2 ("citations are graph traversal; the LLM emits only markers, never a URL") from URLs to markers.
- **No range expander.** `[31]-[60]` is handled as two markers plus a separator. A model emitting a 30-marker span is guessing, not citing; expanding it would manufacture citations it never made.
- **Marker stripping CANNOT raise faithfulness on its own** — the same claims remain, with fewer visible markers. Any faithfulness movement in the eval comes from the prompt change (Task 2). Do not present it otherwise.
- **This runs on every answer the system produces**, not only broken ones. Whitespace repair must not add artefacts to currently-correct answers.
- **Do not strip line breaks or paragraph structure** — repair spaces and tabs only.
- URL stripping is unchanged and still runs first.
- `_build_citations` is unchanged; `cited` keeps its current semantics (ordered-unique markers that resolved).
- `drift.py`'s own prompt is NOT changed this slice — it inherits the shared marker fix, but changing two prompts at once would confound the eval comparison.
- CI gate, clean at every commit: `uv run ruff check src tests` (lints tests too: E702 no semicolons, E402 imports at top), `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.

## Verified current state (do not re-derive)

```python
# src/answer_api/synthesize.py:14-15
_URL_RE = re.compile(r"https?://[^\s\[\]]+", re.IGNORECASE)
_MARKER_RE = re.compile(r"\[(\d+)\]")

# src/answer_api/synthesize.py:29 -- filters `cited` but LEAVES markers in the text
def _finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]:
    text = _URL_RE.sub("", raw).strip()
    cited: list[int] = []
    for m in _MARKER_RE.findall(text):
        n = int(m)
        if n in marker_map and n not in cited:
            cited.append(n)
    return text, cited
```

Shared by `synthesize.py:90` (local), `global_search.py:192` (global reduce), `drift.py:161` (DRIFT).

`tests/unit/test_finalize_answer.py` already exists with 6 tests. They assert only on `cited`, never on the answer text, so **they will still pass** — but `test_keeps_valid_drops_invented_markers` must be strengthened to pin the new contract.

---

### Task 1: Strip unresolvable markers from the answer text

**Files:**
- Modify: `src/answer_api/synthesize.py:29-40`
- Test: `tests/unit/test_finalize_answer.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_strip_markers(text: str, keep: set[int]) -> str`; `_finalize_answer` keeps its existing signature `(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_finalize_answer.py`:

```python
def test_removes_invented_marker_from_the_text_not_just_the_list():
    """The defect this slice fixes: an unresolvable marker was filtered out of
    `cited` but LEFT IN THE PROSE, so readers saw citations the envelope did not
    have."""
    ans, cited = _finalize_answer("Immutable [1], and made-up [9].", MM)
    assert cited == [1]
    assert "[9]" not in ans
    assert "[1]" in ans


def test_range_with_neither_end_resolving_disappears_entirely():
    """Observed in a real global answer: '[31]-[60]'. Both markers and the
    separator must go, leaving no orphaned dash."""
    ans, cited = _finalize_answer("The reports do not compare them [31]-[60].", MM)
    assert cited == []
    assert "[31]" not in ans and "[60]" not in ans
    assert "-" not in ans


def test_range_with_one_end_resolving_keeps_that_end():
    ans, cited = _finalize_answer("Encryption differs [1]-[60].", MM)
    assert cited == [1]
    assert "[1]" in ans and "[60]" not in ans
    assert "-" not in ans


def test_legitimate_range_of_two_valid_markers_is_preserved():
    """Both ends resolve, so this is a real citation span -- leave it alone."""
    ans, cited = _finalize_answer("Both vendors encrypt [1]-[2].", MM)
    assert cited == [1, 2]
    assert "[1]-[2]" in ans


def test_no_whitespace_artefacts_after_removal():
    """This runs on EVERY answer, so sloppy removal would damage correct ones."""
    ans, _ = _finalize_answer("Immutability [9] is supported [1] .", MM)
    assert "  " not in ans
    assert " ." not in ans


def test_url_removal_no_longer_leaves_a_double_space():
    """Pre-existing wart the whitespace repair also fixes."""
    ans, cited = _finalize_answer("See https://evil/x for details [1].", MM)
    assert "  " not in ans and cited == [1]


def test_paragraph_structure_is_preserved():
    """Repair spaces and tabs only -- never line breaks."""
    ans, _ = _finalize_answer("First para [9].\n\nSecond para [1].", MM)
    assert "\n\n" in ans


def test_refusal_string_passes_through_untouched():
    refusal = "I don't have enough information to answer that from the available sources."
    ans, cited = _finalize_answer(refusal, MM)
    assert ans == refusal and cited == []


def test_marker_zero_is_stripped():
    """[0] is falsy -- guard against an `if n:` style membership bug."""
    ans, cited = _finalize_answer("Claimed [0] and real [1].", MM)
    assert cited == [1]
    assert "[0]" not in ans and "[1]" in ans


def test_url_and_unresolvable_markers_together():
    """URL stripping runs first, then marker stripping -- both must land."""
    ans, cited = _finalize_answer("See https://evil/x [9] but really [1].", MM)
    assert "http" not in ans
    assert "[9]" not in ans and "[1]" in ans
    assert "  " not in ans and cited == [1]
```

Then strengthen the existing test so it pins the text contract too — replace `test_keeps_valid_drops_invented_markers` with:

```python
def test_keeps_valid_drops_invented_markers():
    ans, cited = _finalize_answer(
        "Immutable [1], and cross-region [2], and made-up [9].", MM)
    assert cited == [1, 2]        # 9 not in marker_map -> dropped
    assert "[9]" not in ans       # ...and removed from the prose, not just the list
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_finalize_answer.py -q`
Expected: FAIL — several errors of the form `assert "[9]" not in ans`, because the current implementation leaves markers in the text.

- [ ] **Step 3: Implement the stripping helper**

In `src/answer_api/synthesize.py`, add below `_MARKER_RE`:

```python
# A hyphen / en-dash / em-dash with optional surrounding spaces -- the forms a
# model uses to write a marker RANGE like "[31]-[60]".
_SEP = r"[ \t]*[-–—][ \t]*"
_RANGE_RE = re.compile(r"\[(\d+)\]" + _SEP + r"\[(\d+)\]")


def _strip_markers(text: str, keep: set[int]) -> str:
    """Remove every [N] marker whose N is not in `keep`, then repair the spacing.

    Design decision #2 says a citation is graph-derived and the LLM only emits
    markers. Filtering an unresolvable marker out of the `cited` list but leaving
    it in the prose broke that: readers saw citations the envelope did not have.
    A visible marker must always resolve.

    Ranges get no expander on purpose. "[31]-[60]" is two markers and a
    separator; a model writing a 30-marker span is guessing, not citing, so
    expanding it would manufacture citations it never made. When both ends are
    dropped the separator goes too, or the prose is left with an orphaned dash.
    """
    def _range(m: re.Match[str]) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if a in keep and b in keep:
            return m.group(0)          # a real span -- leave it intact
        if a in keep:
            return f"[{a}]"
        if b in keep:
            return f"[{b}]"
        return ""                      # neither resolves: markers AND separator go

    text = _RANGE_RE.sub(_range, text)
    text = _MARKER_RE.sub(
        lambda m: m.group(0) if int(m.group(1)) in keep else "", text)
    # Repair spacing WITHOUT touching line breaks: this runs on every answer, and
    # paragraph structure is part of a correct one.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    return text.strip()
```

- [ ] **Step 4: Use it in `_finalize_answer`**

Replace the body of `_finalize_answer`:

```python
def _finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]:
    """Deterministic design-decision #2 enforcement: strip any URL the model
    emitted (it must never write one), keep the ordered-unique [N] markers that
    map to a retrieved fact, and remove the ones that do not from the TEXT as
    well -- a visible marker must always correspond to a real citation."""
    text = _URL_RE.sub("", raw).strip()
    cited: list[int] = []
    for m in _MARKER_RE.findall(text):
        n = int(m)
        if n in marker_map and n not in cited:
            cited.append(n)
    return _strip_markers(text, set(marker_map)), cited
```

Note `cited` is computed before stripping. That is safe — stripping only removes markers absent from `marker_map`, which were never eligible for `cited` — and it keeps the two concerns readable.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_finalize_answer.py -q`
Expected: PASS (16 passed — the 6 originals plus 10 new).

- [ ] **Step 6: Check the regression surface**

The same function backs local, global and DRIFT. Run their tests:

Run: `uv run --extra dev pytest tests/unit/test_global_map.py tests/unit/test_router_dispatch.py tests/unit/test_answer_api_app.py -q`
Expected: PASS. A failure here means the fix damaged a correct answer path — fix the source, do not relax the test.

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/answer_api/synthesize.py tests/unit/test_finalize_answer.py
git commit -m "fix(answers): strip unresolvable citation markers from the answer text"
```

---

### Task 2: Bound the reduce step to its evidence

**Files:**
- Modify: `src/answer_api/global_search.py:149-154`
- Test: `tests/unit/test_reduce_prompt.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent change, same slice).
- Produces: no signature changes. `_REDUCE_PROMPT` keeps its `{q}` and `{blocks}` format fields and its use of `_REFUSAL`.

**Context:** the traced failure produced a 3,516-character answer with 4 citations, opening with *"The community reports do not include a direct side-by-side comparison"*. The prompt currently permits that: it asks for citations but never forbids commentary about absent evidence, nor requires the answer to stop when the evidence does.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_reduce_prompt.py`:

```python
"""The global reduce prompt must bind the answer to its evidence.

A real run produced 3,516 characters of prose on 4 citations, opening with
meta-commentary about what the community reports did NOT contain. The prompt
permitted all of it. These tests pin the constraints that forbid it -- they are
prompt-shape tests, so the real verification is the eval re-run in Task 3.
"""
from answer_api.global_search import _REDUCE_PROMPT
from answer_api.synthesize import _REFUSAL


def test_still_has_its_format_fields():
    assert "{q}" in _REDUCE_PROMPT and "{blocks}" in _REDUCE_PROMPT


def test_forbids_commentary_about_absent_evidence():
    low = _REDUCE_PROMPT.lower()
    assert "do not" in low or "never" in low
    assert "not contain" in low or "absent" in low or "missing" in low


def test_requires_every_claim_to_carry_a_marker():
    low = _REDUCE_PROMPT.lower()
    assert "every claim" in low
    assert "[n]" in low


def test_offers_the_refusal_when_evidence_is_thin():
    assert _REFUSAL in _REDUCE_PROMPT


def test_still_forbids_urls():
    """Design decision #2 -- the model must never write a URL."""
    assert "url" in _REDUCE_PROMPT.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_reduce_prompt.py -q`
Expected: FAIL on `test_forbids_commentary_about_absent_evidence` — the current prompt says nothing about absent evidence.

- [ ] **Step 3: Rewrite the prompt**

In `src/answer_api/global_search.py`, replace `_REDUCE_PROMPT`:

```python
_REDUCE_PROMPT = (
    "Answer the QUESTION by synthesizing across these community findings, organized "
    "by theme and vendor.\n"
    "Rules:\n"
    "- Cite every claim with the [N] fact markers shown. A sentence with no marker "
    "is not allowed.\n"
    "- Use ONLY these findings. Do NOT use outside knowledge.\n"
    "- Do NOT write any URL.\n"
    "- Do NOT comment on what the findings do not contain, and do not explain what "
    "you cannot compare. Absence of evidence is not a finding.\n"
    "- Let the evidence set the length. Say what the findings support and then stop; "
    "do not pad, hedge, or restate.\n"
    "- If the findings do not support an answer, reply exactly: "
    "\"" + _REFUSAL + "\"\n\nQUESTION: {q}\n\nFINDINGS:\n{blocks}\n\nAnswer:"
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_reduce_prompt.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Confirm the global path still works**

Run: `uv run --extra dev pytest tests/unit/test_global_map.py tests/unit/test_global_rank.py -q`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, full non-live suite**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -q`
Expected: all pass. Run the suite in the FOREGROUND and let it finish (~10 minutes).

- [ ] **Step 7: Commit**

```bash
git add src/answer_api/global_search.py tests/unit/test_reduce_prompt.py
git commit -m "fix(global): bind the reduce step to its evidence"
```

---

### Task 3: Re-run the eval and report honestly

**Files:**
- Modify: `docs/superpowers/router-eval-report.md` (regenerated by the harness, then given an interpretive header)

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: no source interfaces.

**Context:** the 2026-09-08 baseline to beat, from the current report: routing **0.97**, grounding **0.69**, faithfulness **3.76**, and per-mode global faithfulness **1.4** / grounding **0.29** against local's 5.0. The corpus, community layer and judge are unchanged since that run, so global's numbers are a like-for-like comparison.

- [ ] **Step 1: Confirm the graph is unchanged from the baseline run**

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
        for lbl, cy in [('episodes', 'MATCH (n:Episodic) RETURN count(n) AS n'),
                        ('facts', 'MATCH ()-[f:RELATES_TO]->() RETURN count(f) AS n'),
                        ('communities', 'MATCH (c:Community) RETURN count(c) AS n')]:
            r = await sess.run(cy)
            print(f'  {lbl}:', [dict(x) async for x in r][0]['n'])
    await d.close()
asyncio.run(m())
"
```

Expected: `episodes: 385`, `facts: 2173`, `communities: 41`. If any differ, say so in the report — the comparison is then not like-for-like and must be labelled as such.

- [ ] **Step 2: Run the eval**

Run: `uv run --extra dev python -m answer_api.eval_router`

This makes 70+ LLM calls across four tiers and takes a while. Run it in the FOREGROUND; do not background it. It rewrites `docs/superpowers/router-eval-report.md`.

- [ ] **Step 3: Record the numbers**

From the regenerated report, capture: routing accuracy (overall and by intent), grounding precision (overall and by mode), faithfulness (overall and by mode), the comparative block, and `drift_wins`.

- [ ] **Step 4: Write the interpretive header**

Prepend a `>` blockquote header to `docs/superpowers/router-eval-report.md` covering, honestly:

- the comparison against the 2026-09-08 baseline (routing 0.97 / grounding 0.69 / faithfulness 3.76; global 1.4 / 0.29);
- whether global's faithfulness and grounding moved, stated plainly whichever way they went;
- **that marker stripping cannot raise faithfulness by itself** — the same claims remain with fewer visible markers — so any movement is attributable to the prompt change, and a flat result means the reduce step needs more than prompting;
- whether answers still contain markers absent from `citations` (they must not — that is Task 1's guarantee, and a violation is a bug, not a score).

Do not describe a disappointing result as a success. A null result here is a real finding: it says prompt-level constraints are insufficient and the next step is structural.

- [ ] **Step 5: Verify the Task 1 guarantee end to end**

Run:

```
uv run --extra dev python -c "
import json, re
from pathlib import Path
rep = Path('docs/superpowers/router-eval-report.md').read_text()
print('report regenerated:', 'Routing accuracy' in rep)
"
```

Then spot-check one global answer for orphan markers:

```
uv run --extra dev python -c "
import asyncio, re
from neo4j import AsyncGraphDatabase
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder
from answer_api.global_search import global_search, _map_client_and_model
from answer_api.synthesize import _synthesis_client_and_model
async def m():
    s = get_extract_settings()
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    emb = build_embedder(s); mc, mm = _map_client_and_model(s); sc, sm = _synthesis_client_and_model(s)
    try:
        out = await global_search(d, emb, mc, mm, sc, sm,
            q='Compare AWS Backup and Azure Backup database restore workflows.',
            level=s.global_default_level, k=s.global_shortlist_k,
            group_id=s.group_id, relevance_min=s.global_map_relevance_min)
        markers = {int(x) for x in re.findall(r'\[(\d+)\]', out['answer'])}
        cited = {c['marker'] for c in out['citations']}
        print('  answer len:', len(out['answer']))
        print('  markers in text:', sorted(markers))
        print('  markers in citations:', sorted(cited))
        print('  ORPHANS (must be empty):', sorted(markers - cited))
    finally:
        await emb.client.close(); await mc.close(); await sc.close(); await d.close()
asyncio.run(m())
"
```

Expected: `ORPHANS (must be empty): []`. A non-empty list means Task 1 is incomplete — report it as a bug.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/router-eval-report.md
git commit -m "docs(eval): re-run after the citation-integrity fix"
```

---

## Verification checklist

1. A marker visible in an answer always appears in that answer's `citations` (Task 1, Task 3 Step 5).
2. Unresolvable markers, including ranges, are removed from the text (Task 1).
3. No whitespace or punctuation artefacts, and paragraph structure survives (Task 1).
4. `_REDUCE_PROMPT` forbids meta-commentary and requires a marker per claim (Task 2).
5. Local and DRIFT behaviour is otherwise unchanged (Task 1 Step 6).
6. The eval is re-run and global's numbers reported against 1.4 / 0.29, honestly (Task 3).
7. Full non-live suite, ruff and mypy clean.
