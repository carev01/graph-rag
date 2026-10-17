# Answer Precision and List Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** an enumeration counts a vendor only when its evidence really answers the question, and list answers are shorter and faster without losing items.

**Architecture:**
- **Enumeration:** a deterministic relevance floor on each documented vendor's best supporting fact (cross-encoder rerank against the question), plus a stricter cheap-tier check prompt.
- **List mode:** an item-line cap with merging of versions and editions, in the list and completeness prompts.

**Tech Stack:** Python 3.12, FastAPI answer-api, Voyage rerank, uv.

**Spec:** approved in conversation 2026-10-17 (parked items 2 and 3). Measurements:
- **Enumeration:** in run 12, Cohesity was "documented" for "verify that backups can be restored". Its best supporting fact scored 0.448 against the question. Every genuinely documented vendor's best fact scored 0.633–0.824 across the three enumeration questions.
- **List latency:** list answers take 63–351 s. Retrieval is 34–69 s; synthesis of 15–20k-character answers dominates (e.g. four separate "ESXi version" lines).
- **Dropped:** the "overlap the completeness call" idea. The call needs the finished answer, so there is nothing to overlap.

## Global Constraints

- A failure never reads as absence: an unavailable or failed rerank keeps the cheap-tier verdict, records "relevance check unavailable" in `degraded`, and logs a WARNING. It never drops vendors.
- Fixed-wording notes only; the LLM never writes a URL.
- Never write `content or ""`; hermetic unit tests (`ExtractSettings(_env_file=None, ...)` inside tests only); lint `uv run ruff check src tests` (no E702, E402); `uv run mypy src`.

---

### Task 1: Enumeration relevance floor

**Files:** `src/answer_api/enumeration.py`, `src/graph_extract/config.py`; tests `tests/unit/test_enumeration*.py` (find with `grep -l enumerate_vendors tests/unit`), `tests/unit/test_config.py`.

1. Setting `enumeration_min_score: float = 0.55`. Comment: a vendor counts as documented only when its best supporting fact scores at least this against the question; measured 2026-10-17, the false positive's best was 0.448, genuine vendors' best 0.633–0.824.
2. In `enumerate_vendors`, after `supported` is final (after the renderable-text step) and before `documented` is computed, run one rerank per documented vendor, concurrently:
   - call `rerank(q, [texts[i] for i in ids], top_k=len(ids), settings=settings)`;
   - take the best score;
   - when best < `enumeration_min_score`, remove the vendor from `supported`, so it reads as checked and not documented, and log INFO `"enumeration: %s evidence too weak (best %.3f < %.2f); not documented"`.

   With no reranker configured (`rerank_configured(settings)` false), or when a call raises or returns None, keep that vendor and record the vendor-independent degraded entry `"relevance check unavailable"` once in the envelope. It goes in `routing.degraded` via the existing raw `degraded` field, if enumeration exposes one; otherwise add `relevance_unchecked: [vendors]` to the raw envelope. Also log a WARNING. Scores are recorded in the raw envelope as `vendor_scores: {vendor: best}`.
3. Tighten `_CHECK_PROMPT`: after "Use ONLY the evidence.", add `"Mark it documented only when the evidence states the capability the question asks about itself -- a related or adjacent feature does not count. "`.

Tests:
- A vendor whose best score is below the floor is not documented and appears in "Checked:".
- One above the floor stays documented.
- Rerank unavailable keeps both and records the degraded entry.
- A rerank exception for one vendor keeps that vendor.
- The prompt contains the new sentence.
- The setting defaults to 0.55.

- [ ] Tests first; implement; unit suite, ruff, mypy; commit `fix(answer): enumeration counts a vendor only when its best evidence is relevant` (trailer lines).

### Task 2: List item cap and version merging

**Files:** `src/answer_api/list_mode.py`, `src/graph_extract/config.py`; tests `tests/unit/test_list_mode.py`, `tests/unit/test_config.py`.

1. Setting `list_max_items: int = 120`.
2. Add a shared rule constant `_MERGE_RULE` and use it in both `_LIST_PROMPT` and `_COMPLETE_PROMPT`, after the subject rule. Its text:

   `"Give at most {max_items} item lines in total. When several items are versions, editions or minor variants of one thing (for example several versions of one hypervisor), give them ONE line naming them together, with up to three markers. "`

   `{max_items}` is filled from `settings.list_max_items` wherever the prompts are formatted. Escape other braces as the existing format calls require.
3. Reduce the first list synthesis `max_tokens` from 10000 to 8000. The truncation retry at 16000 stays.

Tests:
- Both prompts contain the merge rule with the configured number.
- The first synthesis call uses 8000.
- The settings default is 120.

- [ ] Tests first; implement; unit suite, ruff, mypy; commit `perf(answer): list answers merge versions and cap item lines` (trailer lines).

---

### Task 3: Completeness-pass precision (found in paid run 15)

**Measured:** run 15's "Where can I restore a Windows agent backup?" answer was 31k characters, 24k of them in the completeness pass's "Also documented" section. It included generic facts from the Linux and AIX agent guides ("Veeam Agent supports restoring volumes to the hdisk1 disk", "+data02", "datalv"), renamed as "Veeam Agent for Microsoft Windows (…)". The grader flagged them "needs validation". The attribution judge counted 22 misattributions.

**Files:** `src/answer_api/list_mode.py`, `src/answer_api/item_grades.py`, `src/graph_extract/config.py`; tests `tests/unit/test_list_mode.py`, `tests/unit/test_item_grades.py`, `tests/unit/test_config.py`.

1. **Setting:** `list_completeness_max_items: int = 15`. `_completion_entries` keeps at most that many entries, in reply order.
2. **`_COMPLETE_PROMPT`:** add `"Name each item exactly as its fact names it; never add the question's subject (or any product, platform or operating system the fact does not name) to an item's name. Return only items that are themselves an answer to the question, not details of how an already-listed item works. "`. Test that the sentence is present.
3. **Stricter grading for added items:** `grade_items` gains `strict_from: str | None` (the default `None` changes nothing). `answer_list` passes `"Also documented"`. Every item line after a heading whose text contains `strict_from`, and before the limitations heading, must grade CONFIRMED; a line that would be NEEDS_VALIDATION becomes DROPPED with reason `"completion_unconfirmed"`. The heading-cleanup rule then removes the "Also documented" heading if every item under it was dropped.

Tests:
- A needs-validation line under "### Also documented" is dropped with the new reason.
- A confirmed line there stays.
- The same needs-validation line above the heading stays flagged.
- The cap keeps at most 15.
- Settings default 15.

- [ ] Tests first; implement; unit suite, ruff, mypy; commit `fix(answer): completeness-pass additions must be confirmed, capped and named as the facts name them` (trailer lines).
