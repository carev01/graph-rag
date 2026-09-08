# Answer Citation Integrity — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Stop answers displaying citations that resolve to nothing, and stop the
global reduce step writing prose that outruns its evidence.
**Date:** 2026-09-08
**Status:** Approved design — ready for implementation planning

Arises from the 2026-09-08 router golden-set eval
(`docs/superpowers/router-eval-report.md`), where global mode scored faithfulness
**1.4** and grounding **0.29** against local's 5.0.

This is **Slice A of two**. Slice B (out-of-range dedup indices in the ingest path)
is deliberately separate — see §8.

---

## 1. The two defects

### Defect A — answers display citations that lead nowhere

`_finalize_answer` (`src/answer_api/synthesize.py:29`) enforces design decision #2:
strip any URL the model emitted, then keep the ordered-unique `[N]` markers that map
to a retrieved fact. It filters unresolvable markers out of the returned `cited`
list — but **leaves them in the answer text**.

So a reader sees markers the envelope has no citation for. In the traced example the
answer opened with `... [31]-[60]` while `citations` held **4** entries. The answer
*looks* cited and is not, which inverts the guarantee decision #2 exists to provide:
a visible marker is supposed to mean a real, graph-derived source.

`_MARKER_RE = r"\[(\d+)\]"` also matches only single markers, so a range like
`[31]-[60]` is seen as two markers and a dash.

### Defect B — the reduce step outruns its evidence

`_REDUCE_PROMPT` (`src/answer_api/global_search.py:149`) asks the model to "Cite every
claim with the [N] fact markers shown. Use ONLY these findings."

Traced for *"Compare AWS Backup and Azure Backup database restore workflows"*: the
shortlist returned 10 communities, the map step kept 2 (relevance 6 and 3) exposing
**28 facts**, and the reduce step produced a **3,516-character answer with 4 resolved
citations**, opening with meta-commentary — *"The community reports do not include a
direct side-by-side comparison"* — and padding well past its evidence.

**The pipeline is healthy.** Shortlist, map and fact-surfacing all work; the defect is
in the reduce output. The judge scoring it 0 is correct behaviour, not a judge problem.

## 2. Why one slice fixes three modes

`_finalize_answer` and `_build_citations` are shared by every answering path:

| Mode | Call site |
|---|---|
| local | `synthesize.py:90` |
| global reduce | `global_search.py:192` |
| DRIFT synthesis | `drift.py:161` |

So the marker fix lands in one function and repairs all three. That shared use is the
reason Defects A and B belong together and the ingest-path work does not.

## 3. Fix for Defect A — markers that resolve, or no marker at all

`_finalize_answer` gains one responsibility: **remove unresolvable markers from the
text as well as from the `cited` list.**

Rules:

1. A marker `[N]` survives iff `N in marker_map`. Otherwise it is removed from the
   prose.
2. **Ranges need no special case.** `[31]-[60]` is two markers plus a separator; if
   neither resolves, both are stripped and the orphaned separator is cleaned up with
   them. A dedicated range expander was considered and rejected: a model emitting a
   30-marker span is guessing, not citing, so expanding it would legitimise the
   pattern and manufacture citations the model never really made.
3. **Whitespace must stay clean.** Removing `[31]` mid-sentence can leave a double
   space, a space before punctuation, or a dangling separator. This runs on *every*
   answer the system produces, not only broken ones, so sloppy removal would add
   visible artefacts to currently-correct answers. Specifically: collapse runs of two
   or more spaces to one; remove a space immediately before `.,;:!?`; and where a
   marker was removed from each side of a separator, remove the separator too. A
   "separator" here means a hyphen, en-dash or em-dash (`-`, `–`, `—`) with optional
   surrounding spaces — the forms a model uses to write a marker range. Do NOT strip
   line breaks or paragraph structure.
4. URL stripping is unchanged and still runs first.
5. The returned `cited` list keeps its current semantics: ordered-unique markers that
   resolved. `_build_citations` is unchanged.

**This cannot raise faithfulness by itself.** An answer with fewer visible markers
makes exactly the same claims. Defect A is an integrity fix — it stops the system
misrepresenting how well-grounded an answer is. Any faithfulness movement in the
re-run comes from §4.

## 4. Fix for Defect B — bound the answer to its evidence

`_REDUCE_PROMPT` gains three constraints:

- **Write only what the findings support.** No claim without a `[N]` marker.
- **Never comment on what the reports do not contain.** The traced failure opened with
  precisely this and then padded. Absence of evidence is not a finding.
- **When evidence is thin, say so briefly or refuse.** The `_REFUSAL` string itself is
  **unchanged** — what broadens is when the prompt tells the model to use it: today
  only for "nothing relevant", now also for "not enough to answer well". The
  alternative the model currently takes is hedged bulk, which scores worse and misleads
  more than an honest refusal.

No explicit length cap. A fixed budget is wrong in both directions — too tight when
evidence is rich, still too loose when it is thin — and length is a symptom of
unevidenced claims rather than the disease.

The DRIFT synthesis prompt (`drift.py`) is left alone this slice. It shares
`_finalize_answer` and so gets Defect A's fix, but its prompt is a separate surface
and changing two prompts at once would confound the eval comparison.

## 5. Components

| File | Change |
|---|---|
| `src/answer_api/synthesize.py` | `_finalize_answer` strips unresolvable markers from the text; a small `_strip_markers` helper handles removal plus whitespace repair. |
| `src/answer_api/global_search.py` | `_REDUCE_PROMPT` gains the §4 constraints. |
| `src/answer_api/drift.py` | none (inherits the shared fix). |
| `src/answer_api/router.py` | none. |

## 6. Error handling and edge cases

| Case | Behaviour |
|---|---|
| Answer with no markers at all | Unchanged: text returned as-is, `cited` empty. |
| Every marker unresolvable | All stripped; prose remains, `cited` empty. Reads as uncited, which is the truth. |
| Marker range `[31]-[60]`, neither end resolving | Both markers and the separator removed. |
| Range where one end resolves | The resolving marker stays, the other and the separator go. Result reads as a single citation. |
| Marker `[0]` or a huge number | Not in `marker_map`; stripped. |
| Marker inside a code block | Stripped like any other. Accepted: answers are prose, and a false citation is worse than an edited code sample. |
| Model emits a URL **and** bad markers | URL stripped first (unchanged), then markers. |
| Refusal string returned | No markers present; passes through untouched. |

## 7. Testing

- **Unit (`_finalize_answer`), the core of this slice:** a resolving marker survives in
  both text and list; a non-resolving one is removed from both; a range with neither
  end resolving disappears entirely; a range with one end resolving keeps that end;
  whitespace and punctuation stay clean after removal (no doubled spaces, no space
  before a full stop); URL stripping still works; an answer with no markers is
  unchanged; the refusal string is untouched.
- **Regression:** the existing local and DRIFT tests must still pass unchanged — they
  exercise the same function and are the guard against this fix breaking correct
  answers.
- **End-to-end:** re-run the router golden-set eval and compare global's faithfulness
  and grounding against today's **1.4 / 0.29**. Report the numbers whatever they are.
- Full non-live suite, `uv run ruff check src tests`, `uv run mypy src` clean.

## 8. Deferred to Slice B

**Out-of-range dedup indices.** graphiti logs
`LLM returned invalid duplicate_facts idx values [10,11,14,15] (valid range: 0-9)` at
`edge_operations.py:735` and drops them, so dedup silently misses duplicates and
duplicate facts survive. Our `maxItems` bound caps array *length*, not element
*values*, so it cannot prevent this.

Verified feasible: it is a real `logger.warning` on
`graphiti_core.utils.maintenance.edge_operations` with structured args, so a handler
attached during `ingest_article` can count occurrences per article; and `_tier_for`
(`ingest_driver.py:53`) already selects a tier per article, so "retry this article on
the strong tier" fits the existing shape.

Separate because it is a different subsystem (ingest, not answer), a different test
surface, and validating it requires a re-ingest cycle — which would delay the global
fix that this slice delivers.

## 9. Acceptance criteria

1. A marker left visible in an answer always corresponds to an entry in that answer's
   `citations`.
2. Unresolvable markers, including ranges, are removed from the answer text.
3. Marker removal leaves no whitespace or punctuation artefacts.
4. `_REDUCE_PROMPT` forbids meta-commentary about absent evidence and requires every
   claim to carry a marker.
5. Existing local and DRIFT behaviour is unchanged except for marker stripping.
6. The eval is re-run and global's faithfulness/grounding reported against 1.4 / 0.29.
7. Full non-live suite, ruff and mypy clean.
