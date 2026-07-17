# ling-2.6-flash Viability — Deep Investigation (2026-07-17)

**Question:** the first ling eval was a NO-GO (1 fact on the matrix article,
prompt-echo garbage, 2× latency). Are those issues fixable, or a dead end?
**Answer: fixable-with-tuning, not a drop-in swap.** Two of three root causes are
diagnosed and one is already fixed (committed); the third (over-enumeration →
truncation) is constrainable by prompting + chunk-sizing but not eliminable, and
ling's end-to-end quality/cost is still unproven.

---

## Method matters: Layer-0 vs the real harness

A raw-schema smoke test (my own prompt) had ling extracting cleanly. The failures
only appear inside **graphiti's actual multi-call pipeline**. So the question was
never "can ling extract" — it's "does ling survive graphiti's harness." That
reframing is what made the root-causing productive.

## Root cause 1 — Missing subject → dropped edges (FIXED, committed)

graphiti extracts nodes and edges in separate calls and **drops any edge whose
endpoint isn't in the node set**. ling reliably extracted *features* but omitted
the *subject* (`AWS Backup`), so every "AWS Backup provides X" edge was discarded.

Diagnostic (matrix article, per chunk): chunk with `AWS Backup` node → 9 facts
survived; chunk without it → **13 edges produced, all 13 dropped**.

**Fix:** a one-paragraph instruction — "always extract the backup product as an
entity in every chunk." A/B on 6 chunks:

| | facts survived | edges dropped | subject in node list |
|---|---|---|---|
| baseline | 20 | 234 | 0/6 |
| **nudged** | **312** | 143 | **6/6** |

Validated on gpt-5-mini (production tier): no regression (161 vs 164 facts, 4/4
subject, 0 drops). **Committed to `EXTRACTION_INSTRUCTIONS` (`ef46361`) — a
universal win independent of the ling decision.**

## Root cause 2 — Table over-enumeration → 16k truncation → parse failure (the big one)

This is the `finish_reason: length` the user spotted in OpenRouter's logs.
graphiti caps completions at `DEFAULT_MAX_TOKENS = 16384`. On dense tables ling
tries to enumerate **every cell** (product × region × resource × feature), so its
output grows **super-linearly** with input and blows the cap:

| input chars | max response chars | nodes/edges | LLM calls | truncated |
|---|---|---|---|---|
| 400 | 1.7k | 8 / 0 | 3 | 0 |
| 1,200 | 13.4k | 53 / 54 | 4 | 0 |
| 3,000 | 29–47k | 79–81 / 107–225 | 7–102 | 0–3 |
| ~7,000 (real chunk) | >60k | — | — | **yes → JSON parse fails** |

At real chunk sizes (`max_chunk_tokens=1800` ≈ 7k chars) the response truncates
mid-JSON (`Unterminated string … char 62991`), graphiti's parse fails
(`Error in generating LLM response`), it **retries**, and a dense-article corpus
turns into a retry storm — the full 6-article eval **timed out at 1 hour** without
finishing the ling phase.

## Root cause 3 — dedup/resolution call explosion (cost/latency killer)

Because ling emits so many entities/edges, graphiti makes a resolution/dedup call
**per item**: a single 3,000-char dense chunk produced **102 LLM calls** (3 real
extraction calls + ~99 tiny dedup calls). This is why "cheap per token" is
deceptive — the *number of calls* explodes on exactly the high-value dense docs.

## The lever that helps — a salience/anti-enumeration instruction

Adding "be selective, don't enumerate every table cell, use canonical names,"
same dense 3,000-char snippet:

| | nodes | edges | calls | max response | truncated | names |
|---|---|---|---|---|---|---|
| current | 81 | 107 | 7 | 47k | 3 | ok |
| **+salience** | 41 | 83 | 3 | 26k | **0** | **canonical** (`AWS Backup`, `Amazon S3`, `Amazon EBS`) |

Salience roughly **halved output, cut calls 7→3, eliminated truncation**, and
produced cleaner canonical entity names. It helps materially — but 41 nodes / 83
edges from 3k chars is still heavier than gpt-5-mini, and 26k-char responses would
still risk the cap at full 7k-char chunks. So salience must be **combined with
smaller chunks**.

## Verdict

**Not a dead end — but not a free swap.** ling's failures are integration- and
discipline-shaped, and three independent, cheap levers each measurably help:

1. **Subject nudge** — done, committed, universal.
2. **Salience/anti-enumeration instruction** — halves output, kills truncation on
   the tested snippet, canonicalizes names. Not yet committed (ling-specific;
   untested on gpt-5-mini — could cost gpt-5-mini real facts).
3. **Reduce `max_chunk_tokens`** (e.g. 1800 → ~700–900) — keeps ling under the 16k
   cap; at ≤1,200 chars ling ran clean (finish=stop, no truncation).

**Caveats that keep this in "tuning project," not "done":**
- The core behavior — ling *enumerates* instead of *judging salience* — is
  constrainable by prompting but not eliminable; even tamed it over-extracts vs
  gpt-5-mini and needs heavier downstream dedup/prune.
- Smaller chunks = **more chunks × more calls** → the cost/latency advantage is
  unproven; the per-call explosion could erase the per-token savings.
- **End-to-end quality is still unmeasured** — noise, dedup/false-merge,
  type_precision, and `AvailableIn`/Region coverage never got a clean run (the
  eval timed out). No GO/NO-GO on quality yet.

## Recommended next step (bounded, one session)

If pursuing ling for the cost saving:
1. Commit the salience clause **behind a config flag or ling-only path** (don't
   risk gpt-5-mini) OR validate it on gpt-5-mini first like the subject nudge.
2. Set `max_chunk_tokens ≈ 800`.
3. Re-run the quality gate on **2–3 articles** (not 6) with a generous timeout, to
   get real numbers: noise, dedup, type_precision, `AvailableIn`/Region counts,
   **total LLM calls + wall time + $** vs gpt-5-mini.
4. Decide the tier switch on that quality+cost table.

Until then, production stays gpt-5-mini; the subject nudge is the banked win.

## Repro (scratch, not committed)
`ling_diag.py` (subject/drops), `ling_ab.py` (nudge A/B), `ling_trunc.py`
(size→output scaling), `ling_salience.py` (salience A/B). Isolated `ling-*`
groups, auto-cleaned; production `backup-docs` never touched.
