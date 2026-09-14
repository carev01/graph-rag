# Pilot re-ingest under the new time axis — results

**Date:** 2026-09-14. 83 articles, 655 episodes, **6 h 55 m** (38.1 s/episode).
Raw log: `.superpowers/sdd/pilot-reingest-2026-09-14.log`. The semantic layer was reset
first (1,706 nodes; verified beforehand that it held **0 supersessions and 0 tombstones**,
so no real history was lost — only the 140 phantom invalidations).

One methodological note, because it nearly cost 7x the money: the 83 ingested articles are
**not** contiguous by `sort_order`, and `cli ingest --limit N` takes the first N. Re-ingesting
"the pilot" through the CLI would have processed all **593** structural `:Article` nodes. The
exact ids were captured before the reset and driven individually.

## 1. The contradiction gate delivered, measurably

| | before | after |
|---|---|---|
| dedup calls with an out-of-range index | 7.4–9.5% | **0 of 2,863** |
| `dup_in_invalidation_range` | 154 / 113 across two runs | **0** |
| `parse_failures` | 0 | 0 |
| invalidations written | 140 | **19** |

Spec §9 criterion 5 is delivered: with `existing_edges` always empty the dedup prompt carries
one index range instead of two, so the confusion behind all **267** previously observed bad
indices is structurally impossible rather than merely less likely.

## 2. BACKLOG 33 has its number

**339 same-pair contradictions asserted across 2,863 dedup calls (11.8%)**, yielding the 19
invalidations above. The residual path the gate deliberately does not close is real and
substantial — no longer a suspicion.

## 3. The provider does NOT serialise our concurrent calls

Latency by in-flight count at dispatch, `dedupe_edges.resolve_edge`:

| in-flight | 1 | 2–4 | 5–9 | 10–19 | 20+ |
|---|---|---|---|---|---|
| median | 1231 ms | 1215 ms | 1287 ms | 1469 ms | **1082 ms** |

Flat, with the most-concurrent bucket the fastest. `extract_timestamps` shows the same shape.
**Reducing call count buys nothing**, and batching (BACKLOG 29 / the call-volume review) is
confirmed dead on measurement rather than on argument.

## 4. The throughput lever, now evidenced

Sum of LLM in-call time **25,211 s** against **24,933 s** wall — a ratio of **1.01**. There is
effectively **no overlap across the run**. Four phases are dispatched strictly one at a time:

| phase | share of in-call time | in-flight |
|---|---|---|
| `extract_edges.edge` | 26.4% | always 1 |
| `extract_nodes.extract_summaries_batch` | 13.1% | 273 of 274 at 1 |
| `extract_nodes.extract_text` | 10.0% | always 1 |
| `dedupe_nodes.nodes` | 6.4% | always 1 |
| **total strictly serial** | **55.9%** | |

These are per-episode calls, and `IngestDriver` processes episodes sequentially. Combined with
§3 — the provider absorbs 20-way concurrency without queueing — **concurrent episode
processing is the lever**, and that is now measured rather than assumed. Note the ~1.9x DB
saturation ceiling recorded earlier was measured on the brute-force scan shape that ingest no
longer runs, so it must be re-measured before sizing this.

## 5. The finding against us: the ordering axis is only half-built

We told DocExtractor that `content_changed_at` would become the fact's `valid_at`
(`docs/proposals/2026-09-13-...-reply-3.md`). **It does not.** It sets the *episode* reference
time; the fact's `valid_at` is still produced by graphiti's `extract_timestamps` LLM call.
Measured on the new graph: **817 of 3,590 facts dated (23%)** — the same coverage as before —
and `reference_time` still clusters (152 distinct values over 3,590 facts, largest 164).

That call costs **21.7% of in-call time — 5,476 s, 1.5 h of the 6.9 h run** — to produce a
field that is three-quarters empty and that we have argued should not be the sort key.

Setting `valid_at` from `content_changed_at` directly would give 100% coverage, deliver what
was promised upstream, and remove the second-largest LLM cost in the pipeline. The cost: the
in-text end-dates that call also extracts ("deprecated in 2024") would be lost. That is the
next slice.

## 6. Tier mix, and a retraction

Final: **exact 49, lower_bound 14, first_seen 20 — 59% exact** over 83 articles.

At n=20 this read 35% and I wrote that it looked like "a real property of AWS and Microsoft
content rather than small-sample noise". That was premature: it rose through 47% at n=30 to
59% at completion, against upstream's 70.1% corpus-wide. Twenty articles was not enough to
claim a property of two sources.

## 7. Smaller observations

- **9 `Error parsing valid_at date, skipping`** from graphiti's own timestamp call — it
  returns strings it cannot then parse. Harmless (the edge simply gets no date) and further
  evidence for §5.
- Several `Failed to read from defunct connection ... read timed out` against Neo4j near the
  end of the run. The run completed and exited 0; worth watching if a longer run is attempted.
