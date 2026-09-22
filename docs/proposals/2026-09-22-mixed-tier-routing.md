# Mixing the local fine-tune with solar-pro4 — assessment

**Date:** 2026-09-22
**Status:** Assessment only. Nothing implemented, no `src/` change, no paid call made.
**Question:** does routing SOME extraction prompts to the local fine-tuned
`qwen35-graphrag` and the rest to `upstage/solar-pro4` beat using either alone?

**Answer: no — not per prompt.** Every per-prompt split trades ~$300–$1,400 of a
**$2,176 total** extraction bill for **tens to hundreds of days** of extra wall clock,
because the local box is a single serial resource whose throughput no amount of
scale-out can improve. The split that does pay is by **lane**, not by prompt.

---

## 0. Method and evidence status

Every number below is one of:

- **[measured]** — taken from a run recorded in this repo, cited `file:line`.
- **[computed]** — arithmetic I ran here over `data/ft-12k/*.jsonl`, script shown.
- **[live]** — read from a service just now (the OpenRouter public model catalogue,
  the llama.cpp `/models` endpoint, the llama.cpp `/tokenize` endpoint). No
  generation call was made to either model; `/tokenize` is free and non-generative.
- **[inferred]** — reasoning, labelled as such.

I did **not** call solar-pro4, did not run `ingest`, `theme-build` or
`eval_router`, and made no write to the graph.

---

## 1. Token volume per prompt type — computed, not assumed

### 1.1 Tokens per call

`data/ft-12k/{train,val}.jsonl` hold 10,557 records whose `meta.provenance` is
`captured` for **100%** and `meta.capture_model` is `upstage/solar-pro4` for **100%**
(`[computed]`, counted over both files). The prompts are therefore exactly what went
on the wire — `scripts/build_extraction_dataset.py:1-40` states the prompt is
verbatim even where a target was repaired (190 of 10,557 records carry
`meta.repaired`).

Character counts are exact. To convert to tokens I tokenised **6 sampled records per
prompt type, plus 25 more for `resolve_edge`**, through the local server's
`/tokenize` endpoint `[live]` — the fine-tune's own tokenizer, free, non-generative.
The chars-per-token ratio came out remarkably tight across types:

| prompt | prompt chars/token | completion chars/token |
|---|---:|---:|
| `extract_edges.edge` | 4.03 | 4.29 |
| `extract_nodes.extract_text` | 4.04 | 3.39 |
| `dedupe_edges.resolve_edge` | 4.13 | 2.77 |
| `dedupe_nodes.nodes` | 4.01 | 3.09 |
| `extract_nodes.extract_summaries_batch` | 4.14 | 5.14 |

Applying those ratios to the **full-population** mean character counts (all 10,557
records, not the sample):

| prompt | records | tok in/call | tok out/call |
|---|---:|---:|---:|
| `extract_edges.edge` | 976 | **6,774** | 417 |
| `dedupe_nodes.nodes` | 424 | **6,418** | 68 |
| `extract_nodes.extract_summaries_batch` | 605 | **5,072** | 718 |
| `extract_nodes.extract_text` | 1,105 | **3,276** | 148 |
| `dedupe_edges.resolve_edge` | 7,447 | **758** | 18 |

> **Correction to the brief.** The brief states `resolve_edge` prompts are "~995
> tokens". Measured on 25 records through the model's own tokenizer, the mean is
> **781 tokens** (3,225 chars ÷ 4.13); the full-population mean is **758**. The
> brief's `extract_edges.edge` figure (~6,700) matches mine (6,774) closely, so the
> discrepancy is confined to `resolve_edge`. It moves the conclusion in the brief's
> own direction — `resolve_edge` is *even less* of the token bill than stated.

### 1.2 Tokens per article

Multiplying by the brief's measured calls/article:

| prompt | calls/art | tok in/art | tok out/art |
|---|---:|---:|---:|
| `extract_edges.edge` | 7.9 | **53,512** | 3,291 |
| `dedupe_nodes.nodes` | 6.3 | **40,436** | 426 |
| `extract_nodes.extract_text` | 7.9 | 25,882 | 1,172 |
| `dedupe_edges.resolve_edge` | 31.9 | 24,191 | 576 |
| `extract_nodes.extract_summaries_batch` | 3.2 | 16,230 | 2,296 |
| **total** | **57.1** | **160,252** | **7,761** |

**The call mix and the token mix disagree violently, and this is the single most
decision-relevant fact in the document.** `resolve_edge` is 56% of calls and **15.1%
of input tokens**. `dedupe_nodes` is 11% of calls and **25.2%** of input tokens — the
prompt carries ten candidate entities with their context and returns an index list of
68 tokens. Any intuition built on call counts is wrong by roughly 4x in both
directions.

**Cross-check `[measured]`:** `docs/superpowers/pilot-reingest-2026-09-14.md:46`
records 655 episodes over 83 articles = **7.9 episodes/article**, which is exactly the
per-episode calls/article above — independent confirmation that the brief's call table
and this dataset describe the same pipeline shape.

**Second cross-check:** priced at gpt-5-mini's live rate ($0.25/$2.00 per 1M,
`[live]`), 160,252 in + 7,761 out = **$0.0556/article → $7,027 for 126,405 articles**.
`docs/superpowers/production-readiness-review-2026-09-13.md:361` independently
projects a **$9.7k gpt-5-mini ceiling** for the full corpus from a different
measurement generation. Agreement within 30%, in the expected direction (that
projection predates `install_deterministic_valid_at` and the dropped attribute step,
both of which removed calls). The token model is sound.

---

## 2. Price — found, not assumed

The brief is right that the repo has no price table;
`docs/superpowers/production-readiness-review-2026-09-13.md:191` says solar-pro4's
price "appears nowhere in the repo". It does not have to stay that way: the cheap
tier is OpenRouter (`src/graph_extract/config.py`,
`cheap_llm_base_url = "https://openrouter.ai/api/v1"`) and OpenRouter publishes an
unauthenticated model catalogue. Fetched just now `[live]`, 453 models:

| model | input /1M | output /1M | cached input /1M |
|---|---:|---:|---:|
| `upstage/solar-pro4` (`.env:32`, the cheap tier) | **$0.09** | **$0.36** | $0.018 |
| `openai/gpt-5-mini` (`.env:14`, the strong tier) | $0.25 | $2.00 | — |
| `openai/gpt-5-mini:batch` | $0.125 | $1.00 | — |

These are **measured, not assumed**. The catalogue carries no note of the
2026-10-10 rise, so its magnitude remains unknown — treated as a sensitivity in §6.

Note `input_cache_read` at **$0.018/1M, 20% of input** — solar-pro4 supports prompt
caching and this pipeline does not appear to exploit it. Input is **95.4% of tokens
and 83.8% of the bill** (§3), so that is a larger lever than any routing decision
here. Out of scope, flagged.

---

## 3. Cost per article, per prompt

At the live solar-pro4 rate:

| prompt | $/article | share | local s/article | $ saved per local-second |
|---|---:|---:|---:|---:|
| `extract_edges.edge` | $0.00600 | 34.9% | 82.2 | 7.30 × 10⁻⁵ |
| `dedupe_nodes.nodes` | $0.00379 | 22.1% | 37.8 | **10.03 × 10⁻⁵** |
| `extract_nodes.extract_text` | $0.00275 | 16.0% | 42.7 | 6.45 × 10⁻⁵ |
| `dedupe_edges.resolve_edge` | $0.00238 | 13.9% | 35.1 | 6.80 × 10⁻⁵ |
| `extract_nodes.extract_summaries_batch` | $0.00229 | 13.3% | 33.6 | 6.81 × 10⁻⁵ |
| **total** | **$0.01722** | 100% | **231.3** | |

*(local s/article = calls/article × the brief's fine-tune p50. The sum, 231.3 s,
reproduces the brief's 231 s/article — arithmetic check passed.)*

**Full-corpus extraction at 126,405 articles: $2,176.** That is the entire prize.
Every routing decision below is a fight over a subset of $2,176.

The last column is the one that should decide a split, and it **refuses to
discriminate**: the best prompt to offload (`dedupe_nodes`, huge prompt, 68-token
answer) is worth only **1.6×** the worst (`extract_text`) per second of scarce GPU.
There is no prompt whose offload is obviously worth it and none obviously not. That
flatness is itself the finding: **there is no natural seam here.** The cost profile
does not suggest a split; it suggests the axis is wrong.

---

## 4. Wall clock — the two models do not compose the way the brief hopes

### 4.1 The local model is a global serial resource

Confirmed `[live]` from the server's own argv:

```
llama-server ... --model /models/qwen35-4b-graphrag-mtp-Q8_0.gguf
  --n-gpu-layers all --parallel 1 --split-mode none --ctx-size 32768
  --spec-type draft-mtp --spec-draft-n-max 2 --temperature 0
```

`--parallel 1` is real, not assumed. One slot. Its throughput is **fixed at 1 /
(local s/article)** regardless of how many `semantic-worker` processes exist, how
high `INGEST_ARTICLE_CONCURRENCY` goes, or how many articles are in flight.

(Context headroom is fine and not a constraint: the **largest** single record across
all 10,557, prompt + completion, is **9,188 tokens** against a 32,768 window
`[computed]`. 3.5× headroom.)

### 4.2 Local and API calls for one article cannot overlap

The brief asks whether a mixed router running both at once changes the answer. For a
single article, **no** — and this is checkable rather than arguable.
`graphiti_core/graphiti.py:1122-1160` (`add_episode`, 0.30.1, the pinned version in
`.venv`) is a straight line of sequential `await`s:

```
extract_nodes  →  resolve_extracted_nodes  →  _extract_and_resolve_edges
               →  extract_attributes_from_nodes
```

i.e. `extract_text` → `dedupe_nodes` → `edge` → `resolve_edge` → `summaries_batch`,
one after another. No `semaphore_gather` on that path (the gathers at :842/:875/:902
are the *bulk* path, which ingestion does not use). A mixed split therefore makes
per-article latency **local_time + api_time**, strictly worse than either alone.

Overlap exists only **across concurrently processed articles**. So for the corpus:

> **wall clock ≈ max( local_serial_total , api_total / P )**

where P is effective API-side parallelism. The local term has **no P in it.** It is a
floor that scale-out cannot lower, and every additional worker process makes the local
box relatively worse.

### 4.3 The API side, grounded

From `docs/superpowers/ab-warmup-2026-09-15.md` (the W=8 arm — the production default,
`SEMANTIC_GLOBAL_WARMUP_LOCK` on):

- cold (warm-up-locked) article: **~305 s**, strictly serial globally (`:25`)
- warm article at c=4: **~93 s** marginal (`:25`)
- corpus shape: 280 sources × W=8 = **2,240 cold**, 124,165 warm (`:104-105`)

Check: 16 × 305 + 67 × 93 = 11,111 s predicted vs 11,123 s measured (`:36`), and
11,123/83 = **134 s/article** — exactly the brief's figure. The model reproduces the
measurement to 0.1%.

Projected to the corpus:

| effective parallelism P | warm-up floor | warm phase | **total** |
|---|---:|---:|---:|
| P=4 (1 worker, c=4) | 7.9 d | 133.6 d | **141.5 d** |
| P=16 (4 workers) | 7.9 d | 33.4 d | **41.3 d** |
| P=32 (8 workers) | 7.9 d | 16.7 d | **24.6 d** |
| P→∞ | 7.9 d | → 0 | **7.9 d floor** |

Linear scaling across *processes* is `[inferred]` — `docs/superpowers/ab-concurrency-2026-09-14.md:89-91`
measures diminishing returns above N≈4 *within* one process (shared `max_coroutines`,
max in-flight 36). Separate worker processes hold separate clients, and
`claim_semantic_jobs` uses `FOR UPDATE SKIP LOCKED` (`state_store.py:155`), so they
claim disjointly. Treat P=16/P=32 as optimistic but directionally right; CLAUDE.md
already states the floor dominates past ~32-way.

### 4.4 The comparison

| option | $/article | corpus $ | saved | local floor (days) | beats API at P= |
|---|---:|---:|---:|---:|---|
| **all-API** | $0.01722 | **$2,176** | — | 0 | — |
| all-local | $0 | **$0** | 100% | **338.4** | never |
| all-local except `edge` | $0.00600 | $759 | 65.1% | **218.2** | never |
| index-picking local (`resolve_edge`+`dedupe_nodes`) | $0.01104 | $1,395 | 35.9% | **106.6** | P≤5 only |
| `dedupe_nodes` only | $0.01342 | $1,697 | 22.1% | **55.3** | P≤10 only |
| `resolve_edge` only | $0.01483 | $1,875 | 13.9% | **51.3** | P≤11 only |
| `summaries_batch` only | $0.01493 | $1,887 | 13.3% | **49.2** | P≤11 only |

Read the last two columns together. **The cheapest possible offload — a single
prompt, 13.3% of the bill, $290 saved — pins total corpus throughput at 49 days**,
permanently, no matter how much API concurrency is added. Against an all-API run at
P=16 (41 days) that is already a loss. Against P=32 (25 days) it doubles the
schedule to save $290.

Expressed as a rate: index-picking-local against all-API at P=16 buys **$781 for 65
extra days — about $12/day.** That is not a trade worth making, and the flatness in
§3 means no reshuffling of which prompts go where rescues it.

**Per-prompt mixing is dominated on both axes it was supposed to win on.** It is
recommended against.

---

## 5. Quality consequence — and which splits put `edge` where

From the brief's held-out table (75 records, 17 articles, zero training overlap,
scored by the same `validate_capture` that measured solar-pro4):

| prompt | fine-tune defects | solar-pro4 defects | fine-tune byte-exact |
|---|---:|---:|---:|
| `dedupe_edges.resolve_edge` | 0/15 | 0% | 12/15 (80%) |
| `dedupe_nodes.nodes` | 0/15 | 0% | 13/15 (87%) |
| `extract_nodes.extract_text` | 0/15 | 0.3% | 2/15 |
| `extract_nodes.extract_summaries_batch` | 0/15 | 88% over length | 0/15 |
| `extract_edges.edge` | **2/15** | **20.6%** | 0/15 |

**On `extract_edges.edge` the fine-tune's 2/15 = 13.3% carries a Wilson 95% interval
of 3.7%–37.9% `[computed]`.** That interval **contains** solar-pro4's 20.6%. The
fine-tune has not been shown to beat solar-pro4 on this prompt, and it has not been
shown to lose either — 15 samples cannot separate them. Anyone quoting "13% vs 21%"
is quoting noise. (The 0/15 results are likewise only "≤20.4% at 95%".)

Which splits put `edge` where:

- **all-local**, **all-local-except-`edge`**'s complement, **everything except the
  named exclusions** — put `edge` on the *local* model, i.e. on the one prompt where
  the fine-tune has any observed defect and the widest interval, and simultaneously
  the largest token share (34.9%) and the slowest local call (10.4 s p50).
- **index-picking local**, **`dedupe_nodes` only**, **`resolve_edge` only**,
  **`summaries_batch` only** — keep `edge` on solar-pro4, at its measured 20.6%.
- **`all-local except extract_edges.edge`** is the only listed split that puts `edge`
  on the API *and* takes 65% of the bill. It is also the second-worst on wall clock
  (218 days). It is the best-shaped split in the list and still loses by 5×.

**Both fine-tune defects are `entity_name_not_in_list`** — the model named an endpoint
absent from the prompt's ENTITIES list. `graphiti_core` rejects such edges outright
(`graphiti_core/utils/maintenance/edge_operations.py:217-232` — `continue`,
not append; the same rejection `scripts/build_extraction_dataset.py:80-90` relies on),
so the failure mode is **lost facts, silently**, not corrupt facts. That is the benign
direction, but it is invisible without instrumentation.

### 5.1 Would two models extracting names differently hurt dedup?

This is the invariant-#4 question the parent raised, and the answer is more reassuring
than expected, for two reasons:

1. **Per-prompt routing is corpus-consistent by construction.** Every article's
   `extract_text` would go to the *same* model. Contrast `article_router.is_dense_matrix`
   (`src/graph_extract/article_router.py:11-26`), which is **per-article** — the same
   entity can be named by gpt-5-mini in a dense-matrix article and by solar-pro4 in a
   prose one. Whatever naming inconsistency the system already tolerates, it comes
   from the *existing* router, not from a hypothetical per-prompt one. A per-prompt
   split is strictly safer on this axis.

2. **The fine-tune is a distillation of solar-pro4.** 10,557/10,557 records have
   `meta.capture_model = upstage/solar-pro4` `[computed]`, and byte-exact agreement is
   80%/87% on the two index prompts. Its naming conventions are inherited, not
   independent. The `extract_text` byte-exact rate is only 2/15, but the defect rate
   is 0/15 — it phrases differently, it does not name wrongly.

The residual risk is real but small and is **not** the reason to decline: the
`extract_text` (names) / `dedupe_nodes` (resolution) pair straddling two models means
one model judges strings another generated. Since resolution is by **name and
embedding** and the embedder is a single shared model on both paths
(`build_cheap_graphiti` changes only `llm_base_url`/`llm_model`/`llm_api_key`/
`llm_client_mode`, `graphiti_client.py:256-265`), the embedding space is unchanged and
the comparison is well-posed. **Quality is not what kills per-prompt mixing. Wall
clock is.**

---

## 6. What the split *should* be: lane, not prompt

The local box's weakness is throughput under a 126k-article backlog. It has no such
weakness against a trickle. At 231.3 s/article it sustains **373 articles/day**
`[computed]` at **$0.00/article**.

And the hook already exists. `semantic_jobs.lane` is a column with exactly two values,
`'bootstrap'` and `'incremental'` (`src/graph_sync/state_store.py:32,36,142,153-156`),
already used to preempt backfill with incremental work
(`ORDER BY (lane='bootstrap'), next_attempt_at`, `:154`).

**Recommendation: route by lane.**

| lane | model | cost | wall clock |
|---|---|---|---|
| `bootstrap` (126,405 articles, one-time) | solar-pro4, all five prompts | **$2,176** | **41 days** at P=16 (7.9 d warm-up floor + 33.4 d), 25 d at P=32 |
| `incremental` (steady state) | local fine-tune, all five prompts | **$0** | 231 s/article, capacity **373 articles/day** |

Why this works where per-prompt mixing does not:

- **No contention.** The bootstrap never queues behind the GPU; the GPU never queues
  behind the bootstrap. The `max(local, api/P)` bind in §4.2 disappears because the
  two lanes have disjoint work.
- **No `max_coroutines` fight.** `build_cheap_graphiti` already builds one client per
  tier; a lane router is a third settings view, not a new mechanism.
- **The per-article latency penalty lands where it does not matter.** 231 s vs ~93 s
  is 2.5× worse per article, on a lane whose SLA is "before the next weekly sweep".
- **It respects the warm-up floor rather than competing with it.** The 7.9-day
  `SEMANTIC_GLOBAL_WARMUP_LOCK` floor is a bootstrap-lane property; incremental
  articles land in already-warm sources.
- **It is the failure-tolerant shape.** If the fine-tune degrades on a new vendor's
  prose, the blast radius is one day of incremental deltas, re-runnable, not a
  corpus-wide bootstrap.

**Where it would be implemented, if approved** (not implemented here): not
`article_router` — that is per-article and per-*tier-of-article*, a different
question. The natural hook is the one `dedup_guard` already proves works:
`install_dedup_guard` wraps `graphiti.llm_client.generate_response` in place and
**already dispatches a specific prompt to a different client**
(`src/graph_extract/dedup_guard.py:210-262`; the swap is at `:250`,
`guard.fallback.generate_response`). A
lane router is coarser than that and needs less: pick the tier once, when the worker
claims the job and reads `lane` off the row, and hand `ingest_article` the
corresponding `Graphiti`. No per-call hook at all.

### What would change this answer

- **The October price rise.** Magnitude unknown (§2). At 10× the bootstrap bill goes
  $2,176 → $21,760, and `all-local except edge` (saves 65% = $14,100, costs 177 extra
  days) becomes arguable. Below ~5× it does not. **Bootstrap before 2026-10-10 if the
  schedule allows** — that is worth more than any routing choice here.
- **A second GPU, or `--parallel 4`.** The floor divides. Two boxes at `--parallel 1`
  halve every "local floor" column in §4.4; `index-picking local` at 53 days would
  then beat all-API at P=8. This is the cheapest way to make mixing viable and it is
  a hardware question, not a routing one.
- **Prompt caching on solar-pro4** ($0.018/1M cached input, §2). 83.8% of the bill is
  input tokens. If even half of it caches, all-API drops toward ~$1,200 and the case for
  local weakens further. Worth measuring before the price rise.
- **A faster local model.** The lever is s/article, not $/article. Anything that
  halves 231 s halves every floor in §4.4 proportionally.

---

## 7. Summary of recommendations

1. **Do not mix per prompt.** Dominated on both cost and wall clock, for every split
   considered, by margins of 5×–40× on the axis it loses.
2. **Bootstrap: all-API (solar-pro4).** $2,176, ~41 days at 16-way. Start before
   2026-10-10.
3. **Incremental: all-local.** $0, 373 articles/day capacity, on the existing
   `semantic_jobs.lane` column.
4. **Before trusting the local model on `edge`**, extend the held-out set: 15 records
   cannot distinguish 13% from 21%.
5. **Separately, measure prompt caching.** It is worth more than this entire
   question.
