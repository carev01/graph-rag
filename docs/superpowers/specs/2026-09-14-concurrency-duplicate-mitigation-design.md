# Concurrency Duplicate Mitigation — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Date:** 2026-09-14
**Status:** Approved design — ready for implementation planning
**Follows:** `2026-09-14-concurrent-article-ingest-design.md` (the concurrency spec) and
`docs/superpowers/ab-concurrency-2026-09-14.md` (the A/B that measured its cost)

---

## 1. The problem, measured

Concurrent article ingest is merged and off by default (`ingest_article_concurrency = 1`).
The A/B that gates raising it (same 83 pilot articles, N=4, against the 999-entity
sequential baseline) returned:

| | sequential | concurrent (N=4) |
|---|---|---|
| entities | 999 | **1027** |
| exact-name duplicates | **0** | **30** |
| wall clock | 6 h 55 m | **2 h 10 m** (3.20x) |

Three measured facts shape this design (`ab-concurrency-2026-09-14.md`):

1. **Every one of the 30 duplicates is an exact-name collision** — 1027 entities over 997
   distinct names, against 999/999 sequentially. Repair therefore needs no semantic
   judgement and no LLM.
2. **Duplication concentrates on hub entities.** Duplicated entities span 7.6 articles on
   average against 2.4 for the rest; the names are `AWS Backup` (3x), `Amazon EC2` (3x),
   `Azure Backup` (3x), `Microsoft Azure`, `Amazon RDS`, `Azure Files`, and a run of Azure
   region names. These are the nodes cross-vendor questions resolve through — the ones
   design invariant #4 (one corpus-wide `group_id`) exists to keep whole.
3. **Duplicate risk is front-loaded.** Risk for an entity in `k` articles scales with
   `k-1`; 636 of 999 entities (63.7%) are single-article and carry zero risk. The
   concurrency spec's §2 rejected a warm-up on the flat entity *creation* rate, which is
   dominated by those zero-risk entities. Weighted by risk, hub entities first appear at
   article 1–6 of 83.

### 1.1 The risk curve, recomputed from the baseline graph

The brief's figure ("61.5% of risk weight in the opening 10% of articles") was re-derived
for this spec directly from the live baseline (read-only: article spans via
`(:Article)-[:HAS_EPISODE]->(:Episodic)-[:MENTIONS]->(:Entity)`, positions from
`.superpowers/sdd/pilot-article-ids.json` and from `Article.sort_order`). Total risk weight
`Σ(k-1) = 1538` over 999 entities. Cumulative share of risk weight whose entity first
appears within the first `c` articles:

| c | 1 | 2 | 3 | 4 | 6 | **8** | 10 | 12 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run order, whole 83-article set | 17.4 | 38.2 | 39.1 | 39.1 | 54.6 | 61.0 | **61.5** | 74.6 | 77.2 | 79.8 | 83.0 |
| **per source, by `sort_order`** (first `c` of *each* source) | 38.2 | 53.1 | 59.4 | 71.3 | 73.0 | **79.5** | 81.7 | 82.5 | 83.5 | 85.8 | 94.1 |

The 61.5% reproduces exactly (c=10 in run order). The per-source view — which is the unit
`ingest_source` actually processes, in the order it processes it — is markedly better:
**the first 8 articles of each source carry 79.5% of the risk weight**, and the curve has
its knee at 4 (71.3%). The pilot set is 28 articles of one source and 55 of another, so
"first 8 per source" is 16 sequential articles in total — comparable to the run-order
figure at c=16 (77.2%), which says the per-source gain is mostly real and not an
artefact of counting twice as many articles.

The top hubs by span (Azure Backup 55, Azure 42, Recovery Services vault 37, Backup vault
33, Azure Virtual Machines 32, AWS Backup 30) first appear at per-source position 0 or 1.
This is the mechanism, not a coincidence of the sample: a documentation source in
table-of-contents order opens with overview pages that name the product and its core
concepts. That is an inference from one sample; §9 says how it gets checked.

### 1.2 Why duplicates get worse, not better, if left in place

graphiti's node resolution (`dedup_helpers._resolve_with_similarity`, 0.30.1) tries an
exact normalized-name match against the candidate set before anything else. With one
candidate it resolves deterministically and for free. With **more than one candidate of
the same normalized name it escalates to the LLM** ("Ambiguous: multiple candidates share
the same normalized name"). So once `AWS Backup` exists three times, *every future
mention* of it costs an LLM dedup call and lands on whichever copy the model picks. A
duplicate is not a static blemish: it is a permanent tax on the hottest names in the
corpus and a source of further fragmentation. Repair has to happen before the next
ingest that touches those names, not eventually.

## 2. The decision

**One spec, two components, shipped together, each off by default:**

- **Component 1 — sequential warm-up.** While a source has fewer than `W` articles in
  the semantic graph, its articles run one at a time with nothing else in flight; after
  that they fan out. Targets the front-loaded ~80%.
- **Component 2 — exact-name merge pass.** A deterministic, Cypher-only, report-first
  maintenance command that merges `:Entity` nodes with byte-identical `name` in one
  `group_id` into a single survivor, rewiring every edge by hand. Repairs the remainder.

One spec rather than two because the components are only meaningful together and the
evidence base is one document: warm-up alone leaves ~20% of the risk unaddressed (and the
§1.2 tax on whatever leaks through), while the merge alone lets duplicates exist for the
whole duration of an ingest — during which graphiti is already paying the ambiguity
penalty and splitting new facts across copies. They are separable in *implementation*
(different modules, independently testable, either can ship first) and this spec keeps
them so; they are not separable in *justification*.

Neither changes behaviour at defaults. `ingest_article_concurrency` stays 1; warm-up is
inert at concurrency 1 (see §3.6 for why its default is nonetheless non-zero); the merge
command does nothing destructive without `--apply`. This matches how concurrency shipped.

Quality outranks latency (stated by the user). Where this spec trades speed for exactness
— the strict warm-up barrier, the transaction-per-group merge, the refusal to run
`theme-build` over a graph with known duplicates — it says so and does not apologise.

## 3. Component 1 — sequential warm-up

### 3.1 The unit, and what "the start" means

**Warm-up is per source, and "the start of a source" is a property of the graph, not of
a run.** A source is *cold* while fewer than `W` of its articles have a live episode in
the configured group:

```cypher
MATCH (a:Article {source_id:$source_id})-[r:HAS_EPISODE]->(:Episodic {group_id:$group_id})
WHERE coalesce(r.superseded, false) = false
RETURN count(DISTINCT a) >= $w AS warm
```

`article_source` is a RANGE index on `Article.source_id` (verified: `SHOW INDEXES`), so
this is an index seek plus a small expand. It is scoped by `group_id` deliberately: the
replay gate in `provenance.already_ingested` is *not* group-scoped (the A/B found that the
hard way), and a warm-up predicate that ignored the group would have skipped warm-up on
the A/B's isolated group — precisely the run that needs it.

This definition answers the brief's "what is the unit on BOTH paths" and "does it
persist" with one mechanism:

- **`IngestDriver.ingest_source`** lists a source's articles by `sort_order` and fans out.
  With the predicate, its opening articles run sequentially until the source's live
  article count reaches `W`, then the rest fan out. Because the count is read from the
  graph, an article that was already ingested (skipped at zero cost by
  `already_ingested`) still counts toward warmth — a resumed or partial run does not
  re-pay the warm-up.
- **`run_worker_once`** claims a batch that may span sources and has no notion of order.
  Each claimed article group is classified by its article's `source_id`
  (`MATCH (a:Article {id:$id}) RETURN a.source_id`) and the same predicate. Cold groups
  run alone; warm groups fan out. The worker needs no "start" because the graph carries
  the source's state across batches, restarts and days.
- **Persistence** is therefore free and correct by construction. A source ingested over
  several days warms up once. A quarterly re-extraction of an already-ingested source
  (the user's real cadence) never warms up, which is right: its hub entities already
  exist, so the hazard warm-up guards against is absent. A semantic-layer reset drops
  the count to zero and warm-up runs again, which is also right.

**The article's `Article` node may not exist yet when the worker sees the job.**
`sync_core.apply` calls `enqueue_semantic_job` *before* `apply_structural`
(`sync_core.py:88-89`), so a worker can claim a job in the window before the structural
write lands. A missing article resolves to **cold** — the conservative branch. This is a
rare race, not a steady state, and treating it as warm would silently disable the
protection exactly when the source is newest.

**Remove jobs** (`op == "remove"`) tombstone episodes and extract nothing; they are
never cold.

### 3.2 Strict barrier: a cold article runs with nothing else in flight

When the fan-out reaches a cold item it **waits for every in-flight task to finish**, runs
the cold item alone, then continues. It does not merely run cold items sequentially with
respect to each other while warm items of other sources keep flowing.

The weaker form was considered and rejected: the hazard is two in-flight episodes both
creating an entity that neither can yet see. A cold article of source S running beside
warm articles of source T is safe only if S's new hubs and T's in-flight new entities never
coincide — and cross-source hubs (`Microsoft Azure`, `Amazon S3`) are exactly the
entities invariant #4 cares about. The strict barrier costs some throughput only in the
worker during a bootstrap of a new source, in batches that mix that source with warm
ones; on `ingest_source` the cold articles are the first `W` and nothing is in flight
anyway, so it costs nothing. Quality outranks latency.

Within one run, warmth is monotonic (nothing removes live episodes mid-run), so a source
found warm is cached for the run and never re-queried. A cold source is re-queried per
item, which is one index seek each — negligible against a 300 s article.

### 3.3 How many articles: `W = 8`

From §1.1, per-source: 4 → 71.3%, **8 → 79.5%**, 12 → 82.5%, 16 → 83.5%. The knee is at
4; 8 buys another 8 points; 12 buys 3 more for 50% more sequential articles. Eight is the
point where the curve flattens and is the measured figure, not a rounding of it.

**Cost.** The pilot ran 7.9 episodes/article at 38.0 s/episode sequential and 11.9 s at
N=4: ~300 s versus ~94 s per article. Eight sequential articles per source lose about
8 x 206 s ≈ 27 min per source; over ~194 sources that is ~89 h, under 4 days. The corpus
at N=4 is ~126,405 articles x 94 s ≈ 137 days of one worker. Warm-up adds under 3%.
(Rates are the pilot's measured ones; the corpus and source counts are from
DocExtractor's catalog as of this writing.)

**`W` is uniform — it is not scaled down for small sources.** The two sources in the
live structural layer hold 148 and 445 articles, and the corpus is ~126,405 articles
across ~194 sources, a mean of ~651 per source. Eight articles is therefore roughly
1–5% of a realistic source; the case where warm-up swallows most of a source does not
arise at corpus scale, and a per-source or per-vendor override would be a knob with no
measured use. Caveat: those sizes are two observed sources plus a corpus-wide mean, not
a distribution. If a vendor turns out to publish many tiny sources (release-note feeds,
per-region stubs), the predicate still behaves correctly — the whole source simply runs
sequentially — and the operator can lower `ingest_warmup_articles` for that bootstrap.

**Expected effect.** If duplicates are proportional to risk weight — the model's
assumption, validated once by its correct prediction that duplicates would concentrate on
hubs — warm-up at `W=8` reduces the pilot's 30 duplicates to roughly 6. That is a
prediction, and §9 tests it.

**Confidence.** One 83-article set, two sources (one AWS, one Azure), one concurrent run.
The per-source curve is computed on the *sequential* graph's spans, so it is exact for
the pilot but says nothing directly about a Veeam or Commvault source. The mechanism in
§1.1 (overviews come first in TOC order) is plausible for any vendor documentation but
unmeasured beyond these two. `W` is a knob for that reason, and the merge pass's report
mode (§4.8) gives the residual duplicate count after each concurrent source at zero LLM
cost — the feedback loop for tuning `W` does not depend on another paid A/B.

### 3.4 Architecture

The concurrency spec put the fan-out in one shared helper so the two call sites cannot
drift. Warm-up goes in the same place.

```
src/graph_extract/concurrent_ingest.py
    async def run_concurrently(items, worker, *, limit,
                               is_cold: Callable[[T], Awaitable[bool]] | None = None)
```

Semantics, in input order: for each item, if `is_cold` is given and returns true, await
all launched tasks, then await `worker(item)` directly; otherwise launch `worker(item)` as
a task under the semaphore. Results come back one per item in input order; an exception
from `worker` **or from `is_cold`** is returned in the item's slot (a Neo4j outage in the
predicate fails that item, is logged naming it, and propagates through the existing
after-the-batch contract — nothing is swallowed). `is_cold=None` is byte-for-byte the
current helper, and a test proves it (§7).

```
src/graph_extract/warmup.py
    class WarmupGate:
        def __init__(self, driver: AsyncDriver, group_id: str, threshold: int)
        async def is_cold_source(self, source_id: str) -> bool
        async def is_cold_article(self, article_id: str) -> bool   # article -> source -> is_cold_source
```

`threshold == 0` short-circuits to warm without querying. Warm sources are remembered
for the gate's lifetime (one per run: constructed in `_build_ingest_driver` and
`_build_worker_deps`, attached to the driver as `ingest.warmup_gate`, the same way
`timings` is).

- `ingest_source` passes `is_cold=lambda _aid: gate.is_cold_source(source_id)`.
- `run_worker_once` gains `is_cold` as a keyword argument (default `None`, like
  `concurrency`); the CLI passes `ingest.warmup_gate.is_cold_article` applied to the
  group's `article_id`, returning false for groups whose every job is a remove.

### 3.5 Configuration

```python
# graph_extract/config.py (ExtractSettings -- the object both call sites already read
# ingest_article_concurrency from; graph_sync.config.Settings is the wrong home)
ingest_warmup_articles: int = 8
```

Validated like its sibling: a negative value **raises** at settings construction rather
than being clamped, for the same reason — a `-1` in the environment means someone
believed they configured something. `0` is the explicit "off".

**The default is 8 — on — and the test in §7.1 that pins identical call order at
`ingest_article_concurrency = 1` with `W=0` and `W=8` is what makes that default
defensible.** It is not optional coverage; it is the evidence that shipping the knob on
changes nothing today. If that test is ever weakened to a count assertion, the default
loses its justification.

### 3.6 Why the default is 8 and not 0

Warm-up is inert at `ingest_article_concurrency = 1`: with one slot, awaiting an item
directly and launching it alone under a semaphore of one produce the same call order, and
§7.1 pins that. So `W=8` at the shipped default is no behaviour change. The reason to
default it *on* rather than opt-in is that the operator who raises `N` should get the
protection without knowing a second knob exists; a mitigation that must be discovered
separately from the hazard it mitigates will be forgotten. This is the one place this
spec departs from "opt-in like concurrency", and it is safe because the hazard itself
remains opt-in.

## 4. Component 2 — exact-name merge pass

### 4.1 The criterion: byte-identical `name`, same `group_id`, nothing else

Two `:Entity` nodes are merged iff `a.name = b.name` (Cypher string equality — no
`toLower`, no `trim`, no normalisation) and `a.group_id = b.group_id`. Verified against
the evidence: this catches 30 of 30 observed duplicates.

**Why not graphiti's own normalisation** (`_normalize_string_exact`: lowercase, collapse
whitespace)? Because the sequential baseline — the reference every future comparison is
made against — contains **6 case-insensitive pairs that graphiti's sequential pipeline
left apart** (`Access Control`/`Access control`, `AWS BackInt`/`AWS Backint`,
`point-in-time recovery`/`Point-in-Time Recovery`, `Multi-user authorization`/
`Multi-User Authorization`, `Redundancy`/`redundancy`, `Backup Reader`/`Backup reader`;
verified read-only). Those are not concurrency artefacts, and a pass whose purpose is to
repair concurrency artefacts must not alter the reference graph. Why graphiti left them
apart was not investigated (the candidate search is a cosine search with a floor; a case
variant may fall under it — inference). The report mode lists normalized-name collisions
under `near_duplicates_not_merged` so the operator can see them; merging them is out of
scope until a concurrent run shows them growing (§11).

**Different custom types with the same name** (`AWS Backup:Product` vs `AWS Backup:Tool`)
**are merged.** graphiti would have done the same sequentially: the exact-name branch of
`_resolve_with_similarity` does not consult labels, and the 999/999 baseline shows the
sequential pipeline never keeps two exact-name nodes in this corpus. Label handling is in
§4.4.

### 4.2 The schema being rewired (verified, not assumed)

Probed read-only against the live baseline (999 entities, 655 episodes, 3,590
`RELATES_TO`, 4,479 `MENTIONS`) and against graphiti 0.30.1's Neo4j queries:

| element | properties | notes |
|---|---|---|
| `:Entity` (+ one custom label) | `uuid`, `name`, `group_id`, `labels` (list property mirroring the node labels — 0 mismatches), `created_at` (zoned datetime), `summary`, `name_embedding` (768 floats) | written by `MERGE (n:Entity {uuid}) SET n = $entity_data` then `db.create.setNodeVectorProperty` |
| `RELATES_TO` | `uuid`, `name`, `fact`, `fact_embedding` (768), `episodes` (list of Episodic uuids), `group_id`, `created_at`, `reference_time`, `valid_at` (817 of 3,590), `invalid_at` (19), `expired_at`, **`source_node_uuid`, `target_node_uuid`** | endpoints are ALSO stored as properties (0 of 3,590 disagree with the topology today); the vector is written by `db.create.setRelationshipVectorProperty` |
| `MENTIONS` | `uuid`, `group_id`, `created_at` | `(:Episodic)-[:MENTIONS]->(:Entity)` |
| `:Episodic.entity_edges` | list of `RELATES_TO` uuids (655 episodes, 4,886 refs) | references edges **by uuid** |
| `IN_COMMUNITY` | none | `(:Entity)-[:IN_COMMUNITY]->(:Community)`, written by `theme_builder.writeback`; `:Community.cited_fact_uuids` references facts by uuid |
| `SAME_AS` | none | `(:Vendor|:Product)-[:SAME_AS]->(:Entity)`, written by `reconcile` |

Relationship types touching `:Entity` in the live graph: `MENTIONS` (in) and
`RELATES_TO` (in and out) only; `SAME_AS`, `IN_COMMUNITY` and graphiti's own
`HAS_MEMBER` exist as code paths or indexes but have no instances today. **No vector
indexes exist** — graphiti scores with `vector.similarity.cosine` inline — and **no
uniqueness constraint exists on any graphiti uuid** (only RANGE indexes; `SHOW
CONSTRAINTS` lists the five structural ones). Both procedures
`db.create.setNodeVectorProperty` and `db.create.setRelationshipVectorProperty` are
present on 2026.07.1 Community (`SHOW PROCEDURES`). No APOC.

Two facts about the baseline that decide §4.5: **multiple `RELATES_TO` edges between one
ordered pair are already normal** (440 pairs have more than one, the maximum is 109
between one pair), and **10 self-loops already exist**. Neither is something the merge
introduces; both are things it must not "fix".

### 4.3 Which node survives

**The member with the earliest `created_at`; ties broken by ascending `uuid`.** Total
order, so the survivor is a pure function of the group.

Rationale: this is the node sequential ingestion would have produced. Sequentially, the
first extraction creates the node; every later extraction resolves to it and keeps its
uuid (`state.uuid_map[node.uuid] = match.uuid`). The earliest node is also the one most
likely to be referenced from outside (a community membership, a prior answer's
`center_node_uuid`), so it is the one whose disappearance would hurt most. "Highest
degree" was considered: it is deterministic only with a tie-break anyway, and it does
not mirror anything the pipeline would have done.

### 4.4 What the survivor keeps, and what the loser's fields become

| field | rule | reason |
|---|---|---|
| `uuid`, `name`, `group_id`, `created_at` | survivor's, untouched | identity; `created_at` is what `touched_entities` keys on and must stay honest |
| `name_embedding` | survivor's; loser's dropped | same string through the same embedder → expected identical (one embedding space everywhere, `CLAUDE.md`). The report prints the cosine between the two so this is checked, not trusted. Never copy a vector with plain `SET` — see §4.5 |
| node labels + `labels` property | survivor's, with one exception decided over the **whole group at once**, never loser by loser. **Bare survivor** (no custom label): take the custom-label set of every *typed* loser. If those sets are all identical, promote that set onto the survivor — both the Neo4j labels and the `labels` list, kept in lockstep. If two typed losers disagree with each other, promote **nothing** and report the group under `label_conflicts`. **Typed survivor:** never promoted; a loser carrying a custom label the survivor lacks is dropped and reported under `label_conflicts`. A **bare loser has no say** in either case — it neither promotes nor blocks | the rule at three or more members is explained below the table; at two members it reduces to "bare + typed promotes, typed + different typed is a conflict". `retype` re-derives `:Region` from the gazetteer on the next `cleanup` regardless |
| `summary` | survivor's, unless it is empty and the loser's is not, in which case the loser's | deterministic and loses least. graphiti regenerates an existing node's summary on the next episode that mentions it (`extract_attributes_from_nodes`), so the merged summary self-heals on the next ingest touching the name. The loser's summary text is emitted in the audit so nothing vanishes unseen |
| other properties (`demoted_from_region`, future custom attributes) | survivor's; loser's extra keys listed in the audit, not copied | no silent overwrite in either direction |
| `merged_from`, `merged_at` | **added** to the survivor: `merged_from = coalesce(merged_from, []) + [loser.uuid]`, `merged_at = datetime()` | audit trail on the node itself (invariant #5 allows adding properties to graphiti's nodes). Also the hook §4.7 uses |

**The label rule at three or more members — a deliberate divergence from sequential
resolution, not a reproduction of it.** The production groups are `AWS Backup` ×3,
`Amazon EC2` ×3, `Azure Backup` ×3, so the three-member reading is the one that runs.

At two members the rule is graphiti's own `_promote_resolved_node`
(`graphiti_core/utils/maintenance/dedup_helpers.py:170-189`, graphiti-core 0.30.1): a
typed canonical is returned unchanged (`:175-177`), a label-less extracted node leaves the
canonical unchanged (`:179-181`), and only bare canonical + typed extracted promotes. That
last branch is where a bare loser's "no say" comes from — it is graphiti's boundary, not a
choice made here.

At three or more members there are three candidate rules, and none of them is "what
sequential resolution would have done" for free:

- **First typed loser wins** is what sequential ingest produces: the second typed
  extraction meets an already-typed canonical and is dropped at `:175-177`, silently.
- **Union** is what the two-member exception yields if it is read *per loser* (each typed
  loser meets a survivor that is still bare *from its own point of view*). It manufactures
  a node with two custom labels — a shape graphiti never writes (0 of the 999 baseline
  entities carry more than one custom label), and one that `theme_builder/cli.py:43`
  reads `[0]` of, i.e. types by an unspecified pick in every community report that cites
  it. An earlier draft of this row was read that way and implemented that way; this
  paragraph exists so it is not read that way again.
- **Promote nothing and report** is the rule specified above.

Promote-nothing is chosen over first-typed-wins because this is a one-shot destructive
pass: a bare survivor is a state graphiti repairs on its own — the next ingest that resolves
a typed extraction of the name onto it runs `_promote_resolved_node` again
(`dedup_helpers.py:238`, `:272`, `node_operations.py:610`) and `EntityNode.save` writes
the promoted labels back — whereas an arbitrary winner is permanent and, without a
`label_conflicts` entry, invisible to the operator. The cost is that a disagreeing group
stays bare until such an ingest happens; the audit names it so nobody has to notice by
accident.

### 4.5 Rewiring, in plain Cypher, one transaction per name group

Neo4j cannot re-point a relationship. Each edge is **re-created with an identical
property map — including its `uuid` — then the original is deleted.** Keeping the uuid is
what makes the rest of the graph not notice: `Episodic.entity_edges`,
`Community.cited_fact_uuids`, `resolve_citations` and the sweep all address facts by uuid,
never by relationship identity. This is load-bearing, not defensive: the 655 baseline
episodes carry **4,886 `entity_edges` references by edge uuid** (verified), every one of
which would dangle if a moved edge got a fresh uuid. There is no uniqueness constraint on
`RELATES_TO.uuid` (§4.2), so the moment inside the transaction where old and new share a
uuid is legal.

For every loser `L` in a group with survivor `S`, in **one explicit transaction** per
group (a crash leaves the group either untouched or fully merged, never half-rewired):

**Step 0 — re-verify the plan.** Match `S` and every `L` by `uuid`, `group_id` *and*
`name = $name`. If any is missing or renamed since the plan was computed, abort the group
(raise; the transaction rolls back).

**Step 1 — outgoing facts.** Handles `L→X`, `L→S` (becomes a self-loop on `S`) and `L→L`:

```cypher
MATCH (l:Entity {uuid:$loser, group_id:$g}), (s:Entity {uuid:$survivor, group_id:$g})
MATCH (l)-[old:RELATES_TO]->(t)
WITH s, old, properties(old) AS props, old.fact_embedding AS emb,
     CASE WHEN t.uuid = $loser THEN s ELSE t END AS target
CREATE (s)-[new:RELATES_TO]->(target)
SET new = props
SET new.source_node_uuid = s.uuid, new.target_node_uuid = target.uuid
WITH old, new, emb
CALL db.create.setRelationshipVectorProperty(new, 'fact_embedding', emb)
DELETE old
RETURN count(new) AS moved
```

**Step 2 — incoming facts** (`X→L`, `S→L`): the mirror image, with `new.source_node_uuid`
from the source and `new.target_node_uuid = s.uuid`. Edges already moved in step 1 are
gone, so a loser self-loop is not seen twice.

**Step 3 — `MENTIONS`:** `MATCH (ep:Episodic)-[old:MENTIONS]->(l) CREATE (ep)-[new:MENTIONS]->(s)
SET new = properties(old) DELETE old`. If the same episode already mentions `S`, both
edges are kept — each has its own uuid and `created_at`, and collapsing them is a
judgement the pass does not make.

**Step 4 — `IN_COMMUNITY`:** `MATCH (l)-[old:IN_COMMUNITY]->(c) MERGE (s)-[:IN_COMMUNITY]->(c)
SET c.stale = true DELETE old`. `MERGE`, because membership is a set. `stale` is the flag
`theme_builder.incremental` already treats as dirty (§4.7).

**Step 5 — `SAME_AS`:** `MATCH (v)-[old:SAME_AS]->(l) MERGE (v)-[:SAME_AS]->(s) DELETE old`.

**Step 6 — the guard.** `MATCH (l) WHERE COUNT { (l)--() } = 0` must match; otherwise
raise, naming the surviving relationship types, and roll back. Then **`DELETE l` — never
`DETACH DELETE`.** This is the load-bearing safety property of the whole pass: any
relationship type this spec did not enumerate makes Neo4j refuse the delete and abort the
transaction, instead of being silently dropped with the loser. A future edge type added
to `:Entity` breaks the merge loudly rather than losing data quietly.

**Step 7 — audit stamps** on `S` (§4.4), and label promotion if the rule applies.

Two mechanics that must be exactly right, and that the tests in §7 exist to prove:

- **`source_node_uuid` / `target_node_uuid` are rewritten.** graphiti's read path uses
  `startNode(e).uuid` and pops the stored properties, so a stale value would be invisible
  to graphiti and visible to every plain-Cypher consumer (`lean_edge_search` projects
  them). Today **0 of 3,590 edges** disagree with their topology (verified); that
  invariant is what the rewrite preserves, and it is the reason the rewrite is a
  requirement rather than tidiness.
- **`fact_embedding` is re-set through `db.create.setRelationshipVectorProperty`** after
  `SET new = props` has copied it as a plain float list. `valueType` reports both as
  `LIST<FLOAT NOT NULL>`, so whether a plain `SET` would store the same representation
  graphiti wrote cannot be told from the schema; re-setting through the same procedure
  graphiti uses removes the question. The clause order `WITH ... CALL <void procedure>
  ... DELETE` is Cypher 25 on 2026.07.1 and the integration tests run on that exact
  image; if the parser objects, split the CALL into a second statement in the same
  transaction — do not drop it.

### 4.6 Edges that meet after the merge

`(S)-[f1]->(X)` and `(L)-[f2]->(X)` become two `RELATES_TO` edges from `S` to `X`. **They
are kept.** Each carries its own `fact`, `episodes`, `valid_at` and provenance chain; the
baseline already holds 440 such multi-edge pairs from ordinary sequential ingest, so
graphiti and every consumer already handle it. The same holds for identical fact text on
both: graphiti's *edge* dedup would have appended the second episode to the first edge's
`episodes` had it seen it, and may do so on the next episode that re-extracts the fact;
this pass does not pretend to be that step. `L→S` and `S→L` facts become self-loops on
`S`, also kept (10 exist in the baseline already), counted in the audit as
`self_loops_created` so an operator can inspect them — a fact relating a thing to itself
is likely to be an artefact of the duplication, but deleting it would discard a cited
fact on a guess.

### 4.7 Invariant #3: "updates append, never overwrite" — is deleting a node a violation?

No, and here is the argument rather than the assertion. Invariant #3 protects
*history*: episodes, facts, their validity intervals and their provenance chains, so that
"how did vendor X's treatment change over time" remains answerable. The merge:

- deletes **no episode, no fact, no `HAS_EPISODE` edge, no `MENTIONS`** — every one is
  re-created with its full property map and its uuid, so every provenance chain
  `fact → episodes → article → source_url` resolves identically before and after
  (§7 proves this by comparing full property maps, not counts);
- touches no `valid_at` / `invalid_at` / `expired_at`;
- removes one thing: a second node whose existence recorded nothing about the corpus. It
  recorded that two extractions raced. The loser is an ingest artefact, and its uuid is
  the only information lost — preserved on the survivor as `merged_from` in case anything
  external held it.

What *would* violate the invariant is collapsing the two facts of §4.6 into one, or
choosing between two summaries with an LLM and discarding the other silently. The pass
does neither. Sequential ingest would have produced exactly the survivor with exactly
these edges; the merge restores the state the temporal policy was defined over.

The one honest cost: the loser's `summary` (§4.4) is not history in the invariant's
sense — it is a model's paraphrase that graphiti overwrites on every mention anyway —
but it is dropped, and the audit prints it.

### 4.8 Where it runs

```
uv run --extra dev python -m graph_extract.cli merge-duplicates            # report only (default)
uv run --extra dev python -m graph_extract.cli merge-duplicates --apply    # destructive
```

> **`--apply` must never run while any ingest or semantic worker is running.** graphiti
> saves every fact and mention with `MATCH (source:Entity {uuid: ...}) ... MERGE (...)`.
> If an in-flight episode has resolved an entity to a loser that `--apply` then deletes,
> that `MATCH` yields zero rows and **the edge is silently not written — no error, no log,
> no retry**. This is the only maintenance job for which the runbook's "safe to run
> alongside ingestion" does not hold. The command prints this constraint and the
> condition it cannot check before it applies; checking it is the operator's job.

- **Report mode is the default and writes nothing.** It prints a JSON audit: each group's
  name, members (uuid, `created_at`, labels, degree, summary length), the planned
  survivor, the name-embedding cosine between members, `label_conflicts`,
  `near_duplicates_not_merged` (normalized-name collisions, §4.1), and the totals. The
  same JSON is printed by `--apply`, plus `merged`, `edges_moved` by type,
  `self_loops_created`, and `summary_dropped` texts.
- **`ingest` (CLI) prints the duplicate count at the end of every run** — one aggregation
  over `:Entity` by name, no LLM — so a concurrent source ingest ends with "N exact-name
  duplicates; run `merge-duplicates`" rather than with a number someone has to go and
  compute. The worker does **not** do this per batch: on a million-entity graph the
  aggregation is seconds per batch for a number that only changes meaningfully per
  source; it belongs in the maintenance audit instead.
- **`cleanup` and `maintenance` include the report, never the apply.** A composite
  command that runs weekly on a timer must not delete nodes on its own; the audit surfaces
  the count and the operator runs `--apply` deliberately. This mirrors the retype
  demote-guard's stance that a destructive step wants a human in the loop.
- **`theme-build` refuses to run while duplicates exist**, unless `--allow-duplicates`.
  Community detection over a fragmented entity graph yields wrong communities, not
  merely stale ones: a duplicated `Microsoft Azure` splits one real community in two,
  and the strong-tier reports written over that partition are plausible and wrong in a
  way nothing downstream can detect. Failing loudly beats emitting them, which is the
  project's standing rule that quality outranks convenience. A merge after
  `theme-build` also invalidates every report citing the loser's membership, so
  enforcing the order merge → theme-build at zero cost is cheaper than regenerating.
  `--allow-duplicates` exists for the operator who has read the report and decided the
  duplicates are immaterial to the communities at hand; it is a deliberate override, not
  a default. The incremental path is also covered: step 4 flags every community the loser belonged
  to as `stale`, which `incremental._dirty` already treats as "regenerate", so a merge
  followed by an incremental `theme-build` regenerates exactly the affected communities.
  Without that flag the merge would be invisible to the refresh — `touched_entities`
  keys on `created_at`, and the survivor and its rewired edges keep theirs.
- **Not automatic after ingest.** The merge is a destructive write over the most valuable
  nodes in the graph, and this project has been bitten by destructive writes before. It
  runs when an operator runs it, after reading the report. The recommended sequence
  after a concurrent source ingest: `merge-duplicates` → read → `merge-duplicates
  --apply` → `cleanup` → `theme-build`.

Runbook entry (`docs/superpowers/maintenance-runbook.md`) to be added with the
implementation. It must state, in this order: that `--apply` is the one job that is
**not** safe alongside ingestion and why (silent edge loss through graphiti's
`MATCH`-then-`MERGE` save, above); that the default is report-only; what the job does;
and the recommended sequence. The runbook's existing "the jobs are safe to run
concurrently with ingestion" sentence must be amended to exclude this one explicitly.

### 4.9 Idempotency and safety

- **Safe to run twice.** The second run finds no exact-name groups and reports zero;
  §7 proves the graph is byte-identical after the second run.
- **Safe on a graph with zero duplicates.** No-op, exit 0, count 0.
- **Safe to interrupt.** One transaction per group; groups already merged are complete
  and correct; a re-run merges the rest. On the first failing group the run **stops**
  and raises (rather than continuing through the remaining groups as `ingest_source`
  does): a failure in a destructive pass means something about the graph is not what
  the design assumed, and the operator should look before more is changed.
- **NOT safe to run concurrently with ingestion — the one job in the runbook that is
  not.** graphiti saves an edge with `MATCH (source:Entity {uuid: $source_uuid}) ...
  MERGE (source)-[e:RELATES_TO {uuid}]->(target)`. If an in-flight episode has resolved
  an extracted entity to a loser that the merge then deletes, that `MATCH` yields zero
  rows and **the edge is silently not written — no error, no log.** The same holds for
  `MENTIONS`. The runbook says "quiet window, no ingest and no worker running"; the
  command prints that constraint before it applies. A lock is out of scope (§11).
- **Never crosses groups.** Every `MATCH` in §4.5 carries `group_id` on both nodes;
  §7 has a same-name-other-group test.
- **Reads its plan, re-verifies at apply.** The report and the apply run the same
  planner; apply re-checks each group inside its transaction (step 0).

### 4.10 What could go wrong

Named so they can be tested for, in rough order of severity:

1. **An edge type the spec did not enumerate.** Mitigated by `DELETE` not `DETACH DELETE`
   (§4.5 step 6): the transaction fails and names the type. A test creates a foreign edge
   on a loser and asserts the group is untouched and the run stops.
2. **Silent edge loss from an overlapping ingest** (§4.9). Mitigated operationally only.
3. **A rewired edge with a different property map** — a dropped `episodes` list breaks
   the sweep and citations; a dropped `invalid_at` resurrects an expired fact; a
   re-generated uuid orphans `Episodic.entity_edges` and `Community.cited_fact_uuids`.
   Mitigated by copying `properties(old)` wholesale and proven by full-map comparison.
4. **`fact_embedding` stored in a form graphiti's cosine scoring does not accept, or with
   drifted values.** Mitigated by the vector procedure; proven by an element-wise
   comparison and a `vector.similarity.cosine(new, old_vector) ≈ 1.0` check on the
   testcontainer.
5. **Stale `source_node_uuid`/`target_node_uuid`.** Proven by asserting the 0-mismatch
   invariant over every edge after the merge.
6. **Merging across groups, or merging case variants.** Both proven absent.
7. **Wrong survivor.** Deterministic rule; a test constructs a group whose earliest node
   is neither first by uuid nor highest by degree and asserts it survives.
8. **Label promotion applied when it should not be** (survivor already typed; or a bare
   survivor whose typed losers disagree, where the union or the first loser's labels
   would be promoted). Tested both ways at two members and all three ways at three.
9. **Losing the loser's summary when the survivor's is empty.** Tested.
10. **A community that is not flagged stale.** Tested via `IN_COMMUNITY`.
11. **A partially merged group after a crash.** Transaction per group; a test injects a
    failure after step 1 (via a foreign edge) and asserts the group is fully rolled back.
12. **External holders of the loser uuid** (a client-supplied `center_node_uuid`, a
    golden-set fixture). Nothing in the repository stores entity uuids (the golden sets
    hold questions and fact-level expectations), and `merged_from` on the survivor keeps
    the mapping. Not tested; noted.

## 5. Configuration

```python
ingest_warmup_articles: int = 8        # ExtractSettings; 0 = off; negative rejected
```

The merge pass adds no setting: its only switch is the `--apply` flag, and a
"merge-on-by-default" configuration would contradict §4.8.

## 6. Error handling

| case | behaviour |
|---|---|
| `is_cold` raises for one item (Neo4j unreachable) | that item's slot holds the exception; siblings continue; `ingest_source` re-raises after the batch and the worker's job fails and retries — the existing contracts, unchanged |
| the `Article` node for a worker job does not exist yet | cold (§3.1) |
| `ingest_warmup_articles < 0` | `ValueError` at settings construction |
| a merge group fails step 0 (graph changed since plan) | that group's transaction rolls back; the run stops and raises naming the group |
| a merge group fails step 6 (foreign edge type) | as above, naming the surviving relationship types |
| `--apply` on zero duplicates | no-op, exit 0 |
| `theme-build` with duplicates present | exit non-zero with the count and the remedy, unless `--allow-duplicates` |

## 7. Testing

**Every test must fail with its fix neutralised, proven by mutation and recorded** in the
`.superpowers/sdd/*-mutations.json` shape already used for the dedup guard (mutant name,
passed, failed, failing test ids). Five vacuous tests were caught during the concurrency
work; a test that cannot fail is a defect, not a safety margin. The mutations below are
the minimum list; each test names the mutation(s) that must break it.

### 7.1 Warm-up (hermetic, `tests/unit/`, fakes in the style of `test_ingest_source_concurrency.py`)

| test | must fail under |
|---|---|
| `is_cold=None` produces the identical call and completion order to today's helper | any change to the non-cold path |
| a cold item never overlaps: a slow warm item launched before it has *finished* before the cold item starts, and nothing starts until the cold item finishes | remove either drain around the cold item |
| warm items on BOTH sides of a cold one still overlap each other | replace the whole `is_cold` body with a sequential loop — without this, "everything is a barrier" satisfies every other test and concurrency itself is unpinned |
| a `BaseException` escaping `is_cold` leaves no task running detached | drop the `try/finally` that cancels and drains what was launched |
| the first `W` items of a cold source run one at a time and the rest overlap (the "started when the slow one finished" pattern) | threshold ignored; predicate always warm |
| a source found warm is never queried again in the run; a cold one is re-queried per item | drop the cache (query count assertion); make the cache unconditional |
| `threshold=0` never queries the driver | remove the short-circuit |
| the predicate is group-scoped: episodes in another `group_id` do not warm a source | drop `group_id` from the Cypher (integration, see 7.3) |
| a superseded `HAS_EPISODE` does not count | drop the `coalesce(r.superseded,false)=false` filter (integration) |
| missing `Article` → cold; remove-only group → warm | invert either branch |
| `is_cold` raising lands in the item's slot, siblings complete, `ingest_source` re-raises after the batch | swallow the exception; raise immediately |
| `run_worker_once` runs cold groups alone and warm groups fanned out; results still zip to groups in order | drop the `is_cold` pass-through |
| at `ingest_article_concurrency=1` the call order is identical with `W=0` and `W=8` | (this is the §3.6 claim; it must be a real assertion on order, not on counts) |
| config: default 8, `0` accepted, `-1` rejected with `ValueError` | remove the validator |

### 7.2 Merge pass (integration, `tests/integration/`, Neo4j testcontainer `neo4j:2026.07.1-community`, wipe per test)

What must be **proven** for a destructive merge — a count is never enough:

| test | proves | must fail under |
|---|---|---|
| **full-map conservation**: snapshot every `RELATES_TO` and `MENTIONS` as `(uuid → full property map, start uuid, end uuid)` before; after the merge every uuid is present, every property map is identical (including `episodes`, `valid_at`, `invalid_at`, `expired_at`, `reference_time`, and `fact_embedding` element-wise) except `source_node_uuid`/`target_node_uuid`, which equal the new endpoints | nothing is lost or altered | drop any single key from the copied map; regenerate the uuid. **Not** "skip the vector re-set": measured (Task 5 review), removing the `db.create.setRelationshipVectorProperty` CALL from step 1 survives this test. `SET new = properties(old)` copies the float32 list the procedure stored on the original edge, and that round-trip is lossless, so the element-wise comparison cannot tell the two apart. The CALL is kept because it is what graphiti uses to write the vector, so the stored form is graphiti's rather than a plain property copy whose storage form the schema cannot vouch for; there is no functional consequence either way today -- the live graph has no vector index on `RELATES_TO.fact_embedding`. The cosine row below is the test that observes the vector, and only with the perturbation |
| endpoints: every moved edge starts/ends at the survivor or the original far node; `f.source_node_uuid = startNode(f).uuid` and the target equivalent hold for **every** edge in the group | topology and stored endpoints agree | skip the endpoint `SET` |
| `vector.similarity.cosine(new.fact_embedding, $old_vector)` within 1e-6 of 1.0 on a testcontainer that has the procedure | the embedding remains scorable | replace the procedure with plain `SET` *and* perturb one element (the perturbation is what makes the test discriminate; a plain-`SET`-only mutant may legitimately pass) |
| loser deleted; survivor's `uuid`, `name`, `labels` (node and property), `summary`, `name_embedding`, `created_at` byte-identical to before, plus `merged_from`/`merged_at` | survivor untouched except the audit | copy loser summary unconditionally; drop the stamps |
| survivor = earliest `created_at`, in a group where the earliest is not the smallest uuid and not the highest degree; uuid tie-break with equal `created_at` | deterministic choice | order by uuid; order by degree |
| label promotion: bare `:Entity` survivor + `:Product` loser → survivor gains `:Product` in both places; `:Tool` survivor + `:Product` loser → unchanged and reported in `label_conflicts` | the `_promote_resolved_node` mirror (two members) | apply promotion unconditionally; never |
| three members, bare survivor throughout: `:Product` + `:Product` losers → `:Product` promoted; bare + `:Product` losers → `:Product` promoted (a bare loser has no say); `:Product` + `:Tool` losers → **nothing** promoted, survivor stays bare `:Entity` in both places, group reported in `label_conflicts` by the planner AND `labels_promoted` empty from the merge | §4.4 at 3+ members: agreement promotes, disagreement reports, the union is never manufactured, first-typed-wins is not reproduced | union the losers' labels; promote the first typed loser's labels regardless; count a bare loser as a disagreement |
| empty survivor summary takes the loser's; non-empty keeps its own; dropped text appears in the audit | §4.4 | invert |
| two facts `S→X` and `L→X` survive as two edges with distinct uuids and facts | §4.6 | collapse by `MERGE` on the pair |
| `L→S` and `L→L` facts become self-loops on `S` with their uuids, counted in `self_loops_created` | §4.6 | drop self-loops |
| `MENTIONS` rewired; an episode mentioning both keeps two edges; `Episodic.entity_edges` all still resolve to existing edge uuids | §4.5 step 3, uuid preservation | regenerate MENTIONS uuid; skip step 3 |
| `IN_COMMUNITY` rewired with `MERGE`; the community gets `stale=true`; `cited_fact_uuids` still resolve | §4.8 incremental hook | skip `SET c.stale` |
| `SAME_AS` from a structural `:Product` rewired | step 5 | skip step 5 |
| same name in another `group_id` untouched, byte-identical snapshot | invariant #4 | drop `group_id` from any `MATCH` |
| `Access Control` / `Access control` untouched; listed in `near_duplicates_not_merged` | §4.1 | add `toLower` |
| **foreign edge**: `(L)-[:FOO]->()` — the group's transaction rolls back (edges moved in step 1 are back on the loser, survivor has no `merged_from`), the run stops, the error names `FOO` | `DELETE` not `DETACH DELETE`; transaction per group | `DETACH DELETE`; auto-commit statements instead of one transaction |
| report mode (no `--apply`) leaves a byte-identical graph and lists the group with the planned survivor | §4.8 | any write in the planner |
| second `--apply` run reports zero and leaves a byte-identical graph | idempotent | (a mutant that re-stamps `merged_at` on a no-op must fail this) |
| zero-duplicate graph: no-op, zero counts | §4.9 | — the test must still assert the snapshot, or it is vacuous |
| step 0: rename a member between plan and apply → that group aborts, others merge | re-verification | drop `name = $name` from the re-match |
| `theme-build` guard: refuses with duplicates present; `--allow-duplicates` proceeds; zero duplicates proceeds | §4.8 | remove the guard; invert it |
| `ingest` CLI prints the count; `cleanup`/`maintenance` audits carry the report-mode payload directly under `duplicates` (`groups`, `totals`, `near_duplicates_not_merged`, `label_conflicts`) and never `totals.merged` | §4.8 | — |

"Byte-identical snapshot" means the full node and relationship property maps plus
labels and endpoints for the whole test group, compared as data, not counts.

### 7.3 Live, opt-in (`@live`)

- The warm predicate against the baseline graph returns warm for both pilot sources and
  cold for an unknown source id; the group-scope and superseded filters are checked
  against the real schema rather than a fixture.
- `merge-duplicates` in **report mode only** against the baseline returns 0 groups and
  lists the 6 known near-duplicates. `--apply` is never run against the baseline; it is
  the reference state.

## 8. Success criteria

1. At `ingest_article_concurrency = 1`, call order is unchanged with warm-up at its
   default; the §7.1 order test pins it.
2. With warm-up on, a cold article never overlaps with any other article (§7.1).
3. A source's warmth persists across runs and processes without new state.
4. The merge pass conserves every fact and mention edge with its full property map and
   uuid, leaves stored endpoints equal to topology, never crosses groups, never merges
   non-identical names, and refuses to delete a node that still has any edge.
5. Report mode writes nothing; `--apply` twice equals `--apply` once.
6. `theme-build` cannot run over known duplicates by accident.
7. The next paid concurrent run (§9) reports duplicates before and after warm-up, and
   zero after the merge, against the 999 baseline.

## 9. Validation — the next concurrent run

The same 83 articles, `ingest_article_concurrency = 4`, `ingest_warmup_articles = 8`,
into an isolated group using the A/B runner's supersede-and-restore method (never a
reset of the baseline):

| | measured (N=4, no warm-up) | to measure (N=4, W=8) |
|---|---|---|
| exact-name duplicates | 30 | predicted ~6 (§3.3) |
| near-duplicates (normalized) | not measured | ? — decides whether §4.1's exclusion holds |
| wall clock | 2 h 10 m | predicted ~2 h 25 m (16 sequential articles) |
| after `merge-duplicates --apply` | — | 0 duplicates; entity count 999 ± extraction nondeterminism (the A/B saw ~28% of names differ run to run and cancel) |

If the residual is far above 6 the risk model is wrong for the fan-out shape and `W`
should be re-derived from the run's own spans (the recompute in §1.1 is a 30-line
read-only script and should be committed alongside the implementation). If the
normalized-name count grows materially over the baseline's 6, §11's first item comes
back in scope. Cost roughly $10 and ~2.5 h.

## 10. Rejected alternatives

- **Fixed "first N of this run" warm-up in `ingest_source` only.** Simplest, but it has
  no answer for the worker (which does the bootstrap lane, the case that matters most),
  re-warms on every resumed run, and warms up a re-extraction whose hubs already exist.
  The graph-derived predicate is one query and answers all three.
- **Warm-up as a per-run prefix count handed to the helper** (`warmup=int`). Works for
  `ingest_source`; for the worker it needs a separate partition-and-reorder step, i.e.
  two mechanisms. One predicate, one helper.
- **Per-product or per-vendor warm-up unit.** Would avoid re-warming a product's second
  source, but only sources have an order (`sort_order`), and the §1.1 mechanism is "the
  opening pages of a *source*". Waste is bounded at 8 articles per source.
- **Non-strict barrier** (cold items sequential only among themselves). §3.2.
- **graphiti's `dedupe_nodes_bulk` as the repair.** Costs LLM calls, gives a
  nondeterministic answer to a question that has a deterministic one, and the
  concurrency spec already deferred it for exactly this reason.
- **Normalized-name (case/whitespace) merge.** §4.1: alters the reference baseline,
  no evidence of concurrency producing such pairs, and the report shows them anyway.
- **Merge automatically at the end of `ingest` or in `cleanup`.** Destructive write on a
  timer, over the hub nodes, with no human having read the plan. No.
- **Collapse facts that meet after the merge.** Each has its own provenance; the graph
  already carries 440 multi-edge pairs; and invariant #3 says facts are never
  overwritten. §4.6.
- **`DETACH DELETE` the loser for simplicity.** It is the exact mechanism by which an
  unenumerated edge type would be lost silently. §4.5 step 6.
- **Copy `name_embedding` or `fact_embedding` with plain `SET`.** Cannot be shown from
  the schema to store what graphiti stores; the procedure can.
- **Lower N to 2 instead.** Halves the duplication for 1.9x; keeps the §1.2 tax on what
  remains and forgoes 40% of the measured speedup. Available as a fallback if §9 fails.

## 11. Out of scope

- **Normalized-name / fuzzy / LLM-judged merging.** No evidence; the 6 baseline pairs are
  not concurrency's doing; the report keeps them visible.
- **Regenerating a merged survivor's summary with an LLM.** graphiti does it on the next
  mention; the pass stays Cypher-only.
- **A lock that stops `merge-duplicates --apply` while a worker is running.** Operational
  constraint for now (§4.9). Worth building if the worker becomes an always-on service;
  today it is run in bounded batches.
- **Multiple worker processes.** Both the semaphore and the warm-up barrier are
  per-process; two workers against one graph would reintroduce the hazard. The
  deployment is one worker (`CLAUDE.md`), and nothing here changes that.
- **Re-measuring the DB saturation ceiling and N above 4.** The A/B already shows the two
  slowest prompts degrade 1.5–1.8x at 10–19 in flight; still open, still not this spec.
- **Edge-level dedup of same-fact multi-edges.** §4.6.
- **The 6 near-duplicate pairs in the baseline.** Recorded here; not repaired.
- **A per-source or per-vendor `W` override.** §3.3: no source size in the corpus makes
  the uniform value a problem, and the global knob covers an unusual bootstrap.
