from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    docext_base_url: str
    docext_read_key: str
    docext_admin_key: str = ""  # passed to docext.client.make_docext_client(admin_key=...)
    docext_verify_tls: bool = False
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    # Optional overrides letting the compatibility harness (src/compat/) target a
    # DIFFERENT Neo4j than the one the rest of the stack uses, without editing the
    # neo4j_* values. Empty means "fall back to the neo4j_* value", field by field.
    compat_neo4j_uri: str = ""
    compat_neo4j_user: str = ""
    compat_neo4j_password: str = ""
    llm_base_url: str = "http://srv-llm.home.lan:8080/v1"
    # gpt-5-mini is the chosen extraction tier (Azure Responses API,
    # reasoning=minimal). gpt-oss-120b was evaluated 2026-07 (same Azure
    # endpoint, cheaper) but on an identical 8-article sample it ran ~2x
    # slower and lost AvailableIn/region extraction (0 vs 6 AvailableIn facts,
    # 1 vs 4 regions) -> NO-GO. The pipeline still supports it via .env
    # (LLM_MODEL=gpt-oss-120b, generic_json_schema, reasoning_effort=low) if
    # revisited. Azure endpoint/api-version/key come from .env (never committed).
    llm_model: str = "gpt-5-mini"
    # API key for the LLM endpoint. Local llama-server ignores it ("not-needed");
    # a cloud endpoint (e.g. OpenRouter) needs a real key, supplied via LLM_API_KEY
    # in .env (never committed). Only the LLM/extraction path uses this — the
    # embedder stays local (TEI/Jina, "not-needed").
    llm_api_key: str = "not-needed"
    # Azure OpenAI: when llm_base_url points at *.azure.com/*.cognitiveservices,
    # an AsyncAzureOpenAI client is used (Responses API + structured mode). Set
    # the api version and (for reasoning models like gpt-5-mini) the effort.
    llm_api_version: str = ""
    llm_reasoning_effort: str = "minimal"  # minimal|low|medium|high (gpt-5 family); gpt-oss uses low|medium|high
    embed_base_url: str = "http://srv-llm.home.lan:8082/v1"
    embed_model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    embed_dim: int = 768
    embed_max_batch: int = 32  # TEI/Jina max_client_batch_size; cap embed batches to this
    embed_timeout_seconds: float = 90.0
    chonkie_base_url: str = "http://srv-llm.home.lan:8084"
    chonkie_model: str = "mirth/chonky_modernbert_base_1"
    group_id: str = "backup-docs"
    max_chunk_tokens: int = 1800
    min_chunk_tokens: int = 128
    # Pack consecutive chunks into episodes of up to this many tokens (never above
    # the tier's max_chunk_tokens). 0 = off. Set ONCE before the bootstrap: changing
    # it re-keys an ingested article's episodes on its next re-ingest.
    # MEASURED 2026-09-24, keep 0: packing to 1,200 cut cost 36% and wall clock 40%
    # but lost 28% of facts -- per-call extraction saturates (~4-10 facts per call
    # whatever the input size), partly because CHEAP_TIER_SALIENCE asks for a fixed
    # fact count per chunk. docs/superpowers/chunk-packing-ab-2026-09-24.md.
    pack_target_tokens: int = 0
    max_coroutines: int = 3
    # Default set to generic_json_schema per Task 6 evidence: gpt-oss-20b via
    # llama-server fails the OpenAIClient "structured" (Responses API) path
    # (markdown-fenced/malformed JSON, 0 entities), but OpenAIGenericClient
    # with json_schema extracts cleanly. Task 7's probe compares json_schema
    # vs json_object and confirms this.
    llm_client_mode: Literal["structured", "generic_json_schema", "generic_json_object"] = "generic_json_schema"
    # Repetition penalties for the extraction tier. graphiti never sends these (its
    # client passes only model/messages/temperature/max_tokens/response_format) and we
    # pin temperature=0.0, so a weaker model that starts an ascending-integer run in an
    # unbounded list[int] field (prompts/extract_edges.py: episode_indices has no
    # maxItems) cannot escape it -- under a strict JSON schema the only legal next
    # tokens there are digits, ',' and ']'. It then burns the whole max_tokens budget
    # and truncates into invalid JSON. A frequency penalty makes the repeated digit and
    # comma tokens progressively less attractive so ']' eventually wins.
    # Default 0.0 = inject nothing, preserving the proven gpt-5-mini behaviour; raise
    # only for models that need it.
    # Cap on graphiti's three unbounded array<integer> schema fields
    # (episode_indices, duplicate_facts, contradicted_facts). These are index lists
    # whose legitimate length is tiny (one entry per episode/candidate), so a generous
    # cap is inert for a well-behaved model but makes the ascending-integer runaway
    # unrepresentable. 0 disables the bound.
    llm_max_index_array: int = 25
    # Move each prompt type's static user-message tail into the system message so a
    # provider prefix cache can serve it (graph_extract.cache_layout). OpenAI-compatible
    # chat.completions path only (the Azure Responses path is untouched). Off until
    # the A/B shows cache hits up and extraction unchanged.
    llm_cache_layout: bool = False
    # OpenAI-family models on OpenRouter (BACKLOG 59 follow-up, gpt-6-luna): their
    # structured outputs reject graphiti's schemas unless every object is closed
    # (additionalProperties: false) and lists every property as required (measured
    # 2026-10-14, a 400 otherwise); maxItems is accepted, so the index-array bound
    # stays. Off by default: the solar-pro4 cheap tier never needed it.
    llm_strict_schema: bool = False
    # Reasoning effort sent to OpenRouter as `reasoning: {effort}` on extraction calls
    # (gpt-6-luna: "none" -> 0 reasoning tokens). Empty sends nothing -- the solar-pro4
    # cheap tier's long-standing behaviour, which an effort would change (`low` turns
    # its thinking on). Separate from llm_reasoning_effort, which the Azure strong tier
    # reads and the cheap view inherits.
    llm_openrouter_reasoning_effort: str = ""
    # graphiti always sends `temperature`; reasoning models such as gpt-6-luna do not
    # take it, and with OpenRouter's require_parameters routing that leaves NO
    # endpoint (a 404, measured 2026-10-14). False drops it from extraction calls.
    llm_send_temperature: bool = True
    # Per-request timeout of the strong-tier client (and, through the cheap view, of the
    # cheap tier's via cheap_llm_timeout_seconds). One 90 s value for every tier starved
    # the vast.ai GPU tier: at ~47 generated tokens/s per request an answer over ~4k
    # tokens cannot finish, so the client gave up and every retry failed the same way
    # (BACKLOG 55). The GPU worker Deployment raises the cheap one.
    llm_timeout_seconds: float = 90.0
    # When the CHEAP tier answers graphiti's edge-dedup prompt with an out-of-range
    # candidate index (graphiti drops it silently: a missed dedup or a missed
    # invalidation), re-issue that one prompt on the strong tier and use its reply.
    # Side-effect free -- nothing is written until the reply is resolved.
    #
    # DEFAULT OFF, set from the 2026-09-11 live measurement (17 articles, 1,145
    # dedup calls): the strong tier makes the SAME mistake on 20 of 84 retries
    # (24%), and graphiti's own drop warning still fired 21 times AFTER the retry.
    # This is not a cheap-model quality problem -- all 154 out-of-range indices were
    # invalidation-candidate indices placed in `duplicate_facts`, 0 hallucinated --
    # so the retry treats a symptom of graphiti's shared index space at the price of
    # a slow strong-tier call on ~10% of all dedup calls. Turn it on deliberately if
    # a partial (~76%) recovery is worth that. Detection and counting happen
    # regardless of this switch -- see graph_extract.dedup_guard.
    dedup_retry_on_strong: bool = False
    # graphiti issues an UNFILTERED candidate search per extracted fact to feed
    # contradiction detection. PROFILE shows it scanning every fact
    # (NodeByLabelScan + Expand(All)) while the duplicate search is an index seek:
    # scan_ms = 241 + 0.0699 x facts, so 42.9 s at the projected 610k-fact corpus
    # and past the dedup LLM call's cost at ~3,900 articles -- under 4% of it.
    #
    # DEFAULT OFF. The feature that scan serves does not work: all 140 measured
    # invalidations were contradiction-driven, and the inspectable ones are
    # refinements and near-duplicates ordered by the sequence DocExtractor crawled
    # the pages, because the corpus carries no document revision date. Re-enabling
    # is NOT a flag flip -- see the spec's section 7.
    #
    # Scope: off suspends the scan and CROSS-PAIR invalidation. Turning it on is
    # NOT enough to bring cross-pair invalidation back: with the default
    # ingest_same_pair_contradictions=False, dedup_guard clears EVERY contradicted
    # index, so the scan would run and invalidate nothing (startup logs a WARNING
    # for that combination). Re-enabling needs BOTH flags plus the section 7 review.
    ingest_detect_contradictions: bool = False
    # Ingest-time contradiction invalidation, ALL of it (BACKLOG 33, D4). The name
    # says "same-pair" because that is the only path live while
    # ingest_detect_contradictions=False: the dedup model says a new fact contradicts
    # an existing fact between the SAME two entities and graphiti expires one of
    # them. Measured 0 of 14 genuine on the live graph -- limitations and
    # refinements read as contradictions, removing core facts from answers
    # (pre-bootstrap-decisions-2026-09-23.md D4).
    #
    # False (default) clears the WHOLE contradicted_facts list in dedup_guard --
    # same-pair indices AND, were detection re-enabled, cross-pair ones -- so ingest
    # writes no contradiction-driven invalidation at all. Genuine change is carried
    # by updates and the weekly sweep. True alone restores same-pair only; cross-pair
    # also needs ingest_detect_contradictions=True and the design review in
    # docs/superpowers/specs/2026-09-12-suspend-contradiction-detection-design.md §7.
    ingest_same_pair_contradictions: bool = False
    # Derive a fact's `valid_at` from its episode's reference time
    # (`content_changed_at`) instead of graphiti's per-fact timestamp LLM call.
    #
    # DEFAULT ON. Measured on the 2026-09-14 pilot re-ingest: the LLM call dated
    # only 817 of 3,590 facts (23%), cost 21.7% of all LLM time (5,476 s of a
    # 6.9 h run), and produced semantically mixed dates -- an in-world date from
    # the fact text on about half, the reference time on the rest -- which is the
    # mixture that caused this project's phantom invalidations.
    #
    # The cost of the switch is narrower than it looks: only the per-fact FALLBACK
    # timestamp call is skipped. graphiti's combined edge-extraction prompt still
    # sets `invalid_at` from in-text end dates ("deprecated in 2024" --
    # graphiti_core/prompts/extract_edges.py, parsed at edge_operations.py:282-288),
    # and such an edge is left untouched here. What is lost is the fallback's end
    # date for facts the extraction prompt left undated. Accepted, because
    # invalid_at should record evidence of supersession, not a model's reading of
    # prose. (answer_api.temporal.is_current keeps future end dates current.)
    valid_at_from_content_changed: bool = True
    # Index-backed similarity search (Phase B, spec 2026-09-23). Unbounded
    # edge/node similarity searches -- retrieval and node dedup -- go to tuned
    # Neo4j vector indexes instead of graphiti's full cosine scan, which grows
    # linearly (~12.5 s per call at 250k rows). False restores graphiti's own
    # functions exactly and skips index management.
    vector_search_enabled: bool = True
    # Neighbours requested from the index before the group/score filter and the
    # caller's `limit` are applied. 200 measured 97.4-99.6% of exact search's
    # dedup partners at 250k entities (ann-dedup-probe-2026-09-23.md).
    vector_search_fetch_k: int = 200
    # How long a service start waits for the two vector indexes (only those --
    # not every index in the database) before continuing. One still POPULATING
    # after it is logged at WARNING and the start proceeds; searches fall back
    # to the exact scan until it is ONLINE (after ~30 s of Neo4j waiting on it
    # per call -- vector_search's module docstring). FAILED or mis-configured
    # indexes still refuse to start.
    vector_index_startup_wait_seconds: float = 60.0
    # How many ARTICLES to ingest concurrently. Episodes within an article always
    # stay sequential: consecutive chunks of one document share entities most
    # heavily, and graphiti resolves entities by searching the graph as it
    # currently stands, so two in-flight episodes extracting a not-yet-present
    # entity would each create it under a different uuid.
    #
    # DEFAULT 1 -- byte-for-byte today's call order. Raising it is gated on the
    # A/B in the design spec's section 8: re-ingest the same 83 pilot articles and
    # compare entity count against the sequential baseline of 999.
    ingest_article_concurrency: int = 1
    # Articles at the start of a source that run one at a time, with nothing
    # else in flight, before the fan-out begins. 0 = off.
    #
    # Measured (docs/superpowers/ab-concurrency-2026-09-14.md): at N=4 the
    # pilot produced 30 duplicate entities, all exact-name, concentrated on hub
    # entities (mean article span 7.6 vs 2.4). Duplicate risk for an entity in
    # k articles scales with k-1, and per source by sort_order the first 8
    # articles carry 79.5% of that risk weight -- documentation sources open
    # with overview pages that name the product and its core concepts. The knee
    # is at 4 (71.3%); 12 buys 3 more points for 50% more sequential articles.
    ingest_warmup_articles: int = 8
    llm_frequency_penalty: float = 0.0
    llm_presence_penalty: float = 0.0
    # Opt-in JSONL capture of every LLM call's full prompt/response (design:
    # 2026-09-16 extraction fine-tune dataset, Option B). Empty = OFF, and off
    # must be free: instrument() only opens/writes a file when this is set. See
    # graph_extract.usage.instrument / capture_llm_call. Never commit a path
    # under version control, and never point it at anything that syncs off-box
    # without review -- captured records include full request/response text.
    llm_capture_path: str = ""
    judge_base_url: str = ""
    judge_model: str = ""
    judge_api_key: str = ""  # if empty, the judge reuses llm_api_key (fallback path)
    # --- theme-builder / community layer (design: theme-builder-community-layer) ---
    # Report tier defaults to the synthesis/judge tier (GLM-5.2) when left empty.
    # The faithfulness judge for the router golden-set eval. Falls back to judge_*,
    # but eval_router GUARDS against the fallback resolving to the synthesis tier --
    # a judge grading its own output inflates faithfulness, and the failure is
    # invisible in the score. Point this at a DIFFERENT model family from synthesis
    # so the two do not share failure modes.
    eval_judge_base_url: str = ""
    eval_judge_model: str = ""
    eval_judge_api_key: str = ""
    report_llm_base_url: str = ""
    report_llm_model: str = ""
    report_llm_api_key: str = ""
    # The report verifier. Falls back to eval_judge_*. MUST NOT resolve to the
    # report model -- theme_builder.report._verify_client_and_model raises if it
    # does, because a model grading its own findings inflates the result in a way
    # the result cannot reveal.
    verify_llm_base_url: str = ""
    verify_llm_model: str = ""
    verify_llm_api_key: str = ""
    # Reasoning effort for the verifier tier. Measured 2026-09 on an 11-finding
    # report (~deepseek/deepseek-v4-flash-latest, a REASONING model) via
    # theme_builder.report.verify_report: max_tokens=4000, retried at 12000, both
    # exhausted on reasoning tokens alone (finish_reason='length', content=None) --
    # raising the cap does not help, reasoning just expands to fill it. "low" fixed
    # it (finish_reason='stop', 3028 completion tokens, a real verdict flagging 2 of
    # 11 findings unsupported). Measured "enabled: false" too: it "works"
    # (finish_reason='stop', valid JSON) but in 13 completion tokens it approved
    # everything -- a rubber stamp, not a check. NEVER set this to disable
    # reasoning entirely; "low" is the floor that keeps the verifier a judge rather
    # than a reflex.
    verify_reasoning_effort: str = "low"
    # First-attempt completion budget for the verifier; the retry gets 3x. 4000 (and
    # its 12000 retry) was right when measured above, but how much a reasoning model
    # thinks depends on WHICH OpenRouter provider serves it: re-measured 2026-10-11
    # on one 16-finding report, BaseTen finished in ~2,600 completion tokens while
    # Cohere needed 7,000-9,000 and ignored `effort: low`. At 4000 every Cohere-routed
    # call came back empty and most of a theme-build was staged unverified. Only the
    # tokens actually used are billed, so the headroom costs nothing when unneeded.
    verify_max_tokens: int = 8000
    # Drop dust communities smaller than this. 10, not 3 (2026-10-10, measured on the
    # 45k-entity post-Veeam graph): 3 kept 1,717 communities -- ~25 h of one-at-a-time
    # report writing -- where 10 keeps 284 and still covers 89% of level-0 entities;
    # the 1,433 communities it drops are 3-9-entity fragments holding ~4,300 entities.
    leiden_min_community_size: int = 10
    # Community reports generated at once by an incremental theme-build.
    theme_report_concurrency: int = 8
    leiden_max_levels: int = 3           # cap on intermediate Leiden levels
    # Leiden resolution. 5, not GDS's 1.0 (2026-10-10, measured detection-only on the
    # 45k-entity post-Veeam graph): at 1.0 product hub entities ("Veeam Backup &
    # Replication": 9,230 neighbours, median degree 2) held one 10,706-entity community at
    # level 1 (the level global/DRIFT read) -- a quarter of the graph under one report,
    # and the cause of weak global answers in the Tier 1 eval. 5.0: 295 level-1
    # communities, largest 2,848, level-0 coverage 32k vs 36k. Capping hub degree split
    # it further but orphaned the 7-10k entities attached only to a hub; degree-
    # normalised weights fragmented level 0 and merged levels 1-2 (BACKLOG 57).
    leiden_gamma: float = 5.0
    # Output cap for a community report. 8000 was measured against GLM-5.2 (3000
    # skipped ~24% of communities, 8000 skipped ~0). GLM-5.3-flash reasons more and
    # truncates at 8000 -- and a truncated report is DROPPED silently, because
    # generate_report returns None and writeback only writes communities that have
    # one. Reasoning counts against this budget, so cap the reasoning separately
    # rather than only raising the ceiling.
    report_max_tokens: int = 16000
    # Reasoning effort for the report tier: "low" keeps reasoning (it helps report
    # quality) while stopping it from consuming the whole output budget. Empty
    # string sends no reasoning parameter at all.
    report_reasoning_effort: str = "low"
    report_token_budget: int = 12000     # per-community context budget (~chars/4)
    # Ceiling the budget grows to on fact-rich communities (theme_builder.context.
    # context_budget: ~80 tokens per fact). 12000 fits ~40 facts; a 7,300-fact hub
    # community then gave each of its 30 top entities about one fact (2026-10-11), too
    # few for its report to cover them. 48000 fits ~450.
    report_token_budget_max: int = 48000
    report_top_entities: int = 30        # member entities included in a report's context
    # --- incremental community refresh (design: incremental-community-refresh) ---
    theme_refresh_jaccard_tau: float = 0.5   # min member-set Jaccard to treat a fresh community as the same as a persisted one
    # --- global (map-reduce) search (design: global-search) ---
    map_llm_base_url: str = ""   # map tier; defaults to the judge/synthesis tier when empty
    map_llm_model: str = ""
    map_llm_api_key: str = ""
    # --- answer-path reasoning bounds (design: answer-path-hardening) ---
    # One setting PER TIER, not one shared knob, because the right value is a
    # property of the MODEL a tier points at, and tiers are configured per tier.
    # Measured 2026-09-11: GLM-5.3-flash (synthesis) and deepseek-v4-flash (eval
    # judge) reason by default, so an unbounded effort competes with the answer
    # for max_tokens and returns empty content (8 empty-content retries in ~40
    # reduce calls); "low" bounds it and keeps the answer. upstage/solar-pro4
    # (map + classifier) does NOT reason by default (reasoning_tokens=0), and
    # `effort: low` SWITCHES THINKING ON: at max_tokens=8 every classifier reply
    # was empty; at max_tokens=2000 one of two map calls exhausted the whole
    # budget on reasoning and returned nothing. A shared "low" would have broken
    # both. Empty string = send no reasoning parameter (the model's own default).
    # NEVER "disable" reasoning on a reasoning model instead of bounding it:
    # see verify_reasoning_effort for the 13-token rubber stamp that produced.
    synthesis_reasoning_effort: str = "low"   # judge_* tier: local/drift/reduce synthesis
    map_reasoning_effort: str = ""            # set "low" if map_llm_* is a reasoning model
    eval_judge_reasoning_effort: str = "low"  # the faithfulness judge (eval only)
    # Pre-rerank shortlist size. Inert once a reranker is configured: rerank_top_n
    # bounds what reaches the map step instead (see rerank_top_n below).
    global_shortlist_k: int = 10
    global_default_level: int = 1
    # --- vendor/product scoping, global search (spec 2026-09-25 §4.3, BACKLOG 52) ---
    # Pre-cut candidate pool widened to at least this many before the in-scope
    # share filter runs, so scoping a query doesn't just re-filter the same k
    # cosine winners down to fewer than k.
    global_scope_candidates: int = 24
    # A community is kept only if at least this share of its cited_fact_uuids
    # trace to the scope's vendor(s)/product(s) -- see design spec §4.3 step 3
    # for the measured Azure/AWS example this threshold was picked against.
    global_scope_min_share: float = 0.5
    # --- facet-balanced answers (spec 2026-10-12) ---
    # Comparison: each facet's local search gets max(this, ceil(k / facets)) facts.
    local_min_per_facet: int = 6
    # Scoped local search re-fetches at k*3, k*10, k*30 until k edges survive the
    # scope filter, never above this (the Veeam Agent scope kept 7 of 45, 2026-10-12).
    local_scoped_fetch_cap: int = 450
    # When that post-filter still leaves fewer than k, rank the scope's OWN facts
    # (search._scoped_scan): a small vendor next to a huge one has none in the
    # corpus-wide top-N (Cohesity scoped search returned 0 facts, 2026-10-12). Scopes
    # with more episodes than this skip the scan (served by the post-filter). Live
    # cost is ~0.45 ms/episode (Microsoft, 4.7k episodes: 2.3 s, 2026-10-13), so
    # 6000 keeps the scan inside a 3 s budget.
    local_scoped_scan_max_episodes: int = 6000
    # The scan reranks its top-N cosine candidates with the cross-encoder when one is
    # configured: cosine ranked Cohesity's first DataLock fact ~103rd for "Which backup
    # vendors support immutable backups?"; reranking the top 200 put DataLock/WORM at
    # rank 1 (and top 3 for "Does Cohesity support immutable backups?", 2026-10-13).
    local_scoped_scan_candidates: int = 200
    # Facet searches (comparison/enumeration sides, facet-scoped DRIFT follow-ups) pull
    # this many candidates and rerank them with the cross-encoder down to k: gate run 2
    # showed large scopes (Veeam, Microsoft) filling k from the corpus-wide hybrid
    # search with poor ranking ("PostgreSQL WAL Files Backup" for a ransomware
    # question); reranking a 40-fact pool put the relevant facts on top (2026-10-13).
    # Single-shape searches never rerank this way.
    local_facet_rerank_pool: int = 40
    # --- list mode (spec 2026-10-15) ---
    # Single-vendor "list" questions ("Which workloads ...", "Where can I restore ...")
    # are answered exhaustively: cheap-tier expansion into item-kind queries, a reranked
    # local search per query, near-duplicate collapse, a final rerank, then up to
    # list_supported_k supported facts plus list_limitations_k limitation facts.
    list_mode_enabled: bool = True
    list_expansion_queries_max: int = 10
    list_per_query_pool: int = 40
    list_per_query_k: int = 20
    # Raised 50 -> 80 with diversity-aware (MMR) selection (2026-10-15 amendment §10.3),
    # then 80 -> 250 (2026-10-16): the free gate went 59/67 -> 63/67 expected items, as
    # most missing items were retrieved but ranked past #80.
    list_supported_k: int = 250
    list_limitations_k: int = 12
    # Relation expansion (§10.1): the (subject entity, relation name) pairs of the top
    # list_relation_seed_facts round-one facts are grouped; the list_relation_groups
    # most frequent pairs pull up to list_relation_group_cap current sibling edges each.
    list_relation_seed_facts: int = 40
    list_relation_groups: int = 6
    list_relation_group_cap: int = 300
    # Siblings kept across all groups (group order, then uuid within a group).
    list_relation_total_cap: int = 600
    # MMR: rerank score minus list_mmr_lambda x highest raw cosine to a chosen fact.
    list_mmr_lambda: float = 0.5
    # Documents per final-rerank call (Voyage accepts ~1000 at most).
    list_rerank_batch: int = 900
    # Raw cosine of fact embeddings at or above which two facts are one (sources merged).
    # 0.92 merged "VMware Cloud Director" into a vSphere fact (2026-10-15 amendment).
    list_dedup_cosine: float = 0.96
    # Keyword sweep (spec 2026-10-15 §12.1): a Lucene full-text search of fact text for
    # the question's own distinctive words, run alongside the per-query searches.
    list_sweep_cap: int = 2000
    list_sweep_scoped_overfetch: int = 4
    # Single answers keep the threshold they were reviewed at; lists use list_dedup_cosine.
    local_single_dedup_cosine: float = 0.92
    # Every other single question reranks this many candidates down to k (was raw
    # hybrid order: Q1/Q2 of 2026-10-15 retrieved mostly limitations and duplicates).
    local_single_rerank_pool: int = 80
    # Comparison: communities per facet reaching the map step (single keeps rerank_top_n).
    global_facet_top_n: int = 3
    # Comparison: every facet also gets this many facts from a scoped, reranked local
    # search, rendered under its heading after its community findings. Run 6
    # (2026-10-13): a Veeam Backup for Microsoft 365 side with no usable community
    # finding, and AWS/Veeam sides with only off-point findings, were all reported as
    # "not covered" although the facet's own local search found the right facts.
    global_facet_local_k: int = 8
    # Enumeration: evidence gathered per content vendor before its cheap check.
    enum_communities_per_vendor: int = 2
    enum_facts_per_vendor: int = 8
    # Vendors gathered/checked at once: at 40 vendors the unbounded fan-out issued
    # up to 120 hybrid searches plus 40 scans at once against Neo4j.
    enum_concurrency: int = 8
    # ScopeResolver.load()'s structural catalog is reloaded lazily on use once
    # older than this (plan ruling 4 -- no separate freshness module exists).
    scope_reload_seconds: int = 3600
    # --- DRIFT search (design: drift-search) ---
    drift_primer_level: int = 1      # community level the primer shortlists at
    # Pre-rerank primer shortlist size. Same caveat as global_shortlist_k: once a
    # reranker is configured, rerank_top_n bounds the primer set, not this.
    drift_primer_k: int = 5          # reports shortlisted for the primer
    drift_max_followups: int = 4     # follow-ups kept per round (relevance-budgeted)
    drift_followup_k: int = 8        # local-search k per follow-up
    drift_iterations: int = 1        # follow-up rounds; clamped to [1,2] at call time
    # --- /answer router (design: answer-router) ---
    router_default_mode: str = "drift"   # mode when no heuristic fires and the cheap classifier is absent/uncertain
    # --- hybrid extraction routing (design: hybrid-extraction-router) ---
    # ON by default; degrades to strong-only when cheap_llm_api_key is empty.
    extraction_routing: bool = True
    # Cheap tier = ling-2.6-flash via OpenRouter. api_key from .env (never committed).
    cheap_llm_base_url: str = "https://openrouter.ai/api/v1"
    cheap_llm_model: str = "inclusionai/ling-2.6-flash"
    cheap_llm_api_key: str = ""
    cheap_llm_client_mode: Literal[
        "structured", "generic_json_schema", "generic_json_object"] = "generic_json_schema"
    # The cheap tier's own cache-layout switch (mapped onto llm_cache_layout for the
    # cheap client view). Cached input on solar-pro4 bills $0.018/M vs $0.09/M.
    cheap_llm_cache_layout: bool = False
    cheap_llm_timeout_seconds: float = 90.0   # mapped onto llm_timeout_seconds (cheap view)
    cheap_max_chunk_tokens: int = 900   # smaller chunks for the verbose cheap model
    # Route an article to the STRONG tier when it is a dense table:
    dense_table_line_ratio: float = 0.25
    dense_pipe_count: int = 200
    # Cross-encoder reranker (Voyage AI, Cohere-compatible /v1/rerank). Scores
    # community relevance for the global path, replacing the LLM's improvised
    # 0-10 rating. rerank_top_n / rerank_score_floor are set from measured score
    # distributions -- see the 2026-09-10 spec, not guessed.
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""
    rerank_candidates: int = 50    # cosine pre-cut before the API call
    # Max communities reaching extraction. Once configured, this -- not
    # global_shortlist_k / drift_primer_k -- is what bounds the shortlist.
    rerank_top_n: int = 4          # max communities reaching extraction
    # 0.40, not the provisional 0.45: measured across all 10 golden global/DRIFT
    # questions (docs/superpowers/rerank-threshold-measurement.md). Scores are
    # compressed -- the two communities the inversion complaint was about sit
    # 0.023 apart -- so the floor only removes the clear tail and rerank_top_n
    # does the selecting. 0.45 would have cut the very community that complaint
    # wanted ranked higher; 0.55+ refuses 3 of 10 answerable questions.
    rerank_score_floor: float = 0.40
    # --- MCP server (spec 2026-10-14) ---
    # Bearer token required on /mcp. Empty (the default) leaves /mcp unmounted: a
    # deployment without the token never exposes the MCP endpoint. Set only in the
    # cluster Secret, never in the repo, image or chart values.
    mcp_bearer_token: str = ""
    # Host header values the MCP transport accepts (DNS-rebinding protection): the
    # ingress host, plus localhost for a kubectl port-forward.
    mcp_allowed_hosts: list[str] = ["graphrag.k3s.home.lan", "localhost:*", "127.0.0.1:*"]
    # Seconds between progress notifications while answer_question runs (comparisons
    # and enumerations take 1-7 minutes; clients time out silent calls).
    mcp_progress_seconds: float = 10.0

    @field_validator("ingest_article_concurrency")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        # Rejected, not clamped: a 0 in the environment means someone believes
        # they configured something, and silently running sequentially would hide
        # that from them.
        if v < 1:
            raise ValueError(f"ingest_article_concurrency must be >= 1, got {v}")
        return v

    @field_validator("ingest_warmup_articles")
    @classmethod
    def _not_negative(cls, v: int) -> int:
        # Rejected, not clamped, for the same reason as its sibling: a -1 in
        # the environment means someone believes they configured something.
        if v < 0:
            raise ValueError(f"ingest_warmup_articles must be >= 0, got {v}")
        return v


@lru_cache
def get_extract_settings() -> ExtractSettings:
    return ExtractSettings()  # type: ignore[call-arg]
