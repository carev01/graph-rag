# `/answer` Router — Demonstration Report

**Slice:** Phase 4, slice 2 — the `/answer` router: classify a query to a retrieval
mode (local / global / drift / timeline), dispatch to the existing mode function,
return a uniform envelope with a `mode`/`routing` block.
**Date:** 2026-07-18
**Status:** Implemented, reviewed, merged locally. Full non-live suite green
(423 passed); `@live` smoke green; demonstrated end-to-end on `backup-docs`.

## What it does

`GET /answer?q=&mode=&vendor=` is now a router (its former local-synthesis behavior
is `/answer?mode=local`). For each query it:

1. **Classifies** — `?mode=` override → temporal-regex→timeline → cross-vendor-regex
   →global → cheap-tier LLM (ling) → default `drift`. Heuristics catch the obvious
   cases with zero LLM cost; only the ambiguous ones spend one cheap call.
2. **Dispatches** to the existing mode function (`answer_local` / `global_search` /
   `drift_search` / `timeline_local`), reading per-mode knobs from settings.
3. **Normalizes** every mode's output into one envelope; timeline gets a
   **deterministic** markdown answer + flattened numbered citations (no LLM).
4. **Falls back** once: a `local` classification that retrieves nothing escalates
   to `drift` (`routing.fallback_from="local"`).

The router authors no prose and no URL (design decision #2) — it only picks a mode
label and reshapes; answers/citations come from the mode functions or the
deterministic timeline render.

## Live run on `backup-docs` — four intents, four modes

| Query | → mode | via | citations | source hosts |
|---|---|---|---|---|
| "Does AWS Backup support Vault Lock in compliance mode?" | **local** | llm | 6 | docs.aws.amazon.com |
| "Compare how AWS Backup and Azure Backup handle immutability" | **global** | heuristic | 7 | learn.microsoft.com |
| "What should I consider when planning long-term retention?" | **drift** | llm | 22 | docs.aws.amazon.com, learn.microsoft.com |
| "How has Azure Backup soft-delete changed over time?" | **timeline** | heuristic | 30 | docs.aws.amazon.com, learn.microsoft.com |

Every run: `"http" not in answer` (no LLM-authored URL) and **all citations resolve
to ≥1 real source URL**. Highlights:

- **local (via llm):** the cheap classifier correctly picked `local` for a specific
  product/feature question → *"Yes, AWS Backup supports Vault Lock in compliance
  mode. The `PutBackupVaultLockConfiguration` API can create vault locks in
  compliance mode … [2]"* — 6 citations, all `docs.aws.amazon.com`.
- **global (via heuristic):** the cross-vendor regex caught "Compare … AWS Backup
  and Azure Backup" → map-reduce, theme-organized (*"Theme: Immutability …"*).
- **drift (via llm):** a broad "what should I consider" → the primer→follow-up→
  synthesis motion, 22 citations spanning **both** vendors.
- **timeline (via heuristic):** "changed over time" → the deterministic render:
  *"- backup item 'model' is in soft delete state … — valid_at 2020-01-17…,
  invalid_at … [1]"* — 30 change events numbered, each citation resolving.

The `routing` block on each response records `{chosen, via, fallback_from}` (and
`degraded` when a DRIFT primer degrade is folded in) — the audit trail the plan's
§6 monitoring ("router decision distribution") needs.

## Design-decision alignment (verified)

- **#2 citations are traversal, never LLM:** the router emits only a mode label and
  reshapes. Timeline's answer is deterministically rendered; every mode's
  `answer`/`citations` come from a mode function that resolves URLs via
  `Provenance`. Demonstrated: no URL in any answer, all citations resolve.
- **#4 one embedding space:** unchanged — global/drift/timeline retrieval embeds the
  query with the shared embedder as before.
- **DRIFT degrade normalized:** the empty-primer-shortlist degrade (which returned
  `answer_local`'s divergent shape) is now folded into the uniform envelope with
  `routing.degraded` — closing the deferred item from the DRIFT slice.

## Verification summary

- **Unit — `classify`** (`test_router_classify.py`): override wins; temporal/
  cross-vendor heuristics; cheap-LLM path; `cheap_client=None`→default drift;
  junk/unknown label→default drift.
- **Unit — render/normalize** (`test_router_normalize.py`): deterministic timeline
  markdown + numbered citations; empty timeline; envelope extras present only per
  mode; degrade→`routing.degraded`; `fallback_from` recorded.
- **Unit — orchestrator** (`test_router_dispatch.py`): each mode routes + normalizes;
  local-empty→drift fallback; DRIFT degrade normalized; dispatch signatures verified
  against the real mode functions.
- **App** (`test_answer_api_app.py`): `/answer` returns the uniform envelope with
  `mode`; `?mode=timeline` forces timeline; `?mode=bogus`→422; missing `q`→422;
  lifespan builds/omits the cheap client; pre-existing endpoints unchanged (20 tests).
- **`@live` smoke** (`test_answer_router_live.py`): a temporal query routes to
  timeline via heuristic; no URL in answer; citations resolve.
- Full non-live suite: **423 passed, 9 `@live` deselected**; `ruff check src tests`
  + `mypy` clean.

## Infra note (this run)

The live run was initially blocked by a home-lab outage: the embedder host
(`srv-llm.home.lan`) briefly went off-DNS, and the compose Neo4j bridge
(`br-…`, network `graph-rag_default`, `172.20.0.0/16`) lost its gateway IP
(`172.20.0.1/16`), so the host couldn't reach `localhost:7687` (the docker-proxy
forwarded into an unroutable subnet) even though the DB served Bolt internally.
Re-adding the bridge gateway IP restored host→container routing (`communities=66`
confirmed) with no data loss. Neither the router nor Neo4j was at fault.

## Follow-ups (deferred, per spec §11)

- The `freshness` block (`reports_as_of`/`graph_cursor_time`); richer citation
  fields (vendor/product/section); a reliable single-product→local heuristic
  (currently left to the cheap LLM); additional fallback arcs; a classifier
  confidence score.
- MCP exposure of the modes as separate Copilot tools (Phase 5).
- Minor cleanup: a now-dead `_fake_answer_local` fixture lingers in
  `test_answer_api_app.py` (harmless; the old `/answer` test that used it was
  replaced by router tests).
