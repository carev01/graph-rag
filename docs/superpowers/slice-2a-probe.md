# Slice 2a — Step-0 client-mode decision

**Decision: `llm_client_mode = generic_json_schema`** (set as the config default).

## Evidence (from Task 6's live extraction on `gpt-oss-20b` via llama-server + Neo4j 5.26)

| Mode | Graphiti client | Result |
|---|---|---|
| `structured` (default in the spec) | `OpenAIClient` (Responses API, grammar/structured output) | **Unusable.** Extraction blocked at the `responses.parse` step — gpt-oss-20b returns markdown-fenced / malformed JSON; **0 entities** extracted. |
| `generic_json_schema` | `OpenAIGenericClient(structured_output_mode="json_schema")` | **Works.** Real entities + facts extracted and persisted. Example facts from a Vault-Lock chunk: *"AWS Backup supports immutable backups for Amazon S3"*, *"AWS Backup provides immutability"* — correct and well-formed. |
| `generic_json_object` | `OpenAIGenericClient(structured_output_mode="json_object")` | Not separately timed; available as a fallback. `json_schema` already meets the bar, so it is the default. |

The spec (§5/§7) anticipated exactly this uncertainty and set `structured` as the default **gated by this probe**, with a documented fallback to the generic client. The probe fired early — Task 6's live smoke test *is* the probe for the `structured` vs generic question — and the fallback is now the default.

## Why not spend more GPU time re-confirming

`gpt-oss-20b` runs ~6 min per episode (one GPU, serialized). A full dual-mode probe re-run would cost ~30+ min largely re-confirming the known-broken `structured` path. The `structured` failure is unambiguous (0 entities, parse errors), and `generic_json_schema` is validated end-to-end (episode persisted, provenance-ready, good facts). `src/graph_extract/probe.py` remains available to compare `generic_json_schema` vs `generic_json_object` (or re-evaluate `structured`) if a different model is swapped in later.

## Notes for the viability verdict (Task 11)
- Structured-output reliability metric: `generic_json_schema` produced valid JSON / successful extraction on the sampled chunks; `structured` was 100% failure. Record the parse-fail rate over the pilot run.
- Token usage IS captured in generic mode (`usage.instrument` patches `chat.completions.create`, which the generic client uses; the `structured` path used `responses.parse`, which the tally could not see — moot now).
