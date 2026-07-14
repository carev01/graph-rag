# Slice 1 (Ingestion Foundation) — Open Follow-ups

Forward-looking only. The quality-gate + correctness/robustness + test-hygiene
items originally logged here were resolved in the follow-ups cleanup pass
(commits `3e83132` quality gate + CI, `2a01e6e` slice-1 correctness,
`e7c99fc`/`83795a4` test hygiene + review fixes): mypy/ruff now pass and CI
enforces them; malformed-line dead-lettering, dirty-flag re-pass, webhook 400,
catalog fan-out bounding, lifespan startup cleanup, and idempotent close are all
done. What remains:

## Correctness (low priority)
- **`should_run_source` first-call race** (`state_store.py`): two concurrent
  first-time callers can both pass the `FOR UPDATE` on a not-yet-existing debounce
  row. Mitigated in practice by the advisory lock / in-process single-flight, so
  left as-is; close it if debounce is ever driven outside that guard.

## Performance (scale)
- **N+1 round-trips per article**: `get_content_hash` + `apply_structural` are
  separate autocommit sessions. Fine at slice-1 scale; batch when ingesting the
  full corpus.

## Environment / ops
- **Real webhook cluster→host delivery is unproven end-to-end.** The receiver is
  unit-tested and locally verified, but DocExtractor→host delivery was never
  confirmed: the extraction-trigger endpoint wasn't found (`POST
  /api/extraction/runs` → 405) and cluster→host reachability/TLS is untested.
  Revisit with the correct trigger endpoint when standing up the live deployment.
- **`DOCEXT_VERIFY_TLS=false`** disables TLS verification against the cluster —
  acceptable for the homelab/internal-CA posture, but prefer trusting the internal
  CA (mount the CA cert) where possible.
