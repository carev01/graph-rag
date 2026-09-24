Throwaway spikes behind `docs/superpowers/pre-bootstrap-decisions-2026-09-23.md`. Not CI-covered.
Run from the repo root with OUT=<scratch dir>: `d4_invalidations.py OUT` (read-only Cypher on the
.env graph); `d6_sample.py OUT` (read-only DocExtractor fetches + local chunker), then
`d6_simulate.py OUT`. The per-episode cost fit (54,751 + 19.8 x body prompt chars) was computed
inline from data/ft-captured and is quoted in the doc.
