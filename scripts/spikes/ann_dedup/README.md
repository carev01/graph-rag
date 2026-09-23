Throwaway spike behind `docs/superpowers/ann-dedup-probe-2026-09-23.md`. Not CI-covered.
Run from the repo root with OUT=<scratch dir>, in order:
`extract_pairs.py OUT`, `geom.py OUT` (reads the live graph, read-only), `embed_pairs.py OUT`
(local embedder), then `[HNSW=', <index options>'] [STEPS=...] dedup_probe.py OUT tight|loose`.
`scorecheck.py` is the standalone proof that Neo4j cosine is (1+cos)/2.
