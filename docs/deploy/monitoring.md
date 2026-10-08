# Ingestion monitoring — quick reference

**Dashboard:** Grafana → folder *LLM* → **graph-rag ingestion**
(`http://grafana.k3s.home.lan/d/graph-rag-ingestion`). Pick the vendor at the top.

| row | shows |
|---|---|
| Progress | % done, done / remaining job rows, articles in the last hour, ETA, dead jobs; progress per product and per source |
| Throughput | articles/hour per tier (`api` = OpenRouter workers, `gpu` = vast.ai workers), remaining over time, failures / deferrals / pauses, median time per article |
| LLM and GPU | vLLM reachable, workers up, KV cache, prefix-cache hit rate, exporter data age, vLLM requests and tokens/s, worker tokens/s per tier |
| Cost | OpenRouter balance and spend/hour, GPU rent/hour, GPU cost per article, estimated cost to finish |

Counts are `semantic_jobs` rows: an updated article adds a row, and dead jobs stay in
the total, so a vendor can sit just under 100%.

## Where the numbers come from

- **Workers** (`graph_sync.metrics`, `:9109` on every worker pod): `graphrag_worker_jobs_total{tier,outcome}`
  (done, failed, deferred_lock, deferred_halt), `graphrag_worker_pauses_total{tier,reason}`,
  `graphrag_worker_llm_tokens_total{tier,kind}`, `graphrag_worker_job_seconds`. Prometheus
  finds the pods of both tiers through the headless Service `graph-rag-worker-metrics`.
- **Queue exporter** (`graph-sync metrics-exporter`, Deployment `graph-rag-metrics`, `:9108`):
  `graphrag_queue_jobs{vendor,product,source,status}`, `graphrag_queue_done_last_hour{vendor}`,
  `graphrag_queue_retrying_jobs`, `graphrag_tokens_today`,
  `graphrag_openrouter_credits_remaining_usd`, `graphrag_gpu_hourly_cost_usd`
  (`vast.hourlyCostUsd` in the Helm values). Read-only against Postgres and Neo4j.
- **vLLM** on the vast.ai GPU: its own `vllm:*` metrics through `graph-rag-vast-tunnel:8000`.

## Grafana sizing

Grafana here runs with 1 CPU / 1 Gi and a 5 s probe timeout (raised 2026-10-08 from 0.5
CPU / 512 Mi / 1 s: with this dashboard open, queries queued behind Grafana's own
alerting engine, the 1 s liveness probe failed and the kubelet killed it -- exit 137, the
dashboard "showing nothing"). Prometheus answers these queries in ~0.25 s; Grafana was the
bottleneck. The dashboard refreshes every 5 minutes.

## Changing it

- Panels: edit `deploy/monitoring/build_dashboard.py`, then `deploy/monitoring/apply.sh`.
- Scrape jobs: `deploy/monitoring/prometheus-scrape.yml`, then `apply.sh`. The script
  replaces a marked block at the end of the `prometheus-config` ConfigMap (refusing if the
  jobs would not land in `scrape_configs`), updates the `grafana-dashboards` ConfigMap,
  and reloads Prometheus.
- Alerts are not set up yet (planned: Grafana alerting on throughput stalls, pauses,
  dead jobs, vLLM down, low OpenRouter balance).
