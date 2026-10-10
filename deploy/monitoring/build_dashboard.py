"""Emit the Grafana dashboard "graph-rag ingestion" as JSON (stdout).

Generated rather than hand-edited so a panel is a few readable lines. apply.sh pipes
this into the grafana-dashboards ConfigMap.

    python3 deploy/monitoring/build_dashboard.py > /tmp/graph-rag-ingestion.json
"""
from __future__ import annotations

import json

# Exporter series are pinned to its scrape job (Q): never mixed with anything a worker
# might serve.
Q = 'job="graph-rag-queue"'
V = f'{Q},vendor=~"$vendor"'
REMAINING = f'sum(graphrag_queue_jobs{{status=~"pending|in_progress",{V}}})'
TOTAL = f"sum(graphrag_queue_jobs{{{V}}})"
DONE = f'sum(graphrag_queue_jobs{{status="done",{V}}})'
PER_HOUR = f"sum(graphrag_queue_done_last_hour{{{V}}})"
GPU_ARTICLE_RATE = 'sum(rate(graphrag_worker_jobs_total{outcome="done",tier="gpu"}[1h])) * 3600'
CREDITS = f"max(graphrag_openrouter_credits_remaining_usd{{{Q}}})"
GPU_RATE = f"max(graphrag_gpu_hourly_cost_usd{{{Q}}})"
OR_SPEND = f"clamp_min(-deriv(max(graphrag_openrouter_credits_remaining_usd{{{Q}}})[1h:1m]) * 3600, 0)"
# GPU rent counts only while GPU-tier workers exist (their counters vanish with the pods).
GPU_RENT = (f"scalar({GPU_RATE}) * "
            'scalar(clamp_max(count(graphrag_worker_jobs_total{tier="gpu"}) or vector(0), 1))')

_panels: list[dict] = []
_id = 0


def _next_id() -> int:
    global _id
    _id += 1
    return _id


def row(title: str, y: int) -> None:
    _panels.append({"type": "row", "title": title, "id": _next_id(), "collapsed": False,
                    "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []})


def _targets(exprs: list[tuple[str, str]]) -> list[dict]:
    return [{"refId": chr(65 + i), "expr": e, "legendFormat": legend}
            for i, (e, legend) in enumerate(exprs)]


def stat(title, expr, x, y, w=4, h=4, unit="short", decimals=None, thresholds=None,
         desc="", color_mode="value"):
    steps = thresholds or [{"color": "green", "value": None}]
    _panels.append({
        "type": "stat", "title": title, "id": _next_id(), "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": _targets([(expr, "")]),
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": color_mode, "graphMode": "none", "textMode": "value"},
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                     "thresholds": {"mode": "absolute", "steps": steps}},
                        "overrides": []},
    })


def series(title, exprs, x, y, w=12, h=8, unit="short", desc="", stack=False):
    _panels.append({
        "type": "timeseries", "title": title, "id": _next_id(), "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": _targets(exprs),
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
        "fieldConfig": {"defaults": {"unit": unit, "custom": {
            "lineWidth": 2, "fillOpacity": 10,
            "stacking": {"mode": "normal" if stack else "none"}}}, "overrides": []},
    })


def bars(title, expr, legend, x, y, w=12, h=8, unit="percentunit", desc=""):
    _panels.append({
        "type": "bargauge", "title": title, "id": _next_id(), "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": "A", "expr": expr, "legendFormat": legend, "instant": True}],
        "options": {"orientation": "horizontal", "displayMode": "gradient",
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "showUnfilled": True},
        "fieldConfig": {"defaults": {"unit": unit, "min": 0, "max": 1 if unit == "percentunit"
                                     else None,
                                     "thresholds": {"mode": "absolute", "steps": [
                                         {"color": "orange", "value": None},
                                         {"color": "green", "value": 0.99}]}},
                        "overrides": []},
    })


RED_IF_ANY = [{"color": "green", "value": None}, {"color": "red", "value": 1}]

# --- Progress ----------------------------------------------------------------------
row("Progress ($vendor) -- counts are semantic_jobs rows (an updated article adds one)", 0)
stat("Progress", f"{DONE} / {TOTAL}", 0, 1, unit="percentunit", decimals=1)
stat("Done", DONE, 4, 1)
stat("Remaining", REMAINING, 8, 1, desc="pending + in progress")
stat("Articles / hour", PER_HOUR, 12, 1, desc="completed in the last hour")
stat("ETA", f"{REMAINING} / clamp_min({PER_HOUR}, 1) * 3600", 16, 1, unit="s",
     desc="remaining / last hour's rate")
stat("Dead jobs", f'sum(graphrag_queue_jobs{{status="dead",{V}}}) or vector(0)', 20, 1,
     thresholds=RED_IF_ANY, color_mode="background",
     desc="jobs that spent all attempts; each needs a look")
bars("Progress by product",
     f'sum by (product) (graphrag_queue_jobs{{status="done",{V}}})'
     f" / sum by (product) (graphrag_queue_jobs{{{V}}})", "{{product}}", 0, 5)
bars("Progress by source",
     f'sum by (product, source) (graphrag_queue_jobs{{status="done",{V}}})'
     f" / sum by (product, source) (graphrag_queue_jobs{{{V}}})",
     "{{product}} / {{source}}", 12, 5)

# --- Throughput --------------------------------------------------------------------
row("Throughput", 13)
series("Articles per hour, by tier",
       [('sum by (tier) (rate(graphrag_worker_jobs_total{outcome="done"}[15m])) * 3600',
         "{{tier}}")], 0, 14, desc="api = OpenRouter workers, gpu = vast.ai workers")
series("Remaining over time", [(REMAINING, "remaining")], 12, 14)
series("Failures, deferrals and pauses (per hour)",
       [('sum by (outcome, route) (rate(graphrag_worker_jobs_total{outcome!="done"}[15m])) * 3600',
         "{{outcome}} ({{route}})"),
        ("sum by (reason) (rate(graphrag_worker_pauses_total[15m])) * 3600",
         "pause: {{reason}}")], 0, 22, w=8,
       desc="failed spends an attempt; deferred_* and pauses do not. Empty = none.")
stat("Pending jobs with spent attempts", f"max(graphrag_queue_retrying_jobs{{{Q}}})", 8, 22,
     h=8, desc="will retry; a job reaching 5 attempts goes dead")
series("Median time per article, by tier",
       [('histogram_quantile(0.5, sum by (le, tier) '
         '(rate(graphrag_worker_job_seconds_bucket{outcome="done"}[30m])))', "{{tier}}")],
       12, 22, unit="s")

# --- LLM and GPU -------------------------------------------------------------------
row("LLM and GPU", 30)
stat("vLLM", 'max(up{job="vllm-vast"}) or vector(0)', 0, 31,
     thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}],
     color_mode="background", desc="1 = reachable through the cluster tunnel")
stat("Workers up", 'count(up{job="graph-rag-workers"} == 1) or vector(0)', 4, 31)
stat("KV cache in use", "max(vllm:kv_cache_usage_perc)", 8, 31, unit="percentunit",
     decimals=1)
stat("Prefix cache hit rate",
     "sum(rate(vllm:prefix_cache_hits_total[15m])) / sum(rate(vllm:prefix_cache_queries_total[15m]))",
     12, 31, unit="percentunit", decimals=1)
stat("Exporter data age", f"time() - max(graphrag_exporter_last_refresh_timestamp_seconds{{{Q}}})",
     16, 31, unit="s", thresholds=[{"color": "green", "value": None},
                                   {"color": "red", "value": 300}],
     desc="seconds since the queue gauges were refreshed")
stat("Tokens today", f"max(graphrag_tokens_today{{{Q}}})", 20, 31, unit="short")
series("vLLM requests", [("sum(vllm:num_requests_running)", "running"),
                         ("sum(vllm:num_requests_waiting)", "waiting")], 0, 35)
series("vLLM tokens per second",
       [("sum(rate(vllm:prompt_tokens_total[5m]))", "prompt"),
        ("sum(rate(vllm:generation_tokens_total[5m]))", "generation")], 12, 35)
series("Worker LLM tokens per second, by tier",
       [("sum by (tier, kind) (rate(graphrag_worker_llm_tokens_total[5m]))",
         "{{tier}} {{kind}}")], 0, 43, w=24)

# --- Cost --------------------------------------------------------------------------
row("Cost", 51)
stat("OpenRouter balance", CREDITS, 0, 52,
     unit="currencyUSD", decimals=2,
     thresholds=[{"color": "red", "value": None}, {"color": "orange", "value": 5},
                 {"color": "green", "value": 15}])
stat("OpenRouter spend / hour", OR_SPEND, 4, 52, unit="currencyUSD", decimals=2,
     desc="from the balance's slope over the last hour")
stat("GPU rental / hour", GPU_RATE, 8, 52, unit="currencyUSD",
     decimals=3)
stat("GPU cost per article",
     f"scalar({GPU_RATE}) / clamp_min({GPU_ARTICLE_RATE}, 1)", 12, 52,
     unit="currencyUSD", decimals=4, desc="rental / GPU-tier articles per hour")
stat("Estimated cost to finish",
     f"{REMAINING} / clamp_min({PER_HOUR}, 1) * "
     f"({GPU_RENT} + scalar({OR_SPEND} or vector(0)))",
     16, 52, w=8, unit="currencyUSD", decimals=2,
     desc="hours left for the selected vendor at last hour's rate x (GPU rent while GPU "
          "workers run + OpenRouter spend per hour). The OpenRouter rate is account-wide, "
          "so it includes answer-api and other traffic.")

dashboard = {
    "uid": "graph-rag-ingestion", "title": "graph-rag ingestion", "tags": ["graph-rag"],
    "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "5m",
    "time": {"from": "now-12h", "to": "now"},
    "templating": {"list": [{
        "name": "vendor", "label": "Vendor", "type": "query",
        "query": {"query": "label_values(graphrag_queue_jobs, vendor)", "refId": "vendor"},
        "definition": "label_values(graphrag_queue_jobs, vendor)",
        "refresh": 2, "includeAll": True, "multi": True, "allValue": ".*",
        "current": {"text": ["Veeam"], "value": ["Veeam"]}, "sort": 1}]},
    "panels": _panels,
}

if __name__ == "__main__":
    print(json.dumps(dashboard, indent=1))
