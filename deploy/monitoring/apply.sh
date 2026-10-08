#!/bin/bash
# Install the graph-rag ingestion scrape jobs and Grafana dashboard into the cluster's
# existing monitoring stack (namespace `monitoring`). Idempotent: the scrape jobs live
# between `# >>> graph-rag` / `# <<< graph-rag` markers at the end of prometheus.yml and
# are replaced on every run (the rest of the file, comments included, is untouched);
# the dashboard is one key of the grafana-dashboards ConfigMap.
#   deploy/monitoring/apply.sh
set -euo pipefail
NS="${NS:-monitoring}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

python3 "$HERE/build_dashboard.py" > "$TMP/graph-rag-ingestion.json"

kubectl -n "$NS" get cm prometheus-config -o jsonpath='{.data.prometheus\.yml}' > "$TMP/current.yml"
python3 - "$TMP/current.yml" "$HERE/prometheus-scrape.yml" "$TMP/new.yml" <<'PY'
import re, sys
cur, block, out = (open(p).read() if i < 2 else p for i, p in enumerate(sys.argv[1:]))
cur = re.sub(r"\n?  # >>> graph-rag ingestion.*?# <<< graph-rag ingestion <<<\n?", "\n", cur, flags=re.S)
open(out, "w").write(cur.rstrip("\n") + "\n\n" + block)
PY
python3 -c "import sys,yaml; d=yaml.safe_load(open(sys.argv[1])); print('scrape jobs:', [j['job_name'] for j in d['scrape_configs']])" "$TMP/new.yml"

python3 - "$TMP/new.yml" "$TMP/graph-rag-ingestion.json" > "$TMP/patch-prom.json" <<'PY'
import json, sys
print(json.dumps({"data": {"prometheus.yml": open(sys.argv[1]).read()}}))
PY
kubectl -n "$NS" patch cm prometheus-config --type merge --patch-file "$TMP/patch-prom.json" >/dev/null
python3 - "$TMP/graph-rag-ingestion.json" > "$TMP/patch-dash.json" <<'PY'
import json, sys
print(json.dumps({"data": {"graph-rag-ingestion.json": open(sys.argv[1]).read()}}))
PY
kubectl -n "$NS" patch cm grafana-dashboards --type merge --patch-file "$TMP/patch-dash.json" >/dev/null
echo "ConfigMaps patched; the kubelet syncs them into the pods within ~1-2 minutes."

# Reload Prometheus once the new config is visible inside its pod (--web.enable-lifecycle).
POD=$(kubectl -n "$NS" get pod -l app=prometheus -o name 2>/dev/null | head -1)
[ -n "$POD" ] || POD=$(kubectl -n "$NS" get pods -o name | grep prometheus | head -1)
for _ in $(seq 1 30); do
  kubectl -n "$NS" exec "$POD" -- grep -q "graph-rag-queue" /etc/prometheus/prometheus.yml 2>/dev/null && break
  sleep 5
done
kubectl -n "$NS" exec "$POD" -- wget -q -O- --post-data= http://localhost:9090/-/reload && echo "Prometheus reloaded"
