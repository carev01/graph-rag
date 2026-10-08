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
# Refuse to patch unless our jobs parse as members of scrape_configs (appending at the
# end is only right while scrape_configs is the file's last top-level key).
python3 - "$TMP/new.yml" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
jobs = [j.get("job_name") for j in d.get("scrape_configs") or []]
missing = {"graph-rag-queue", "graph-rag-workers", "vllm-vast"} - set(jobs)
if missing:
    sys.exit(f"refusing to patch: {sorted(missing)} would not land in scrape_configs "
             "(is scrape_configs still the last top-level key of prometheus.yml?)")
print("scrape jobs:", jobs)
PY

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
POD=$( (kubectl -n "$NS" get pods -o name || true) | grep -m1 prometheus || true)
[ -n "$POD" ] || { echo "no prometheus pod found in $NS: reload it by hand"; exit 1; }
synced=no
for _ in $(seq 1 36); do
  if kubectl -n "$NS" exec "$POD" -- grep -q "graph-rag-queue" /etc/prometheus/prometheus.yml 2>/dev/null; then
    synced=yes; break
  fi
  sleep 5
done
[ "$synced" = yes ] || { echo "the new config never reached $POD (subPath mount?): restart it"; exit 1; }
kubectl -n "$NS" exec "$POD" -- wget -q -O- --post-data= http://localhost:9090/-/reload
echo "Prometheus reloaded"
