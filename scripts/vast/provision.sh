#!/bin/bash
# Provision (or re-provision) a vast.ai vLLM instance for the cheap extraction tier, and
# optionally point the cluster's tunnel at it. Quick reference: docs/deploy/vast-gpu.md.
#
#   scripts/vast/provision.sh <ssh-host> <ssh-port> [--cluster]
#   e.g. scripts/vast/provision.sh ssh6.vast.ai 16301 --cluster
#
# Prompts for the Hugging Face token (hidden; used once on the instance, never stored).
# Keys live OUTSIDE the repo, in $VAST_DIR (default ~/.config/graph-rag/vast, mode 700):
#   api_key            vLLM API key (generated on first run, reused afterwards)
#   tunnel_ed25519     the cluster tunnel's SSH key pair (generated on first run)
# --cluster: refresh the graph-rag-vast Secret (host, port, host key, both keys) and
# restart the tunnel Deployment. Needs kubectl access to the graph-rag namespace.
set -euo pipefail

HOST="${1:?usage: provision.sh <ssh-host> <ssh-port> [--cluster]}"
PORT="${2:?usage: provision.sh <ssh-host> <ssh-port> [--cluster]}"
CLUSTER="${3:-}"
NS="${NS:-graph-rag}"
SECRET="${SECRET:-graph-rag-vast}"
VAST_DIR="${VAST_DIR:-$HOME/.config/graph-rag/vast}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$PORT" "root@$HOST")

umask 077
mkdir -p "$VAST_DIR"
[ -s "$VAST_DIR/api_key" ] || python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$VAST_DIR/api_key"
[ -s "$VAST_DIR/tunnel_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -C graph-rag-vast-tunnel -f "$VAST_DIR/tunnel_ed25519"

echo "== reaching root@$HOST:$PORT (your own SSH key must be on the vast account)"
"${SSH[@]}" -n true 2>/dev/null || { echo "cannot SSH to the instance"; exit 1; }

read -r -s -p "Hugging Face token (read access to the model repo): " HF_TOKEN; echo
scp -q -P "$PORT" "$HERE/remote_setup.sh" "root@$HOST:/workspace/remote_setup.sh"
printf '%s\n%s\n' "$HF_TOKEN" "$(cat "$VAST_DIR/api_key")" | "${SSH[@]}" \
  "TUNNEL_PUBKEY='$(cat "$VAST_DIR/tunnel_ed25519.pub")' bash /workspace/remote_setup.sh" \
  2>&1 | grep -v -E "^(Welcome to vast|Have fun|AI agents)"
unset HF_TOKEN

echo "== the tunnel key can forward, and nothing else"
ssh -n -o BatchMode=yes -o IdentitiesOnly=yes -i "$VAST_DIR/tunnel_ed25519" -p "$PORT" \
  -o ExitOnForwardFailure=yes -f -N -L 18099:127.0.0.1:18000 "root@$HOST"
sleep 2
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat "$VAST_DIR/api_key")" \
  http://127.0.0.1:18099/v1/models)
pkill -f "18099:127.0.0.1:18000" || true
[ "$code" = 200 ] || { echo "tunnel check failed (HTTP $code)"; exit 1; }
echo "   /v1/models via the tunnel key: HTTP 200"

if [ "$CLUSTER" = "--cluster" ]; then
  echo "== refreshing secret $NS/$SECRET and restarting the tunnel"
  known=$(ssh-keyscan -p "$PORT" -t ed25519 "$HOST" 2>/dev/null)
  [ -n "$known" ] || { echo "ssh-keyscan returned nothing"; exit 1; }
  kubectl -n "$NS" create secret generic "$SECRET" \
    --from-file=id_ed25519="$VAST_DIR/tunnel_ed25519" \
    --from-literal=known_hosts="$known" \
    --from-literal=ssh_host="$HOST" --from-literal=ssh_port="$PORT" \
    --from-file=api_key="$VAST_DIR/api_key" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  if kubectl -n "$NS" get deploy graph-rag-vast-tunnel >/dev/null 2>&1; then
    kubectl -n "$NS" rollout restart deploy/graph-rag-vast-tunnel >/dev/null
    kubectl -n "$NS" rollout status deploy/graph-rag-vast-tunnel --timeout=120s
  else
    echo "   tunnel Deployment not installed yet: set vast.enabled=true in the Helm release"
  fi
fi
echo "done"
