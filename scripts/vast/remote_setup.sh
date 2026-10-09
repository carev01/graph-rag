#!/bin/bash
# Runs ON the vast.ai instance (uploaded and invoked by scripts/vast/provision.sh).
# Reads two lines on stdin -- the Hugging Face token, then the vLLM API key -- so neither
# ever appears in a process list or a shell history. The HF token is used once (only if
# the model is not already on disk) and never written.
#
# vLLM runs as the template's own SUPERVISED service (`supervisorctl ... vllm`), pointed at
# our model, so it comes back by itself after a container restart. (2026-10-08 21:53: the
# container restarted; a hand-launched vLLM did not return, the template's stock
# Qwen3.5-9B service started instead and failed, and the GPU tier sat idle.)
# The service runs `vllm serve $VLLM_MODEL $VLLM_ARGS $(cat /etc/vllm-args.conf)` with
# /etc/environment and /workspace/.env loaded (/opt/supervisor-scripts/vllm.sh).
#
# Idempotent: re-running only restarts vLLM if the API key or the serve arguments changed,
# and never re-downloads a complete model.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-carev01/qwen35-4b-graphrag}"
MODEL_DIR=/workspace/hf/model
TUNNEL_PUBKEY="${TUNNEL_PUBKEY:-}"
ARGS_FILE=/etc/vllm-args.conf
KEY_ENV=/workspace/.env   # sourced by the supervised service; mode 600

read -r HF_TOKEN
read -r VLLM_KEY
source /venv/main/bin/activate
serving() { curl -sf -o /dev/null http://127.0.0.1:18000/health; }

echo "== 1/5 make room: the template's stock vLLM"
# A fresh instance's supervised vllm service starts downloading the template's stock
# Qwen3.5-9B (18 GB) at boot; on a 24 GB disk that leaves no room for our 9.3 GB model
# (2026-10-09: "No space left on device"). Stop it and drop its download BEFORE fetching.
if ! grep -q "^VLLM_MODEL=\"$MODEL_DIR\"" /etc/environment; then
  supervisorctl stop vllm model-ui >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
fi
# The stock service's --download-dir; its layout varies (models--*/ or bare blobs/), and
# nothing of ours lives there: empty it.
find /workspace/models -mindepth 1 -delete 2>/dev/null || true
echo "   disk free: $(df -h / | awk 'NR==2 {print $4}')"

echo "== 2/5 model files"
model_complete() {
  python - "$MODEL_DIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
try:
    shards = set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values())
    ok = os.path.getsize(os.path.join(d, "config.json")) > 0 and all(
        os.path.getsize(os.path.join(d, s)) > 0 for s in shards)
except Exception:
    ok = False
sys.exit(0 if ok else 1)
EOF
}
if model_complete; then
  echo "   $MODEL_DIR is complete: no download"
else
  # No token: anonymous download (the model repo is public since 2026-10-09).
  echo "   fetching $MODEL_REPO ($([ -n "$HF_TOKEN" ] && echo "with token" || echo "anonymous"))"
  mkdir -p "$MODEL_DIR"
  HF_TOKEN="$HF_TOKEN" python - "$MODEL_REPO" "$MODEL_DIR" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], token=os.environ.get("HF_TOKEN") or False)
EOF
  model_complete || { echo "   download incomplete"; exit 1; }
fi
unset HF_TOKEN

echo "== 3/5 point the supervised vllm service at the model"
fingerprint() { { cat /etc/environment "$ARGS_FILE" "$KEY_ENV" 2>/dev/null || true; } | sha256sum | cut -d' ' -f1; }
BEFORE=$(fingerprint)
setenv() {   # set KEY="VALUE" in /etc/environment, replacing any existing line
  grep -v "^$1=" /etc/environment > /etc/environment.new || true
  printf '%s="%s"\n' "$1" "$2" >> /etc/environment.new
  chmod 644 /etc/environment.new && mv /etc/environment.new /etc/environment
}
setenv VLLM_MODEL "$MODEL_DIR"
setenv MODEL_NAME "$MODEL_DIR"     # model-ui keys off MODEL_NAME
setenv VLLM_ARGS ""                # the template's defaults (9B tool/reasoning parsers) do not apply
# --max-model-len 65536: graphiti asks for max_tokens=16384, so a 32k window overflows on
# prompts over ~16k tokens. The template's --compilation-config (CUDA graphs for batch
# sizes 1-8 only) is replaced: we run up to 64 sequences.
cat > "$ARGS_FILE" <<EOF
--served-model-name qwen35-graphrag --host 127.0.0.1 --port 18000 --max-model-len ${MAX_MODEL_LEN:-65536} --max-num-seqs ${MAX_SEQS:-64} --gpu-memory-utilization 0.90 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' --default-chat-template-kwargs '{"enable_thinking":false}'
EOF
chmod 644 "$ARGS_FILE"
( umask 077; printf 'VLLM_API_KEY=%s\n' "$VLLM_KEY" > "$KEY_ENV" )
chmod 600 "$KEY_ENV"
rm -f /workspace/vllm.key /workspace/serve.sh   # the pre-supervisor layout
CHANGED=no
[ "$BEFORE" = "$(fingerprint)" ] || CHANGED=yes

echo "== 4/5 tunnel key (restricted to forwarding 127.0.0.1:18000)"
if [ -n "$TUNNEL_PUBKEY" ]; then
  # authorized_keys2, never authorized_keys: vast REWRITES authorized_keys from the account
  # keys (seen 2026-10-08 20:21, silently dropping the tunnel key; the tunnels survived on
  # open sessions until a network blip, then could not log back in). sshd here reads both
  # files (`sshd -T`: authorizedkeysfile .ssh/authorized_keys .ssh/authorized_keys2).
  # restrict = no pty/agent/X11/forwarding; port-forwarding + permitopen re-allow the one
  # forward; command= stops the key running anything (ssh -N never asks to).
  # No `grep -q` under pipefail: it exits on the first match, sshd -T dies of SIGPIPE
  # and the pipeline "fails" -- a false warning (2026-10-09).
  /usr/sbin/sshd -T 2>/dev/null | grep -i '^authorizedkeysfile.*authorized_keys2' >/dev/null \
    || echo "   WARNING: sshd does not read authorized_keys2; the tunnel key will not work"
  mkdir -p /root/.ssh
  ( umask 077; printf '%s\n' "restrict,port-forwarding,permitopen=\"127.0.0.1:18000\",command=\"/bin/false\" $TUNNEL_PUBKEY" \
    > /root/.ssh/authorized_keys2 )
  # Remove an entry an older version of this script put in vast's file.
  if grep -qF "$TUNNEL_PUBKEY" /root/.ssh/authorized_keys 2>/dev/null; then
    grep -vF "$TUNNEL_PUBKEY" /root/.ssh/authorized_keys > /root/.ssh/authorized_keys.new || true
    chmod 600 /root/.ssh/authorized_keys.new && mv /root/.ssh/authorized_keys.new /root/.ssh/authorized_keys
  fi
fi

echo "== 5/5 vLLM"
# A vLLM launched by hand (the pre-supervisor layout) is not supervised: replace it.
HAND=$(ps -eo pid=,ppid=,args= | awk '$2 == 1 && /vllm serve \/workspace\/hf\/model/ {print $1}')
state=$( (supervisorctl status vllm 2>/dev/null || true) | awk '{print $2}')
if [ -n "$HAND" ] || [ "$state" != RUNNING ] || [ "$CHANGED" = yes ] || ! serving; then
  echo "   (re)starting the supervised service (drain the GPU workers first if they are busy)"
  if [ -n "$HAND" ]; then kill $HAND 2>/dev/null || true; fi
  supervisorctl stop vllm >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 3
  supervisorctl start vllm >/dev/null
  for _ in $(seq 1 120); do serving && break; sleep 5; done
else
  echo "   already serving with these settings"
fi
serving || { supervisorctl status vllm; tail -30 /var/log/portal/vllm.log 2>/dev/null; exit 1; }
echo "   vLLM healthy ($( (supervisorctl status vllm || true) | awk '{print $1, $2}'))"
if [ -n "${PUBLIC_IPADDR:-}" ] && [ -n "${VAST_TCP_PORT_22:-}" ]; then
  echo "   direct ssh (bypasses vast's shared proxy -- use it for --cluster): $PUBLIC_IPADDR $VAST_TCP_PORT_22"
fi
