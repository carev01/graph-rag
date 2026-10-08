#!/bin/bash
# Runs ON the vast.ai instance (uploaded and invoked by scripts/vast/provision.sh).
# Reads two lines on stdin -- the Hugging Face token, then the vLLM API key -- so neither
# ever appears in a process list or a shell history. The API key is stored in
# /workspace/vllm.key (mode 600); the HF token is used once and never written.
#
# Idempotent: on an instance already serving the model it only re-checks the key, the serve
# command and the tunnel key (restarting vLLM if the key or serve command changed); it
# never re-downloads.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-carev01/qwen35-4b-graphrag}"
MODEL_DIR=/workspace/hf/model
TUNNEL_PUBKEY="${TUNNEL_PUBKEY:-}"

read -r HF_TOKEN
read -r VLLM_KEY
source /venv/main/bin/activate

serving() { curl -sf -o /dev/null http://127.0.0.1:18000/health; }
SERVING=no
if serving && [ -s "$MODEL_DIR/config.json" ]; then SERVING=yes; fi

if [ "$SERVING" = yes ]; then
  echo "== 1-2/5 already serving $MODEL_DIR: skipping stop and download"
else
echo "== 1/5 stop the template's stock vLLM (it serves Qwen3.5-9B and takes the GPU)"
supervisorctl stop model-ui vllm >/dev/null 2>&1 || true
# Stopping the service leaves its engine process holding ~20 GB of VRAM.
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
sleep 3
rm -rf /workspace/models/models--Qwen--*     # the stock model's 18 GB download
echo "   GPU memory in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo "== 2/5 fetch $MODEL_REPO"
mkdir -p "$MODEL_DIR" /workspace/logs
HF_TOKEN="$HF_TOKEN" python - "$MODEL_REPO" "$MODEL_DIR" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], token=os.environ["HF_TOKEN"])
EOF
fi
unset HF_TOKEN

echo "== 3/5 API key and serve script"
umask 077
KEY_CHANGED=no
if [ "$(cat /workspace/vllm.key 2>/dev/null)" != "$VLLM_KEY" ]; then KEY_CHANGED=yes; fi
printf '%s\n' "$VLLM_KEY" > /workspace/vllm.key
OLD_SERVE=$(sha256sum /workspace/serve.sh 2>/dev/null | cut -d' ' -f1 || true)
cat > /workspace/serve.sh <<'EOF'
#!/bin/bash
# Fine-tuned qwen35-4b-graphrag on vLLM. Localhost only: reached through SSH tunnels.
source /venv/main/bin/activate
export VLLM_API_KEY="$(cat /workspace/vllm.key)"
exec vllm serve /workspace/hf/model \
  --served-model-name qwen35-graphrag \
  --host 127.0.0.1 --port 18000 \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${MAX_SEQS:-64}" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
EOF
chmod 700 /workspace/serve.sh
SERVE_CHANGED=no
[ "$OLD_SERVE" = "$(sha256sum /workspace/serve.sh | cut -d' ' -f1)" ] || SERVE_CHANGED=yes

echo "== 4/5 tunnel key (restricted to forwarding 127.0.0.1:18000)"
if [ -n "$TUNNEL_PUBKEY" ]; then
  # authorized_keys2, never authorized_keys: vast REWRITES authorized_keys from the account
  # keys (seen 2026-10-08 20:21, silently dropping the tunnel key; the tunnels survived on
  # open sessions until a network blip, then could not log back in). sshd here reads both
  # files (`sshd -T`: authorizedkeysfile .ssh/authorized_keys .ssh/authorized_keys2).
  # restrict = no pty/agent/X11/forwarding; port-forwarding + permitopen re-allow the one
  # forward; command= stops the key running anything (ssh -N never asks to).
  sshd -T 2>/dev/null | grep -qi '^authorizedkeysfile.*authorized_keys2' \
    || echo "   WARNING: sshd does not read authorized_keys2; the tunnel key will not work"
  mkdir -p /root/.ssh
  printf '%s\n' "restrict,port-forwarding,permitopen=\"127.0.0.1:18000\",command=\"/bin/false\" $TUNNEL_PUBKEY" \
    > /root/.ssh/authorized_keys2
  chmod 600 /root/.ssh/authorized_keys2
  # Remove an entry an older version of this script put in vast's file.
  if grep -qF "$TUNNEL_PUBKEY" /root/.ssh/authorized_keys 2>/dev/null; then
    grep -vF "$TUNNEL_PUBKEY" /root/.ssh/authorized_keys > /root/.ssh/authorized_keys.new || true
    chmod 600 /root/.ssh/authorized_keys.new && mv /root/.ssh/authorized_keys.new /root/.ssh/authorized_keys
  fi
fi

echo "== 5/5 start vLLM"
if [ "$SERVING" = yes ] && { [ "$KEY_CHANGED" = yes ] || [ "$SERVE_CHANGED" = yes ]; }; then
  echo "   API key or serve command changed: restarting vLLM (drain the GPU workers first)"
  pkill -f "vllm serve" || true
  pkill -f "VLLM::EngineCore" || true
  for _ in $(seq 1 30); do serving || break; sleep 2; done
  SERVING=no
fi
if [ "$SERVING" = yes ]; then
  echo "   already serving with this key"
else
  setsid nohup /workspace/serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    serving && break
    sleep 5
  done
fi
serving || { tail -30 /workspace/logs/vllm.log; exit 1; }
grep -E "GPU KV cache size|Maximum concurrency" /workspace/logs/vllm.log | tail -2 | sed 's/^.*\] /   /'
echo "   vLLM healthy"
if [ -n "${PUBLIC_IPADDR:-}" ] && [ -n "${VAST_TCP_PORT_22:-}" ]; then
  echo "   direct ssh (bypasses vast's shared proxy -- use it for --cluster): $PUBLIC_IPADDR $VAST_TCP_PORT_22"
fi
