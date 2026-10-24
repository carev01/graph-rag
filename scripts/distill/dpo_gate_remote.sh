#!/bin/bash
# dpo4 gate, ON the Vast instance (spec 2026-10-24 section 6). The local driver copies
# dpo_gate.py, chat_format.py and val.jsonl to $W, then runs:
#   ssh ... 'bash /workspace/dpo4/dpo_gate_remote.sh gate <adapter-revision>' < hf-token
#   ssh ... 'bash /workspace/dpo4/dpo_gate_remote.sh push'                    < hf-token
# The HF token is read from stdin (one line) into the environment: never on a command
# line, never written to disk. ckpt100 is already on the instance's disk from the capture.
set -euo pipefail
CMD="${1:?gate <adapter-revision> | push}"
W=/workspace/dpo4
BASE=/workspace/hf/model-opus-distill-2026-10-ckpt100
ADAPTER_REPO=carev01/qwen35-4b-graphrag-dpo-adapters
EXPORT_REPO=carev01/qwen35-4b-graphrag
EXPORT_REV=opus-distill-2026-10-dpo4
read -r HF_TOKEN
export HF_TOKEN
source /venv/main/bin/activate
cd "$W"
[ -s "$BASE/config.json" ] || { echo "ckpt100 is not at $BASE"; exit 1; }

if [ "$CMD" = gate ]; then
  REV="${2:?adapter revision, e.g. dpo4-ckpt40}"
  supervisorctl stop vllm >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
  echo "GPU memory in use after stopping the server: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  python -c "import peft" 2>/dev/null || pip install -q peft
  rm -rf adapter merged
  hf download "$ADAPTER_REPO" --revision "$REV" --local-dir adapter \
      --include adapter_model.safetensors adapter_config.json
  unset HF_TOKEN
  python dpo_gate.py merge --base "$BASE" --adapter adapter --out merged
  python dpo_gate.py generate --model "$BASE" --tokenizer "$BASE" --val val.jsonl --out smoke-base.jsonl --limit 2
  python dpo_gate.py generate --model merged --tokenizer "$BASE" --val val.jsonl --out smoke-dpo.jsonl --limit 2
  python dpo_gate.py score --smoke --val val.jsonl --base smoke-base.jsonl --dpo smoke-dpo.jsonl --out smoke.json \
    || { echo "smoke: both models' continuations are token-identical -- adapter not engaged; no gate"; exit 2; }
  python dpo_gate.py generate --model "$BASE" --tokenizer "$BASE" --val val.jsonl --out gen-base.jsonl
  python dpo_gate.py generate --model merged --tokenizer "$BASE" --val val.jsonl --out gen-dpo.jsonl
  python dpo_gate.py generate --model merged --tokenizer "$BASE" --val val.jsonl --out smoke-json.jsonl --full --limit 3
  set +e
  python dpo_gate.py score --val val.jsonl --base gen-base.jsonl --dpo gen-dpo.jsonl --out gate.json
  rc=$?
  set -e
  echo "gate exit code: $rc (0 pass, 1 fail)"
  exit $rc
elif [ "$CMD" = push ]; then
  python -c "import json; assert json.load(open('gate.json'))['passed'], 'gate did not pass'"
  python dpo_gate.py push --merged merged --repo "$EXPORT_REPO" --revision "$EXPORT_REV" --smoke-json smoke-json.jsonl
else
  echo "unknown command $CMD"; exit 1
fi
