# Cheap extraction tier on a vast.ai GPU — quick reference

The cheap extraction tier can run on a rented GPU instead of OpenRouter: the fine-tuned
**qwen35-4b-graphrag** served by **vLLM** on a vast.ai RTX 4090 and reached from the
cluster through an SSH tunnel. Workers on this tier run as a second Deployment,
`graph-rag-worker-gpu`; the strong tier, the embedder and everything else are unchanged.

```
graph-rag-worker-gpu ──http──▶ Service graph-rag-vast-tunnel:8000
                                  │ (pod: ssh -L 0.0.0.0:8000:127.0.0.1:18000)
                                  ▼
                       vast.ai instance  127.0.0.1:18000  vLLM (API key required)
```

**Pilot (2026-10-08, RTX 4090 at $0.383/h):** on 12 Veeam articles through the production
pipeline it was **1.85× faster than solar-pro4 per article** (33 s vs 61 s, sequential),
with 126 vs 131 distinct supported claims (blind judge), 5 vs 2 unsupported facts, and
half the redundant restatements. The load test saturated at ~1.7 req/s
(~7,800 prompt tok/s) at concurrency 48, about 420 articles/h projected; at
production's ~66k prompt tokens/article that is ~$0.001/article against ~$0.0066 on
solar-pro4.

## When you need this page

- **A new instance** (first rent, or the old one was destroyed/recycled): §1 + §2.
- **Instance stopped and started again:** the filesystem survives, but nothing restarts
  vLLM — run §1 again (idempotent; it skips the model download, and restarts vLLM only
  if the API key or the serve command changed — drain the GPU workers first, §3). The ssh port can change on
  restart: if it did, run §1 with `--cluster`.
- **Switching workers between tiers:** §3.

Model weights: private HF repo **`carev01/qwen35-4b-graphrag`** (bf16 safetensors). Its
shard 2 was rebuilt from the Q8_0 GGUF on 2026-10-08 — see §5.

## 1. Provision the instance

1. Rent a GPU on vast.ai with the **vLLM template** (24 GB VRAM is enough: the model takes
   8 GB, the rest is KV cache). Disk ≥ 24 GB. Make sure **your own SSH public key** is on
   your vast account, then copy the instance's ssh host and port from its card (e.g.
   `ssh -p 16301 root@ssh6.vast.ai`).
2. Run, from the repo root:

   ```bash
   scripts/vast/provision.sh ssh6.vast.ai 16301 --cluster
   ```

   It prompts for a Hugging Face token with read access to the model repo (used once on
   the instance, never stored), then:
   - stops the template's stock vLLM (it serves Qwen3.5-9B) and kills the engine process
     it leaves holding ~20 GB of VRAM, and deletes that model's 18 GB download;
   - downloads `carev01/qwen35-4b-graphrag` and starts vLLM on `127.0.0.1:18000` with an
     API key, prefix caching, thinking off, vision inputs disabled;
   - installs the cluster's tunnel public key on the instance, **restricted to forwarding
     `127.0.0.1:18000`** with no shell (`restrict,port-forwarding,permitopen=…,command="/bin/false"`), and proves the
     tunnel key can reach `/v1/models`;
   - with `--cluster`: rewrites the `graph-rag-vast` Secret (ssh host, port, host key,
     tunnel private key, vLLM API key) and restarts the tunnel pod.

   Keys live outside the repo in `~/.config/graph-rag/vast/` (`api_key`,
   `tunnel_ed25519`); the first run generates them and later runs reuse them.

3. Check: `kubectl -n graph-rag get pods -l app=graph-rag-vast-tunnel` shows `1/1 Ready`
   (readiness probes vLLM's `/health` through the forward).

For local testing, open your own tunnel: `ssh -N -p <port> -L 18080:127.0.0.1:18000
root@<host>`, then `scripts/vast/loadtest.py --key-file ~/.config/graph-rag/vast/api_key`.

## 2. Enable the tier in the Helm release

```bash
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set image.tag=sha-<deployed> \
  --set vast.enabled=true --set vast.workerReplicas=4 --set worker.replicas=4
```

`vast.enabled` adds the tunnel Deployment and Service and the `graph-rag-worker-gpu`
Deployment. The GPU workers get `CHEAP_LLM_BASE_URL=http://graph-rag-vast-tunnel:8000/v1`,
`CHEAP_LLM_API_KEY` from the Secret and `CHEAP_LLM_MODEL=qwen35-graphrag` as `env`, which
overrides the shared Secret/ConfigMap for those pods only.

**Before scaling GPU workers up, prove the path with a real call** (a failing GPU worker
spends one of each job's 5 attempts per failure): run a one-off pod with the worker image,
the same `envFrom`, and the three `env` overrides above, that POSTs one
`/chat/completions` to `$CHEAP_LLM_BASE_URL` with `$CHEAP_LLM_API_KEY` and expects 200.
Then scale up and confirm vLLM's `vllm:request_success_total` rises.

## 3. Moving workers between tiers

Both Deployments claim from the same queue (`FOR UPDATE SKIP LOCKED`), so any split works
and changing it never loses work (SIGTERM finishes the in-flight article).

| goal | `worker.replicas` | `vast.workerReplicas` |
|---|---|---|
| canary | 4 | 4 |
| all on the GPU | 0 | 8 |
| back to OpenRouter | 8 | 0 |

```bash
kubectl -n graph-rag scale deploy/graph-rag-worker-gpu --replicas=8
kubectl -n graph-rag scale deploy/graph-rag-worker --replicas=0
```

(`kubectl scale` is quick; record the final split in the Helm values too so the next
`helm upgrade` does not revert it.)

Compare tiers during a canary from the worker logs: `semantic batch llm tokens` lines per
pod, jobs done per hour, and `dead` counts in `semantic_jobs`. vLLM's own counters are at
`/metrics` on the tunnel Service (needs the API key).

**If the instance goes away**, GPU-tier jobs fail and back off (5 attempts, 30 s base) —
a long outage can push jobs to `dead`. Scale `graph-rag-worker-gpu` to 0 and the API
tier back up first, then fix the instance.

## 4. Troubleshooting

| symptom | cause / fix |
|---|---|
| ~20 GB VRAM in use with nothing serving | the stock vLLM's `VLLM::EngineCore` survived `supervisorctl stop`; `pkill -f VLLM::EngineCore` (the script does this) |
| tunnel pod CrashLoop, `Host key verification failed` | the instance changed (new host/port/host key): re-run §1 with `--cluster` |
| every login refused after provisioning | an entry got glued onto the last account key (vast writes `authorized_keys` without a final newline; `remote_setup.sh` now adds one first). Fix from the vast console → Jupyter terminal: put your public key back on its own line in `/root/.ssh/authorized_keys` |
| tunnel pod `Permission denied (publickey)` | the restricted tunnel key is missing from `/root/.ssh/authorized_keys` (vast may rewrite it on recycle): re-run §1 |
| GPU workers log `Connection error.` but vLLM's request counter does not move | a trailing newline in the Secret's `api_key` (httpx refuses the header and graphiti reports it as a connection error); `provision.sh --cluster` now strips it. Test from a pod before scaling: see §2 |
| GPU workers fail with `httpx.ConnectError: All connection attempts failed`, tunnel pod `Ready=False` | an HTTP readiness probe through the tunnel (~1.2 s round trip) timed out under load and pulled the only tunnel pod from the Service. Readiness is now a TCP check on the local listener, liveness probes `/health` with a 10 s timeout, and there are 2 tunnel replicas (`vast.tunnelReplicas`) |
| tunnel Ready but workers get 401 | API key mismatch between the Secret and `/workspace/vllm.key`: re-run §1 with `--cluster` |
| a job fails with `maximum context length is 32768 tokens. However, you requested 16384 output tokens` | graphiti asks for `max_tokens=16384`, so a prompt over ~16k tokens overflows a 32k window; `serve.sh` uses `--max-model-len 65536` (`MAX_MODEL_LEN`) |
| vLLM log: `Mamba cache mode is set to 'align'` | expected: prefix caching on Qwen3.5's linear-attention layers is experimental in vLLM 0.23 |
| `/workspace` lost after recycle | expected without a vast volume: §1 re-downloads everything |

The vLLM log is `/workspace/logs/vllm.log` on the instance; the serve command is
`/workspace/serve.sh` (written by `scripts/vast/remote_setup.sh`).

## 5. Why the weights were rebuilt (history)

The fine-tune's original export had an empty shard 2 (the Google Drive upload had
failed), and vLLM 0.23 / transformers 5.12 cannot load a Qwen3.5 GGUF. Shard 2 was
rebuilt from `carev01/qwen35-4b-graphrag-mtp-gguf` with `scripts/vast/gguf_rebuild/`:

- `inverse.py` inverts llama.cpp's Qwen3.5 converter (`conversion/qwen.py`): V-head
  reorder (grouped↔tiled) on `in_proj_qkv/z/a/b`, `A_log`, `dt_bias`, `conv1d`,
  `out_proj`; `A_log` stored as `-exp(A_log)`; `conv1d` squeezed; +1 on every RMSNorm
  except `linear_attn.norm` — including MTP's `pre_fc_norm_*`, which the converter renames
  to `enorm`/`hnorm` before applying it.
- `roundtrip.py` runs the converter's own forward transform on base Qwen3.5-4B tensors
  and inverts it: 441/441 tensors back within 6e-8. It also derives the tensor-name map,
  so no name is hand-mapped.
- `rebuild.py` checks the inverse on the real GGUF (against the shard-1 tensors that did
  survive: 0.5–0.6% error = Q8 rounding; every rebuilt tensor at cosine ≥ 0.9996 to the
  base, while skipping the un-reorder drops V-head tensors to between −0.35 and 0.60),
  then writes shard 2 with the vision tower copied from the base model.

These run on the instance (paths under `/workspace`) and need llama.cpp's `gguf-py` and
`conversion/` on `sys.path`. Re-run them only if the HF repo is ever lost and the GGUF is
all that remains.
