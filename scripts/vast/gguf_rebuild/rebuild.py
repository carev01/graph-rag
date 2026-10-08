"""Rebuild the missing shard 2 of the fine-tune from its Q8_0 GGUF.

Text + MTP tensors: GGUF -> dequantize -> invert() (round-trip proven) -> base dtype.
Vision tensors: base Qwen/Qwen3.5-4B (the fine-tune never touched them).
Checks: (A) GGUF vs the shard-1 tensors we DO have = Q8 error only;
(B) every rebuilt tensor vs base: cosine near 1 if the inverse is right (fine-tune is a
small delta), and the reorder-free alternative must score worse on V-head tensors."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = ["/workspace/llama.cpp/gguf-py", str(Path(__file__).parent)]
import numpy as np, torch, gguf
from safetensors import safe_open
from safetensors.torch import save_file
from inverse import invert

FT = Path("/workspace/hf/model"); BASE = Path("/workspace/base")
S1, S2 = "model.safetensors-00001-of-00002.safetensors", "model.safetensors-00002-of-00002.safetensors"
idx = json.load(open(FT / "model.safetensors.index.json"))["weight_map"]
assert idx == json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
name_of = json.load(open("/workspace/namemap.json"))          # gguf name -> hf name
ft1 = safe_open(str(FT / S1), "pt"); base2 = safe_open(str(BASE / S2), "pt")

cos = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()
kind = lambda k: re.sub(r"\.\d+\.", ".N.", k)
A = defaultdict(list); B = defaultdict(list); alt = defaultdict(list); out = {}

r = gguf.GGUFReader("/workspace/gguf/qwen35-4b-graphrag-mtp-Q8_0.gguf")
seen = set()
for t in r.tensors:
    k = name_of[t.name]; seen.add(k)
    raw = torch.from_numpy(np.ascontiguousarray(
        gguf.quants.dequantize(t.data, t.tensor_type).reshape([int(x) for x in reversed(t.shape)]))).float()
    got = invert(k, raw)
    if idx[k] == S1:                      # (A) we have the real tensor
        ref = ft1.get_tensor(k).float()
        assert got.shape == ref.shape, (k, got.shape, ref.shape)
        A[kind(k)].append(((got - ref).norm() / ref.norm()).item())
        continue
    b = base2.get_tensor(k)               # (B) rebuilt tensor vs base
    assert got.shape == b.shape, (k, got.shape, b.shape)
    B[kind(k)].append(cos(got, b))
    if "linear_attn." in k and not k.endswith("norm.weight"):
        if k.endswith(".A_log"): raw = torch.log(-raw)
        if ".conv1d" in k: raw = raw.unsqueeze(1)
        if raw.shape == b.shape: alt[kind(k)].append(cos(raw, b))
    out[k] = got.to(b.dtype)
missing = [k for k in idx if "visual" not in k and k not in seen]
assert not missing, missing[:10]
for k, f in idx.items():
    if f == S2 and "visual" in k:
        out[k] = base2.get_tensor(k)
assert set(out) == {k for k, f in idx.items() if f == S2}

print("== A: GGUF vs shard-1 tensors we have (expect Q8 error only)")
for k, v in sorted(A.items()): print(f"  rel_err max {max(v):.4f}  {k}")
print("== B: rebuilt text tensors vs base (cosine); V-head tensors also WITHOUT un-reorder")
for k, v in sorted(B.items()):
    a = f"  no-unreorder {min(alt[k]):.4f}" if k in alt else ""
    print(f"  cos min {min(v):.5f}{a}  {k}")
if "--write" in sys.argv:
    save_file(out, str(FT / S2), metadata={"format": "pt"})
    print("wrote", len(out), "tensors")
