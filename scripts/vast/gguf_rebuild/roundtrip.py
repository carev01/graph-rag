"""Round-trip proof: HF tensor -> llama.cpp converter's OWN forward transform -> invert()
must reproduce the HF tensor, for every text/MTP tensor of the base model. Also yields
the converter's name map (gguf name -> hf name) used by the rebuild, so names are never
hand-mapped."""
import json, re, sys
from pathlib import Path
sys.path[:0] = ["/workspace/llama.cpp", "/workspace/llama.cpp/gguf-py", str(Path(__file__).parent)]
import torch, gguf
from safetensors import safe_open
from conversion.base import ModelBase, ModelType, get_model_architecture
from convert_hf_to_gguf import get_model_class
from inverse import invert

D = Path("/workspace/rt")
hp = ModelBase.load_hparams(D, False)
cls = get_model_class(get_model_architecture(hp, ModelType.TEXT))
print("converter class:", cls.__name__, [c.__name__ for c in cls.__mro__[:6]])
m = cls(D, gguf.LlamaFileType.ALL_F32, Path("/tmp/rt.gguf"), eager=True, dry_run=True)

idx = json.load(open(D / "model.safetensors.index.json"))["weight_map"]
files = {f: safe_open(str(D / f), "pt") for f in set(idx.values())}
raw = lambda k: files[idx[k]].get_tensor(k)

# filtered-name -> original HF name: re-run filter on each original key
orig_of = {}
for k in idx:
    t = cls.filter_tensors((k, lambda: None))
    if t:
        orig_of[t[0]] = k

names, worst = {}, {}
for fname in m.model_tensors:
    hf = orig_of[fname]
    if "visual" in hf:
        continue
    src = raw(hf).float()
    mm = re.search(r"layers\.(\d+)\.", fname)
    bid = int(mm.group(1)) if mm else None
    outs = list(m.modify_tensors(src.clone(), fname, bid))
    assert len(outs) == 1, (fname, [o[0] for o in outs])
    gname, g = outs[0]
    g = torch.as_tensor(g).float()
    names[gname] = hf
    back = invert(hf, g)
    assert back.shape == src.shape, (hf, back.shape, src.shape)
    err = (back - src).abs().max().item()
    kind = re.sub(r"\.\d+\.", ".N.", hf)
    worst[kind] = max(worst.get(kind, 0.0), err)

for k, v in sorted(worst.items()):
    print(f"{v:.3e}  {k}")
json.dump(names, open("/workspace/namemap.json", "w"), indent=0)
print("mapped", len(names), "tensors; max err overall", max(worst.values()))
