"""Builds notebooks/distill_dpo.ipynb. Cells shared with distill_train.ipynb (install, HF
token, Drive mount) are copied from it verbatim so the notebooks never drift apart."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELECT_SRC = (REPO / "scripts/distill/dpo_select.py").read_text()
train_nb = json.loads((REPO / "notebooks/distill_train.ipynb").read_text())


def src_of(i: int) -> str:
    return "".join(train_nb["cells"][i]["source"])


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


assert "pip_install" in src_of(3) and "get_hf_token" in src_of(5), "training notebook moved"
assert "drive.mount" in src_of(9) and "CKPT_DIR" in src_of(9), "training notebook moved"

cells = [
    md("""
# DPO round dpo4 -- qwen3.5-4b extraction model (Colab)

Starts from the SFT checkpoint `opus-distill-2026-10-ckpt100` and runs one round of
Direct Preference Optimisation on coverage-balanced pairs. It then uploads ONE selected
LoRA adapter to the private repo `carev01/qwen35-4b-graphrag-dpo-adapters`. The gate and
the export run on the Vast GPU (`scripts/distill/dpo_gate.py`), never from Colab.

**Why dpo4 (2026-10-24).** dpo3's pairs were all deletions (`chosen` = `rejected` minus the
unsupported edges), so "write fewer edges" scored on every pair. Its checkpoint-10 cut the
removed edges to 0.71 of ckpt100's but the kept ones to 0.85, failing the 0.90 bar.

**The dataset** (`scripts/distill/dpo_coverage.py`, labels by Claude subagents) has three
pair types, each from ckpt100's own replies:
- **A** drops the confirmed-unsupported edges;
- **B** keeps a supported edge that `rejected` drops;
- **C** appends the supported edges ckpt100 missed.

The mean edge-count difference between chosen and rejected is within +/-0.1, so length
carries no signal.

**Checkpoint selection by likelihood, not val loss** (dpo3: val loss kept falling while
`logps/chosen` collapsed). The rule is the last checkpoint whose eval `logps/chosen` is
within `LOGPS_TOLERANCE` (1 nat) of step 0's. `LikelihoodStop` ends training at the first
eval below that line, since nothing after it could be selected.

**Guards kept from dpo3:** causal_conv1d kernel off for training, torch.compile off, the
gradient guard, `eval_on_start`, and the check that the trainer trains the serving prompt.

**Use:** set the HF token secret (section 2), check section 3, then **Runtime -> Run all**.
A dead runtime resumes from the last checkpoint on Drive. Hand back the adapter revision
that section 10 prints.
"""),
    md("## 1. Install packages\n\nSame install as `distill_train.ipynb` section 1."),
    code(src_of(3)),
    md("""
## 1a. Fast causal_conv1d kernel -- OFF for training (`USE_HUB_CAUSAL_CONV1D`)

**This cell also turns torch.compile off.** dpo3's first attempt sat for over two hours
in torch.compile tracing the decoder layers of training step 1 (one CPU core at 100%,
GPU at 0%, no step finished). Every layer now runs eagerly.

**Off since dpo3 (2026-10-23).** dpo2 trained with this kernel and its backward pass was
broken at real sequence sizes: 40 of 72 optimiser steps had a NaN gradient norm (the
trainer skipped them) and the rest ran at ~480k. The proof is steps 1-2. dpo1 and dpo2
shared weights, seed and data order, and the learning rate cannot affect those steps. dpo1
(PyTorch path) got gradient norms of 136k and 220k there, dpo2 got 545k and NaN. Its
forward pass is fine (inference numbers were identical with and without it), so the
inference-only notebook `distill_dpo_eval.ipynb` keeps it on.

What follows is kept for that notebook and for a future fixed build:

The Gated DeltaNet layers run a short causal convolution. Without the `causal_conv1d`
package transformers uses its reference PyTorch version ("correct but much slower"; dpo1
ran that way, ~2.5 min per step on an L4). PyPI ships only a source build and the
package's GitHub wheels stop at torch 2.10, but the HF kernel hub
(`kernels-community/causal-conv1d`) has a prebuilt build for this exact torch/CUDA.

This cell loads that build and checks it against the package's own reference
implementation (forward, backward and the one-token decode step). It then registers it
as `causal_conv1d` **before** transformers imports the Qwen3.5 modelling code, which looks
the package up once, at import time. Any failure leaves the PyTorch path in place, so
the run continues, only slower. Section 6 then confirms which implementation the
loaded model actually calls.
"""),
    code("""
import importlib.util
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

# torch.compile OFF (dpo3, 2026-10-23): it spent 2 h of CPU tracing the first training
# step's decoder layers, GPU at 0%, no step finished. Checked per call, so every
# torch.compile'd function from here on (Unsloth's layer compile included) runs eagerly.
torch._dynamo.config.disable = True
print("torch.compile: off (torch._dynamo.config.disable = True)")

CC1D_REPO, CC1D_REVISION = "kernels-community/causal-conv1d", "v1"
_QWEN_MOD = "transformers.models.qwen3_5.modeling_qwen3_5"


def load_causal_conv1d():
    tv = "".join(torch.__version__.split("+")[0].split(".")[:2])
    variant = f"torch{tv}-cxx11-cu{torch.version.cuda.replace('.', '')}-x86_64-linux"
    root = Path(snapshot_download(CC1D_REPO, revision=CC1D_REVISION,
                                  allow_patterns=[f"build/{variant}/*"]))
    pkg = root / "build" / variant
    if not (pkg / "__init__.py").exists():
        raise FileNotFoundError(f"{CC1D_REPO}@{CC1D_REVISION} has no {variant} build")
    spec = importlib.util.spec_from_file_location(
        "causal_conv1d", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["causal_conv1d"] = mod
    spec.loader.exec_module(mod)
    return mod, variant


def check_causal_conv1d(mod) -> None:
    # Compare against the reference implementations shipped in the same package.
    ref = mod.causal_conv1d_interface
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2, 96, 333, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(96, 4, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(96, device="cuda", dtype=torch.bfloat16, generator=g)
    xk, xr = x.clone().requires_grad_(), x.float().clone().requires_grad_()
    out = mod.causal_conv1d_fn(xk, w, b, activation="silu")
    want = ref.causal_conv1d_ref(xr, w.float(), b.float(), activation="silu")
    assert torch.allclose(out.float(), want, atol=0.1, rtol=0.05), "forward mismatch"
    out.float().sum().backward()
    want.sum().backward()
    assert torch.allclose(xk.grad.float(), xr.grad, atol=0.1, rtol=0.05), "backward mismatch"
    state = torch.randn(2, 96, 4, device="cuda", dtype=torch.bfloat16, generator=g)
    step = torch.randn(2, 96, 1, device="cuda", dtype=torch.bfloat16, generator=g)
    s_k, s_r = state.clone(), state.float().clone()
    out = mod.causal_conv1d_update(step, s_k, w, b, activation="silu")
    want = ref.causal_conv1d_update_ref(step.float(), s_r, w.float(), b.float(), activation="silu")
    assert torch.allclose(out.float(), want, atol=0.1, rtol=0.05), "decode-step mismatch"
    assert torch.allclose(s_k.float(), s_r, atol=0.05), "decode-step state mismatch"


USE_HUB_CAUSAL_CONV1D = False   # dpo2: its BACKWARD gave NaN / ~5e5 gradient norms
CC1D_LOADED = False
if not USE_HUB_CAUSAL_CONV1D:
    print("causal_conv1d: hub kernel off (USE_HUB_CAUSAL_CONV1D = False) -- PyTorch path")
elif _QWEN_MOD in sys.modules:
    print("transformers' Qwen3.5 code is already imported, so a kernel registered now would "
          "be ignored -- Runtime -> Restart session, then Run all")
else:
    try:
        _cc1d, _variant = load_causal_conv1d()
        check_causal_conv1d(_cc1d)
        CC1D_LOADED = True
        print(f"causal_conv1d: {CC1D_REPO} {_variant} loaded and matches its reference")
    except Exception as e:
        for _k in [k for k in sys.modules if k.split(".")[0] == "causal_conv1d"]:
            del sys.modules[_k]
        print(f"causal_conv1d: hub kernel NOT used ({type(e).__name__}: {e}) -- "
              "training runs on the PyTorch path (correct, slower)")
"""),
    md(src_of(4)),
    code(src_of(5)),
    md("""
## 3. Configuration

The DPO hyper-parameters are starting points, not tuned values: the held-out eval and the
end-to-end trial decide. **Learning rate 5e-5, not 5e-6 (2026-10-22).** The dpo1 run at
5e-6 separated the val pairs (reward accuracy 0.97) but moved them too little to change
what the model writes. At the point where `chosen` and `rejected` split, the mean log-odds
of skipping the unsupported edge went from -3.29 to -2.95, and greedy decoding skipped it
in 4 of 39 pairs both before and after. A LoRA that starts at zero needs roughly ten times
a full fine-tune's DPO rate.

**dpo3 (2026-10-23) repeats dpo2 with clean gradients.** dpo2 (5e-5) ran with the hub
causal_conv1d kernel, whose backward pass was broken (section 1a): 40 of 72 steps skipped
on NaN, the rest at ~480k gradient norm, so 5e-5 has never had a clean test. dpo3 runs on
the PyTorch path, as dpo1 did, and section 9 stops on the first non-finite gradient.

dpo1 also showed gradient norms above 100k for exactly steps 1-10, then 2.5-5.6 from step
11 on: the first evaluation and checkpoint save happen at step 10. `eval_on_start` runs an
evaluation before step 1, which tests whether that resets it, and records the starting
point. Section 9 prints the first steps' norms; past `GRAD_NORM_FROM_STEP` a norm above
`GRAD_NORM_LIMIT` stops the run (dpo1 never exceeded 6 there).
"""),
    code("""
from pathlib import Path

# --- dataset: dpo_pairs.py output (train.jsonl / val.jsonl), "drive" or "hf_dataset" -------
DATA_SOURCE = "hf_dataset"
HF_DATASET_REPO = "carev01/ft-opus-distill"   # private
HF_DATASET_REVISION = "dpo4"
DRIVE_DATA_DIR = "/content/drive/MyDrive/ft-opus-dpo4"   # only when DATA_SOURCE == "drive"

# --- Google Drive: checkpoints, final adapter, metrics (survive a runtime death) ----------
RUN_NAME = "opus-distill-dpo4"     # dpo3: 5e-5, deletion-only pairs taught brevity
DRIVE_RUN_DIR = Path("/content/drive/MyDrive/graph-rag-distill") / RUN_NAME
CKPT_DIR = DRIVE_RUN_DIR / "checkpoints"
FINAL_ADAPTER_DIR = DRIVE_RUN_DIR / "final-adapter"
FINAL_MARKER = FINAL_ADAPTER_DIR / "COMPLETE"   # written LAST: adapter fully saved

# --- the model DPO starts from: the merged SFT checkpoint the pairs were generated with ---
HF_MODEL_REPO = "carev01/qwen35-4b-graphrag"
START_REVISION = "opus-distill-2026-10-ckpt100"
START_DIR = Path("/content/start-model")    # adapter_config points here: keep it fixed
MAX_SEQ_LENGTH = 20480
LOAD_IN_4BIT = True

# --- LoRA (same shape as the SFT run) ----------------------------------------------------
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# --- DPO -----------------------------------------------------------------------------------
BETA = 0.1                 # strength of the pull towards the reference (ckpt100)
NLL_ALPHA = 2.0            # NLL on chosen ("sft" loss, a per-token mean); 1.0 did not stop dpo3's displacement
LEARNING_RATE = 2e-5       # dpo3 5e-5 displaced chosen by step 20; dpo1 5e-6 did nothing
NUM_TRAIN_EPOCHS = 2
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "cosine"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_STEPS = 10            # also the save interval, so the best checkpoint is on Drive
SAVE_TOTAL_LIMIT = None     # adapters are small; selection needs every checkpoint
SEED = 20261021
GRAD_NORM_LIMIT = 1000.0   # training guard: stop on a larger norm from GRAD_NORM_FROM_STEP on
GRAD_NORM_FROM_STEP = 12   # dpo1: >100k for steps 1-10 (before the first eval), then <6

LOGPS_TOLERANCE = 1.0      # selection: eval logps/chosen may fall at most this far below step 0

# --- hand-off: ONE adapter to a PRIVATE repo; the gate and export run on Vast ---------------
HF_ADAPTER_REPO = "carev01/qwen35-4b-graphrag-dpo-adapters"

OUT_DIR = Path("/content/outputs/opus-distill-dpo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("config set")
"""),
    md("## 3a. Mount Google Drive\n\nSame as `distill_train.ipynb` section 3a."),
    code(src_of(9)),
    md("## 4. Get the dataset"),
    code("""
DATA_DIR = Path("/content/data/ft-opus-dpo4")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if DATA_SOURCE == "drive":
    DATA_DIR = Path(DRIVE_DATA_DIR)
elif DATA_SOURCE == "hf_dataset":
    from huggingface_hub import hf_hub_download
    for fname in ("train.jsonl", "val.jsonl", "val_eval.jsonl", "MANIFEST.json", "sample.json"):
        try:
            p = hf_hub_download(repo_id=HF_DATASET_REPO, repo_type="dataset",
                                filename=fname, revision=HF_DATASET_REVISION, token=HF_TOKEN)
        except Exception:
            if fname in ("sample.json", "MANIFEST.json"):
                print(f"  {fname} not in {HF_DATASET_REPO}@{HF_DATASET_REVISION} -- skipping")
                continue
            raise
        (DATA_DIR / fname).write_bytes(Path(p).read_bytes())
else:
    raise ValueError(f"unknown DATA_SOURCE {DATA_SOURCE!r}")

assert all((DATA_DIR / f).exists() for f in ("train.jsonl", "val.jsonl", "val_eval.jsonl")), DATA_DIR
print(f"dataset ready at {DATA_DIR}")
"""),
    md("""
## 5. Load the pairs and check their shape

Every record: `messages` = `[system, user]` exactly as vLLM received them; `chosen` and
`rejected` = the reply JSON. An A pair's chosen is a strict subset of rejected; a B or C
pair's rejected is a strict subset of chosen. `meta.delta` must match, and the train set's
mean delta must be within +/-0.1. `val_eval.jsonl` is the trainer's fixed eval subset.
"""),
    code("""
import json
from collections import Counter


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


train_records = read_jsonl(DATA_DIR / "train.jsonl")
val_records = read_jsonl(DATA_DIR / "val.jsonl")
val_eval_records = read_jsonl(DATA_DIR / "val_eval.jsonl")
for _r in train_records + val_records:
    assert [m["role"] for m in _r["messages"]] == ["system", "user"], _r["meta"]
    _c, _x = json.loads(_r["chosen"])["edges"], json.loads(_r["rejected"])["edges"]
    _small, _big = (_c, _x) if _r["meta"]["type"] == "A" else (_x, _c)
    assert 0 < len(_small) < len(_big), _r["meta"]
    assert all(e in _big for e in _small), _r["meta"]
    assert _r["meta"]["delta"] == len(_c) - len(_x), _r["meta"]
_val_ids = {r["meta"]["call_id"] for r in val_records}
assert all(r["meta"]["call_id"] in _val_ids for r in val_eval_records), "val_eval is not a subset of val"
for _name, _rows in (("train", train_records), ("val", val_records), ("val_eval", val_eval_records)):
    _types = Counter(r["meta"]["type"] for r in _rows)
    _mean = sum(r["meta"]["delta"] for r in _rows) / len(_rows)
    print(f"{_name}: {len(_rows)} pairs {dict(sorted(_types.items()))}, mean edge delta {_mean:+.3f}")
assert abs(sum(r["meta"]["delta"] for r in train_records) / len(train_records)) <= 0.1, (
    "train set is not length-balanced (spec: |mean delta| <= 0.1)")
assert len(train_records) >= 50, "too few pairs for a DPO round -- capture more first"
"""),
    md("""
## 6. Download the starting model and attach LoRA

The merged bf16 weights of `START_REVISION` go to `START_DIR`. A fixed path matters: the
saved adapter records it as its base.
"""),
    code("""
from huggingface_hub import snapshot_download
from unsloth import FastLanguageModel
import torch

snapshot_download(repo_id=HF_MODEL_REPO, revision=START_REVISION, local_dir=str(START_DIR),
                  ignore_patterns=["*.gguf"], token=HF_TOKEN)
assert any(START_DIR.glob("*.safetensors")), f"no safetensors in {HF_MODEL_REPO}@{START_REVISION}"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(START_DIR),
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
print(f"{START_REVISION} loaded, LoRA attached (the reference model is the same weights "
      "with the adapter disabled)")
import transformers.models.qwen3_5.modeling_qwen3_5 as _mq


def calls_package(fn, package: str) -> bool:
    # Follow transformers' kernel wrappers down to the implementation they call.
    seen, stack = set(), [fn]
    while stack:
        f = stack.pop()
        if f is None or id(f) in seen or not callable(f):
            continue
        seen.add(id(f))
        if str(getattr(f, "__module__", "")).split(".")[0] == package:
            return True
        stack.append(getattr(f, "__wrapped__", None))
        for c in getattr(f, "__closure__", None) or ():
            try:
                stack.append(c.cell_contents)
            except ValueError:
                pass
    return False


_fast = {"causal_conv1d": calls_package(_mq.causal_conv1d_fn, "causal_conv1d"),
         "fla": calls_package(getattr(_mq, "torch_chunk_gated_delta_rule", None), "fla")}
print(f"fast kernels in use: {_fast}")
assert USE_HUB_CAUSAL_CONV1D or not _fast["causal_conv1d"], (
    "the hub causal_conv1d kernel is off, yet the model calls a causal_conv1d package -- "
    "its backward pass broke dpo2; restart the session without it")
if CC1D_LOADED and not _fast["causal_conv1d"]:
    print("WARNING: the causal_conv1d kernel loaded in section 1a but the model does not "
          "call it -- this run is on the PyTorch path")
"""),
    md("""
## 7. Build prompt / chosen / rejected text

`prompt` is the serving prompt: the chat template with `add_generation_prompt=True,
enable_thinking=False`, which already ends in the empty think block. Completions get
`<|im_end|>` unless this TRL version appends the end-of-turn token itself (section 8
decides). Pairs longer than `MAX_SEQ_LENGTH` are dropped, never truncated: a truncated
prompt would train on a prompt vLLM never sees.
"""),
    code("""
from datasets import Dataset

END_MARK = "<|im_end|>"
# Unsloth may hand back a multimodal processor for this architecture; its __call__ batches
# (len(ids) would be 1) and it has no eos_token. Count and decode with the text tokenizer.
TEXT_TOK = getattr(tokenizer, "tokenizer", tokenizer)
RESPONSE_PART = tokenizer.apply_chat_template(
    [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
    enable_thinking=False).rsplit(END_MARK + "\\n", 1)[-1]
print(f"serving prompt ends with {RESPONSE_PART!r}")


def to_pair(r: dict, mark: str) -> dict:
    prompt = tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    assert prompt.endswith(RESPONSE_PART), prompt[-80:]
    return {"prompt": prompt, "chosen": r["chosen"] + mark, "rejected": r["rejected"] + mark}


def build(records: list[dict], mark: str) -> Dataset:
    rows, dropped = [], 0
    for r in records:
        p = to_pair(r, mark)
        n = max(len(TEXT_TOK(p["prompt"] + p[k], add_special_tokens=False)["input_ids"])
                for k in ("chosen", "rejected"))   # a C pair's chosen is the longer one
        if n > MAX_SEQ_LENGTH:
            dropped += 1
            continue
        rows.append(p)
    print(f"  {len(rows)} pairs kept, {dropped} over {MAX_SEQ_LENGTH} tokens dropped")
    return Dataset.from_list(rows)


_lens = sorted(len(TEXT_TOK(to_pair(r, "")["prompt"], add_special_tokens=False)["input_ids"])
               for r in train_records)
MAX_PROMPT_TOKENS = min(_lens[-1], MAX_SEQ_LENGTH)
print(f"prompt tokens: median {_lens[len(_lens) // 2]}, max {_lens[-1]}")
"""),
    md("""
## 8. Build the trainer -- and check what it will actually train on (fail loudly)

TRL versions differ: some append the tokenizer's EOS to every completion, and some
tokenise the prompt with special tokens added. So this cell decodes the trainer's own
first tokenised row and requires prompt == serving prompt and chosen == answer +
`<|im_end|>` (a trailing newline is accepted). If this TRL appended `<|im_end|>` a
second time, the trainer is rebuilt once with the completions unmarked.
"""),
    code("""
import dataclasses
import inspect

try:  # older Unsloth needs the explicit patch; newer ones patch on import
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
except ImportError:
    pass
from transformers import TrainerCallback
from trl import DPOConfig, DPOTrainer

_fields = {f.name for f in dataclasses.fields(DPOConfig)}
_extra = {}
if "max_prompt_length" in _fields:
    _extra["max_prompt_length"] = MAX_PROMPT_TOKENS
# transformers 5.x removed warmup_ratio; its warmup_steps reads a float < 1 as a ratio.
_extra["warmup_ratio" if "warmup_ratio" in _fields else "warmup_steps"] = WARMUP_RATIO
if NLL_ALPHA and "rpo_alpha" in _fields:      # TRL < 1.0
    _extra["rpo_alpha"] = NLL_ALPHA
elif NLL_ALPHA:
    # TRL 1.x dropped rpo_alpha. The same term is the "sft" loss: the mean per-token NLL
    # of the chosen completion, added to the sigmoid (DPO) loss with weight NLL_ALPHA.
    assert "loss_weights" in _fields, ("this TRL has neither rpo_alpha nor loss_weights: "
                                       "set NLL_ALPHA = 0 only if you accept plain DPO "
                                       "(coverage risk)")
    _extra["loss_type"] = ["sigmoid", "sft"]
    _extra["loss_weights"] = [1.0, NLL_ALPHA]
print(f"DPOConfig extras: {_extra}")

dpo_config = DPOConfig(
    output_dir=str(CKPT_DIR),
    beta=BETA,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    optim="adamw_8bit",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=False,   # dpo4: selection by logps/chosen (section 10), not val loss
    max_length=MAX_SEQ_LENGTH,
    seed=SEED,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    eval_on_start=True,        # baseline eval; also tests whether it ends dpo1's step 1-10 norms
    **_extra,
)
_tok_kw = ("processing_class" if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters
           else "tokenizer")


class GradGuard(TrainerCallback):
    # Stop on the first non-finite gradient, BEFORE the optimiser step: dpo2 skipped 40 of
    # 72 steps on NaN for hours without a word. Print the first steps' norms (dpo1: >100k
    # until the first eval) and stop on a huge finite norm once training has settled.
    checked = 0   # optimiser steps inspected: proves the hook fires under Unsloth's loop

    def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
        if not self.checked:
            print("  grad guard: inspecting gradients before every optimiser step", flush=True)
        self.checked += 1
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise RuntimeError(f"step {state.global_step + 1}: non-finite gradients in "
                               f"{len(bad)} tensors (first: {bad[0]}) -- stopping")

    def on_log(self, args, state, control, logs=None, **kw):
        g = (logs or {}).get("grad_norm")
        if g is None:
            return
        if not self.checked:
            raise RuntimeError("grad guard never ran before an optimiser step (this trainer "
                               "loop skips on_pre_optimizer_step) -- the NaN check is not live")
        if state.global_step <= GRAD_NORM_FROM_STEP:
            print(f"  step {state.global_step}: grad_norm {g:.4g}", flush=True)
        elif g > GRAD_NORM_LIMIT:
            raise RuntimeError(f"step {state.global_step}: grad_norm {g:.4g} > "
                               f"{GRAD_NORM_LIMIT} -- stopping")

""" + SELECT_SRC + """

class LikelihoodStop(TrainerCallback):
    # The dpo4 selection rule, live: stop at the first eval whose logps/chosen falls more
    # than LOGPS_TOLERANCE below step 0's -- nothing after it could be selected. Stateless
    # (reads state.log_history), so a resumed run keeps its step-0 baseline.
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if LOGPS_KEY not in (metrics or {}):
            raise RuntimeError(f"eval metrics have no {LOGPS_KEY}: the selection rule cannot run")
        series = eval_logps(state.log_history)
        if not series or series[0][0] != 0:
            raise RuntimeError("no step-0 eval in the log history -- is eval_on_start off?")
        start, (step, value) = series[0][1], series[-1]
        if step and below_line(start, value, LOGPS_TOLERANCE):
            print(f"  step {step}: eval logps/chosen {value:.2f} < {start:.2f} - "
                  f"{LOGPS_TOLERANCE} -- stopping (selection rule)", flush=True)
            control.should_training_stop = True
        return control


def make_trainer(mark: str) -> DPOTrainer:
    # TRL 1.x copies the untrained adapter into a frozen "ref" adapter (= ckpt100, since
    # LoRA starts at zero) for the reference pass; a rebuild must not find one already.
    if "ref" in getattr(model, "peft_config", {}):
        model.delete_adapter("ref")
    print(f"building datasets (completion mark {mark!r})")
    return DPOTrainer(model=model, ref_model=None, args=dpo_config,
                      train_dataset=build(train_records, mark),
                      eval_dataset=build(val_eval_records, mark),
                      callbacks=[GradGuard(), LikelihoodStop()],
                      **{_tok_kw: tokenizer})


def first_row(t: DPOTrainer) -> tuple[str, str]:
    row = t.train_dataset[0]
    for p_col, c_col in (("prompt_input_ids", "chosen_input_ids"), ("prompt_ids", "chosen_ids")):
        if p_col in row and c_col in row:
            dec = lambda ids: TEXT_TOK.decode(ids, skip_special_tokens=False)
            return dec(row[p_col]), dec(row[c_col])
    raise AssertionError(f"unknown tokenised layout {sorted(row)} -- check this TRL version's "
                         "DPOTrainer before training")


trainer = make_trainer(END_MARK)
_p, _c = first_row(trainer)
_answer = train_records[0]["chosen"]
if _c.startswith(_answer + END_MARK + END_MARK) or _c.startswith(_answer + END_MARK + "\\n" + END_MARK):
    print("TRL appends <|im_end|> itself -- rebuilding without our mark")
    trainer = make_trainer("")
    _p, _c = first_row(trainer)
_serving = to_pair(train_records[0], "")["prompt"]
assert _p == _serving, f"tokenised prompt != serving prompt: {_p[-120:]!r} vs {_serving[-120:]!r}"
_OK_TAILS = {END_MARK, END_MARK + "\\n"}
_eos = TEXT_TOK.eos_token or ""
_tail = _c[len(_answer):] if _c.startswith(_answer) else None
assert _tail is not None, f"chosen does not start with the answer: {_c[:120]!r}"
if _tail not in _OK_TAILS:
    # An EOS other than <|im_end|> appended after it: vLLM stops at <|im_end|>, harmless.
    assert _eos and _eos != END_MARK and _tail.endswith(_eos) and _tail[:-len(_eos)] in _OK_TAILS, (
        f"chosen does not decode to answer + <|im_end|>: tail {_tail!r}")
    print(f"note: TRL also trains on {_eos!r} after <|im_end|> (harmless)")
print(f"OK: trains on serving prompt + answer + {_tail!r}")
"""),
    md("""
## 9. Train (resumable)

Watch `eval_logps/chosen` at every eval: section 10 keeps the last checkpoint within
`LOGPS_TOLERANCE` of step 0's value, and training stops by itself at the first eval below
that line ("stopping (selection rule)"). `eval_rewards/accuracies` well above 0.5 means
the model separates the pairs.

The gradient guard prints "grad guard: inspecting gradients..." once, then the gradient
norm of each of the first `GRAD_NORM_FROM_STEP` steps; it stops on a NaN/inf gradient, or
a norm above `GRAD_NORM_LIMIT` later.
"""),
    code("""
import json as _json

_CKPT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")


def last_complete_checkpoint(ckpt_dir: Path) -> str | None:
    for c in sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]),
                    reverse=True):
        if all((c / f).exists() for f in _CKPT_FILES) and any(c.glob("adapter_model.*")):
            return str(c)
        print(f"skipping incomplete {c.name}")
    return None


if FINAL_MARKER.exists():
    print(f"final adapter already on Drive ({FINAL_ADAPTER_DIR}) -- skipping to export. "
          "Delete that folder to train again.")
else:
    _last = last_complete_checkpoint(CKPT_DIR)
    print(f"resuming from {_last}" if _last else "no checkpoint on Drive -- starting fresh")
    train_result = trainer.train(resume_from_checkpoint=_last)
    model.save_pretrained(str(FINAL_ADAPTER_DIR), selected_adapters=["default"])  # not TRL's "ref"
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    (DRIVE_RUN_DIR / "train_result.json").write_text(_json.dumps(train_result.metrics, indent=2))
    (DRIVE_RUN_DIR / "log_history.json").write_text(
        _json.dumps(trainer.state.log_history, indent=2))
    FINAL_MARKER.write_text("adapter + tokenizer fully saved\\n")
"""),
    md("""
## 10. Select the checkpoint and hand it off

Applies the rule to `log_history.json` and prints the trajectory. If no checkpoint after
step 0 qualifies, the round fails here, with no gate. Otherwise the selected checkpoint's
adapter goes to the PRIVATE repo `HF_ADAPTER_REPO`, revision `dpo4-ckpt<step>`.
"""),
    code("""
import json as _json

from huggingface_hub import HfApi

_hist = _json.loads((DRIVE_RUN_DIR / "log_history.json").read_text())
for _s, _v in eval_logps(_hist):
    print(f"eval @ {_s:>4}: logps/chosen {_v:.2f}")
SELECTED_STEP = select_step(_hist, LOGPS_TOLERANCE)
assert SELECTED_STEP is not None, (
    f"no checkpoint after step 0 kept logps/chosen within {LOGPS_TOLERANCE} nat of step 0's "
    "-- the round fails here, no gate (spec section 5)")
_ckpt = CKPT_DIR / f"checkpoint-{SELECTED_STEP}"
assert (_ckpt / "adapter_model.safetensors").exists(), f"{_ckpt} has no adapter"
ADAPTER_REVISION = f"dpo4-ckpt{SELECTED_STEP}"
api = HfApi(token=HF_TOKEN)
api.create_repo(HF_ADAPTER_REPO, repo_type="model", private=True, exist_ok=True)
assert api.model_info(HF_ADAPTER_REPO).private, f"{HF_ADAPTER_REPO} is not private -- refusing"
api.create_branch(HF_ADAPTER_REPO, repo_type="model", branch=ADAPTER_REVISION, exist_ok=True)
for _f in ("adapter_model.safetensors", "adapter_config.json"):
    api.upload_file(path_or_fileobj=str(_ckpt / _f), path_in_repo=_f, repo_id=HF_ADAPTER_REPO,
                    repo_type="model", revision=ADAPTER_REVISION)
api.upload_file(path_or_fileobj=str(DRIVE_RUN_DIR / "log_history.json"),
                path_in_repo="log_history.json", repo_id=HF_ADAPTER_REPO, repo_type="model",
                revision=ADAPTER_REVISION)
print(f"DONE: {HF_ADAPTER_REPO}@{ADAPTER_REVISION} -- hand this name back")
"""),
    md("""
## 11. Next -- outside this notebook

Hand back the adapter revision. The gate runs on the Vast GPU against the bf16 merge
(`scripts/distill/dpo_gate_remote.sh gate <revision>`), with the pass bar unchanged from
dpo3. Only after a pass, and with authorisation, the same merged weights are pushed to
`carev01/qwen35-4b-graphrag@opus-distill-2026-10-dpo4` and the adoption trial runs.
"""),
]

nb = {"cells": cells, "metadata": {**train_nb["metadata"],
                                   "colab": {"name": "distill_dpo.ipynb", "provenance": []}},
      "nbformat": train_nb["nbformat"], "nbformat_minor": train_nb["nbformat_minor"]}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "notebooks/distill_dpo.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print(f"wrote {out} ({len(cells)} cells); all code cells compile", file=sys.stderr)
