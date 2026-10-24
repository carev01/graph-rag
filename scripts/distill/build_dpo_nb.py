"""Builds notebooks/distill_dpo.ipynb. Cells shared with distill_train.ipynb (install, HF
token, Drive mount) are copied from it verbatim so the notebooks never drift apart."""
import json
import sys
from pathlib import Path

REPO = Path("/home/openclaw/Documents/Shared/projects/graph-rag")
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
# DPO round -- qwen3.5-4b extraction model (Colab)

Starts from the SFT checkpoint `opus-distill-2026-10-ckpt100` and runs one round of
Direct Preference Optimisation against the student's OWN unsupported facts, then pushes
the merged model as a new revision of `carev01/qwen35-4b-graphrag`.

**Why (2026-10-21).** The teacher judge showed Opus's training targets are clean (0.8%
unsupported on held-out chunks). The unsupported facts the distilled model writes come
from the student itself, some from pretraining knowledge of the vendor docs. SFT only
ever shows good outputs, so it cannot train that out. DPO shows the model its own reply
with the invented fact (`rejected`) next to the same reply without it (`chosen`).

**The dataset** comes from `scripts/distill/dpo_pairs.py`. ckpt100 served on vLLM
extracted 700 fresh articles, none of them in the distillation or trial samples
(`haiku_extract_ab.py --capture`). Each edge reply was judged twice, and a fact counts as
unsupported only when both passes agree. `chosen` differs from `rejected` only by those
edges, so preference moves away from the invented facts and not from extraction in
general.

**Guards against the known DPO failure modes:**
- *Losing coverage*: an NLL term on `chosen` (`NLL_ALPHA`: TRL's `rpo_alpha`, or its `"sft"` loss) keeps the
  likelihood of the good edges up. The end-to-end trial gates on distinct facts too.
- *Broken JSON*: guided decoding enforces the schema at serving time. This notebook also
  smoke-tests three held-out prompts before pushing.
- *Training a different completion than vLLM serves*: section 8 decodes the trainer's own
  tokenised row and checks it equals serving prompt + answer + `<|im_end|>`.
- *Overfitting* (v2 overfit after step 100): the best checkpoint by eval loss is kept
  (`load_best_model_at_end`), and checkpoints are saved at every eval so it is never
  rotated out.

**Runtime:** L4 or A100. Edge prompts run to several thousand tokens, and DPO runs each
pair's chosen, rejected and reference passes. It **never overwrites** a revision that
already holds weights.

**Use:** set the HF token secret (section 2), check section 3, then **Runtime -> Run all**.
A dead runtime resumes from the last checkpoint on Drive.
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
HF_DATASET_REVISION = "dpo1"
DRIVE_DATA_DIR = "/content/drive/MyDrive/ft-opus-dpo1"   # only when DATA_SOURCE == "drive"

# --- Google Drive: checkpoints, final adapter, metrics (survive a runtime death) ----------
RUN_NAME = "opus-distill-dpo3"     # dpo1: 5e-6, too weak; dpo2: 5e-5 on a broken kernel
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
NLL_ALPHA = 1.0            # NLL on chosen (rpo_alpha / "sft" loss): protects coverage; 0 = plain DPO
LEARNING_RATE = 5e-5       # dpo1's 5e-6 moved the split-point log-odds by only +0.34
NUM_TRAIN_EPOCHS = 2
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "cosine"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_STEPS = 10            # also the save interval, so the best checkpoint is on Drive
SAVE_TOTAL_LIMIT = 4
SEED = 20261021
GRAD_NORM_LIMIT = 1000.0   # training guard: stop on a larger norm from GRAD_NORM_FROM_STEP on
GRAD_NORM_FROM_STEP = 12   # dpo1: >100k for steps 1-10 (before the first eval), then <6

# --- export ----------------------------------------------------------------------------------
HF_MODEL_REVISION = "opus-distill-2026-10-dpo3"
MAX_REMOVED_SHARE = 0.75   # gate: DPO writes at most this share of ckpt100's removed edges
MIN_KEPT_SHARE = 0.90      # ... and at least this share of ckpt100's kept edges
MIN_BASELINE_REMOVED = 10  # ckpt100 must write this many removed edges, or nothing is measurable
HF_MODEL_PRIVATE = False   # the repo is public by design; a branch cannot be more private

OUT_DIR = Path("/content/outputs/opus-distill-dpo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("config set")
"""),
    md("## 3a. Mount Google Drive\n\nSame as `distill_train.ipynb` section 3a."),
    code(src_of(9)),
    md("## 4. Get the dataset"),
    code("""
DATA_DIR = Path("/content/data/ft-opus-dpo1")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if DATA_SOURCE == "drive":
    DATA_DIR = Path(DRIVE_DATA_DIR)
elif DATA_SOURCE == "hf_dataset":
    from huggingface_hub import hf_hub_download
    for fname in ("train.jsonl", "val.jsonl", "MANIFEST.json", "sample.json"):
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

assert (DATA_DIR / "train.jsonl").exists() and (DATA_DIR / "val.jsonl").exists(), DATA_DIR
print(f"dataset ready at {DATA_DIR}")
"""),
    md("""
## 5. Load the pairs and check their shape

Every record: `messages` = `[system, user]` exactly as vLLM received them, `chosen` and
`rejected` = the reply JSON. `chosen` must parse, hold fewer edges than `rejected`, and
every chosen edge must appear in `rejected` unchanged.
"""),
    code("""
import json
from collections import Counter


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


train_records = read_jsonl(DATA_DIR / "train.jsonl")
val_records = read_jsonl(DATA_DIR / "val.jsonl")
for _r in train_records + val_records:
    assert [m["role"] for m in _r["messages"]] == ["system", "user"], _r["meta"]
    _c, _x = json.loads(_r["chosen"])["edges"], json.loads(_r["rejected"])["edges"]
    assert 0 < len(_c) < len(_x), _r["meta"]
    assert all(e in _x for e in _c), _r["meta"]
_cats = Counter(c for r in train_records for f in r["meta"]["removed"] for c in f["categories"][:1])
print(f"train pairs {len(train_records)}, val pairs {len(val_records)}")
print(f"facts removed (train): {sum(len(r['meta']['removed']) for r in train_records)}; "
      f"categories {dict(_cats)}")
assert len(train_records) >= 50, "too few pairs for a DPO round -- capture more first"
"""),
    md("""
## 6. Download the starting model and attach LoRA

The merged bf16 weights of `START_REVISION` go to `START_DIR`. A fixed path matters: the
saved adapter records it as its base, and the export (section 10) reloads from there.
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
        n = len(TEXT_TOK(p["prompt"] + p["rejected"], add_special_tokens=False)["input_ids"])
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
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
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


def make_trainer(mark: str) -> DPOTrainer:
    # TRL 1.x copies the untrained adapter into a frozen "ref" adapter (= ckpt100, since
    # LoRA starts at zero) for the reference pass; a rebuild must not find one already.
    if "ref" in getattr(model, "peft_config", {}):
        model.delete_adapter("ref")
    print(f"building datasets (completion mark {mark!r})")
    return DPOTrainer(model=model, ref_model=None, args=dpo_config,
                      train_dataset=build(train_records, mark),
                      eval_dataset=build(val_records, mark), callbacks=[GradGuard()],
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
## 9. Train (resumable), then evaluate on the held-out pairs

`eval_rewards/accuracies` is the share of held-out pairs where the model now prefers the
reply without the invented fact more than ckpt100 did. Look for it well above 0.5, and
for `eval_logps/chosen` not collapsing (that is the coverage risk).

The gradient guard prints "grad guard: inspecting gradients..." once, then the gradient
norm of each of the first `GRAD_NORM_FROM_STEP` steps. dpo1 had >100k there until the
first eval; with `eval_on_start` they should be single digits. Either way the run
continues; it stops on a NaN/inf gradient, or a norm above `GRAD_NORM_LIMIT` later.
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
    eval_metrics = trainer.evaluate()   # the best checkpoint (load_best_model_at_end)
    print(_json.dumps(eval_metrics, indent=2))
    model.save_pretrained(str(FINAL_ADAPTER_DIR), selected_adapters=["default"])  # not TRL's "ref"
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    (DRIVE_RUN_DIR / "train_result.json").write_text(_json.dumps(train_result.metrics, indent=2))
    (DRIVE_RUN_DIR / "eval_metrics.json").write_text(_json.dumps(eval_metrics, indent=2))
    (DRIVE_RUN_DIR / "log_history.json").write_text(
        _json.dumps(trainer.state.log_history, indent=2))
    FINAL_MARKER.write_text("adapter + tokenizer fully saved\\n")
    print(f"best checkpoint: {trainer.state.best_model_checkpoint}")
    _acc = eval_metrics.get("eval_rewards/accuracies")
    if _acc is not None and _acc < 0.6:
        print(f"WARNING: held-out reward accuracy {_acc:.2f} -- DPO barely separated the pairs")
"""),
    md("""
## 10. Gate -- does the DPO model stop writing the removed edges?

Reloads the DPO model (ckpt100 + the final adapter) and ckpt100 itself. For each val pair,
both continue greedily from the prefix `chosen` and `rejected` share up to the split, and
finish the reply. Edges are counted by (source, target, relation):

- **removed edges written**: the unsupported edges the pair deleted (57 in dpo1's val set);
- **kept edges written**: the edges `chosen` keeps after the split (141);
- **valid JSON** replies.

**Pass:** the DPO model writes at most `MAX_REMOVED_SHARE` of ckpt100's removed edges, at
least `MIN_KEPT_SHARE` of its kept edges, and no fewer valid JSON replies; ckpt100 must
write at least `MIN_BASELINE_REMOVED` removed edges, or the set cannot show a reduction.
(dpo2 reference: ckpt100 wrote 27 removed / 59 kept, 39 JSON; ~30 min per model on an L4.)

**Why not the old gate.** It checked whether greedy picks `chosen`'s token at the split.
But there `chosen` starts the NEXT edge, so a model that stops writing the removed edge
and opens a different one scored zero, and a removed edge that came back later was never
seen. `starts_removed` below is that token-level view, for reference only.
"""),
    code("""
import gc
import re
import time

for _name in ("trainer", "model"):
    globals().pop(_name, None)
gc.collect()
torch.cuda.empty_cache()

assert FINAL_MARKER.exists(), f"no complete final adapter at {FINAL_ADAPTER_DIR}"
assert any(START_DIR.glob("*.safetensors")), "run section 6 first (the adapter's base)"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(FINAL_ADAPTER_DIR), max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT, dtype=None)
FastLanguageModel.for_inference(model)
base, base_tok = FastLanguageModel.from_pretrained(
    model_name=str(START_DIR), max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT, dtype=None)
FastLanguageModel.for_inference(base)

_TRIPLE = re.compile(r'"source_entity_name": "(.*?)", "target_entity_name": "(.*?)", "relation_type": "(.*?)"')


def _triples(text: str) -> tuple[set, bool]:
    try:
        return {(e["source_entity_name"], e["target_entity_name"], e["relation_type"])
                for e in json.loads(text)["edges"]}, True
    except (ValueError, KeyError, TypeError):
        return set(_TRIPLE.findall(text)), False


@torch.inference_mode()
def rest_from_split(m, r: dict) -> dict | None:
    p = to_pair(r, "")
    n_prompt = len(TEXT_TOK(p["prompt"], add_special_tokens=False)["input_ids"])
    ch = TEXT_TOK(p["prompt"] + p["chosen"], add_special_tokens=False)["input_ids"]
    rj = TEXT_TOK(p["prompt"] + p["rejected"], add_special_tokens=False)["input_ids"]
    split = next((i for i, (a, b) in enumerate(zip(ch, rj)) if a != b), None)
    if split is None:
        return None
    ids = torch.tensor([ch[:split]], device=m.device)
    out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                     max_new_tokens=min(len(rj) - split + 256, 4096))
    new = out[0][split:].tolist()
    prefix = TEXT_TOK.decode(ch[n_prompt:split], skip_special_tokens=True)
    gen, ok = _triples(prefix + TEXT_TOK.decode(new, skip_special_tokens=True))
    done = set(_TRIPLE.findall(prefix))            # edges already complete before the split
    removed = _triples(r["rejected"])[0] - _triples(r["chosen"])[0]
    kept = _triples(r["chosen"])[0] - done
    return {"removed": len(removed), "removed_written": len(removed & gen),
            "kept": len(kept), "kept_written": len(kept & gen), "json": ok,
            "starts_removed": bool(new) and new[0] == rj[split]}


_tot, _per_pair = {}, {}
for _label, _m in (("ckpt100", base), ("dpo", model)):
    _t0 = time.time()
    _per_pair[_label] = []
    for _i, _r in enumerate(val_records, 1):
        _per_pair[_label].append(rest_from_split(_m, _r))
        if _i % 5 == 0 or _i == len(val_records):
            print(f"  [{_label}] {_i}/{len(val_records)} pairs, {time.time() - _t0:.0f}s", flush=True)
    _rows = [x for x in _per_pair[_label] if x]
    _tot[_label] = {k: sum(x[k] for x in _rows) for k in _rows[0]}
    print(f"[{_label}] {len(_rows)} pairs, {time.time() - _t0:.0f}s: {_tot[_label]}")
(DRIVE_RUN_DIR / "gate_continuation.json").write_text(
    json.dumps({"totals": _tot, "per_pair": _per_pair}, indent=1))

_b, _d = _tot["ckpt100"], _tot["dpo"]
print(f"removed edges written: dpo {_d['removed_written']} vs ckpt100 {_b['removed_written']} "
      f"(of {_b['removed']}); kept: dpo {_d['kept_written']} vs {_b['kept_written']} "
      f"(of {_b['kept']}); JSON: dpo {_d['json']} vs {_b['json']}")
assert _b["removed_written"] >= MIN_BASELINE_REMOVED, (
    f"ckpt100 writes only {_b['removed_written']} of the removed edges greedily -- too few "
    "for this val set to measure a reduction; the gate cannot decide")
GATE_PASSED = (_d["removed_written"] <= MAX_REMOVED_SHARE * _b["removed_written"]
               and _d["kept_written"] >= MIN_KEPT_SHARE * _b["kept_written"]
               and _d["json"] >= _b["json"])
assert GATE_PASSED, "gate failed -- not exporting (results saved to gate_continuation.json on Drive)"
print("GATE PASSED")
"""),
    md("""
## 10a. Export -- merged bf16 safetensors to a NEW revision

Runs only after the gate passed. Refuses a revision that already has its own commits,
checks three short held-out prompts still come back as JSON (both models' edge counts side
by side: a large drop means DPO is cutting supported edges too), merges to bf16, uploads,
and checks the Hub's shards equal the merged index.
"""),
    code("""
import gc
import json as _json

from huggingface_hub import HfApi

assert globals().get("GATE_PASSED"), "run the gate first; export only after it passes"
api = HfApi(token=HF_TOKEN)
# A new branch is a copy of main, weights included, so "the branch holds weights" cannot
# be the test: refuse only a branch that has moved away from main (an earlier export).
_refs = {b.name: b.target_commit
         for b in api.list_repo_refs(HF_MODEL_REPO, repo_type="model").branches}
assert _refs.get(HF_MODEL_REVISION, _refs["main"]) == _refs["main"], (
    f"{HF_MODEL_REPO}@{HF_MODEL_REVISION} already has its own commits (an earlier export?)"
    " -- choose a new HF_MODEL_REVISION")
api.create_branch(repo_id=HF_MODEL_REPO, repo_type="model", branch=HF_MODEL_REVISION,
                  exist_ok=True)


def edges_of(m, t, prompt: str) -> int | None:
    ids = t(text=prompt, return_tensors="pt").to(m.device)
    out = m.generate(**ids, max_new_tokens=4096, do_sample=False)   # 1024 cut an 11-edge reply
    reply = t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return len(_json.loads(reply)["edges"])
    except (ValueError, KeyError, TypeError):
        print(f"  not JSON: {reply[:160]!r} ... {reply[-160:]!r}")
        return None


for _r in sorted(val_records, key=lambda r: len(r["messages"][1]["content"]))[:3]:
    _prompt = to_pair(_r, "")["prompt"]
    _n_dpo, _n_base = edges_of(model, tokenizer, _prompt), edges_of(base, base_tok, _prompt)
    print(f"edges: dpo {_n_dpo}, ckpt100 {_n_base}, chosen {len(_json.loads(_r['chosen'])['edges'])}")
    assert _n_dpo is not None, "the DPO model did not answer in JSON -- do not upload this export"
print("smoke test passed")
del base, base_tok
gc.collect()
torch.cuda.empty_cache()

merged_dir = OUT_DIR / "merged-safetensors"
model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
_index = json.loads((merged_dir / "model.safetensors.index.json").read_text())
_missing = sorted(set(_index["weight_map"].values())
                  - {p.name for p in merged_dir.glob("*.safetensors")})
assert not _missing, f"merged save is missing shards: {_missing}"
api.upload_folder(repo_id=HF_MODEL_REPO, repo_type="model", folder_path=str(merged_dir),
                  revision=HF_MODEL_REVISION,
                  delete_patterns=["*.safetensors", "*.gguf"],   # no shard of main's survives
                  commit_message=f"{RUN_NAME}: DPO on {START_REVISION}, merged bf16")
_pushed = {f for f in api.list_repo_files(HF_MODEL_REPO, repo_type="model",
                                          revision=HF_MODEL_REVISION)
           if f.endswith((".safetensors", ".gguf"))}
assert _pushed == set(_index["weight_map"].values()), f"Hub shards != merged index: {_pushed}"
print(f"DONE: {HF_MODEL_REPO}@{HF_MODEL_REVISION}")
"""),
    md("""
## 11. Next -- outside this notebook

Hand the revision name back. It is validated like ckpt100, on the SAME 60 trial articles
(none of them in the DPO data): held-out eval (`scripts/eval_finetune.py --guided`), the
45-chunk faithfulness check, then `scripts/haiku_extract_ab.py --arms v2 --reuse
<adoption-trial rows>` against the adoption trial's routed rows, judged twice.
"""),
]

nb = {"cells": cells, "metadata": {**train_nb["metadata"],
                                   "colab": {"name": "distill_dpo.ipynb", "provenance": []}},
      "nbformat": train_nb["nbformat"], "nbformat_minor": train_nb["nbformat_minor"]}
out = REPO / "notebooks/distill_dpo.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print(f"wrote {out} ({len(cells)} cells); all code cells compile", file=sys.stderr)
