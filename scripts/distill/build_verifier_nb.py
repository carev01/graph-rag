"""Builds notebooks/distill_verifier.ipynb. Cells shared with distill_train.ipynb (install,
HF token, Drive mount, serving check) and distill_dpo.ipynb (causal_conv1d off, compile
off) are copied verbatim, so the notebooks never drift apart."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELECT_SRC = (REPO / "scripts/distill/verifier_select.py").read_text()
train_nb = json.loads((REPO / "notebooks/distill_train.ipynb").read_text())
dpo_nb = json.loads((REPO / "notebooks/distill_dpo.ipynb").read_text())


def src_of(i: int) -> str:
    return "".join(train_nb["cells"][i]["source"])


def dpo_cell(marker: str) -> str:
    hits = ["".join(c["source"]) for c in dpo_nb["cells"]
            if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


assert "pip_install" in src_of(3) and "get_hf_token" in src_of(5), "training notebook moved"
assert "drive.mount" in src_of(9) and "CKPT_DIR" in src_of(9), "training notebook moved"
assert "def build_training_text" in src_of(17), "training notebook moved"
assert "training text does not equal serving prompt" in src_of(19), "training notebook moved"

cells = [
    md("""
# Fact verifier verifier1 -- qwen3.5-4b (Colab)

Trains a LoRA adapter on `Qwen/Qwen3.5-4B` that labels every fact ckpt100 extracted from a
chunk: `stated` (with a quote from the chunk), `unsupported` (with a category) or
`previous_only`. Training data: the dpo4 Opus labels (`scripts/distill/verifier_data.py`).
Spec: `docs/superpowers/specs/2026-10-27-fact-verifier-design.md`.

**Selection:** the checkpoint with the lowest dev loss. Training stops by itself once the
last `DEV_PATIENCE` evals all sit above the best before them.

**Guards from the DPO notebook:** causal_conv1d kernel off for training, torch.compile off,
the gradient guard, the serving-prompt check and the loss-mask check.

**Use:** set the HF token secret (section 2), then **Runtime -> Run all**. A dead runtime
resumes from the last checkpoint on Drive, and the predictions resume where they stopped.
Hand back the adapter revision section 10 prints; the test predictions are uploaded with it
and scored locally (`scripts/distill/verifier_eval.py`).
"""),
    md("## 1. Install packages\n\nSame install as `distill_train.ipynb` section 1."),
    code(src_of(3)),
    md("## 1a. causal_conv1d kernel OFF for training, torch.compile OFF\n\n"
       "Verbatim from `distill_dpo.ipynb` section 1a (dpo2: the hub kernel's backward gave NaN "
       "gradients; dpo3: torch.compile hung for 2 h in step 1)."),
    code(dpo_cell("USE_HUB_CAUSAL_CONV1D = False")),
    md(src_of(4)),
    code(src_of(5)),
    md("## 3. Configuration"),
    code("""
from pathlib import Path

# --- dataset: verifier_data.py output (train / val = dev / test) ----------------------------
DATA_SOURCE = "hf_dataset"
HF_DATASET_REPO = "carev01/ft-opus-distill"   # private
HF_DATASET_REVISION = "verifier1"
DRIVE_DATA_DIR = "/content/drive/MyDrive/ft-opus-verifier1"   # only when DATA_SOURCE == "drive"

# --- Google Drive: checkpoints, final adapter, predictions (survive a runtime death) ---------
RUN_NAME = "opus-verifier1"
DRIVE_RUN_DIR = Path("/content/drive/MyDrive/graph-rag-distill") / RUN_NAME
CKPT_DIR = DRIVE_RUN_DIR / "checkpoints"
FINAL_ADAPTER_DIR = DRIVE_RUN_DIR / "final-adapter"
FINAL_MARKER = FINAL_ADAPTER_DIR / "COMPLETE"   # written LAST: adapter fully saved

# --- model -------------------------------------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"   # the plain model, not ckpt100: no shared extraction habits
MAX_SEQ_LENGTH = 20480
LOAD_IN_4BIT = True

# --- LoRA (same shape as the SFT run) ----------------------------------------------------
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# --- training (SFT settings of distill_train.ipynb) --------------------------------------
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 2
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16   # 32 on an L4 if 16 runs out of memory
EVAL_STEPS = 25            # also the save interval, so the selected checkpoint is on Drive
SAVE_TOTAL_LIMIT = None    # SFT v2's best checkpoint was rotated out; adapters are small
DEV_PATIENCE = 3           # stop once the last 3 evals all sit above the best before them
SEED = 20261027
GRAD_NORM_LIMIT = 1000.0   # training guard: stop on a larger norm from GRAD_NORM_FROM_STEP on
GRAD_NORM_FROM_STEP = 12

# --- predictions on test, and the hand-off ---------------------------------------------------
PRED_BATCH = 8
HF_ADAPTER_REPO = "carev01/qwen35-4b-graphrag-verifier"   # private

OUT_DIR = Path("/content/outputs/opus-verifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("config set")
"""),
    md("## 3a. Mount Google Drive\n\nSame as `distill_train.ipynb` section 3a."),
    code(src_of(9)),
    md("## 4. Get the dataset"),
    code("""
DATA_DIR = Path("/content/data/ft-opus-verifier1")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if DATA_SOURCE == "drive":
    DATA_DIR = Path(DRIVE_DATA_DIR)
elif DATA_SOURCE == "hf_dataset":
    from huggingface_hub import hf_hub_download
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl", "MANIFEST.json"):
        p = hf_hub_download(repo_id=HF_DATASET_REPO, repo_type="dataset",
                            filename=fname, revision=HF_DATASET_REVISION, token=HF_TOKEN)
        (DATA_DIR / fname).write_bytes(Path(p).read_bytes())
else:
    raise ValueError(f"unknown DATA_SOURCE {DATA_SOURCE!r}")

assert all((DATA_DIR / f).exists() for f in ("train.jsonl", "val.jsonl", "test.jsonl")), DATA_DIR
print(f"dataset ready at {DATA_DIR}")
"""),
    md("## 5. Load the splits and check their shape (fail loudly)"),
    code("""
import json
from collections import Counter


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


train_records = read_jsonl(DATA_DIR / "train.jsonl")
val_records = read_jsonl(DATA_DIR / "val.jsonl")      # the dev split
test_records = read_jsonl(DATA_DIR / "test.jsonl")
for _name, _rows in (("train", train_records), ("dev", val_records), ("test", test_records)):
    for _r in _rows:
        assert [m["role"] for m in _r["messages"]] == ["system", "user", "assistant"], _r["meta"]
        _labels = json.loads(_r["messages"][2]["content"])["labels"]
        assert len(_labels) == len(_r["meta"]["final"]), _r["meta"]["call_id"]
    _finals = Counter(f for r in _rows for f in r["meta"]["final"])
    print(f"{_name}: {len(_rows)} replies, facts {dict(sorted(_finals.items()))}")
_groups = {s: {r["meta"]["group"] for r in rows}
           for s, rows in (("train", train_records), ("dev", val_records), ("test", test_records))}
assert not (_groups["train"] & _groups["test"]) and not (_groups["dev"] & _groups["test"]), (
    "an article is on both sides of the test split")
"""),
    md("## 6. Load the base model and attach LoRA"),
    code("""
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=None,
    token=HF_TOKEN,
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
print("base model loaded, LoRA attached")
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


_uses_cc1d = calls_package(_mq.causal_conv1d_fn, "causal_conv1d")
print(f"model calls a causal_conv1d package: {_uses_cc1d}")
assert USE_HUB_CAUSAL_CONV1D or not _uses_cc1d, (
    "the hub causal_conv1d kernel is off, yet the model calls a causal_conv1d package -- "
    "its backward pass broke dpo2; restart the session without it")
"""),
    md("## 7. Build the chat-formatted text (thinking-off serving prompt + answer)"),
    code(src_of(17).replace(
        'train_ds = Dataset.from_list(train_records).map(to_text, remove_columns=["messages"])\n'
        'val_ds = Dataset.from_list(val_records).map(to_text, remove_columns=["messages"])',
        'train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_records]).map(\n'
        '    to_text, remove_columns=["messages"])\n'
        'val_ds = Dataset.from_list([{"messages": r["messages"]} for r in val_records]).map(\n'
        '    to_text, remove_columns=["messages"])')),
    md("## 8. Verify: training text == serving prompt + answer + end-of-turn (fail loudly)"),
    code(src_of(19)),
    md("""
## 9. Train -- completion-only loss, resumable

The gradient guard prints the first `GRAD_NORM_FROM_STEP` gradient norms, stops on a NaN or
inf gradient before the optimiser step, and on a norm above `GRAD_NORM_LIMIT` later.
`DevLossStop` ends training once the last `DEV_PATIENCE` evals all sit above the best one
before them.
"""),
    code("""
import dataclasses
import inspect

from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

""" + SELECT_SRC + """

class GradGuard(TrainerCallback):
    # Stop on the first non-finite gradient, BEFORE the optimiser step (dpo2 skipped 40 of
    # 72 steps on NaN without a word).
    checked = 0

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
            raise RuntimeError("grad guard never ran before an optimiser step -- the NaN check "
                               "is not live")
        if state.global_step <= GRAD_NORM_FROM_STEP:
            print(f"  step {state.global_step}: grad_norm {g:.4g}", flush=True)
        elif g > GRAD_NORM_LIMIT:
            raise RuntimeError(f"step {state.global_step}: grad_norm {g:.4g} > "
                               f"{GRAD_NORM_LIMIT} -- stopping")


class DevLossStop(TrainerCallback):
    # Stateless (reads state.log_history), so a resumed run keeps its history.
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        losses = [v for _, v in dev_losses(state.log_history)]
        if should_stop(losses, DEV_PATIENCE):
            print(f"  step {state.global_step}: dev loss above its best for {DEV_PATIENCE} "
                  "evals -- stopping", flush=True)
            control.should_training_stop = True
        return control


_probe_prefix = tokenizer.apply_chat_template(
    [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
    enable_thinking=False)
RESPONSE_PART = _probe_prefix.rsplit(END_OF_TURN, 1)[-1]
INSTRUCTION_PART = "<|im_start|>user\\n"
_fields = {f.name for f in dataclasses.fields(SFTConfig)}
_extra = {"max_seq_length" if "max_seq_length" in _fields else "max_length": MAX_SEQ_LENGTH,
          "warmup_ratio" if "warmup_ratio" in _fields else "warmup_steps": WARMUP_RATIO}
print(f"SFTConfig extras: {_extra}")

sft_config = SFTConfig(
    output_dir=str(CKPT_DIR),
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    optim="adamw_8bit",
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=False,   # selection is section 10's, from log_history
    dataset_text_field="text",
    packing=False,                  # train_on_responses_only masks per example
    seed=SEED,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    **_extra,
)
_tok_kw = ("processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
           else "tokenizer")
trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=sft_config,
                     callbacks=[GradGuard(), DevLossStop()], **{_tok_kw: tokenizer})
trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART,
                                  response_part=RESPONSE_PART)

# --- mask check: the loss covers exactly answer + "<|im_end|>" ---------------------
_labels = list(trainer.train_dataset[0]["labels"])
_unmasked = [t for t in _labels if t != -100]
_decoded = tokenizer.decode(_unmasked, skip_special_tokens=False)
_expected = train_records[0]["messages"][-1]["content"] + "<|im_end|>"
# Current TRL appends the tokenizer's EOS (<|im_end|>) to text that already ends with the
# template's "<|im_end|>\\n": harmless, generation stops at the first <|im_end|>.
assert _decoded in (_expected, _expected + "\\n", _expected + "\\n<|im_end|>"), (
    f"train_on_responses_only left the wrong tokens unmasked: {_decoded[:200]!r}")
if _decoded.endswith("\\n<|im_end|>"):
    print("note: TRL appends a second <|im_end|> after the template's end of turn (harmless)")
print(f"OK: unmasked labels == answer + '<|im_end|>' ({len(_unmasked)} of {len(_labels)} tokens)")
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
    print(f"final adapter already on Drive ({FINAL_ADAPTER_DIR}) -- skipping to the selection "
          "(section 10). Delete that folder to train again.")
else:
    _last = last_complete_checkpoint(CKPT_DIR)
    print(f"resuming from {_last}" if _last else "no checkpoint on Drive -- starting fresh")
    train_result = trainer.train(resume_from_checkpoint=_last)
    model.save_pretrained(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    (DRIVE_RUN_DIR / "train_result.json").write_text(_json.dumps(train_result.metrics, indent=2))
    (DRIVE_RUN_DIR / "log_history.json").write_text(
        _json.dumps(trainer.state.log_history, indent=2))
    FINAL_MARKER.write_text("adapter + tokenizer fully saved\\n")
"""),
    md("""
## 10. Select the checkpoint, predict the test split, hand off

Picks the lowest dev loss from `log_history.json` and loads that checkpoint into a FRESH
model (the training model still holds the final-step adapter). Before the test split it
checks itself on dev records: replies must parse, and batched generation must give the
same verdicts as batch 1 (left padding on this hybrid-attention model), else predictions
run at batch 1. Then it writes greedy, thinking-off replies for every test record to Drive
(resumable: done call ids are skipped, a half-written last line is dropped) and uploads
the adapter, the log and the predictions to the PRIVATE repo `HF_ADAPTER_REPO`, revision
`verifier1-ckpt<step>`.
"""),
    code("""
import gc
import json as _json
import os

from huggingface_hub import HfApi

_hist = _json.loads((DRIVE_RUN_DIR / "log_history.json").read_text())
for _s, _v in dev_losses(_hist):
    print(f"eval @ {_s:>4}: dev loss {_v:.4f}")
SELECTED_STEP = select_dev_step(_hist)
assert SELECTED_STEP is not None, "no eval in the log history -- nothing to select"
_ckpt = CKPT_DIR / f"checkpoint-{SELECTED_STEP}"
assert (_ckpt / "adapter_model.safetensors").exists(), f"{_ckpt} has no adapter"
ADAPTER_REVISION = f"verifier1-ckpt{SELECTED_STEP}"

# A fresh model with the selected checkpoint's adapter (distill_train's reload pattern).
del trainer
del model
gc.collect()
torch.cuda.empty_cache()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(_ckpt), max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=LOAD_IN_4BIT,
    dtype=None, token=HF_TOKEN)
FastLanguageModel.for_inference(model)
TEXT_TOK = getattr(tokenizer, "tokenizer", tokenizer)
TEXT_TOK.padding_side = "left"
if TEXT_TOK.pad_token is None:
    TEXT_TOK.pad_token = TEXT_TOK.eos_token
END_ID = TEXT_TOK.convert_tokens_to_ids("<|im_end|>")
print(f"loaded {_ckpt.name} into a fresh model")


def serving_prompt(r: dict) -> str:
    return tokenizer.apply_chat_template(r["messages"][:-1], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def generate(batch: list[dict]) -> list[str]:
    enc = TEXT_TOK([serving_prompt(r) for r in batch], return_tensors="pt", padding=True,
                   add_special_tokens=False).to("cuda")
    cap = 64 + 160 * max(len(r["meta"]["final"]) for r in batch)   # facts, not answers
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cap, do_sample=False,
                             eos_token_id=END_ID, pad_token_id=TEXT_TOK.pad_token_id)
    return [TEXT_TOK.decode(ids, skip_special_tokens=True)
            for ids in out[:, enc["input_ids"].shape[1]:]]


def verdicts(text: str, n: int):
    try:
        labels = _json.loads(text)["labels"]
        by_n = {lab["n"]: lab["verdict"] for lab in labels}
    except (ValueError, KeyError, TypeError):
        return None
    return tuple(by_n.get(k) for k in range(1, n + 1)) if len(by_n) == n == len(labels) else None


# --- self-check on dev: parse rate, agreement with the targets, batched == batch 1 ---------
_probe = sorted(val_records, key=lambda r: len(serving_prompt(r)))[:8]
_single = [generate([r])[0] for r in _probe]
_batched = [t for i in range(0, len(_probe), PRED_BATCH) for t in generate(_probe[i:i + PRED_BATCH])]
_n_facts = [len(r["meta"]["final"]) for r in _probe]
_v1 = [verdicts(t, n) for t, n in zip(_single, _n_facts)]
_vb = [verdicts(t, n) for t, n in zip(_batched, _n_facts)]
_want = [verdicts(r["messages"][2]["content"], n) for r, n in zip(_probe, _n_facts)]
_parsed = sum(v is not None for v in _v1)
_match = sum(a == b for v, w in zip(_v1, _want) if v is not None for a, b in zip(v, w))
_same = sum((a == b) if a is not None else (s == t)
            for a, b, s, t in zip(_v1, _vb, _single, _batched))
print(f"dev agreement: {_parsed}/{len(_probe)} replies parse at batch 1; verdicts match the "
      f"targets on {_match}/{sum(n for v, n in zip(_v1, _n_facts) if v is not None)} facts; "
      f"batched == batch 1 on {_same}/{len(_probe)}")
assert _parsed > 0, "no dev reply parses -- adapter or prompt is wrong; not predicting the test split"
if _same < len(_probe):
    print(f"WARNING: batch {PRED_BATCH} changes the verdicts -- predicting at batch 1")
    PRED_BATCH = 1


def drop_partial_tail(path: Path) -> None:
    # A runtime that died mid-write leaves a half line; appending to it would corrupt the file.
    if not path.exists():
        return
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    good = []
    for x in lines:
        try:
            row = _json.loads(x)
        except ValueError:
            continue
        if isinstance(row, dict) and "call_id" in row and "reply" in row:
            good.append(x)
    if len(good) != len(lines):
        print(f"dropping {len(lines) - len(good)} unusable line(s) from {path.name}")
        path.write_text("".join(g + "\\n" for g in good), encoding="utf-8")


PRED_PATH = DRIVE_RUN_DIR / f"test_predictions-ckpt{SELECTED_STEP}.jsonl"
drop_partial_tail(PRED_PATH)
_done = set()
if PRED_PATH.exists():
    _done = {_json.loads(x)["call_id"] for x in PRED_PATH.read_text().splitlines() if x.strip()}
_todo = sorted((r for r in test_records if r["meta"]["call_id"] not in _done),
               key=lambda r: len(serving_prompt(r)))
print(f"{len(_done)} predictions already on Drive, {len(_todo)} to go (batch {PRED_BATCH})")
# One open/append/close per batch: Colab's Drive mount uploads on close, so a file held
# open for the whole loop lost ~650 written rows when a runtime died (2026-10-27).
for _i in range(0, len(_todo), PRED_BATCH):
    _batch = _todo[_i:_i + PRED_BATCH]
    _rows = [_json.dumps({"call_id": _r["meta"]["call_id"], "reply": _text,
                          "step": SELECTED_STEP}, ensure_ascii=False) + "\\n"
             for _r, _text in zip(_batch, generate(_batch))]
    with PRED_PATH.open("a", encoding="utf-8") as _fh:
        _fh.write("".join(_rows))
        _fh.flush()
        os.fsync(_fh.fileno())
    print(f"  {min(_i + PRED_BATCH, len(_todo))}/{len(_todo)}", flush=True)
_n = len({_json.loads(x)["call_id"] for x in PRED_PATH.read_text().splitlines() if x.strip()})
assert _n == len(test_records), f"{_n} predictions for {len(test_records)} test records"

api = HfApi(token=HF_TOKEN)
api.create_repo(HF_ADAPTER_REPO, repo_type="model", private=True, exist_ok=True)
assert api.model_info(HF_ADAPTER_REPO).private, f"{HF_ADAPTER_REPO} is not private -- refusing"
api.create_branch(HF_ADAPTER_REPO, repo_type="model", branch=ADAPTER_REVISION, exist_ok=True)
for _src, _dst in ((_ckpt / "adapter_model.safetensors", "adapter_model.safetensors"),
                   (_ckpt / "adapter_config.json", "adapter_config.json"),
                   (DRIVE_RUN_DIR / "log_history.json", "log_history.json"),
                   (PRED_PATH, "test_predictions.jsonl")):
    api.upload_file(path_or_fileobj=str(_src), path_in_repo=_dst, repo_id=HF_ADAPTER_REPO,
                    repo_type="model", revision=ADAPTER_REVISION)
print(f"DONE: {HF_ADAPTER_REPO}@{ADAPTER_REVISION} -- hand this name back")
"""),
    md("""
## 11. Next -- outside this notebook

Hand back the adapter revision. Scoring is local and free:
`uv run --extra dev python -m scripts.distill.verifier_eval --data <dataset dir>
--predictions <test_predictions.jsonl> --out <report.json>` applies the pre-registered bar
(spec section 6). Only on a pass does the production hook get its own spec.
"""),
]

assert any('[{"messages": r["messages"]} for r in train_records]' in "".join(c["source"])
           for c in cells), "section 7's dataset line was not rewritten -- distill_train.ipynb moved"

nb = {"cells": cells, "metadata": {**train_nb["metadata"],
                                   "colab": {"name": "distill_verifier.ipynb", "provenance": []}},
      "nbformat": train_nb["nbformat"], "nbformat_minor": train_nb["nbformat_minor"]}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "notebooks/distill_verifier.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print(f"wrote {out} ({len(cells)} cells); all code cells compile", file=sys.stderr)
