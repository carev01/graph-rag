"""build_verifier_nb.py: the verifier notebook carries the approved settings, the guards
from the DPO notebook, the embedded selection module, and the prediction hand-off (spec
2026-10-27-fact-verifier-design.md section 4)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.distill import verifier_select as vs

REPO = Path(__file__).resolve().parents[2]


def _cells(tmp_path: Path) -> list[str]:
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(REPO / "scripts/distill/build_verifier_nb.py"), str(out)],
                   check=True, cwd=REPO)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"]]


def test_settings_guards_and_handoff(tmp_path: Path) -> None:
    text = "\n".join(_cells(tmp_path))
    for needle in ('BASE_MODEL_ID = "Qwen/Qwen3.5-4B"', 'HF_DATASET_REVISION = "verifier1"',
                   "LEARNING_RATE = 2e-4", "NUM_TRAIN_EPOCHS = 2", "WARMUP_RATIO = 0.03",
                   "EVAL_STEPS = 25", "SAVE_TOTAL_LIMIT = None", "DEV_PATIENCE = 3",
                   "MAX_SEQ_LENGTH = 20480", "USE_HUB_CAUSAL_CONV1D = False",
                   "torch._dynamo.config.disable = True", "class GradGuard", "class DevLossStop",
                   "load_best_model_at_end=False", '"test.jsonl"', "train_on_responses_only",
                   'HF_ADAPTER_REPO = "carev01/qwen35-4b-graphrag-verifier"',
                   'ADAPTER_REVISION = f"verifier1-ckpt{SELECTED_STEP}"',
                   "test_predictions-ckpt", "enable_thinking=False", 'padding_side = "left"'):
        assert needle in text, needle
    assert "save_pretrained_merged" not in text and "push_to_hub_gguf" not in text


def test_shared_cells_are_verbatim(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    train = ["".join(c["source"]) for c in json.loads(
        (REPO / "notebooks/distill_train.ipynb").read_text())["cells"]]
    dpo = ["".join(c["source"]) for c in json.loads(
        (REPO / "notebooks/distill_dpo.ipynb").read_text())["cells"]]
    for src in (train[3], train[5], train[9], train[19]):
        assert src in cells
    assert next(c for c in dpo if "USE_HUB_CAUSAL_CONV1D = False" in c) in cells


def test_embedded_selection_is_the_module(tmp_path: Path) -> None:
    src = (REPO / "scripts/distill/verifier_select.py").read_text()
    assert any(src.strip() in c for c in _cells(tmp_path))


def test_selection_rules() -> None:
    hist = [{"step": 25, "eval_loss": 0.30}, {"step": 25, "eval_loss": 0.30},
            {"step": 50, "eval_loss": 0.20}, {"step": 75, "eval_loss": 0.22},
            {"step": 10, "loss": 1.0}]
    assert vs.dev_losses(hist) == [(25, 0.30), (50, 0.20), (75, 0.22)]
    assert vs.select_dev_step(hist) == 50
    assert vs.select_dev_step([]) is None
    assert not vs.should_stop([0.3, 0.2, 0.21, 0.22], 3)
    assert vs.should_stop([0.3, 0.2, 0.21, 0.22, 0.25], 3)
    assert not vs.should_stop([0.3, 0.2, 0.21, 0.19, 0.25], 3)


def test_prediction_reloads_the_selected_checkpoint_and_checks_batching(tmp_path: Path) -> None:
    cell = next(c for c in _cells(tmp_path) if "SELECTED_STEP = select_dev_step" in c)
    assert "FastLanguageModel.from_pretrained(" in cell and "model_name=str(_ckpt)" in cell
    assert "load_adapter" not in cell and "set_adapter" not in cell
    assert "del trainer" in cell
    assert "PRED_BATCH = 1" in cell                 # fallback when batched != batch-1 on dev
    assert "dev agreement" in cell
    assert "drop_partial_tail(PRED_PATH)" in cell   # a half-written line from a dead runtime
