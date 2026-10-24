"""build_dpo_nb.py: the dpo4 notebook carries the approved settings, embeds dpo_select
verbatim, and no longer gates or exports from Colab (spec 2026-10-24 section 5)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _cells(tmp_path: Path) -> list[str]:
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(REPO / "scripts/distill/build_dpo_nb.py"), str(out)],
                   check=True, cwd=REPO)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"]]


def test_dpo4_settings_and_no_colab_gate(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    text = "\n".join(cells)
    for needle in ('HF_DATASET_REVISION = "dpo4"', "LEARNING_RATE = 2e-5", "NLL_ALPHA = 2.0",
                   "SAVE_TOTAL_LIMIT = None", "load_best_model_at_end=False",
                   "LOGPS_TOLERANCE = 1.0",
                   'HF_ADAPTER_REPO = "carev01/qwen35-4b-graphrag-dpo-adapters"',
                   '"val_eval.jsonl"', "class LikelihoodStop", "eval_dataset=build(val_eval_records",
                   'ADAPTER_REVISION = f"dpo4-ckpt{SELECTED_STEP}"'):
        assert needle in text, needle
    for gone in ("def rest_from_split", "## 10a. Export", "save_pretrained_merged",
                 'metric_for_best_model="eval_loss"', "trainer.evaluate()"):
        assert gone not in text, gone


def test_embedded_selection_is_the_module_and_runs(tmp_path: Path) -> None:
    src = (REPO / "scripts/distill/dpo_select.py").read_text()
    cell = next(c for c in _cells(tmp_path) if "def select_step" in c)
    assert src.strip() in cell
    ns: dict = {}
    exec(src, ns)
    hist = [{"step": 0, "eval_logps/chosen": -45.0}, {"step": 10, "eval_logps/chosen": -45.5}]
    assert ns["select_step"](hist, 1.0) == 10


def test_length_balance_is_asserted_for_train_and_val_not_val_eval(tmp_path: Path) -> None:
    cell = next(c for c in _cells(tmp_path) if "def read_jsonl" in c)
    assert 'for _name, _rows in (("train", train_records), ("val", val_records)):' in cell
    assert "set is not length-balanced (spec: |mean delta| <= 0.1)" in cell
    assert cell.count("length-balanced") == 1       # val_eval is print-only
    assert '("val_eval", val_eval_records)' in cell


def test_no_export_wording_left_in_the_resume_message(tmp_path: Path) -> None:
    text = "\n".join(_cells(tmp_path))
    assert "skipping to export" not in text
    assert "skipping to the selection (section 10)" in text
