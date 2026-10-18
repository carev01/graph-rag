"""`package_dataset.py`: zips train/val/sample into one archive with a manifest of
per-prompt counts, token estimates and the loss-weight mix, and pushes to a private
HF dataset repo through a dependency-injected, fake `HfApi` -- no network, and
`huggingface_hub` need not even be installed for any of this to run.

Also: `notebooks/distill_train.ipynb` is valid JSON (and valid nbformat, when the
optional `nbformat` package is importable).
"""
from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import pytest

from scripts.distill import package_dataset as pkg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _record(prompt_name: str, *, loss_weight: float, article_id: str = "art-1",
            prompt_chars: int = 40, target_chars: int = 8) -> dict:
    return {
        "prompt_name": prompt_name,
        "response_schema": "Whatever",
        "messages": [
            {"role": "system", "content": "x" * prompt_chars},
            {"role": "assistant", "content": "y" * target_chars},
        ],
        "meta": {"loss_weight": loss_weight, "article_id": article_id,
                 "provenance": "graph_derived", "fidelity": []},
    }


def _tiny_dataset(tmp_path: Path, *, with_sample: bool = True) -> Path:
    in_dir = tmp_path / "ft-tiny"
    in_dir.mkdir()
    train = [
        _record("extract_edges.edge", loss_weight=1.5),
        _record("extract_edges.edge", loss_weight=1.5),
        _record("dedupe_nodes.nodes", loss_weight=0.4),
    ]
    val = [_record("extract_edges.edge", loss_weight=1.5, article_id="art-2")]
    _write_jsonl(in_dir / "train.jsonl", train)
    _write_jsonl(in_dir / "val.jsonl", val)
    if with_sample:
        (in_dir / "sample.json").write_text(
            json.dumps({"articles": [{"article_id": "art-1"}, {"article_id": "art-2"}]}),
            encoding="utf-8")
    return in_dir


# --------------------------------------------------------------------------- #
# package(): the zip and its manifest
# --------------------------------------------------------------------------- #

def test_package_writes_zip_with_manifest_counts(tmp_path):
    in_dir = _tiny_dataset(tmp_path)
    out_zip = tmp_path / "dist" / "ft-tiny.zip"

    manifest = pkg.package(in_dir, out_zip)

    assert out_zip.exists()
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
    assert names == {"train.jsonl", "val.jsonl", "sample.json", "MANIFEST.json"}

    assert manifest["counts"]["train"] == {
        "extract_edges.edge": 2, "dedupe_nodes.nodes": 1, "_total": 3}
    assert manifest["counts"]["val"] == {"extract_edges.edge": 1, "_total": 1}
    assert manifest["loss_weight_mix"] == {
        "extract_edges.edge": 1.5, "dedupe_nodes.nodes": 0.4}
    assert manifest["sample_included"] is True
    assert manifest["tokens"]["train"]["_total"] > 0
    assert set(manifest["tokens"]["train"]) == {
        "extract_edges.edge", "dedupe_nodes.nodes", "_total"}
    assert isinstance(manifest["git_commit"], str) and manifest["git_commit"]

    # The manifest is also written to disk in the dataset directory, not just zipped.
    on_disk = json.loads((in_dir / "MANIFEST.json").read_text())
    assert on_disk == manifest


def test_package_token_estimate_matches_chars_over_four(tmp_path):
    in_dir = tmp_path / "ft-tok"
    in_dir.mkdir()
    rec = _record("extract_edges.edge", loss_weight=1.0, prompt_chars=40, target_chars=8)
    _write_jsonl(in_dir / "train.jsonl", [rec])
    _write_jsonl(in_dir / "val.jsonl", [])

    manifest = pkg.package(in_dir, tmp_path / "out.zip")

    # 40 // 4 + 8 // 4 = 10 + 2 = 12
    assert manifest["tokens"]["train"]["extract_edges.edge"] == 12
    assert manifest["tokens"]["train"]["_total"] == 12


def test_package_missing_sample_warns_but_succeeds(tmp_path, caplog):
    in_dir = _tiny_dataset(tmp_path, with_sample=False)

    with caplog.at_level(logging.WARNING):
        manifest = pkg.package(in_dir, tmp_path / "out.zip")

    assert manifest["sample_included"] is False
    with zipfile.ZipFile(tmp_path / "out.zip") as zf:
        assert "sample.json" not in zf.namelist()
    assert any("sample.json" in r.message for r in caplog.records)


def test_package_missing_train_or_val_raises(tmp_path):
    in_dir = tmp_path / "ft-empty"
    in_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        pkg.package(in_dir, tmp_path / "out.zip")


def test_main_cli_packages_without_hf_repo(tmp_path):
    in_dir = _tiny_dataset(tmp_path)
    out_zip = tmp_path / "out.zip"

    rc = pkg.main(["--in", str(in_dir), "--out", str(out_zip)])

    assert rc == 0
    assert out_zip.exists()


# --------------------------------------------------------------------------- #
# push_to_hub(): fake HfApi, no network, token never logged
# --------------------------------------------------------------------------- #

class FakeHfApi:
    def __init__(self):
        self.created: list[dict] = []
        self.uploaded: list[dict] = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_file(self, **kwargs):
        self.uploaded.append(kwargs)


def test_push_to_hub_uses_injected_fake_api(tmp_path, monkeypatch):
    in_dir = _tiny_dataset(tmp_path)
    pkg.package(in_dir, tmp_path / "out.zip")  # writes MANIFEST.json into in_dir
    monkeypatch.setenv("HF_TOKEN", "secret-token-value")
    fake = FakeHfApi()

    pkg.push_to_hub(in_dir, "carev01/opus-distill-test", True, hf_api=fake)

    assert fake.created == [
        {"repo_id": "carev01/opus-distill-test", "repo_type": "dataset",
         "private": True, "exist_ok": True}]
    uploaded_names = {u["path_in_repo"] for u in fake.uploaded}
    assert uploaded_names == {"train.jsonl", "val.jsonl", "sample.json", "MANIFEST.json"}
    for u in fake.uploaded:
        assert u["repo_id"] == "carev01/opus-distill-test"
        assert u["repo_type"] == "dataset"


def test_push_to_hub_without_token_raises_and_touches_no_api(tmp_path, monkeypatch):
    for var in pkg.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    fake = FakeHfApi()

    with pytest.raises(RuntimeError):
        pkg.push_to_hub(tmp_path, "carev01/opus-distill-test", True, hf_api=fake)

    assert fake.created == []
    assert fake.uploaded == []


def test_push_to_hub_never_logs_the_token(tmp_path, monkeypatch, caplog):
    in_dir = _tiny_dataset(tmp_path)
    pkg.package(in_dir, tmp_path / "out.zip")
    secret = "sk-hf-do-not-print-this-0123456789"
    monkeypatch.setenv("HF_TOKEN", secret)
    fake = FakeHfApi()

    with caplog.at_level(logging.DEBUG):
        pkg.push_to_hub(in_dir, "carev01/opus-distill-test", True, hf_api=fake)

    assert secret not in caplog.text
    # The token IS passed into upload_file's kwargs (the real HfApi needs it per
    # call) -- what must never happen is logging it, which the assertion above covers.


def test_push_to_hub_warns_when_public(tmp_path, monkeypatch, caplog):
    in_dir = _tiny_dataset(tmp_path)
    pkg.package(in_dir, tmp_path / "out.zip")
    monkeypatch.setenv("HF_TOKEN", "secret-token-value")
    fake = FakeHfApi()

    with caplog.at_level(logging.WARNING):
        pkg.push_to_hub(in_dir, "carev01/opus-distill-test", False, hf_api=fake)

    assert any("PUBLIC" in r.message for r in caplog.records)
    assert fake.created[0]["private"] is False


def test_push_to_hub_private_logs_no_warning(tmp_path, monkeypatch, caplog):
    in_dir = _tiny_dataset(tmp_path)
    pkg.package(in_dir, tmp_path / "out.zip")
    monkeypatch.setenv("HF_TOKEN", "secret-token-value")
    fake = FakeHfApi()

    with caplog.at_level(logging.WARNING):
        pkg.push_to_hub(in_dir, "carev01/opus-distill-test", True, hf_api=fake)

    assert not any("PUBLIC" in r.message for r in caplog.records)
    assert fake.created[0]["private"] is True


def test_main_cli_with_hf_repo_defaults_to_private(tmp_path, monkeypatch):
    in_dir = _tiny_dataset(tmp_path)
    out_zip = tmp_path / "out.zip"
    calls = []
    monkeypatch.setattr(pkg, "push_to_hub",
                        lambda in_dir, repo_id, private, **kw: calls.append(
                            (in_dir, repo_id, private)))

    rc = pkg.main(["--in", str(in_dir), "--out", str(out_zip),
                  "--hf-repo", "carev01/opus-distill-test"])

    assert rc == 0
    assert calls == [(in_dir, "carev01/opus-distill-test", True)]


def test_main_cli_with_public_flag_passes_private_false(tmp_path, monkeypatch):
    in_dir = _tiny_dataset(tmp_path)
    out_zip = tmp_path / "out.zip"
    calls = []
    monkeypatch.setattr(pkg, "push_to_hub",
                        lambda in_dir, repo_id, private, **kw: calls.append(
                            (in_dir, repo_id, private)))

    rc = pkg.main(["--in", str(in_dir), "--out", str(out_zip),
                  "--hf-repo", "carev01/opus-distill-test", "--public"])

    assert rc == 0
    assert calls == [(in_dir, "carev01/opus-distill-test", False)]


def test_main_cli_without_hf_repo_never_calls_push(tmp_path, monkeypatch):
    in_dir = _tiny_dataset(tmp_path)
    out_zip = tmp_path / "out.zip"
    monkeypatch.setattr(pkg, "push_to_hub",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("push_to_hub must not be called")))

    rc = pkg.main(["--in", str(in_dir), "--out", str(out_zip)])

    assert rc == 0


# --------------------------------------------------------------------------- #
# notebook validity
# --------------------------------------------------------------------------- #

def test_notebook_is_valid_json_and_nbformat_if_available():
    nb_path = REPO_ROOT / "notebooks" / "distill_train.ipynb"
    assert nb_path.exists(), f"missing {nb_path}"
    raw = json.loads(nb_path.read_text(encoding="utf-8"))
    assert raw.get("nbformat") == 4
    assert raw.get("cells")

    try:
        import nbformat
    except ImportError:
        return
    nb = nbformat.reads(nb_path.read_text(encoding="utf-8"), as_version=4)
    nbformat.validate(nb)
