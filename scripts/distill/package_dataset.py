#!/usr/bin/env python
"""Zip the packaged fine-tune dataset for Colab upload, and optionally push it to a
private Hugging Face dataset repo (spec 2026-10-17 section 7, Task 6).

    uv run --extra dev python -m scripts.distill.package_dataset \\
        --in data/ft-opus --out dist/ft-opus.zip \\
        [--hf-repo carev01/<name> [--public]]

READ-ONLY on the dataset directory except for writing `MANIFEST.json` into it.
Makes no Neo4j, LLM or embedder call. The `--hf-repo` push is the only network
I/O this script performs.

ZIP CONTENTS
------------
Exactly `train.jsonl`, `val.jsonl`, `sample.json` and a freshly-written
`MANIFEST.json`, read from `--in`. `train.jsonl`/`val.jsonl` are
`build_extraction_dataset.py`'s own output (record shape: `prompt_name`,
`response_schema`, `messages`, `meta.loss_weight`/`meta.repeat_index`/...; see
its `_record` and `assign_loss_weights`). `sample.json` is Task 3 harness
`select`'s output -- the builder writes no such file itself, so the runbook has
the operator copy or point `select --out` directly at `<in>/sample.json`
before packaging. It is OPTIONAL here (a missing one only warns) because the
builder's own files are what a trainer actually consumes; `sample.json` is
provenance, not training data.

`train.jsonl` and `val.jsonl` are required: packaging without them would ship
an empty or mislabelled dataset silently.

MANIFEST.json records, per the brief: counts per `prompt_name` (train and val
separately), the loss-weight mix (`meta.loss_weight` per `prompt_name`, read
off the train split -- `assign_loss_weights` sets one constant weight per
prompt type, so the first record of each type carries it), token estimates
(chars // 4 over every message, matching the builder's own `full`-basis
yardstick in `_tokens_by_prompt`), and the git commit this was packaged at
(mirrors `extraction_proof.py`'s `_commit()`: `git rev-parse --short HEAD`,
`"?"` if git is unavailable -- never fatal).

HF PUSH (`--hf-repo`)
----------------------
`huggingface_hub` is NOT a base dependency of this repo (nothing else here
needs it) and is imported lazily, only inside `push_to_hub`, only when the
caller has not injected a fake `hf_api` -- so importing this module, and every
code path except an actual `--hf-repo` push, needs nothing beyond the stdlib.
Install it first (`uv pip install huggingface_hub` or add it to a venv) before
using `--hf-repo`.

The token comes ONLY from the `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
environment variable -- never a CLI flag, never logged, never written to
MANIFEST.json. The pushed repo is PRIVATE by default (fail closed): `--public`
is the explicit opt-out, and omitting both flags keeps it private. Passing
`--public` logs a warning, since it is now a deliberate, consequential choice
rather than an easy-to-miss default.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
SAMPLE_FILE = "sample.json"
MANIFEST_FILE = "MANIFEST.json"
REQUIRED_FILES = (TRAIN_FILE, VAL_FILE)
OPTIONAL_FILES = (SAMPLE_FILE,)
# Order matters only for the zip's directory listing.
ZIP_MEMBERS = (TRAIN_FILE, VAL_FILE, SAMPLE_FILE, MANIFEST_FILE)

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _commit() -> str:
    """`git rev-parse --short HEAD`, `"?"` on any failure. Mirrors
    `scripts/extraction_proof.py`'s `_commit()` -- duplicated rather than
    imported so this script has no dependency on that one's internals."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _counts_by_prompt(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.get("prompt_name", "unknown")] += 1
    out = dict(counts)
    out["_total"] = len(records)
    return out


def _tokens_by_prompt(records: list[dict]) -> dict[str, int]:
    """chars // 4 over every message, summed per `prompt_name` -- the same
    yardstick as `build_extraction_dataset._tokens_by_prompt(basis="full")`."""
    tokens: dict[str, int] = defaultdict(int)
    for r in records:
        tokens[r.get("prompt_name", "unknown")] += sum(
            max(1, len(m.get("content", "")) // 4) for m in r.get("messages", ()))
    out = dict(tokens)
    out["_total"] = sum(out.values())
    return out


def _loss_weight_mix(records: list[dict]) -> dict[str, float]:
    """One `meta.loss_weight` per `prompt_name`, from the train split. The
    builder sets a constant weight per prompt type (`assign_loss_weights`), so
    the first record seen of each type carries the value for all of them."""
    mix: dict[str, float] = {}
    for r in records:
        name = r.get("prompt_name", "unknown")
        weight = r.get("meta", {}).get("loss_weight")
        if name not in mix and weight is not None:
            mix[name] = weight
    return mix


def build_manifest(in_dir: Path) -> dict[str, Any]:
    train = _read_jsonl(in_dir / TRAIN_FILE)
    val = _read_jsonl(in_dir / VAL_FILE)
    return {
        "git_commit": _commit(),
        "counts": {"train": _counts_by_prompt(train), "val": _counts_by_prompt(val)},
        "tokens": {"train": _tokens_by_prompt(train), "val": _tokens_by_prompt(val)},
        "loss_weight_mix": _loss_weight_mix(train),
        "sample_included": (in_dir / SAMPLE_FILE).exists(),
    }


def package(in_dir: Path, out_path: Path) -> dict[str, Any]:
    """Build MANIFEST.json (written into `in_dir`) and zip it with the dataset
    files into `out_path`. Returns the manifest dict."""
    missing_required = [name for name in REQUIRED_FILES if not (in_dir / name).exists()]
    if missing_required:
        raise FileNotFoundError(
            f"{in_dir} is missing {missing_required} -- run "
            "build_extraction_dataset.py first")
    manifest = build_manifest(in_dir)
    manifest_path = in_dir / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ZIP_MEMBERS:
            member = in_dir / name
            if not member.exists():
                if name in OPTIONAL_FILES:
                    logger.warning(
                        "  %s not found in %s -- packaging without it (the runbook's "
                        "harness select step should write it there)", name, in_dir)
                    continue
                raise FileNotFoundError(f"{member} vanished mid-package")
            zf.write(member, arcname=name)
    return manifest


def _hf_token() -> str:
    for var in HF_TOKEN_ENV_VARS:
        token = os.environ.get(var)
        if token:
            return token
    raise RuntimeError(
        f"no HF token in the environment -- set one of {HF_TOKEN_ENV_VARS} before "
        "using --hf-repo; this script never takes a token on the command line")


def push_to_hub(in_dir: Path, repo_id: str, private: bool, *, hf_api: Any = None) -> None:
    """Push the packaged files to a private HF dataset repo via `huggingface_hub`.

    The token is read ONLY from the environment and is never logged or
    returned -- only whether one was found, via the `RuntimeError` above, not
    its value. `hf_api` is dependency-injected so tests can pass a fake and
    never import the real library or touch the network.
    """
    token = _hf_token()
    if hf_api is None:
        from huggingface_hub import HfApi  # lazy: not a base dependency
        hf_api = HfApi(token=token)
    if not private:
        logger.warning("  --public passed: %s will be a PUBLIC dataset repo", repo_id)
    hf_api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for name in ZIP_MEMBERS:
        member = in_dir / name
        if not member.exists():
            continue
        hf_api.upload_file(path_or_fileobj=str(member), path_in_repo=name,
                           repo_id=repo_id, repo_type="dataset", token=token)
    logger.info("  pushed %s -> hf://datasets/%s (private=%s)", in_dir, repo_id, private)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.distill.package_dataset",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", type=Path, required=True,
                   help="dataset directory holding train.jsonl / val.jsonl "
                        "[/ sample.json]")
    p.add_argument("--out", type=Path, required=True, help="output zip path")
    p.add_argument("--hf-repo", default=None,
                   help="push the packaged files to this HF dataset repo id "
                        "(e.g. carev01/opus-distill-2026-10), PRIVATE by default; "
                        "token from HF_TOKEN / HUGGING_FACE_HUB_TOKEN only")
    p.add_argument("--public", action="store_true",
                   help="create the HF repo PUBLIC instead of private. The default "
                        "(this flag omitted) is private -- fail closed. Pass this "
                        "only when you deliberately want a public dataset repo.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    manifest = package(args.in_dir, args.out)
    logger.info("packaged %s -> %s", args.in_dir, args.out)
    logger.info("  train: %d records, val: %d records",
                manifest["counts"]["train"]["_total"], manifest["counts"]["val"]["_total"])
    if args.hf_repo:
        push_to_hub(args.in_dir, args.hf_repo, not args.public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
