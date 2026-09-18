#!/usr/bin/env python
"""Re-filter an already-built dataset to a shorter context window.

Why this exists separately from `build_extraction_dataset.py --max-prompt-tokens`:
that path rebuilds from the raw capture, and the capture is a 64 MB scratch file
that does not survive a reboot. This works from the shipped `train.jsonl` /
`val.jsonl`, which do.

It undoes the upsampling (every copy carries `meta.repeat_index`, so the distinct
set is recoverable exactly), drops records whose prompt cannot fit the window,
recomputes the loss weights over what remains, and re-upsamples the TRAIN split
only. Recomputing rather than reusing matters: dropping the long tail removes
tokens unevenly across prompts, so weights derived from the old distribution
would aim at a mix that no longer exists.

    uv run --extra dev python scripts/refilter_dataset.py \\
        --in data/ft-captured --out data/ft-12k --max-prompt-tokens 12288
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# `pythonpath = ["."]` in pyproject is a PYTEST setting; running this file
# directly puts scripts/ on sys.path, not the repo root. Add the root so the
# sibling builder imports as a package module rather than being duplicated here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_extraction_dataset import (  # noqa: E402
    assign_loss_weights,
    drop_overlong,
    upsample_train,
)

logger = logging.getLogger("refilter_dataset")


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _distinct(records: list[dict]) -> list[dict]:
    """Undo upsampling. Copy 0 is the original; the rest are repeats."""
    return [r for r in records if r.get("meta", {}).get("repeat_index", 0) == 0]


def _table(title: str, records: list[dict]) -> None:
    c = Counter(r["prompt_name"] for r in records)
    logger.info("  %s: %d records", title, len(records))
    for name, n in c.most_common():
        logger.info("      %-45s %6d", name, n)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prompt-tokens", type=int, required=True)
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--max-repeat", type=int, default=12)
    ap.add_argument("--loss-basis", choices=("full", "completion"), default="full")
    args = ap.parse_args()

    train_in = _distinct(_read(args.src / "train.jsonl"))
    val_in = _read(args.src / "val.jsonl")
    logger.info("=== input (upsampling undone) ===")
    _table("train distinct", train_in)
    _table("val", val_in)

    train, dropped_t = drop_overlong(train_in, args.max_prompt_tokens, args.chars_per_token)
    val, dropped_v = drop_overlong(val_in, args.max_prompt_tokens, args.chars_per_token)
    logger.info("\n=== dropped over %d tokens (%.1f chars/token) ===",
                args.max_prompt_tokens, args.chars_per_token)
    for name in sorted(set(dropped_t) | set(dropped_v), key=lambda k: -(dropped_t.get(k, 0))):
        base = sum(1 for r in train_in + val_in if r["prompt_name"] == name)
        gone = dropped_t.get(name, 0) + dropped_v.get(name, 0)
        logger.info("    %-45s %5d of %5d  (%.1f%%)", name, gone, base, 100 * gone / base)
    total_gone = sum(dropped_t.values()) + sum(dropped_v.values())
    total_base = len(train_in) + len(val_in)
    logger.info("    %-45s %5d of %5d  (%.1f%%)", "TOTAL", total_gone, total_base,
                100 * total_gone / total_base)

    # Weights over what SHIPS, not over what was built: the drop is uneven across
    # prompts, so the old weights would target a distribution that no longer exists.
    assign_loss_weights(train + val, basis=args.loss_basis)
    upsampled = upsample_train(train, args.max_repeat, args.loss_basis)

    args.out.mkdir(parents=True, exist_ok=True)
    src_schemas = args.src / "schemas.json"
    if src_schemas.exists():
        (args.out / "schemas.json").write_text(src_schemas.read_text(encoding="utf-8"),
                                               encoding="utf-8")
    _write(args.out / "train.jsonl", upsampled)
    _write(args.out / "val.jsonl", val)

    logger.info("\n=== output ===")
    _table("train (upsampled)", upsampled)
    _table("val (never upsampled)", val)
    longest = max((sum(len(m["content"]) for m in r["messages"])
                   for r in upsampled + val), default=0)
    logger.info("\n  longest surviving prompt: %d chars = %d tokens at %.1f chars/token",
                longest, int(longest / args.chars_per_token), args.chars_per_token)
    logger.info("  wrote %d train / %d val -> %s", len(upsampled), len(val), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
