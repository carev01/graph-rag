#!/usr/bin/env python
"""Move each prompt's static tail to the FRONT, so a prefix cache can use it.

Measured on this corpus: a working prompt cache is worth ~nothing as prompts are
currently laid out. `extract_edges.edge` shares 22 characters with the next call;
`dedupe_nodes.nodes` shares 36. The cache hits 9.9x on a literal repeat and never
gets the chance, because graphiti puts the per-call content first
(PREVIOUS_MESSAGES / CURRENT_MESSAGE / ENTITIES) and the static rules last.

Moving the static block to the front turns that shared content into a shared
PREFIX. Measured movable fractions: extract_text 89.4%, resolve_edge 69.7%,
extract_edges.edge 40.0%, summaries 9.7%, dedupe_nodes 5.6% -- projected 67
s/article, 28%.

The block is moved WITHIN the user message rather than into the system message.
Both give the same cacheable prefix, but keeping every token in the same role is
the smaller distribution shift, which matters because the model was fine-tuned on
the current layout and we want to know whether it tolerates the change without a
retrain.

The static tail is derived empirically: the longest common suffix across records
of a prompt type IS its static content, and it is byte-identical by construction.
The cut is aligned to the blank line after the preceding block's closing tag so
the moved text is a self-contained section rather than a fragment.

    uv run --extra dev python scripts/reorder_prompts.py \
        --in data/ft-12k/val.jsonl --out data/ft-12k/val-reordered.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path


def common_suffix(strings: list[str]) -> str:
    rev = [s[::-1] for s in strings]
    return os.path.commonprefix(rev)[::-1]


def clean_block(suffix: str) -> str:
    """Trim the leading fragment so the moved text starts at a section boundary.

    Two distinct fragments have to come off, and missing either corrupts the
    prompt:

    1. **A partial line.** The longest common suffix can begin MID-TOKEN, because
       a varying value may end in a constant. `extract_edges.edge` is the real
       case: REFERENCE_TIME is `2026-08-17 23:53:21.843656+00:00`, the date
       varies, and the suffix therefore starts at `+00:00`. Moving that splits
       the timestamp in half and leaves `...843656` stranded. So always advance
       to the next line boundary first.
    2. **A closing tag.** What remains often opens with the closer of the
       preceding variable block (`</TEXT>`, `</REFERENCE_TIME>`). That belongs to
       the content staying behind, not to the static rules.
    """
    i = suffix.find("\n")
    s = suffix[i + 1:] if i >= 0 else suffix
    if s.lstrip().startswith("</"):
        j = s.find("\n\n")
        if j >= 0:
            s = s[j + 2:]
    return s.lstrip("\n")


def reorder(user: str, block: str) -> str:
    """Static block first, then everything else, with the block removed from the end."""
    if not user.endswith(block):
        return user  # untouched; caller counts these
    head = user[: len(user) - len(block)]
    return f"{block.rstrip()}\n\n{head.rstrip()}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="derive the static blocks from this file instead of --in; "
                         "use the TRAIN split so the blocks do not depend on the "
                         "records being transformed")
    args = ap.parse_args()

    with args.src.open() as fh:
        records = [json.loads(line) for line in fh]
    corpus_path = args.corpus or args.src
    with corpus_path.open() as fh:
        corpus = [json.loads(line) for line in fh]

    by_name: dict[str, list[str]] = collections.defaultdict(list)
    for r in corpus:
        if r.get("meta", {}).get("repeat_index", 0) == 0:
            by_name[r["prompt_name"]].append(
                next(m["content"] for m in r["messages"] if m["role"] == "user"))
    blocks = {n: clean_block(common_suffix(v[:400])) for n, v in by_name.items()}

    print(f"static blocks derived from {corpus_path} ({len(corpus)} records):")
    for n, b in sorted(blocks.items()):
        print(f"  {n:42} {len(b):>7,} chars")

    moved = skipped = 0
    out = []
    for r in records:
        block = blocks.get(r["prompt_name"], "")
        msgs = []
        for m in r["messages"]:
            if m["role"] == "user" and block:
                new = reorder(m["content"], block)
                if new != m["content"]:
                    moved += 1
                else:
                    skipped += 1
                m = {**m, "content": new}
            msgs.append(m)
        out.append({**r, "messages": msgs,
                    "meta": {**r["meta"], "prompt_layout": "static-first"}})

    # Self-check: a reorder is a PERMUTATION. If the word multiset changed, the
    # cut landed mid-token and the prompt is corrupted -- refuse to write it
    # rather than leave a plausible-looking file for a benchmark to consume.
    corrupted = []
    for before, after in zip(records, out):
        ub = next(m["content"] for m in before["messages"] if m["role"] == "user")
        ua = next(m["content"] for m in after["messages"] if m["role"] == "user")
        if sorted(ub.split()) != sorted(ua.split()):
            corrupted.append(before["prompt_name"])
    if corrupted:
        counts = collections.Counter(corrupted)
        print(f"\nREFUSING TO WRITE: {len(corrupted)} records changed content, "
              f"not just order:", file=sys.stderr)
        for name, n in counts.most_common():
            print(f"    {name:42} {n:>5}", file=sys.stderr)
        print("  The static block is being cut mid-token; fix clean_block().",
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nreordered {moved} of {moved+skipped} user messages "
          f"({skipped} did not end with their block and were left as-is)")
    print(f"wrote {len(out)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
