"""dpo4 gate (spec docs/superpowers/specs/2026-10-24-dpo-coverage-pairs-design.md section 6).
Runs ON the Vast GPU instance (scripts/distill/dpo_gate_remote.sh drives it); the scoring
and statistics are pure and unit-tested, the GPU commands import torch / transformers /
peft / vllm lazily.

    python dpo_gate.py merge --base B --adapter A --out M --revision REV  # bf16 merge, adapter engaged?
    python dpo_gate.py generate --model M --tokenizer B --val V --out G [--limit N] [--full]
    python dpo_gate.py score --base G0 --dpo G1 --val V --out gate.json --revision REV [--smoke]
    python dpo_gate.py push --merged M --repo R --revision EXPORT_REV --smoke-json G --gate gate.json
                                                               # only if gate.json passed for M's adapter

Each model continues every val pair greedily from the token prefix chosen and rejected
share, and edges are compared as exact (source, target, relation) triples. The pass bar,
unchanged from dpo3, is on the A pairs: removed written <= 0.75 x ckpt100's, kept written
>= 0.90 x, valid JSON not fewer, ckpt100's removed written >= 10. B and C are reported.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from scripts.distill.chat_format import serving_prompt
except ImportError:   # on the instance: chat_format.py is copied next to this file
    from chat_format import serving_prompt   # type: ignore[no-redef]

MAX_REMOVED_SHARE = 0.75
MIN_KEPT_SHARE = 0.90
MIN_BASELINE_REMOVED = 10
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_TRIPLE = re.compile(r'"source_entity_name": "(.*?)", "target_entity_name": "(.*?)", '
                     r'"relation_type": "(.*?)"')
MARKER = "adapter_revision.txt"   # written into the merged dir: which adapter revision it is
SMOKE_IDENTICAL = 2    # `score --smoke`: every continuation is token-identical (adapter not engaged)
SMOKE_NO_PAIRS = 3     # `score --smoke`: no pair has both generations and a split point
COUNTS = ("removed", "removed_written", "added", "added_written", "kept", "kept_written", "json")


def triples(text: str) -> tuple[set, bool]:
    try:
        return {(e["source_entity_name"], e["target_entity_name"], e["relation_type"])
                for e in json.loads(text)["edges"]}, True
    except (ValueError, KeyError, TypeError):
        return set(_TRIPLE.findall(text)), False


def split_point(a: list[int], b: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def prepare(record: dict, tok: Any) -> dict | None:
    """The shared token prefix up to where chosen and rejected first differ, its reply
    part as text, and the generation budget; None when they tokenise identically."""
    prompt = serving_prompt(tok, record["messages"])
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    ch = tok(prompt + record["chosen"], add_special_tokens=False)["input_ids"]
    rj = tok(prompt + record["rejected"], add_special_tokens=False)["input_ids"]
    split = split_point(ch, rj)
    if split is None:
        return None
    return {"call_id": record["meta"]["call_id"], "type": record["meta"]["type"],
            "prefix_ids": ch[:split], "reply_prefix_ids": ch[n_prompt:split],
            "prefix_text": tok.decode(ch[n_prompt:split], skip_special_tokens=True),
            "max_tokens": min(max(len(ch), len(rj)) - split + 256, 4096)}


def split_texts(tok: Any, reply_prefix_ids: list[int], gen_ids: list[int]) -> tuple[str, str]:
    """(prefix_text, gen_text) with prefix_text + gen_text == the reply decoded as one
    sequence: decoding the two halves separately garbles a character the split cuts in
    two. prefix_text ends before any character the prefix alone leaves incomplete."""
    whole = tok.decode([*reply_prefix_ids, *gen_ids], skip_special_tokens=True)
    cut = len(tok.decode(reply_prefix_ids, skip_special_tokens=True).rstrip("\ufffd"))
    return whole[:cut], whole[cut:]


def score_pair(record: dict, prefix_text: str, gen_text: str) -> dict:
    gen, ok = triples(prefix_text + gen_text)
    done = set(_TRIPLE.findall(prefix_text))   # edges complete before the split
    ch, rj = triples(record["chosen"])[0], triples(record["rejected"])[0]
    removed, added, kept = rj - ch, ch - rj, ch - done
    return {"call_id": record["meta"]["call_id"], "type": record["meta"]["type"],
            "removed": len(removed), "removed_written": len(removed & gen),
            "added": len(added), "added_written": len(added & gen),
            "kept": len(kept), "kept_written": len(kept & gen), "json": ok}


def totals(rows: list[dict], kind: str) -> dict:
    sel = [r for r in rows if r["type"] == kind]
    return {"pairs": len(sel), **{k: sum(int(r[k]) for r in sel) for k in COUNTS}}


def decide(base: dict, dpo: dict) -> tuple[bool, list[str]]:
    why = []
    if base["removed_written"] < MIN_BASELINE_REMOVED:
        why.append(f"baseline: ckpt100 writes only {base['removed_written']} removed edges "
                   f"(< {MIN_BASELINE_REMOVED}) -- the val set cannot show a reduction")
    if dpo["removed_written"] > MAX_REMOVED_SHARE * base["removed_written"]:
        why.append(f"removed: {dpo['removed_written']} > {MAX_REMOVED_SHARE} x "
                   f"{base['removed_written']}")
    if dpo["kept_written"] < MIN_KEPT_SHARE * base["kept_written"]:
        why.append(f"kept: {dpo['kept_written']} < {MIN_KEPT_SHARE} x {base['kept_written']}")
    if dpo["json"] < base["json"]:
        why.append(f"json: {dpo['json']} < {base['json']}")
    return not why, why


def sign_test(base: list[int], dpo: list[int]) -> dict:
    down = sum(d < b for b, d in zip(base, dpo))
    up = sum(d > b for b, d in zip(base, dpo))
    n = down + up
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(down, up) + 1)) / 2 ** n) if n else 1.0
    return {"down": down, "same": len(base) - n, "up": up, "p": p}


def _ratio(rows_b: list[dict], rows_d: list[dict], key: str) -> float | None:
    den = sum(r[key] for r in rows_b)
    return sum(r[key] for r in rows_d) / den if den else None


def bootstrap(base_rows: list[dict], dpo_rows: list[dict], n: int = 10_000,
              seed: int = 20261024) -> dict:
    """Pairs resampled together: CIs for removed ratio, kept ratio and their quotient."""
    rng = random.Random(seed)
    idx = range(len(base_rows))
    rem, kep, sel = [], [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        rb, rd = [base_rows[i] for i in s], [dpo_rows[i] for i in s]
        r, k = _ratio(rb, rd, "removed_written"), _ratio(rb, rd, "kept_written")
        if r is None or k is None or not k:
            continue
        rem.append(r)
        kep.append(k)
        sel.append(r / k)

    def ci(xs: list[float]) -> list[float | None]:
        if not xs:   # no resample had a usable denominator (no A pairs, or ckpt100 wrote none)
            return [None, None]
        xs = sorted(xs)
        return [xs[int(0.025 * len(xs))], xs[min(len(xs) - 1, int(0.975 * len(xs)))]]

    r0, k0 = _ratio(base_rows, dpo_rows, "removed_written"), _ratio(base_rows, dpo_rows, "kept_written")
    return {"removed_ratio": r0, "removed_ratio_ci": ci(rem), "kept_ratio": k0,
            "kept_ratio_ci": ci(kep),
            "selectivity": r0 / k0 if r0 is not None and k0 else None,
            "selectivity_ci": ci(sel), "resamples": len(rem)}


def differing(names: Iterable[str], equal: Callable[[str], bool]) -> list[str]:
    return [n for n in names if any(f".{t}." in n for t in TARGET_MODULES) and not equal(n)]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]


# --- GPU-side commands (lazy imports; not unit-tested) -------------------------------------

def copyable(name: str) -> bool:
    """Base-dir files carried into the merged dir: tokenizer, chat template, configs. Never
    weights, and never the base's shard index (save_pretrained writes the merged one)."""
    return not name.endswith((".safetensors", ".safetensors.index.json"))


def check_gate(gate: dict, marker: str | None) -> list[str]:
    """Why `gate` does not license pushing the merged model whose marker file says `marker`
    (empty list: it does)."""
    why = []
    if not gate.get("passed"):
        why.append("gate did not pass")
    if marker is None:
        why.append("the merged model has no adapter-revision marker")
    elif gate.get("adapter_revision") != marker:
        why.append(f"gate.json judged adapter revision {gate.get('adapter_revision')!r}, "
                   f"the merged model is {marker!r}")
    return why


def merge(base: Path, adapter: Path, out: Path, revision: str) -> None:
    import torch
    import transformers
    from peft import PeftModel
    from safetensors import safe_open

    arch = json.loads((base / "config.json").read_text())["architectures"][0]
    model = getattr(transformers, arch).from_pretrained(str(base), torch_dtype=torch.bfloat16)
    peft = PeftModel.from_pretrained(model, str(adapter))
    live = sum(int(p.detach().abs().sum().item() > 0)
               for n, p in peft.named_parameters() if "lora_B" in n)
    assert live > 0, "every lora_B is zero after loading: the adapter's keys did not match"
    print(f"adapter loaded: {live} non-zero lora_B tensors", flush=True)
    peft.merge_and_unload().save_pretrained(str(out), safe_serialization=True)
    for f in base.iterdir():   # tokenizer, chat template, processor configs
        if f.is_file() and copyable(f.name) and not (out / f.name).exists():
            shutil.copy2(f, out / f.name)

    def weight_map(d: Path) -> dict[str, str]:
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            return json.loads(idx.read_text())["weight_map"]
        return {k: "model.safetensors" for k in safe_open(str(d / "model.safetensors"), "pt").keys()}

    wb, wm = weight_map(base), weight_map(out)
    common = sorted(set(wb) & set(wm))
    targets = [n for n in common if any(f".{t}." in n for t in TARGET_MODULES)]
    assert targets, "no LoRA target tensor has the same name in ckpt100 and the merged model"

    def equal(name: str) -> bool:
        with safe_open(str(base / wb[name]), "pt") as fb, safe_open(str(out / wm[name]), "pt") as fm:
            return torch.equal(fb.get_tensor(name), fm.get_tensor(name))

    diff = differing(targets, equal)
    assert diff, "the merged model's LoRA target weights equal ckpt100's: the adapter is not engaged"
    print(f"merged: {len(diff)} of {len(targets)} target tensors differ from ckpt100", flush=True)
    (out / MARKER).write_text(revision, encoding="utf-8")   # last: only a complete merge has one


def generate(model: Path, tokenizer: Path, val: Path, out: Path, limit: int, full: bool,
             max_model_len: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    records = _read_jsonl(val)
    prepared, skipped = [], 0
    if full:   # whole replies, the shortest prompts first (the export's JSON smoke test)
        for r in sorted(records, key=lambda r: len(r["messages"][1]["content"]))[: limit or 3]:
            ids = tok(serving_prompt(tok, r["messages"]), add_special_tokens=False)["input_ids"]
            prepared.append({"call_id": r["meta"]["call_id"], "type": r["meta"]["type"],
                             "prefix_ids": ids, "reply_prefix_ids": [], "max_tokens": 4096})
    else:
        for r in records[: limit or None]:
            p = prepare(r, tok)
            if p is None:
                skipped += 1
            else:
                prepared.append(p)
    llm = LLM(model=str(model), tokenizer=str(tokenizer), dtype="bfloat16",
              max_model_len=max_model_len, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.generate([TokensPrompt(prompt_token_ids=p["prefix_ids"]) for p in prepared],
                        [SamplingParams(temperature=0.0, max_tokens=p["max_tokens"])
                         for p in prepared])
    with Path(out).open("w", encoding="utf-8") as fh:
        for p, o in zip(prepared, outs):
            gen_ids = list(o.outputs[0].token_ids)
            prefix_text, gen_text = split_texts(tok, p["reply_prefix_ids"], gen_ids)
            fh.write(json.dumps({"call_id": p["call_id"], "type": p["type"],
                                 "prefix_text": prefix_text, "gen_text": gen_text,
                                 "gen_ids": gen_ids}, ensure_ascii=False) + "\n")
    print(f"generated {len(prepared)} ({skipped} pairs with no split point skipped) -> {out}")


def push(merged: Path, repo: str, revision: str, smoke_json: Path, gate_json: Path) -> None:
    import os

    from huggingface_hub import HfApi

    marker_file = merged / MARKER
    marker = marker_file.read_text(encoding="utf-8").strip() if marker_file.exists() else None
    refusal = check_gate(json.loads(Path(gate_json).read_text(encoding="utf-8")), marker)
    assert not refusal, "refusing to push: " + "; ".join(refusal)
    smoke = _read_jsonl(smoke_json)
    assert len(smoke) == 3, f"expected 3 smoke replies, got {len(smoke)}"
    for row in smoke:
        assert isinstance(json.loads(row["gen_text"])["edges"], list), row["call_id"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    refs = {b.name: b.target_commit for b in api.list_repo_refs(repo, repo_type="model").branches}
    assert refs.get(revision, refs["main"]) == refs["main"], (
        f"{repo}@{revision} already has its own commits -- choose a new revision")
    api.create_branch(repo_id=repo, repo_type="model", branch=revision, exist_ok=True)
    index = json.loads((merged / "model.safetensors.index.json").read_text())
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(merged), revision=revision,
                      delete_patterns=["*.safetensors", "*.gguf"],
                      commit_message="dpo4: DPO on opus-distill-2026-10-ckpt100, merged bf16")
    pushed = {f for f in api.list_repo_files(repo, repo_type="model", revision=revision)
              if f.endswith((".safetensors", ".gguf"))}
    assert pushed == set(index["weight_map"].values()), f"Hub shards != merged index: {pushed}"
    print(f"DONE: {repo}@{revision}")


def score(val: Path, base_gen: Path, dpo_gen: Path, out: Path, smoke: bool,
          revision: str = "") -> int:
    records = {r["meta"]["call_id"]: r for r in _read_jsonl(val)}
    gb = {r["call_id"]: r for r in _read_jsonl(base_gen)}
    gd = {r["call_id"]: r for r in _read_jsonl(dpo_gen)}
    ids = [cid for cid in records if cid in gb and cid in gd]
    if smoke:
        if not ids:
            Path(out).write_text(json.dumps({"pairs": 0, "identical": None}))
            print("smoke: no pair has both generations (a pair with no split point is "
                  "skipped) -- nothing to compare")
            return SMOKE_NO_PAIRS
        same = all(gb[c]["gen_ids"] == gd[c]["gen_ids"] for c in ids)
        Path(out).write_text(json.dumps({"pairs": len(ids), "identical": same}))
        print(f"smoke: {len(ids)} pairs, token-identical: {same}")
        return SMOKE_IDENTICAL if same else 0
    rows = {lbl: [score_pair(records[c], g[c]["prefix_text"], g[c]["gen_text"]) for c in ids]
            for lbl, g in (("ckpt100", gb), ("dpo", gd))}
    tot = {k: {lbl: totals(rows[lbl], k) for lbl in rows} for k in ("A", "B", "C")}
    passed, why = decide(tot["A"]["ckpt100"], tot["A"]["dpo"])
    a_b = [r for r in rows["ckpt100"] if r["type"] == "A"]
    a_d = [r for r in rows["dpo"] if r["type"] == "A"]
    gate = {"passed": passed, "adapter_revision": revision, "why": why, "pairs_scored": len(ids),
            "pairs_missing": len(records) - len(ids), "totals": tot,
            "stats": {**bootstrap(a_b, a_d),
                      "sign_removed": sign_test([r["removed_written"] for r in a_b],
                                                [r["removed_written"] for r in a_d]),
                      "sign_kept": sign_test([r["kept_written"] for r in a_b],
                                             [r["kept_written"] for r in a_d])},
            "bar": {"MAX_REMOVED_SHARE": MAX_REMOVED_SHARE, "MIN_KEPT_SHARE": MIN_KEPT_SHARE,
                    "MIN_BASELINE_REMOVED": MIN_BASELINE_REMOVED},
            "per_pair": rows}
    Path(out).write_text(json.dumps(gate, indent=1))
    print(json.dumps({k: v for k, v in gate.items() if k != "per_pair"}, indent=1))
    print("GATE PASSED" if passed else "GATE FAILED: " + "; ".join(why))
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dpo_gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--base", type=Path, required=True)
    m.add_argument("--adapter", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--revision", required=True)
    gn = sub.add_parser("generate")
    gn.add_argument("--model", type=Path, required=True)
    gn.add_argument("--tokenizer", type=Path, required=True)
    gn.add_argument("--val", type=Path, required=True)
    gn.add_argument("--out", type=Path, required=True)
    gn.add_argument("--limit", type=int, default=0)
    gn.add_argument("--full", action="store_true")
    gn.add_argument("--max-model-len", type=int, default=32768)
    s = sub.add_parser("score")
    s.add_argument("--val", type=Path, required=True)
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--dpo", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--revision", default="")
    p = sub.add_parser("push")
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--smoke-json", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        merge(a.base, a.adapter, a.out, a.revision)
    elif a.cmd == "generate":
        generate(a.model, a.tokenizer, a.val, a.out, a.limit, a.full, a.max_model_len)
    elif a.cmd == "score":
        return score(a.val, a.base, a.dpo, a.out, a.smoke, a.revision)
    elif a.cmd == "push":
        push(a.merged, a.repo, a.revision, a.smoke_json, a.gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
