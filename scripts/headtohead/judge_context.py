"""Build the blind judging tasks for the head-to-head comparison (spec §7).

For every question the judge gets the question, its type and gold facts, and the two
answers under shuffled anonymous labels (A/B), each with the source document text behind
every citation:
- vendorkb: the cited chunk's text;
- graph (answer-api): the text of the episode(s) the cited fact came from, labelled with
  their article titles; the fact text is shown too, marked as the system's paraphrase.
  Every answer-api mode's citations carry `fact_uuid`, so community-report answers resolve
  the same way (report -> facts -> episodes is already what their citations are).

Outputs, under `<run>/pass-N/`:
- `tasks.jsonl`: every task, with the `labels` mapping (the judges never see this file);
- `batches/batch-NNN.jsonl`: 2 blinded judge views per file (no labels, no system ids);
- `meta.json`: the seed and question count.

Run: uv run --extra dev python -m scripts.headtohead.judge_context \
    --run eval/headtohead/runs/<date> --pass 1 --seed 1 [--only-qids FILE]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SIDES = ("graph", "vendorkb")
BATCH_SIZE = 2  # ruling R14: 4 tasks at the cap is ~120k tokens per judge call
MAX_SOURCE_CHARS = 60000
MAX_EPISODES_PER_FACT = 5
PARAPHRASE_NOTE = "system paraphrase (not evidence)"
TRUNCATION_NOTE = "\n[... source text truncated by the harness ...]"

EPISODES_FOR_FACTS = (
    "MATCH ()-[r:RELATES_TO {uuid: $u}]->() UNWIND r.episodes AS eid "
    "MATCH (e:Episodic {uuid: eid}) OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e) "
    "RETURN r.fact AS fact, e.content AS text, a.id AS article_id, a.title AS title, "
    "e.uuid AS episode_uuid ORDER BY episode_uuid, article_id")


def latest_answers(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    """(qid, side) -> row; a resumed run may repeat a pair, and the LAST line wins."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        out[(str(r["qid"]), str(r["side"]))] = r
    return out


def read_answers(path: Path) -> dict[tuple[str, str], dict]:
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i == len(lines) - 1:  # a torn final line (crashed run) is not an answer
                print(f"WARNING: skipping torn final line in {path}", file=sys.stderr)
                continue
            raise
    return latest_answers(rows)


def _graph_sources(row: dict, graph_sources: dict[str, list[dict]]) -> list[dict]:
    out = []
    for c in sorted(row.get("citations") or [], key=lambda c: c.get("marker", 0)):
        uuid = c.get("fact_uuid")
        rows = graph_sources.get(uuid, []) if uuid else []
        fact = c.get("fact") or next((r.get("fact") for r in rows if r.get("fact")), None)
        own = {str(x.get("article_id")) for x in c.get("sources") or []
               if x.get("article_id") is not None}
        # the citation's own article(s) first; otherwise the query's stable order
        ordered = sorted(rows, key=lambda r: str(r.get("article_id")) not in own)
        docs: list[dict] = []
        for r in ordered:  # an episode linked to two articles comes back twice: show once
            if isinstance(r.get("text"), str) and all(d["text"] != r["text"] for d in docs):
                docs.append({"title": r.get("title"), "text": r["text"]})
        out.append({"n": c.get("marker"), "paraphrase": fact,
                    "paraphrase_note": PARAPHRASE_NOTE, "unresolvable": not docs,
                    "documents": docs[:MAX_EPISODES_PER_FACT],
                    "documents_omitted": max(0, len(docs) - MAX_EPISODES_PER_FACT)})
    return out


def _vendorkb_sources(row: dict, chunk_texts: dict[str, str]) -> list[dict]:
    out = []
    for c in sorted(row.get("citations") or [], key=lambda c: c.get("marker", 0)):
        cid = c.get("chunk_id")
        text = chunk_texts.get(str(cid)) if cid else None
        docs = [] if text is None else [{"title": c.get("title"), "text": text}]
        out.append({"n": c.get("marker"), "paraphrase": None, "paraphrase_note": None,
                    "unresolvable": not docs, "documents": docs, "documents_omitted": 0})
    return out


def fair_shares(needs: list[int], budget: int) -> list[int]:
    """Max-min fair split of `budget` chars over sources needing `needs` chars: every source
    gets an equal share, a source needing less hands the rest back, and the leftover is
    shared again (round-robin, one char at a time once shares round to zero) among the
    sources that still have text. Every source with text gets some while budget >= count."""
    alloc = [0] * len(needs)
    remaining = budget
    active = [i for i, n in enumerate(needs) if n > 0]
    while active and remaining > 0:
        share = remaining // len(active)
        for i in active:
            give = min(share or 1, needs[i] - alloc[i], remaining)
            alloc[i] += give
            remaining -= give
            if remaining == 0:
                break
        active = [i for i in active if alloc[i] < needs[i]]
    return alloc


def _cap(sources: list[dict], max_chars: int, originals: list[list[str]]) -> bool:
    needs = [sum(len(t) for t, d in zip(orig, s["documents"], strict=True)
                 if d["same_text_as_source"] is None)
             for s, orig in zip(sources, originals, strict=True)]
    alloc = fair_shares(needs, max_chars)
    truncated = False
    for s, orig, budget in zip(sources, originals, alloc, strict=True):
        kept = []
        for text, d in zip(orig, s["documents"], strict=True):
            if d["same_text_as_source"] is not None:
                d["text"], d["truncated"] = "", False
                kept.append(d)
            elif budget <= 0:
                s["documents_omitted"] += 1  # nothing of it fits: not shown at all
                truncated = True
            elif len(text) <= budget:
                d["text"], d["truncated"] = text, False
                budget -= len(text)
                kept.append(d)
            else:
                d["text"], d["truncated"] = text[:budget] + TRUNCATION_NOTE, True
                budget = 0
                truncated = True
                kept.append(d)
        s["documents"] = kept
    return truncated


def _dedupe_and_cap(sources: list[dict], max_chars: int) -> bool:
    """In place. A document text already shown for an earlier source of the same answer is
    replaced by a pointer (`same_text_as_source`) -- but only while that earlier copy is
    shown whole: a pointer to a truncated or dropped copy gets its own text back. Then each
    source gets a fair share of `max_chars` (`fair_shares`; ruling R13), cutting within a
    source from its tail. Returns whether anything was truncated."""
    originals = [[d["text"] for d in s["documents"]] for s in sources]
    no_pointer: set[tuple[int, int]] = set()
    while True:
        work = [dict(s, documents=[dict(d, _pos=(si, di)) for di, d in enumerate(s["documents"])])
                for si, s in enumerate(sources)]
        targets: dict[tuple[int, int], tuple[int, int]] = {}
        seen: dict[str, tuple[int, int]] = {}
        for si, s in enumerate(work):
            for di, d in enumerate(s["documents"]):
                first = seen.get(originals[si][di])
                if first is not None and first[0] != si and (si, di) not in no_pointer:
                    d["same_text_as_source"] = sources[first[0]]["n"]
                    targets[(si, di)] = first
                else:
                    d["same_text_as_source"] = None
                    seen.setdefault(originals[si][di], (si, di))
        truncated = _cap(work, max_chars, originals)
        whole = {d["_pos"] for s in work for d in s["documents"]
                 if d["same_text_as_source"] is None and not d["truncated"]}
        broken = {pos for pos, target in targets.items() if target not in whole}
        if not broken:
            break
        no_pointer |= broken
    for s, w in zip(sources, work, strict=True):
        for d in w["documents"]:
            del d["_pos"]
        s["documents"] = w["documents"]
        s["documents_omitted"] = w["documents_omitted"]
    return truncated


def _has_evidence(source: dict) -> bool:
    """Some document text is shown for this citation (directly or via a pointer)."""
    return any(d["same_text_as_source"] is not None
               or d["text"].replace(TRUNCATION_NOTE, "").strip()
               for d in source["documents"])


def build_tasks(questions: list[dict], answers: Iterable[dict] | dict, *,
                graph_sources: dict[str, list[dict]], chunk_texts: dict[str, str],
                seed: int, max_source_chars: int = MAX_SOURCE_CHARS) -> list[dict]:
    """Pure: one task per question, in qid-sorted order, labels drawn from `seed` (one
    draw per task). `answers` is the answers.jsonl rows (last (qid, side) row wins) or the
    already-reduced mapping. Raises ValueError naming every question missing a side."""
    latest = answers if isinstance(answers, dict) else latest_answers(answers)
    qs = sorted(questions, key=lambda q: str(q["id"]))
    missing = [f"{q['id']}/{s}" for q in qs for s in SIDES if (str(q["id"]), s) not in latest]
    if missing:
        raise ValueError(f"answers.jsonl lacks rows for: {', '.join(missing)}")
    rng = random.Random(seed)
    tasks = []
    for q in qs:
        qid = str(q["id"])
        graph_label = "A" if rng.random() < 0.5 else "B"
        labels = {"graph": graph_label, "vendorkb": "B" if graph_label == "A" else "A"}
        task: dict[str, Any] = {
            "key": qid, "qid": qid, "type": q["type"], "question": q["question"],
            "gold_facts": list(q.get("gold_facts") or []), "labels": labels, "seed": seed,
            "answers": {}, "truncated": {}, "errored": {}, "unresolvable": {},
            "zero_evidence": {}}
        for side in SIDES:
            row, label = latest[(qid, side)], labels[side]
            if row.get("answer") is None:
                task["answers"][label] = {"errored": True, "text": None, "sources": []}
                task["truncated"][label] = False
                task["errored"][label] = True
                task["unresolvable"][label] = 0
                task["zero_evidence"][label] = 0
                continue
            sources = (_graph_sources(row, graph_sources) if side == "graph"
                       else _vendorkb_sources(row, chunk_texts))
            task["truncated"][label] = _dedupe_and_cap(sources, max_source_chars)
            task["answers"][label] = {"errored": False, "text": row["answer"],
                                      "sources": sources}
            task["errored"][label] = False
            task["unresolvable"][label] = sum(1 for s in sources if s["unresolvable"])
            task["zero_evidence"][label] = sum(1 for s in sources if not s["unresolvable"]
                                               and not _has_evidence(s))
        tasks.append(task)
    return tasks


def judge_view(task: dict) -> dict:
    """What a judge sees: no labels mapping, no system ids, the same keys on both sides."""
    answers = {}
    for label in ("A", "B"):
        a = task["answers"][label]
        answers[label] = {
            "errored": a["errored"], "text": a["text"],
            "sources": [{"n": s["n"], "paraphrase": s["paraphrase"],
                         "paraphrase_note": s["paraphrase_note"],
                         "unresolvable": s["unresolvable"],
                         "documents_omitted": s["documents_omitted"],
                         "documents": [{"title": d["title"], "text": d["text"],
                                        "same_text_as_source": d["same_text_as_source"],
                                        "truncated": d["truncated"]}
                                       for d in s["documents"]]}
                        for s in a["sources"]]}
    return {"key": task["key"], "type": task["type"], "question": task["question"],
            "gold_facts": task["gold_facts"], "answers": answers}


def needs_judge(task: dict) -> bool:
    """A task where both answers errored needs no judge: both are not perfect."""
    return not all(task["errored"].values())


def check_pass_dir(pass_dir: Path, *, force: bool) -> str | None:
    """None if pass_dir may be (re)written. An existing pass is never silently replaced:
    rewriting tasks.jsonl under existing judgements would swap labels under them. `force`
    allows a rewrite only while judgements/ is absent or empty."""
    tasks, jdir = pass_dir / "tasks.jsonl", pass_dir / "judgements"
    if jdir.is_dir() and any(jdir.iterdir()):
        return (f"{jdir} holds judgements; refusing to rewrite this pass even with --force "
                "(use a new pass number)")
    if (tasks.exists() or jdir.exists()) and not force:
        return f"{pass_dir} already exists (tasks.jsonl or judgements/); pass --force to rewrite"
    return None


def write_pass(pass_dir: Path, tasks: list[dict]) -> None:
    pass_dir.mkdir(parents=True, exist_ok=True)
    with (pass_dir / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    bdir = pass_dir / "batches"
    bdir.mkdir(exist_ok=True)
    for old in bdir.glob("batch-*.jsonl"):
        old.unlink()
    judged = [t for t in tasks if needs_judge(t)]
    for i in range(0, len(judged), BATCH_SIZE):
        path = bdir / f"batch-{i // BATCH_SIZE + 1:03d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for t in judged[i:i + BATCH_SIZE]:
                fh.write(json.dumps(judge_view(t), ensure_ascii=False) + "\n")


# --- I/O -----------------------------------------------------------------------------


async def fetch_graph_sources(fact_uuids: set[str]) -> dict[str, list[dict]]:
    """Read-only: EPISODES_FOR_FACTS for each cited fact uuid."""
    from neo4j import AsyncGraphDatabase, RoutingControl

    from graph_extract.config import ExtractSettings

    s = ExtractSettings()
    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    sem = asyncio.Semaphore(8)
    out: dict[str, list[dict]] = {}

    async def one(u: str) -> None:
        async with sem:
            r = await driver.execute_query(EPISODES_FOR_FACTS, {"u": u},
                                           routing_=RoutingControl.READ)
        out[u] = [rec.data() for rec in r.records]

    try:
        await asyncio.gather(*(one(u) for u in sorted(fact_uuids)))
    finally:
        await driver.close()
    return out


async def fetch_chunk_texts(chunk_ids: set[str]) -> dict[str, str]:
    from vendorkb.config import VendorkbSettings
    from vendorkb.db import connect

    conn = await connect(VendorkbSettings().vendorkb_dsn)
    try:
        cur = await conn.execute(
            "SELECT chunk_id::text, text FROM chunks WHERE chunk_id::text = ANY(%s)",
            (sorted(chunk_ids),))
        return {str(cid): text for cid, text in await cur.fetchall()}
    finally:
        await conn.close()


def _other_pass_seeds(run_dir: Path, pass_n: int) -> dict[int, int]:
    out = {}
    for meta in run_dir.glob("pass-*/meta.json"):
        n = int(meta.parent.name.split("-", 1)[1])
        if n != pass_n:
            out[n] = json.loads(meta.read_text(encoding="utf-8"))["seed"]
    return out


async def _amain(args: argparse.Namespace) -> int:
    from scripts.headtohead.questions import load_questions

    run_dir = Path(args.run)
    clash = {n: s for n, s in _other_pass_seeds(run_dir, args.pass_n).items()
             if s == args.seed}
    if clash:
        print(f"seed {args.seed} already used by pass {sorted(clash)}; each pass needs its "
              "own seed", file=sys.stderr)
        return 2
    blocked = check_pass_dir(run_dir / f"pass-{args.pass_n}", force=args.force)
    if blocked:
        print(blocked, file=sys.stderr)
        return 2
    questions = load_questions(Path(args.questions))
    if args.only_qids:
        wanted = {x.strip() for x in Path(args.only_qids).read_text().splitlines()
                  if x.strip()}
        unknown = wanted - {str(q["id"]) for q in questions}
        if unknown:
            print(f"--only-qids names unknown questions: {sorted(unknown)}", file=sys.stderr)
            return 2
        questions = [q for q in questions if str(q["id"]) in wanted]
    latest = read_answers(run_dir / "answers.jsonl")
    qids = {str(q["id"]) for q in questions}
    rows = [r for (qid, _s), r in latest.items() if qid in qids]
    fact_uuids = {c["fact_uuid"] for r in rows if r["side"] == "graph" and r.get("answer")
                  for c in r.get("citations") or [] if c.get("fact_uuid")}
    chunk_ids = {str(c["chunk_id"]) for r in rows
                 if r["side"] == "vendorkb" and r.get("answer")
                 for c in r.get("citations") or [] if c.get("chunk_id")}
    graph_sources = await fetch_graph_sources(fact_uuids)
    chunk_texts = await fetch_chunk_texts(chunk_ids)
    tasks = build_tasks(questions, latest, graph_sources=graph_sources,
                        chunk_texts=chunk_texts, seed=args.seed,
                        max_source_chars=args.max_source_chars)
    pass_dir = run_dir / f"pass-{args.pass_n}"
    write_pass(pass_dir, tasks)
    (pass_dir / "meta.json").write_text(json.dumps(
        {"seed": args.seed, "questions": len(tasks),
         "max_source_chars": args.max_source_chars}, indent=2) + "\n", encoding="utf-8")
    n_trunc = {s: sum(t["truncated"][t["labels"][s]] for t in tasks) for s in SIDES}
    n_unres = {s: sum(t["unresolvable"][t["labels"][s]] for t in tasks) for s in SIDES}
    n_zero = {s: sum(t["zero_evidence"][t["labels"][s]] for t in tasks) for s in SIDES}
    n_batches = len(list((pass_dir / "batches").glob("batch-*.jsonl")))
    print(f"pass {args.pass_n}: {len(tasks)} tasks, {n_batches} batches; "
          f"fact uuids {len(fact_uuids)} (resolved {sum(1 for v in graph_sources.values() if v)})"
          f", chunks {len(chunk_ids)} (found {len(chunk_texts)}); truncated {n_trunc}; "
          f"unresolvable citations {n_unres}; citations with zero evidence text {n_zero} "
          "(expected 0)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pass", dest="pass_n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--questions", default="eval/headtohead/questions.yaml")
    ap.add_argument("--only-qids", default=None, help="file with one question id per line")
    ap.add_argument("--max-source-chars", type=int, default=MAX_SOURCE_CHARS)
    ap.add_argument("--force", action="store_true",
                    help="rewrite an existing pass that has no judgements yet")
    return asyncio.run(_amain(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
