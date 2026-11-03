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
- `batches/batch-NNN.jsonl`: 4 blinded judge views per file (no labels, no system ids);
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
BATCH_SIZE = 4
MAX_SOURCE_CHARS = 60000
PARAPHRASE_NOTE = "system paraphrase (not evidence)"
TRUNCATION_NOTE = "\n[... source text truncated by the harness ...]"

EPISODES_FOR_FACTS = (
    "MATCH ()-[r:RELATES_TO {uuid: $u}]->() UNWIND r.episodes AS eid "
    "MATCH (e:Episodic {uuid: eid}) OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e) "
    "RETURN r.fact AS fact, e.content AS text, a.id AS article_id, a.title AS title")


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
        docs: list[dict] = []
        for r in rows:  # an episode linked to two articles comes back twice: show it once
            if isinstance(r.get("text"), str) and all(d["text"] != r["text"] for d in docs):
                docs.append({"title": r.get("title"), "text": r["text"]})
        out.append({"n": c.get("marker"), "paraphrase": fact,
                    "paraphrase_note": PARAPHRASE_NOTE, "unresolvable": not docs,
                    "documents": docs})
    return out


def _vendorkb_sources(row: dict, chunk_texts: dict[str, str]) -> list[dict]:
    out = []
    for c in sorted(row.get("citations") or [], key=lambda c: c.get("marker", 0)):
        cid = c.get("chunk_id")
        text = chunk_texts.get(str(cid)) if cid else None
        docs = [] if text is None else [{"title": c.get("title"), "text": text}]
        out.append({"n": c.get("marker"), "paraphrase": None, "paraphrase_note": None,
                    "unresolvable": not docs, "documents": docs})
    return out


def _dedupe_and_cap(sources: list[dict], max_chars: int) -> bool:
    """In place: a document text already shown for an earlier source of the same answer is
    replaced by a pointer; then the total shown text is capped at `max_chars`, cutting the
    tail. Returns whether anything was truncated."""
    seen: dict[str, Any] = {}
    for s in sources:
        for d in s["documents"]:
            first = seen.setdefault(d["text"], s["n"])
            if first != s["n"]:
                d["text"] = ""
                d["same_text_as_source"] = first
            else:
                d["same_text_as_source"] = None
    remaining = max_chars
    truncated = False
    for s in sources:
        for d in s["documents"]:
            if len(d["text"]) <= remaining:
                remaining -= len(d["text"])
                d["truncated"] = False
            else:
                d["text"] = d["text"][:remaining] + TRUNCATION_NOTE
                d["truncated"] = True
                remaining = 0
                truncated = True
    return truncated


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
            "answers": {}, "truncated": {}, "errored": {}, "unresolvable": {}}
        for side in SIDES:
            row, label = latest[(qid, side)], labels[side]
            if row.get("answer") is None:
                task["answers"][label] = {"errored": True, "text": None, "sources": []}
                task["truncated"][label] = False
                task["errored"][label] = True
                task["unresolvable"][label] = 0
                continue
            sources = (_graph_sources(row, graph_sources) if side == "graph"
                       else _vendorkb_sources(row, chunk_texts))
            task["truncated"][label] = _dedupe_and_cap(sources, max_source_chars)
            task["answers"][label] = {"errored": False, "text": row["answer"],
                                      "sources": sources}
            task["errored"][label] = False
            task["unresolvable"][label] = sum(1 for s in sources if s["unresolvable"])
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
    n_batches = len(list((pass_dir / "batches").glob("batch-*.jsonl")))
    print(f"pass {args.pass_n}: {len(tasks)} tasks, {n_batches} batches; "
          f"fact uuids {len(fact_uuids)} (resolved {sum(1 for v in graph_sources.values() if v)})"
          f", chunks {len(chunk_ids)} (found {len(chunk_texts)}); truncated {n_trunc}; "
          f"unresolvable citations {n_unres}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pass", dest="pass_n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--questions", default="eval/headtohead/questions.yaml")
    ap.add_argument("--only-qids", default=None, help="file with one question id per line")
    ap.add_argument("--max-source-chars", type=int, default=MAX_SOURCE_CHARS)
    return asyncio.run(_amain(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
