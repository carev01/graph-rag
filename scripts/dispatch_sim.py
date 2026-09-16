"""Offline replay of concurrent ingest: what does an ordering or a knob cost?

Answers "how many duplicate entities would concurrency `c`, warm-up `W` and
dispatch order X produce?" **without spending money**. The section 9 A/B cost
~$10 and 3.1 h; this reproduces it in under a second, so ordering and knob
questions stop being paid experiments.

Everything is derived from a graph an ingest already wrote -- nothing is
hard-coded and no fixture is needed:

- **dispatch order** = articles sorted by their first episode's `created_at`.
  The pilot ran sequentially, so execution order IS dispatch order (verified
  against the recorded list: identical for all 83).
- **per-chunk durations** = gaps between consecutive `Episodic.created_at`
  stamps. graphiti sets `created_at = utc_now()` at the top of `add_episode`
  (`graphiti.py:1068`), so a stamp is the chunk's START. The final chunk gets
  the mean.
- **entity spans** = `(:Article)-[:HAS_EPISODE]->(:Episodic)-[:MENTIONS]->(:Entity)`,
  keeping the FIRST chunk of each (entity, article) -- the chunk that would
  create the node.

The model
---------
A chunk occupies `[s, s+d]`. Its node resolution happens at `s + alpha*d` and
its nodes commit at `s + d` (graphiti writes them in one transaction at the end
of `add_episode`). For each entity, order its first-mention chunks by resolution
time: the first creates the node, and each later chunk creates ANOTHER COPY iff
no earlier-resolving chunk has committed by the time it resolves. `excess` is
`sum(copies - 1)` -- the same quantity `merge_duplicates` counts and the A/B
reported.

The scheduler replays `run_concurrently` (`concurrent_ingest.py`): items in
dispatch order, a warm item takes the next free slot, a cold item drains
everything launched, runs alone, then the fan-out resumes. `is_cold` replays
`WarmupGate.is_cold_source`: fewer than `W` articles of the source have a
committed chunk.

Known approximations, stated rather than hidden:

- The dispatch clock only advances on cold items, so a warm item's predicate is
  evaluated at the last drain. This only matters at the barrier's edge -- once a
  source is warm it stays warm -- and the calibration below absorbs it.
- No provider degradation under load, so simulated wall clock runs ~10% short of
  measured (the A/B saw latency rise in the highest in-flight buckets).
- `alpha` -- where in a chunk resolution happens -- is not measured. The
  calibration is flat across 0.1/0.3/0.5, so conclusions do not rest on it.

Calibration (the reason to trust it)
------------------------------------
Against the 999-entity `backup-docs` baseline it must reproduce BOTH paid arms
of the 2026-09-14/15 A/B at c=4 -- see CALIBRATION below. It gets the residual's
*population* right too: at W=8 most of the excess comes from dispatch pair
(71, 72), the immutable-vault pair the A/B identified by name. If these rows
drift, the script is wrong or the graph is no longer that baseline -- do not
trust its other numbers.

Writes nothing: the session is opened READ-only, so the server itself refuses
a write.

    uv run --extra dev python scripts/dispatch_sim.py
    uv run --extra dev python scripts/dispatch_sim.py --sweep        # c = 1..16
    uv run --extra dev python scripts/dispatch_sim.py -c 8 -W 8
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import statistics
import sys
from collections import defaultdict

from neo4j import AsyncGraphDatabase, RoutingControl

from graph_extract.config import get_extract_settings

# (warmup, alpha) -> (excess, measured) for the recorded order at c=4.
# Pinned the way risk_curve.py pins the spec's 79.5%: these are what the two
# paid arms measured (30 and 16), and what this model must land near.
CALIBRATION = {0: 30, 8: 16}
CALIBRATION_TOLERANCE = 8

_ARTICLES = """
MATCH (a:Article)-[r:HAS_EPISODE]->(e:Episodic {group_id:$g})
WHERE coalesce(r.superseded, false) = false
WITH a, r, e ORDER BY r.chunk_index
RETURN a.id AS id, a.source_id AS source_id, a.sort_order AS sort_order,
       collect({chunk: r.chunk_index, epoch: e.created_at.epochMillis}) AS chunks
"""

_SPANS = """
MATCH (a:Article)-[r:HAS_EPISODE]->(ep:Episodic {group_id:$g})-[:MENTIONS]->(e:Entity {group_id:$g})
WHERE coalesce(r.superseded, false) = false
WITH e, a, min(r.chunk_index) AS first_chunk
RETURN e.name AS name, collect({article: a.id, first_chunk: first_chunk}) AS mentions
"""


class Pilot:
    """The recorded run, reduced to what the replay needs."""

    def __init__(self, articles: list[dict], spans: list[dict]) -> None:
        self.spans = [s for s in spans if len(s["mentions"]) > 1]
        self.source = {a["id"]: a["source_id"] for a in articles}
        self.sort_order = {a["id"]: a["sort_order"] for a in articles}
        self.chunks = {a["id"]: [c["chunk"] for c in a["chunks"]] for a in articles}

        # Dispatch order = execution order of the sequential run.
        first_stamp = {a["id"]: min(c["epoch"] for c in a["chunks"]) for a in articles}
        self.order = sorted(first_stamp, key=lambda a: first_stamp[a])

        # Chunk durations: the gap to the next chunk STARTED anywhere in the run.
        starts = sorted((c["epoch"] / 1000.0, a["id"], c["chunk"])
                        for a in articles for c in a["chunks"])
        self.duration: dict[tuple[str, int], float] = {}
        for (t, aid, ci), nxt in zip(starts, starts[1:]):
            self.duration[(aid, ci)] = nxt[0] - t
        mean = statistics.fmean(self.duration.values()) if self.duration else 1.0
        last = starts[-1]
        self.duration[(last[1], last[2])] = mean
        self.mean_chunk = mean

    def by_source(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for aid in self.order:
            out[self.source[aid]].append(aid)
        return out


def schedule(p: Pilot, order: list[str], c: int, warmup: int, alpha: float,
             rng: random.Random | None = None, sigma: float = 0.3
             ) -> dict[tuple[str, int], tuple[float, float]]:
    """Replay run_concurrently + WarmupGate. Returns per-chunk (resolve, commit)."""
    times: dict[tuple[str, int], tuple[float, float]] = {}
    slots = [0.0] * c
    launched: list[float] = []
    now = 0.0
    committed_first: dict[str, float] = {}

    def live(source: str, at: float) -> int:
        return sum(1 for a, t in committed_first.items()
                   if p.source[a] == source and t <= at)

    def run(aid: str, start: float) -> float:
        t = start
        for ci in p.chunks[aid]:
            d = p.duration[(aid, ci)]
            if rng is not None:
                d *= rng.lognormvariate(0.0, sigma)
            times[(aid, ci)] = (t + alpha * d, t + d)
            t += d
        committed_first[aid] = times[(aid, p.chunks[aid][0])][1]
        return t

    for aid in order:
        if warmup > 0 and live(p.source[aid], now) < warmup:
            if launched:
                now = max(now, max(launched))
                launched.clear()
            now = run(aid, now)
            slots = [now] * c
            continue
        i = min(range(c), key=lambda k: slots[k])
        end = run(aid, max(slots[i], now))
        slots[i] = end
        launched.append(end)
    return times


def excess(p: Pilot, times) -> tuple[int, dict[str, int]]:
    """Duplicate entities created: sum(copies - 1), and the per-name breakdown."""
    total = 0
    per_name: dict[str, int] = {}
    for span in p.spans:
        ms = sorted((times[(m["article"], m["first_chunk"])] for m in span["mentions"]),
                    key=lambda rc: rc[0])
        copies = 1
        earliest_commit = ms[0][1]
        for resolve, commit in ms[1:]:
            if resolve < earliest_commit:
                copies += 1
            earliest_commit = min(earliest_commit, commit)
        if copies > 1:
            per_name[span["name"]] = copies - 1
            total += copies - 1
    return total, per_name


def wall(times) -> float:
    return max(commit for _, commit in times.values())


def evaluate(p: Pilot, order: list[str], c: int, warmup: int, alpha: float,
             seeds: int) -> tuple[int, float, float, float]:
    """Deterministic excess, noisy mean/sd over `seeds` replays, and wall hours."""
    det = schedule(p, order, c, warmup, alpha)
    noisy = [excess(p, schedule(p, order, c, warmup, alpha, random.Random(s)))[0]
             for s in range(seeds)]
    return (excess(p, det)[0],
            statistics.fmean(noisy) if noisy else float("nan"),
            statistics.pstdev(noisy) if noisy else float("nan"),
            wall(det) / 3600)


# --- orderings ---------------------------------------------------------------

def hybrid(p: Pilot, permute, warmup: int) -> list[str]:
    """What ingest_driver.spread_siblings does: the first `warmup` of each source
    in sort_order, then the remainder permuted."""
    head, rest = [], []
    for members in p.by_source().values():
        by_so = sorted(members, key=lambda a: p.sort_order[a])
        head += by_so[:warmup]
        rest += by_so[warmup:]
    return head + permute(rest)


def orderings(p: Pilot, warmup: int) -> dict[str, list[str]]:
    src_order = [a for members in p.by_source().values()
                 for a in sorted(members, key=lambda x: p.sort_order[x])]
    return {
        "recorded (the A/B's own order)": p.order,
        "sort_order, source by source (the OLD ingest_source)": src_order,
        "ascending id, no warm-up prefix": sorted(p.order),
        "hybrid: warm-up by sort_order + rest ascending id (SHIPPED)":
            hybrid(p, sorted, warmup),
        "hybrid + rest by sha256(id)":
            hybrid(p, lambda r: sorted(r, key=lambda a: hashlib.sha256(a.encode()).digest()),
                   warmup),
    }


async def load(group_id: str) -> Pilot:
    s = get_extract_settings()
    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        arts = await driver.execute_query(_ARTICLES, {"g": group_id},
                                          routing_=RoutingControl.READ)
        spans = await driver.execute_query(_SPANS, {"g": group_id},
                                           routing_=RoutingControl.READ)
    finally:
        await driver.close()
    return Pilot([dict(r) for r in arts.records], [dict(r) for r in spans.records])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="backup-docs")
    ap.add_argument("-c", "--concurrency", type=int, default=4)
    ap.add_argument("-W", "--warmup", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep concurrency 1..16 for the shipped ordering")
    args = ap.parse_args()

    p = asyncio.run(load(args.group))
    if not p.order:
        print(f"group {args.group!r} has no articles with live episodes", file=sys.stderr)
        return 1
    sources = {s: len(v) for s, v in p.by_source().items()}
    print(f"group {args.group!r}: {len(p.order)} articles over {len(sources)} source(s) "
          f"({', '.join(str(n) for n in sorted(sources.values(), reverse=True))}), "
          f"{len(p.duration)} chunks, {len(p.spans)} multi-article entities")
    print(f"mean chunk {p.mean_chunk:.1f}s; sequential wall "
          f"{sum(p.duration.values())/3600:.2f} h\n")

    # --- calibration ---------------------------------------------------------
    print("== calibration: the recorded order at c=4, against the two paid arms ==")
    ok = True
    for warmup, measured in sorted(CALIBRATION.items()):
        row = []
        for alpha in (0.1, 0.3, 0.5):
            row.append(evaluate(p, p.order, 4, warmup, alpha, 0)[0])
        drift = abs(row[1] - measured)
        flag = "" if drift <= CALIBRATION_TOLERANCE else "   <-- DRIFTED"
        ok = ok and not flag
        print(f"  W={warmup}: simulated {row} at alpha=0.1/0.3/0.5, measured {measured}{flag}")
    if not ok:
        print("\n  The model no longer reproduces the measured arms. Either this is not\n"
              "  the 999-entity baseline, or the model is wrong -- do not trust the rows\n"
              "  below until this is resolved.", file=sys.stderr)
    print()

    if args.sweep:
        print(f"== concurrency sweep, W={args.warmup}, shipped ordering, "
              f"{args.seeds} noisy seeds ==")
        print(f"  {'c':>3}  {'dupes':>5}  {'noisy mean':>10}  {'wall h':>7}  speedup")
        base = None
        for c in (1, 2, 4, 8, 12, 16, 24):
            order = hybrid(p, sorted, args.warmup)
            det, mean, _sd, hours = evaluate(p, order, c, args.warmup, args.alpha, args.seeds)
            base = base or hours
            print(f"  {c:>3}  {det:>5}  {mean:>10.1f}  {hours:>7.2f}  {base/hours:.2f}x")
        return 0 if ok else 1

    print(f"== orderings at c={args.concurrency}, W={args.warmup}, "
          f"alpha={args.alpha}, {args.seeds} noisy seeds ==")
    print(f"  {'ordering':58} {'dupes':>5} {'noisy':>6} {'sd':>5} {'wall h':>7}")
    for name, order in orderings(p, args.warmup).items():
        assert sorted(order) == sorted(p.order), f"{name} lost or duplicated an article"
        det, mean, sd, hours = evaluate(p, order, args.concurrency, args.warmup,
                                        args.alpha, args.seeds)
        print(f"  {name:58} {det:>5} {mean:>6.1f} {sd:>5.1f} {hours:>7.2f}")

    rng = random.Random(1000)
    vals = [excess(p, schedule(p, hybrid(p, lambda r: rng.sample(r, len(r)), args.warmup),
                               args.concurrency, args.warmup, args.alpha, rng))[0]
            for _ in range(args.seeds)]
    vals.sort()
    print(f"  {'hybrid + rest uniform random (why NOT to shuffle)':58} "
          f"{'':>5} {statistics.fmean(vals):>6.1f} {statistics.pstdev(vals):>5.1f}"
          f"     p95={vals[int(0.95 * len(vals))]} max={vals[-1]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
