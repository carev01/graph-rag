"""Sample fresh articles for a DPO round's on-policy capture. READ-ONLY on DocExtractor;
no LLM, no Neo4j.

    uv run --extra dev python -m scripts.distill.dpo_sample --out DIR/dpo_sample.json \\
        --exclude DISTILL/sample.json --exclude TRIAL/d6_sample.json [--articles 700]

Sources are drawn with probability proportional to their article count (as the d6 trial
sample was), `--per-source` articles from each. An article is skipped when it is in any
`--exclude` file (a distillation `sample.json` -- every split, held-out included -- or a
trial sample list: those articles must stay unseen so they can score the DPO model), a
navigation page, empty, longer than `--max-chars`, or a dense matrix the router sends to
the strong tier (the student never serves those in production, so its replies there are
not worth preferring or rejecting). The output is a `haiku_extract_ab --sample` file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from docext.client import make_docext_client
from graph_extract import content_fetch
from graph_extract.article_filter import is_navigation_article
from graph_extract.article_router import is_dense_matrix
from graph_extract.config import get_extract_settings


def excluded_ids(paths: list[Path]) -> set[str]:
    """Article ids in distillation samples (`{"articles": [{"article_id"}]}`) and trial
    samples (`[{"id"}]`)."""
    ids: set[str] = set()
    for p in paths:
        data = json.loads(p.read_text())
        rows = data["articles"] if isinstance(data, dict) else data
        ids |= {r.get("article_id") or r["id"] for r in rows}
    return ids


def skip_reason(article_id: str, title: str, markdown: str, *, excluded: set[str],
                max_chars: int, ratio: float, pipes: int) -> str:
    """Why an article stays out of the sample ("" when it goes in)."""
    if article_id in excluded:
        return "excluded"
    if is_navigation_article(title) or not markdown.strip():
        return "navigation_or_empty"
    if len(markdown) > max_chars:
        return "too_long"
    if is_dense_matrix(markdown, ratio_threshold=ratio, pipe_threshold=pipes):
        return "dense"
    return ""


def pick_sources(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """`n` distinct sources, probability proportional to article count."""
    def count(x: dict) -> int:
        return x.get("article_count") or x.get("articles") or 0
    pool = [x for x in rows if count(x) > 0]
    chosen: list[dict] = []
    while pool and len(chosen) < n:
        i = rng.choices(range(len(pool)), weights=[count(x) for x in pool])[0]
        chosen.append(pool.pop(i))
    return chosen


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, action="append", default=[])
    ap.add_argument("--articles", type=int, default=700)
    ap.add_argument("--per-source", type=int, default=7)
    ap.add_argument("--max-chars", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    s = get_extract_settings()
    rng = random.Random(a.seed)
    excluded = excluded_ids(a.exclude)
    print(f"excluding {len(excluded)} articles from {len(a.exclude)} files", flush=True)
    out: list[dict] = []
    skipped: Counter = Counter()
    async with make_docext_client(base_url=s.docext_base_url, read_key=s.docext_read_key,
                                  admin_key=s.docext_admin_key,
                                  verify_tls=s.docext_verify_tls) as dx:
        r = await dx.get("/api/dashboard/sources")
        r.raise_for_status()
        d = r.json()
        rows = d if isinstance(d, list) else d.get("sources") or d.get("items") or []
        # Enough sources for the target even after skips; the loop stops at --articles.
        sources = pick_sources(rows, 2 * -(-a.articles // a.per_source), rng)
        for src in sources:
            if len(out) >= a.articles:
                break
            sid = src.get("id") or src.get("source_id")
            vendor = src.get("vendor_name") or src.get("vendor")
            lr = await dx.get("/api/articles", params={"source_id": sid, "limit": 200})
            lr.raise_for_status()
            items = lr.json().get("items") or lr.json().get("articles") or []
            rng.shuffle(items)
            taken = 0
            for it in items:
                if taken >= a.per_source or len(out) >= a.articles:
                    break
                if it["id"] in excluded:
                    skipped["excluded"] += 1
                    continue
                art = await content_fetch.fetch_article(dx, it["id"])
                why = skip_reason(art.id, art.title or "", art.content_markdown,
                                  excluded=excluded, max_chars=a.max_chars,
                                  ratio=s.dense_table_line_ratio, pipes=s.dense_pipe_count)
                if why:
                    skipped[why] += 1
                    continue
                out.append({"id": art.id, "vendor": vendor, "source": src.get("name"),
                            "title": art.title, "chars": len(art.content_markdown)})
                taken += 1
            print(f"{len(out)} articles ({vendor} / {src.get('name')})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    print(f"wrote {len(out)} articles from {len({x['vendor'] for x in out})} vendors and "
          f"{len({(x['vendor'], x['source']) for x in out})} sources; skipped {dict(skipped)}")


if __name__ == "__main__":
    asyncio.run(main())
