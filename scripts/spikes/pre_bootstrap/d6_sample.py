"""D6 spike: sample articles across the corpus, fetch (read-only), neural-chunk (local)."""
import asyncio, json, random, sys
import httpx
from graph_extract.config import get_extract_settings
from graph_extract import content_fetch, chonkie_client
from graph_extract.article_filter import is_navigation_article
from docext.client import make_docext_client
OUT = sys.argv[1]; N_SOURCES = 25; PER_SOURCE = 8
async def main():
    s = get_extract_settings(); rng = random.Random(23)
    dx = make_docext_client(base_url=s.docext_base_url, read_key=s.docext_read_key,
                            admin_key=s.docext_admin_key, verify_tls=s.docext_verify_tls)
    r = await dx.get("/api/dashboard/sources"); r.raise_for_status(); d = r.json()
    rows = d if isinstance(d, list) else d.get("sources") or d.get("items") or []
    def cnt(x): return x.get("article_count") or x.get("articles") or 0
    rows = [x for x in rows if cnt(x) > 0]
    print("sources with articles:", len(rows), "total articles:", sum(cnt(x) for x in rows), flush=True)
    chosen = set()
    while len(chosen) < N_SOURCES:  # probability proportional to article count
        chosen.add(rng.choices(range(len(rows)), weights=[cnt(x) for x in rows])[0])
    out = []
    async with httpx.AsyncClient(base_url=s.chonkie_base_url, timeout=120) as ch:
        for i in sorted(chosen):
            src = rows[i]; sid = src.get("id") or src.get("source_id")
            lr = await dx.get("/api/articles", params={"source_id": sid, "limit": 200}); lr.raise_for_status()
            items = lr.json().get("items") or lr.json().get("articles") or []
            for a in rng.sample(items, min(PER_SOURCE, len(items))):
                art = await content_fetch.fetch_article(dx, a["id"])
                if is_navigation_article(art.title or "") or not art.content_markdown.strip():
                    continue
                chunks = await chonkie_client.neural_chunk(ch, art.content_markdown, s.chonkie_model)
                out.append({"id": art.id, "vendor": src.get("vendor_name") or src.get("vendor"),
                            "source": src.get("name"), "title": art.title,
                            "chars": len(art.content_markdown),
                            "chunks": [{"t": c.text, "tok": c.token_count} for c in chunks]})
            print(f"{len(out)} articles ({src.get('vendor_name') or src.get('vendor')} / {src.get('name')})", flush=True)
    json.dump(out, open(f"{OUT}/d6_sample.json", "w"))
    await dx.aclose()
asyncio.run(main())
