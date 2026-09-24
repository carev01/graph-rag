import json, statistics as st, sys
from graph_extract.config import get_extract_settings
from graph_extract.chonkie_client import Chunk
from graph_extract.episode_builder import build_episodes
from graph_extract.article_router import is_dense_matrix
s = get_extract_settings()
arts = json.load(open(sys.argv[1] + "/d6_sample.json"))
AP, BP, AC, BC = 54751, 19.8, 1263, 2.23      # fitted per-episode prompt/completion chars
def pack(eps, target):
    out = []
    for e in eps:
        if out and out[-1][0] + e.token_count <= target:
            out[-1] = (out[-1][0] + e.token_count, out[-1][1] + len(e.body))
        else:
            out.append((e.token_count, len(e.body)))
    return out
rows = {}
dense_n = 0
for a in arts:
    chunks = [Chunk(text=c["t"], start_index=0, end_index=0, token_count=c["tok"]) for c in a["chunks"]]
    md = "\n".join(c["t"] for c in a["chunks"])
    dense = is_dense_matrix(md, ratio_threshold=s.dense_table_line_ratio, pipe_threshold=s.dense_pipe_count)
    dense_n += dense
    cap = s.max_chunk_tokens if dense else s.cheap_max_chunk_tokens
    for label, target in (("today", None), ("pack 600", 600), ("pack 900", 900), ("pack 1200", 1200), ("pack 1800", 1800)):
        c2 = max(cap, target or 0)
        eps = build_episodes(article_id=a["id"], title=a["title"], chapter_path="", content_hash="x"*16,
                             chunks=chunks, max_chunk_tokens=c2, min_chunk_tokens=s.min_chunk_tokens)
        units = [(e.token_count, len(e.body)) for e in eps] if target is None else pack(eps, target)
        r = rows.setdefault(label, {"eps": 0, "tok": [], "p": 0.0, "c": 0.0})
        r["eps"] += len(units); r["tok"] += [u[0] for u in units]
        r["p"] += sum(AP + BP * u[1] for u in units); r["c"] += sum(AC + BC * u[1] for u in units)
print(f"{len(arts)} articles, {dense_n} dense (strong tier, cap {s.max_chunk_tokens}), rest cheap (cap {s.cheap_max_chunk_tokens})")
base = rows["today"]
print(f"{'variant':10s} {'episodes':>8s} {'eps/art':>7s} {'median tok':>10s} {'prompt vol':>10s} {'compl vol':>9s}")
for k, r in rows.items():
    print(f"{k:10s} {r['eps']:8d} {r['eps']/len(arts):7.2f} {st.median(r['tok']):10.0f} "
          f"{r['p']/base['p']:9.0%} {r['c']/base['c']:9.0%}")
