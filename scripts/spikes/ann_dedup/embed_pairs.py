import asyncio, json, sys
import numpy as np
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder
sp = sys.argv[1]
async def main():
    pairs = json.load(open(sp + "/dedup_pairs.json"))
    emb = build_embedder(get_extract_settings())
    # graphiti embeds node.name.replace('\n',' ') for the query AND stores name_embedding of the name
    q = await emb.create_batch([a.replace("\n", " ") for a, _ in pairs])
    t = await emb.create_batch([b.replace("\n", " ") for _, b in pairs])
    Q = np.array(q, dtype=np.float32); T = np.array(t, dtype=np.float32)
    Q /= np.linalg.norm(Q, axis=1)[:, None]; T /= np.linalg.norm(T, axis=1)[:, None]
    np.save(sp + "/pair_q.npy", Q); np.save(sp + "/pair_t.npy", T)
    c = (Q * T).sum(1)
    print("pairs", len(c), "pair cosine p5/p25/p50/p75", np.percentile(c, [5, 25, 50, 75]).round(3),
          "below 0.6:", int((c <= 0.6).sum()))
    E = np.load(sp + "/entity_vecs.npy"); names = json.load(open(sp + "/entity_names.json"))
    # sanity: does the embedder reproduce stored name_embeddings?
    idx = [i for i, n in enumerate(names)][:20]
    re = np.array(await emb.create_batch([names[i].replace("\n", " ") for i in idx]), dtype=np.float32)
    re /= np.linalg.norm(re, axis=1)[:, None]
    print("stored vs re-embedded cosine min", float((re * E[idx]).sum(1).min().round(4)))
asyncio.run(main())
