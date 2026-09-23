"""THROWAWAY spike: does ANN find real node-dedup partners as the entity set grows?

Background = the 999 real entity name vectors + synthetic vectors jittered around
them, with per-dim sigma calibrated to real geometry (tight: cos-to-seed ~0.81 =
real nearest-neighbour median; loose: ~0.60). Targets = the 'existing' side of 418
real duplicate pairs judged by solar-pro4. Ground truth = exact cosine in numpy,
the same computation as graphiti's brute query (top-15, score > 0.6).
"""
import asyncio, json, sys, time
import numpy as np
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

SP = sys.argv[1]; BG = sys.argv[2]; SIGMA = {"tight": 0.0265, "loose": 0.0481}[BG]
import os
STEPS = [int(x) for x in os.environ.get("STEPS", "10000,50000,100000,250000").split(",")]
LIMIT, MIN = 15, 0.6   # Neo4j scale: (1+cos)/2
RAW_MIN = 2 * MIN - 1  # = raw cosine 0.2, what graphiti's 0.6 really means
DEPTHS = (15, 50, 200)

def main_sync():
    E = np.load(f"{SP}/entity_vecs.npy"); Q = np.load(f"{SP}/pair_q.npy"); T = np.load(f"{SP}/pair_t.npy")
    import json as _j
    names = [b for _, b in _j.load(open(f'{SP}/dedup_pairs.json'))]
    uniq = sorted(set(names)); first = {n: names.index(n) for n in uniq}
    TI = np.array([uniq.index(n) for n in names])  # pair -> target node
    T = T[[first[n] for n in uniq]]
    E = E[(E @ T.T).max(1) < 0.9999]  # drop exact twins of targets: a twin under another uuid is a false miss
    rng = np.random.default_rng(7)
    return E, Q, T, TI, rng

async def insert(d, rows):
    for i in range(0, len(rows), 1000):
        await d.execute_query("UNWIND $r AS row CREATE (:Entity {uuid:row.u, group_id:'p', name_embedding:row.v})",
                              r=rows[i:i+1000])

async def run():
    E, Q, T, TI, rng = main_sync()
    own = (Q * T[TI]).sum(1) > RAW_MIN
    with Neo4jContainer("neo4j:2026.07.1-community").with_env("NEO4J_server_memory_heap_max__size", "4G") \
            .with_env("NEO4J_server_memory_pagecache_size", "4G") as neo:
        d = AsyncGraphDatabase.driver(neo.get_connection_url(), auth=("neo4j", neo.password))
        await d.execute_query("CREATE VECTOR INDEX px IF NOT EXISTS FOR (n:Entity) ON (n.name_embedding) "
                              "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'" + os.environ.get('HNSW', '') + "}}")
        r = await d.execute_query("SHOW INDEXES YIELD name, options WHERE name='px' RETURN options")
        print('index config:', r.records[0]['options']['indexConfig'], flush=True)
        base = np.vstack([E, T]).astype(np.float32)
        uu = [f"e{i}" for i in range(len(E))] + [f"t{i}" for i in range(len(T))]
        await insert(d, [{"u": u, "v": v.tolist()} for u, v in zip(uu, base)])
        allv = [base]; alluu = list(uu); have = len(base)
        print(f"hnsw={os.environ.get('HNSW','default')} background={BG} sigma={SIGMA} pairs={len(Q)} own-searchable={int(own.sum())}", flush=True)
        for n in STEPS:
            add = n - have
            if add > 0:
                seeds = E[rng.integers(0, len(E), add)]
                X = seeds + rng.normal(0, SIGMA, seeds.shape).astype(np.float32)
                X /= np.linalg.norm(X, axis=1)[:, None]
                new_uu = [f"b{have + i}" for i in range(add)]
                await insert(d, [{"u": u, "v": v.tolist()} for u, v in zip(new_uu, X)])
                allv.append(X.astype(np.float32)); alluu += new_uu; have = n
            await d.execute_query("CALL db.awaitIndexes(3600)")
            M = np.vstack(allv); t0 = time.time()
            brute_hit = np.zeros(len(Q), bool); rank = np.full(len(Q), -1)
            for i, q in enumerate(Q):
                s = M @ q; tgt = len(E) + TI[i]
                better = int((s > s[tgt]).sum())
                rank[i] = better
                brute_hit[i] = s[tgt] > RAW_MIN and better < LIMIT
            hits = {k: np.zeros(len(Q), bool) for k in DEPTHS}; lat = {k: [] for k in DEPTHS}
            for i, q in enumerate(Q):
                for k in DEPTHS:
                    t1 = time.perf_counter()
                    r = await d.execute_query("CALL db.index.vector.queryNodes('px', $k, $v) YIELD node, score "
                                              "WHERE score > $min RETURN node.uuid AS u ORDER BY score DESC LIMIT $lim",
                                              k=k, v=q.tolist(), min=MIN, lim=LIMIT)
                    lat[k].append((time.perf_counter() - t1) * 1000)
                    hits[k][i] = f"t{TI[i]}" in {x["u"] for x in r.records}
            b = brute_hit & own
            line = (f"N={have:>7,} brute-finds {b.sum()}/{own.sum()} "
                    f"(median rank of partner among own-searchable {int(np.median(rank[own]))})")
            for k in DEPTHS:
                cond = hits[k][b].mean() if b.any() else float('nan')
                line += f" | k={k}: ANN finds {int((hits[k] & b).sum())}/{b.sum()} = {cond:.3f} ({np.median(lat[k]):.0f} ms)"
            print(line, flush=True)
        await d.close()

asyncio.run(run())
