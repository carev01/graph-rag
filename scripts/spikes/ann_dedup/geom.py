import asyncio, json, random, sys
import numpy as np
from neo4j import AsyncGraphDatabase, RoutingControl
from graph_extract.config import get_extract_settings
sys.path.insert(0, "scripts")
from vector_crossover import build_pool, resample_one

async def main():
    s = get_extract_settings()
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    r = await d.execute_query("MATCH (n:Entity {group_id:$g}) WHERE n.name_embedding IS NOT NULL "
                              "RETURN n.name AS name, n.name_embedding AS v", g=s.group_id,
                              routing_=RoutingControl.READ)
    await d.close()
    names = [x["name"] for x in r.records]; V = np.array([x["v"] for x in r.records], dtype=np.float32)
    np.save(sys.argv[1] + "/entity_vecs.npy", V); json.dump(names, open(sys.argv[1] + "/entity_names.json", "w"))
    norms = np.linalg.norm(V, axis=1); print("entities", len(V), "norm min/max", norms.min(), norms.max())
    U = V / norms[:, None]; S = U @ U.T; np.fill_diagonal(S, -1)
    nn = S.max(1); print("real NN cosine  p10/p50/p90", np.percentile(nn, [10, 50, 90]).round(3))
    print("real random-pair cosine median", np.median(S[np.triu_indices(len(U), 1)]).round(3))
    rng = random.Random(0); pool = build_pool(10, 768, rng, seeds=[list(u) for u in U])
    c = [float(np.dot(U[i], resample_one(type(pool)([list(U[i])], pool.provenance, False), rng)))
         for i in range(200)]
    print("cos(seed, resample_one(sigma=0.15)) median", round(float(np.median(c)), 3))
asyncio.run(main())
