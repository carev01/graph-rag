import asyncio, numpy as np
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer
async def run():
    rng=np.random.default_rng(1); X=rng.normal(size=(50,768)).astype(np.float32); X/=np.linalg.norm(X,axis=1)[:,None]
    with Neo4jContainer("neo4j:2026.07.1-community") as neo:
        d=AsyncGraphDatabase.driver(neo.get_connection_url(), auth=("neo4j", neo.password))
        await d.execute_query("CREATE VECTOR INDEX px FOR (n:E) ON (n.v) OPTIONS {indexConfig:{`vector.dimensions`:768,`vector.similarity_function`:'cosine'}}")
        await d.execute_query("UNWIND $r AS r CREATE (:E {i:r.i, v:r.v})", r=[{"i":i,"v":x.tolist()} for i,x in enumerate(X)])
        await d.execute_query("CALL db.awaitIndexes(60)")
        r=await d.execute_query("CALL db.index.vector.queryNodes('px',3,$q) YIELD node,score RETURN node.i AS i, score, vector.similarity.cosine(node.v,$q) AS fn", q=X[0].tolist())
        for x in r.records: print(x["i"], "index score", round(x["score"],4), "vector.similarity.cosine", round(x["fn"],4), "numpy", round(float(X[0]@X[x["i"]]),4))
        await d.close()
asyncio.run(run())
