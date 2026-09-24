import asyncio, json, sys
from neo4j import AsyncGraphDatabase, RoutingControl
from graph_extract.config import get_extract_settings
Q = """
MATCH (a:Entity)-[v:RELATES_TO {group_id:$g}]->(b:Entity) WHERE v.invalid_at IS NOT NULL
OPTIONAL MATCH (a)-[o:RELATES_TO {group_id:$g}]-(b) WHERE o.uuid <> v.uuid
WITH a, b, v, collect({fact:o.fact, name:o.name, created:toString(o.created_at), valid:toString(o.valid_at),
     invalid:toString(o.invalid_at), eps:o.episodes}) AS others
OPTIONAL MATCH (art:Article)-[:HAS_EPISODE]->(e:Episodic) WHERE e.uuid IN v.episodes
RETURN a.name AS src, b.name AS tgt, v.name AS rel, v.fact AS fact, toString(v.valid_at) AS valid,
       toString(v.invalid_at) AS invalid, toString(v.expired_at) AS expired, toString(v.created_at) AS created,
       v.episodes AS eps, collect(DISTINCT art.title) AS victim_articles, others
"""
async def main():
    s = get_extract_settings()
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    r = await d.execute_query(Q, g=s.group_id, routing_=RoutingControl.READ)
    rows = [dict(x) for x in r.records]
    # article titles for the others' episodes
    for row in rows:
        for o in row["others"]:
            if o["eps"]:
                rr = await d.execute_query("MATCH (art:Article)-[:HAS_EPISODE]->(e:Episodic) WHERE e.uuid IN $u RETURN collect(DISTINCT art.title) AS t", u=o["eps"], routing_=RoutingControl.READ)
                o["articles"] = rr.records[0]["t"]
            o.pop("eps", None)
        row.pop("eps", None)
    json.dump(rows, open(sys.argv[1] + "/d4_rows.json", "w"), indent=1, default=str)
    print(len(rows), "invalidated facts")
    await d.close()
asyncio.run(main())
