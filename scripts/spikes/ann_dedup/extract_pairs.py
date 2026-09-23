"""Step 1: real node-dedup duplicate pairs from the LLM capture dataset.

Usage: python extract_pairs.py OUT_DIR   (reads data/ft-captured/{train,val}.jsonl)
Writes OUT_DIR/dedup_pairs.json: [extracted_name, existing_name] pairs the model
judged duplicates, identical names excluded (graphiti resolves those before the LLM).
"""
import json
import re
import sys

out = sys.argv[1]
pairs = set()
for f in ["data/ft-captured/train.jsonl", "data/ft-captured/val.jsonl"]:
    for line in open(f):
        r = json.loads(line)
        if r["prompt_name"] != "dedupe_nodes.nodes":
            continue
        user = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        ans = json.loads([m for m in r["messages"] if m["role"] == "assistant"][0]["content"])
        ents = re.search(r"<ENTITIES>\s*(\[.*?\])\s*</ENTITIES>", user, re.S)
        exi = re.search(r"<EXISTING ENTITIES>\s*(\[.*?\])\s*</EXISTING ENTITIES>", user, re.S)
        if not (ents and exi):
            continue
        E = json.loads(ents.group(1))
        X = {c["candidate_id"]: c["name"] for c in json.loads(exi.group(1))}
        names = {e.get("id", i): e["name"] for i, e in enumerate(E)}
        for res in ans["entity_resolutions"]:
            d = res["duplicate_candidate_id"]
            if d != -1 and res["id"] in names and d in X and names[res["id"]] != X[d]:
                pairs.add((names[res["id"]], X[d]))
json.dump(sorted(pairs), open(f"{out}/dedup_pairs.json", "w"))
print(len(pairs), "pairs")
