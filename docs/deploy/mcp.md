# MCP server (answer-api /mcp) — LAN setup

answer-api serves an MCP endpoint at `https://graphrag.k3s.home.lan/mcp` (streamable
HTTP, stateless) when `MCP_BEARER_TOKEN` is set in `graph-rag-secret`. Without it, /mcp
is not served. Tools: `answer_question` (the /answer router; may take minutes),
`search_facts` (raw cited facts, free), `list_vendors_products` (catalog, free).

## 1. Add the token to the Secret (operator)

Generate and add it without echoing it (it never touches disk or the repo):

```bash
TOKEN=$(openssl rand -hex 32)
kubectl -n graph-rag get secret graph-rag-secret -o json \
  | T="$TOKEN" python3 -c 'import json,sys,base64,os; d=json.load(sys.stdin); d["data"]["MCP_BEARER_TOKEN"]=base64.b64encode(os.environ["T"].encode()).decode(); json.dump(d,sys.stdout)' \
  | kubectl apply -f -
kubectl -n graph-rag rollout restart deploy/graph-rag-answer
unset TOKEN
```

Read it back when configuring a client:
`kubectl -n graph-rag get secret graph-rag-secret -o jsonpath='{.data.MCP_BEARER_TOKEN}' | base64 -d`

## 2. Trust the internal CA

The endpoint uses the cluster's internal CA (`rootca.home.lan`). For Claude Code:
`export NODE_EXTRA_CA_CERTS=/path/to/rootca.pem` before starting `claude`.

## 3. Add the server to Claude Code

```bash
claude mcp add --transport http graph-rag https://graphrag.k3s.home.lan/mcp \
  --header "Authorization: Bearer <token>"
```

Comparisons and "which vendors" questions take 1-7 minutes. The server sends progress
notifications every 10 s; if your client still times out, raise its tool timeout
(Claude Code: `export MCP_TOOL_TIMEOUT=600000`, in ms).

## 4. Cost

`search_facts` and `list_vendors_products` make no LLM call. Each `answer_question` costs
what one /answer call costs (roughly $0.005-0.02; more for comparisons and enumerations).

## 5. Rotate or disable

Rotate: repeat step 1 and update clients. Disable: remove the key from the Secret and
restart the deployment — /mcp then returns 404.
