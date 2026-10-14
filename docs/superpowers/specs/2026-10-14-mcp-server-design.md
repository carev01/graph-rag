# MCP server for answer-api (LAN slice of Phase 5) — design

**Status:** approved in conversation 2026-10-14 (sections 1–3); this document records it.
**Goal:** the user can query the graph from Claude Code / Claude Desktop on the home LAN
through an MCP server, and judge answer quality before more money goes into ingestion.
Copilot Studio, public exposure and OAuth are later slices; this one must not block them.

## 1. Architecture

- A new module `src/answer_api/mcp_server.py` builds an MCP server with the official MCP
  Python SDK (`mcp`, 2.x) and mounts its **streamable-HTTP** app at **`/mcp`** on the
  existing answer-api FastAPI app. The SDK's session manager is started and stopped inside
  the app's lifespan.
- The bodies of the `/answer` and `/search/local` routes move into plain async functions
  (`run_answer(app, ...)`, `run_search_local(app, ...)`) that both the REST routes and the
  MCP tools call. REST behaviour and responses are unchanged.
- Tools reuse `app.state` (graphiti, Neo4j driver, embedder, LLM clients, scope resolver):
  no second set of clients, no extra network hop.

## 2. Tools

### 2.1 `answer_question`
Inputs: `question` (string, required); `vendors`, `products` (lists of names, optional);
`mode` (`auto` | `local` | `global` | `drift` | `timeline`, default `auto` = the router).
Runs exactly what `GET /answer` runs (scope resolution, question shape, router, fallbacks).

Output:
- **Text content:** the answer as written (with its `[N]` markers), then a `Sources` list,
  one line per cited marker: `[N] <title> — <vendor> · <product> — <url>`. URLs come only
  from the deterministic citation resolver (design decision #2: the LLM never writes a
  URL). Notes the router already emits (degraded sides, "Could not check X", the
  enumeration "Checked / Documented for" line) are part of the answer text and pass through.
- **Structured content:** `{mode, scope, shape, facets, applies_to, citations}` from the
  router envelope, for clients that use it.

Latency: comparisons and enumerations take 1–7 minutes. While the call runs, the tool
sends an MCP **progress notification every ~10 s** so clients that honour progress do not
time out; the setup notes document raising the client's tool timeout (`MCP_TOOL_TIMEOUT`
for Claude Code).

### 2.2 `search_facts`
Inputs: `query` (required); `vendors`, `products` (optional); `k` (default 10, max 25);
`include_superseded` (default false). Runs what `GET /search/local` runs. No LLM call.
Output: text listing each fact with its valid-from / valid-until dates and source
article(s) (title, vendor · product, URL); structured content = the facts list.

### 2.3 `list_vendors_products`
No inputs. Output: the vendors that have extracted content, each with its products and
article counts (from the structural layer, the same catalog the scope resolver loads), so
the user or the client model can discover valid filter names.

### 2.4 Tool descriptions
Each description says when to use the tool: `answer_question` for any question (it picks
the retrieval mode itself; can take minutes); `search_facts` to inspect raw evidence;
`list_vendors_products` to find filter names. All instruct the client to show sources.

## 3. Auth

- `/mcp` requires `Authorization: Bearer <token>`. The token is the new setting
  `MCP_BEARER_TOKEN`, stored only in the cluster Secret (never in the repo, image or chart
  values). Comparison is constant-time. A missing or wrong token returns **401** with
  `WWW-Authenticate: Bearer`.
- **Fail closed:** with `MCP_BEARER_TOKEN` unset, `/mcp` is not mounted (404).
- REST routes are unchanged (LAN-only, unauthenticated); BACKLOG 47 stays open for them.

## 4. Transport and deployment

- **Stateless** streamable HTTP (no server-side session): each call is self-contained, so
  a pod restart or a second replica loses nothing. Progress notifications stream back on
  the call's response. DNS-rebinding protection (Origin/Host validation) is enabled with
  the cluster host name allowed.
- Same host and ingress as today: `https://graphrag.k3s.home.lan/mcp`, internal CA. The
  plan verifies Traefik neither buffers nor times out a long streamed response on that
  route, and raises the route's timeouts if it does.
- Helm: one new Secret key and env var. No other structural chart change.
- User setup (documented in `docs/deploy/k3s.md` or a new `docs/deploy/mcp.md`):
  `claude mcp add --transport http graph-rag https://graphrag.k3s.home.lan/mcp --header
  "Authorization: Bearer <token>"`, trusting the internal CA (`NODE_EXTRA_CA_CERTS` for
  Claude Code; the equivalent for Desktop), and the tool-timeout setting.

## 5. Errors

A failure never reads as an answer.
- Unknown vendor/product names: an MCP tool error (`isError`) naming the unknown names and
  listing the valid ones (the REST 422's content).
- Backend failures (Neo4j, an LLM provider): a tool error saying what failed; logged with
  the cause.
- Client cancellation propagates and stops the underlying work.
- A refusal or a partial answer is a result, not an error.
- No exception is swallowed: each is converted to a tool error or re-raised, and logged.

## 6. Testing

Free tests (no LLM spend):
- Unit: each tool against the app with fake `app.state` backends, through a real MCP client
  session over the ASGI app in-process — tool listing (names, descriptions, input schemas),
  text and structured output, the Sources list built only from citations, errors and the
  unknown-name message, progress notifications on a slow fake answer.
- Auth: no token → 401, wrong token → 401, right token → 200, token unset → 404; REST routes
  unaffected.
- REST regression: the existing `/answer` and `/search/local` tests pass unchanged after the
  handler extraction.
Live (after deploy): list tools, `list_vendors_products` and `search_facts` (free), then one
`answer_question` (~1¢) to confirm progress notifications survive the ingress on a long call.

## 7. Out of scope

Public exposure (tunnel / APIM), OAuth, Copilot Studio onboarding, per-user auth, separate
per-mode tools, answer caching.
