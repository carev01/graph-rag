# graph-rag on srv-k3s — cut-over runbook

Companion to `docs/superpowers/specs/2026-09-24-k3s-deployment-design.md` and the
chart at `deploy/helm/graph-rag`. Run each step in order and verify it before moving
to the next — nothing here is safe to run out of sequence.

**Never** paste `.env`, a secret file, or a real credential into a chat, ticket,
commit, or this document. Every credential below is either generated at runtime
into a shell variable that is never echoed (`$pw`, `$newpw`) or a placeholder
you fill in (`<N>`, `<revision>`, `<private-dir>`, `<new-size>`, `<pod>`, `<tries>`,
`sha-abc1234`).

## 1. Prerequisites

- `kubectl` context pointed at `srv-k3s.home.lan`.
- `helm` (v4 on this host; `helm version` to confirm).
- The image built by the `image` GitHub Actions workflow (`.github/workflows/image.yml`):
  `ghcr.io/carev01/graph-rag:sha-abc1234` (7-char short SHA of the commit on `main`).
  The chart enforces the format: `image.tag` must match `^sha-[0-9a-f]{7}$` (lowercase
  hex) or `helm upgrade`/`helm template` fails fast with a clear message — a bare commit
  SHA, an uppercase hex digit, or a moving tag like `:main` is rejected before anything
  is applied.

Verify the image pulls anonymously — the repo and its GHCR package are public and the
image holds no secret:

```
docker logout ghcr.io
docker pull ghcr.io/carev01/graph-rag:sha-abc1234
```

If that fails with an auth error, the package defaulted to private: set its visibility
to public in GitHub → the repo → Packages → `graph-rag` → Package settings → Change
visibility.

### Exposure — confirm before installing

The chart's Ingress (`graphrag.k3s.home.lan` → `graph-rag-answer`) has **no
authentication and no allow-list**, and it rides on Traefik's LoadBalancer, which this
cluster shares with a host that *is* published to the internet. The answer API is
therefore only as private as the cluster IP's ports 80/443. Before step 4, confirm from
**outside** the home network (e.g. a phone off Wi-Fi, or an external port checker) that
80 and 443 on the cluster's public-facing address do not route to Traefik for
`graphrag.k3s.home.lan` — a request with that `Host` header must not return the API. If
it does, stop: do not install until that is closed at the router/firewall. A Traefik
`ipAllowList` middleware is deliberately **not** relied on: behind k3s ServiceLB the
client source IP can be SNATed to a node address, so the allow-list may see every client
as "internal". Public exposure and auth are Phase F (BACKLOG item 47).

### Helm values on upgrade

Every `helm upgrade` below uses `--reuse-values`: it keeps the release's stored values
(the `--set`s from earlier steps) and merges in the new `--set`s. It does **not** pick
up keys that a *newer* chart version added to `values.yaml` — those render as empty.
When upgrading to a chart version whose `values.yaml` changed, use Helm 4's
`--reset-then-reuse-values` instead (chart defaults first, then the stored values, then
the command line), and re-check `helm get values graph-rag -n graph-rag` afterwards.

## 2. Namespace

```
kubectl create namespace graph-rag
kubectl label namespace graph-rag stack.stage=production
```

The label enrols the namespace in Kasten's daily `production-apps-backup` policy
(backup + export) — the same convention as `docextractor`.

## 3. Secret, from `.env`, never committed

The chart reads no credential itself (`values.yaml` holds none); everything lives in
one secret, `graph-rag-secret`, built once from the dev host's `.env`. Run this from
the repo root, so `.env` resolves to the project's own file.

First confirm what `.env` already carries — list key **names only**, never values:

```
grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env
```

`.env` already defines its own `POSTGRES_DSN` (the dev host's, pointed at the
Docker-Compose Postgres) — that line and any `POSTGRES_PASSWORD` must be dropped
before this cluster's own values are appended, or
`kubectl create secret --from-env-file` fails on the duplicate key. Do the copy,
strip, and rewrite in one subshell so `umask 077` (and the `$tmpfile` variable)
don't leak into the rest of the session:

```
( umask 077
  tmpfile=$(mktemp /tmp/graph-rag-secret.XXXXXX.env)
  cp .env "$tmpfile"

  # strip surrounding quotes from every value: --from-env-file keeps quotes
  # literally, which breaks anything that parses the value later (DSNs, URLs).
  sed -i -E 's/^([A-Za-z_][A-Za-z0-9_]*)="?([^"]*)"?$/\1=\2/' "$tmpfile"

  # drop the dev host's own POSTGRES_DSN/POSTGRES_PASSWORD before appending
  # this cluster's own values below.
  sed -i '/^POSTGRES_DSN=/d; /^POSTGRES_PASSWORD=/d' "$tmpfile"

  # keep the DocExtractor ADMIN key out of every production pod: only
  # `graph_sync.cli register-webhook` uses it, and nothing in the cluster runs that.
  sed -i '/^DOCEXT_ADMIN_KEY=/d' "$tmpfile"

  # a fresh Postgres password for the in-cluster StatefulSet, and the DSN that
  # points at it. graphsync/graphsync are the chart's postgres.user /
  # postgres.database defaults -- see deploy/helm/graph-rag/values.yaml.
  pw=$(openssl rand -hex 24)
  echo "POSTGRES_PASSWORD=${pw}" >> "$tmpfile"
  echo "POSTGRES_DSN=postgresql://graphsync:${pw}@graph-rag-postgres:5432/graphsync" >> "$tmpfile"

  kubectl -n graph-rag create secret generic graph-rag-secret --from-env-file="$tmpfile"
  shred -u "$tmpfile"
)
```

Never paste `.env`, `$tmpfile`'s contents, or the secret's contents into a chat,
ticket, or commit.

Because `DOCEXT_ADMIN_KEY` is not in the secret, `register-webhook` cannot run in the
cluster; if it is ever needed, run it from a trusted host that holds the admin key
(see step 13).

**After changing `graph-rag-secret`** (rotation, or updating a key), every pod that
reads it must be restarted — `envFrom` is only read at container start, and the
chart's `checksum/config` pod-template annotation covers the ConfigMap, not this
out-of-band secret, so a plain `kubectl apply`/`helm upgrade` with no other change
will not roll these pods on its own:

```
kubectl -n graph-rag rollout restart deploy/graph-rag-sync deploy/graph-rag-answer deploy/graph-rag-worker
```

**Rotating `POSTGRES_PASSWORD` specifically is not that alone.** The postgres image
only applies `POSTGRES_PASSWORD` once, at `initdb` (first cluster init on an empty
`PGDATA`); once the StatefulSet's PVC already holds a database, changing the secret
and restarting the `postgres` pod does **not** change the live role's password —
the running database keeps the old one. Rotate the actual role first.

Generate the new password the same way the initial one was generated
(`openssl rand -hex 24`), not an arbitrary string: it gets embedded directly in
`POSTGRES_DSN` as a URL, and a DSN-unsafe character in it (`@`, `/`, `:`, `#`, a
literal space) would silently corrupt the connection string unless percent-encoded.
Hex has none of those:

```
openssl rand -hex 24
```

Copy that value; you will enter it twice below, never as a command-line argument,
so it is never echoed, logged, or captured in shell history or in `kubectl`'s own
argument list:

```
kubectl -n graph-rag exec -it graph-rag-postgres-0 -- psql -U graphsync -d graphsync
```

At the `graphsync=#` prompt, `\password` prompts twice and hides the input:

```
\password graphsync
```

Then update the secret's `POSTGRES_PASSWORD` and `POSTGRES_DSN` (the DSN embeds the
same password) to match, entering the same value once more via a hidden prompt
rather than a command-line argument, and merge it into the existing secret (so
every other key is left untouched):

```
read -rs -p "new POSTGRES_PASSWORD (same value just set with \password): " newpw; echo
kubectl -n graph-rag get secret graph-rag-secret -o json \
  | jq --arg pw "$newpw" '
      .data.POSTGRES_PASSWORD = ($pw | @base64)
      | .data.POSTGRES_DSN = ("postgresql://graphsync:" + $pw + "@graph-rag-postgres:5432/graphsync" | @base64)
    ' \
  | kubectl apply -f -
unset newpw
kubectl -n graph-rag rollout restart deploy/graph-rag-sync deploy/graph-rag-answer deploy/graph-rag-worker
```

The `postgres` StatefulSet pod itself does not need restarting for this — its
running `postgres` process already has the new password from the `\password` step,
and the secret is now merely consistent with it for the next time `postgres`
actually reads `POSTGRES_PASSWORD` (a future `initdb` against a fresh PVC).

## 4. Install, quiesced

Nothing may spend tokens, poll DocExtractor, or write to Neo4j or Postgres until
connectivity is proven and state is migrated. Install with **every** writer at zero or
suspended:

- `sync.replicas=0` — the poller (it runs `init_schema` on Neo4j and Postgres at start);
- `answer.replicas=0` — the answer API writes to Neo4j at startup
  (`ensure_vector_indexes`), and it would only become Ready *after* connecting, so it
  cannot host the connectivity check either;
- `cron.jobs.cleanup.suspend=true`, `cron.jobs.maintenance.suspend=true` — both write
  to Neo4j on a schedule;
- workers already default to 0 and `theme-build` is suspended by default in
  `values.yaml`.

```
helm upgrade --install graph-rag deploy/helm/graph-rag -n graph-rag \
  --set image.tag=sha-abc1234 \
  --set sync.replicas=0 \
  --set answer.replicas=0 \
  --set cron.jobs.cleanup.suspend=true \
  --set cron.jobs.maintenance.suspend=true
```

Only the Postgres StatefulSet starts. Wait for it (bounded — a PVC that cannot be
provisioned fails here instead of hanging):

```
kubectl -n graph-rag rollout status statefulset/graph-rag-postgres --timeout=300s
```

## 5. Connectivity

### Running one-off pods (used here and in steps 7–9)

Every one-off command in this runbook runs as its own **Pod** with
`--restart=Never`: it runs once, reaches `Succeeded` (or `Failed`) and stays there —
no restart loop, and nothing borrowed from a Deployment's pod (whose memory limit and
lifecycle belong to its own process). The pod gets the same environment and security
posture as the chart's containers (`envFrom`: the secret, then the ConfigMap, in that
order; the same pod/container `securityContext`; `automountServiceAccountToken:
false`) and explicit resources, all via `--overrides`.

Three rules, each learned the hard way:

- **All `kubectl` flags must come before `--`; everything after `--` is the
  container's command**, passed through verbatim and never parsed by `kubectl`.
  `--dry-run=client -o yaml` placed *after* `--command --` is silently swallowed into
  the container's argv and `kubectl` performs a real `create`.
- **The image tag and command appear in both the flags and the override.**
  `--overrides` is a JSON *merge patch*, which replaces the whole `containers` array:
  anything the override's container omits (image, command, resources) comes out
  **missing**, not inherited from `--image`/`--command`. Keep the two in sync.
- **Validate client-side first, as its own complete command.** `--dry-run=client`
  may issue a read-only API-discovery `GET`, but it cannot issue the `POST` a create
  needs. Read the rendered YAML — `envFrom` order, `restartPolicy: Never`, both
  `securityContext` blocks, `resources`, and `image:` being the tag you meant — then
  run the identical command without `--dry-run=client -o yaml`.

Each pod is followed by the same bounded wait, which must end in `phase=Succeeded`
(`<pod>` and `<tries>` are given with each use; one try is 5 s):

```
pod=<pod>
phase=
for i in $(seq 1 <tries>); do
  phase=$(kubectl -n graph-rag get pod "$pod" -o jsonpath='{.status.phase}')
  case "$phase" in Succeeded|Failed) break ;; esac
  sleep 5
done
echo "phase=$phase"
kubectl -n graph-rag logs "$pod"
```

### The check

Read-only and free (no LLM tokens, no writes). Build the override once in a shell
variable (a quoted heredoc: nothing inside it is expanded):

```
ovr=$(cat <<'JSON'
{
  "apiVersion": "v1",
  "spec": {
    "automountServiceAccountToken": false,
    "securityContext": {"runAsNonRoot": true, "runAsUser": 10001, "runAsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"}},
    "containers": [{
      "name": "graph-rag-connectivity",
      "image": "ghcr.io/carev01/graph-rag:sha-abc1234",
      "command": ["python", "-m", "graph_sync.connectivity"],
      "envFrom": [{"secretRef": {"name": "graph-rag-secret"}},
                  {"configMapRef": {"name": "graph-rag-config"}}],
      "resources": {"requests": {"cpu": "50m", "memory": "256Mi"},
                    "limits": {"cpu": "500m", "memory": "512Mi"}},
      "securityContext": {"allowPrivilegeEscalation": false, "capabilities": {"drop": ["ALL"]}}
    }]
  }
}
JSON
)
```

Validate (creates nothing):

```
kubectl -n graph-rag run graph-rag-connectivity --restart=Never \
  --dry-run=client -o yaml \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.connectivity
```

Then the real run:

```
kubectl -n graph-rag run graph-rag-connectivity --restart=Never \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.connectivity
```

Run the wait loop above with `pod=graph-rag-connectivity` and `<tries>` = 24, read the
output, then clean up:

```
kubectl -n graph-rag delete pod graph-rag-connectivity
```

The pod must end `phase=Succeeded` and every line must read `OK`. A `FAIL` line names
the dependency and a redacted error — no credential is ever printed, including inside a
partially-quoted DSN. `docextractor` requires HTTP 200 from `/api/health`, and
`docext-auth` makes one **authenticated** read (`GET /api/articles?limit=1`, key in the
`X-API-Key` header) that must also return 200 — so a revoked or rotated read key fails
here, not on the first poll. The optional tiers (`judge`, `report`, `map`, `rerank`,
`eval-judge`, `verify`) are probed only when their `*_base_url` is non-empty; unset
reports `OK  <tier>  not configured`.

### Start the answer API

Only now that Neo4j is proven reachable with the right credentials, let the answer
API start (its startup verifies/creates the vector indexes — an idempotent write):

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set answer.replicas=1
kubectl -n graph-rag rollout status deploy/graph-rag-answer --timeout=300s
```

(300 s covers the startup probe's 180 s budget plus an image pull.)

## 6. Migrate the state

**Stop anything on the dev host that could still write first.** Today nothing does
— the services run from a shell, not as a daemon — but confirm it before dumping,
since a concurrent writer would make the dump/restore comparison in this step
meaningless:

```
pgrep -fl 'graph_sync|graph_extract|theme_builder|uvicorn' || echo "nothing running"
```

On the dev host, the stopped `graphrag-postgres` container holds the sync cursor,
bootstrap progress, and token ledger built so far. Start it and wait for it to
actually accept connections — `docker start` returns as soon as the container
process launches, well before Postgres has finished recovery:

```
docker start graphrag-postgres
ready=no
for i in $(seq 1 60); do
  if docker exec graphrag-postgres pg_isready -q; then ready=yes; break; fi
  sleep 1
done
echo "ready=$ready"
```

`ready=no` after 60 s means Postgres is not recovering normally — stop and look at
`docker logs graphrag-postgres`; do not dump a database that is not accepting
connections.

Dump using the container's own `POSTGRES_USER`/`POSTGRES_DB` env vars rather than a
hardcoded user/db name. Three independent checks must all pass before going any further:
`pg_dump` exited 0 (a failure mid-stream still leaves a non-empty, truncated file), the
file is non-empty, and `pg_restore -l` can read the archive's table of contents (run
inside the container, so the dev host needs no Postgres client tools; the dump is fed
on stdin):

```
dumpfile=<private-dir>/graphsync.dump
: "${dumpfile:?set dumpfile first}"
if docker exec graphrag-postgres sh -c 'pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB"' > "$dumpfile"
then echo "pg_dump: exit 0"
else echo "pg_dump FAILED -- do not proceed past this point" >&2
fi
test -s "$dumpfile" || echo "dump is EMPTY -- do not proceed past this point" >&2
if docker exec -i graphrag-postgres pg_restore -l < "$dumpfile" > /dev/null
then echo "pg_restore -l: archive readable"
else echo "dump archive UNREADABLE -- do not proceed past this point" >&2
fi
```

Record every state table's contents to compare against after restore — all seven
tables `src/graph_sync/state_store.py`'s `CREATE TABLE` statements define
(`sync_cursor`, `bootstrap_progress`, `webhook_delivery`, `source_debounce`,
`dead_letter`, `semantic_jobs`, `token_ledger`). A row *count* is enough for the
five append-only/queue tables, but not for `sync_cursor` (always exactly one row,
by its own `CHECK (id = 1)`) or `bootstrap_progress` (a fixed row per shard whose
`watermark`/`last_id`/`status` still change) — hash their actual values instead.
`semantic_jobs`'s `GROUP BY` output has no defined row order, so it also gets an
explicit `ORDER BY` to make the two runs byte-for-byte comparable; the
`bootstrap_progress` hash uses `row(...)::text` rather than `||`-concatenation so a
NULL `last_id` and a boundary shift between columns (e.g. `shard='ab', last_id='c'`
vs. `shard='a', last_id='bc'`) can't produce the same hash by coincidence:

```
docker exec -i graphrag-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT lane, status, count(*) FROM semantic_jobs GROUP BY 1, 2 ORDER BY 1, 2;
SELECT count(*) FROM webhook_delivery;
SELECT count(*) FROM source_debounce;
SELECT count(*) FROM dead_letter;
SELECT count(*) FROM token_ledger;
SELECT md5(cursor) FROM sync_cursor;
SELECT md5(string_agg(row(shard, watermark, last_id, status)::text, ',' ORDER BY shard)) FROM bootstrap_progress;
SQL
docker stop graphrag-postgres
```

Copy the dump into the pod and restore it as a single all-or-nothing operation —
stop on the first error rather than leaving a half-restored database:

```
: "${dumpfile:?set dumpfile first}"
kubectl -n graph-rag cp "$dumpfile" graph-rag-postgres-0:/tmp/graphsync.dump
kubectl -n graph-rag exec graph-rag-postgres-0 -- \
  sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-acl --exit-on-error --single-transaction /tmp/graphsync.dump'
```

Re-run the exact same seven queries against the pod and compare every line to the
dev-host output — they must match exactly:

```
kubectl -n graph-rag exec -i graph-rag-postgres-0 -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT lane, status, count(*) FROM semantic_jobs GROUP BY 1, 2 ORDER BY 1, 2;
SELECT count(*) FROM webhook_delivery;
SELECT count(*) FROM source_debounce;
SELECT count(*) FROM dead_letter;
SELECT count(*) FROM token_ledger;
SELECT md5(cursor) FROM sync_cursor;
SELECT md5(string_agg(row(shard, watermark, last_id, status)::text, ',' ORDER BY shard)) FROM bootstrap_progress;
SQL
```

Then delete the dump on both sides:

```
: "${dumpfile:?set dumpfile first}"
kubectl -n graph-rag exec graph-rag-postgres-0 -- rm /tmp/graphsync.dump
rm "$dumpfile"
```


## 7. Verify the migrated state against Neo4j (free, read-only)

The restore in step 6 proves the cluster's Postgres equals the dev host's. It does not
prove that state describes the **Neo4j graph the cluster's secret points at** — if it
does not (a different instance, a graph that was lost or rebuilt since the state was
written), the poller and workers would act on a false picture of what is already
applied. Both checks below are reads; nothing is running yet that could write.

### 7a. The cursor migrated

```
kubectl -n graph-rag exec -i graph-rag-postgres-0 -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tA' <<'SQL'
SELECT coalesce(bool_and(cursor IS NOT NULL AND cursor <> ''), false) FROM sync_cursor;
SQL
```

It must print `t`. (This is `SELECT cursor IS NOT NULL FROM sync_cursor`, hardened so
that an empty table — zero rows — and an empty string both print `f` rather than
nothing.) **Why this gates everything after it:** with no cursor, the incremental pull
calls `GET /api/articles/delta` with no `since`, which DocExtractor serves as a
**full-corpus snapshot** (`CLIENT-USAGE-GUIDE.md` §6). `SyncCore` enqueues every article
not already in Neo4j with the same content hash — for this corpus, nearly all ~105k —
into the **`incremental`** lane, and the worker's claim query
(`StateStore.claim_semantic_jobs`: `AND ($2 OR lane='incremental')`) takes the
incremental lane **regardless of the daily token budget**; only the `bootstrap` lane is
budget-gated. An empty cursor therefore turns the first poll into an unmetered full
bootstrap. On `f`: stop, do not start the poller, and re-check the dump/restore.

### 7b. Postgres state versus the graph

`python -m graph_sync.state_crosscheck` (in the image) reads, inside a `READ ONLY`
Postgres transaction and a READ-access Neo4j session:

- every `bootstrap_progress` row with status `complete`, and counts the `:Article`
  nodes for its shard. A shard is what `SyncCore.bootstrap` keyed it by
  (`source_id or vendor_id or "global"`), so it is matched as a Source id
  (`Article.source_id`), else a Vendor id (`Vendor→Product→Source→Article`), else
  `global` (all Articles). A complete shard with **0** Articles fails;
- a random sample (`--sample`, default 50) of `semantic_jobs` with status `done` and op
  `upsert`, and checks each article exists and has at least one `HAS_EPISODE` edge.
  Navigation pages (`is_navigation_article`: release notes, "what's new", link farms)
  are completed without extraction by design and are reported separately, not failed.

It prints only shard/article ids and counts. Override and dry-run:

```
ovr=$(cat <<'JSON'
{
  "apiVersion": "v1",
  "spec": {
    "automountServiceAccountToken": false,
    "securityContext": {"runAsNonRoot": true, "runAsUser": 10001, "runAsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"}},
    "containers": [{
      "name": "graph-rag-crosscheck",
      "image": "ghcr.io/carev01/graph-rag:sha-abc1234",
      "command": ["python", "-m", "graph_sync.state_crosscheck", "--sample", "50"],
      "envFrom": [{"secretRef": {"name": "graph-rag-secret"}},
                  {"configMapRef": {"name": "graph-rag-config"}}],
      "resources": {"requests": {"cpu": "50m", "memory": "256Mi"},
                    "limits": {"cpu": "500m", "memory": "512Mi"}},
      "securityContext": {"allowPrivilegeEscalation": false, "capabilities": {"drop": ["ALL"]}}
    }]
  }
}
JSON
)
```

```
kubectl -n graph-rag run graph-rag-crosscheck --restart=Never \
  --dry-run=client -o yaml \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.state_crosscheck --sample 50
```

Then the real run (same command without `--dry-run=client -o yaml`), the step-5 wait
with `pod=graph-rag-crosscheck` and `<tries>` = 36, and
`kubectl -n graph-rag delete pod graph-rag-crosscheck`.

```
kubectl -n graph-rag run graph-rag-crosscheck --restart=Never \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.state_crosscheck --sample 50
```

It must end `phase=Succeeded` with every line `OK`. **On any `FAIL`: stop. Do not start
the poller or any worker.** The state and the graph disagree — most likely the secret's
`NEO4J_URI` is not the instance the dev-host state was built against (the dev graph has
already been lost once when the Neo4j instance moved), or the wrong dump was restored.
Nothing needs undoing: the only thing running is the read-only answer API. Resolve the
mismatch (right instance, or deliberately reset the state for a fresh bootstrap — a
separate, reviewed decision) before continuing.

## 8. First pull, then start the poller

Record the queue by lane before the first pull:

```
kubectl -n graph-rag exec -i graph-rag-postgres-0 -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT lane, status, count(*) FROM semantic_jobs GROUP BY 1, 2 ORDER BY 1, 2;
SQL
```

Pull once, as a one-off pod (step 5's pattern; worker-sized resources, since a large
delta is streamed and mapped in-process). The poller Deployment stays at 0, so this is
the only writer:

```
ovr=$(cat <<'JSON'
{
  "apiVersion": "v1",
  "spec": {
    "automountServiceAccountToken": false,
    "securityContext": {"runAsNonRoot": true, "runAsUser": 10001, "runAsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"}},
    "containers": [{
      "name": "graph-rag-sync-once",
      "image": "ghcr.io/carev01/graph-rag:sha-abc1234",
      "command": ["python", "-m", "graph_sync.cli", "sync-once"],
      "envFrom": [{"secretRef": {"name": "graph-rag-secret"}},
                  {"configMapRef": {"name": "graph-rag-config"}}],
      "resources": {"requests": {"cpu": "100m", "memory": "384Mi"},
                    "limits": {"cpu": "1", "memory": "1Gi"}},
      "securityContext": {"allowPrivilegeEscalation": false, "capabilities": {"drop": ["ALL"]}}
    }]
  }
}
JSON
)
```

```
kubectl -n graph-rag run graph-rag-sync-once --restart=Never \
  --dry-run=client -o yaml \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.cli sync-once
```

```
kubectl -n graph-rag run graph-rag-sync-once --restart=Never \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.cli sync-once
```

Wait with `pod=graph-rag-sync-once`, `<tries>` = 120 (10 min; if it is still `Running`,
run the loop again — a pull is structural writes only, no LLM spend). It must end
`phase=Succeeded` with a `sync-once complete: applied=… advanced=True …` line. Then
`kubectl -n graph-rag delete pod graph-rag-sync-once`. (Deleting it mid-pull is safe:
the cursor advances only after the stream's terminal cursor line.)

Re-run the lane query and compare. **Stop and review before any worker is scaled** if
the `incremental` lane's `pending` count grew by more than the articles DocExtractor
could plausibly have changed since the dev host's last pull (a handful to a few hundred
after a re-extraction; the `applied=` count of the output above says how many the pull
touched). Growth in the thousands, approaching the corpus size, is the empty/wrong-cursor
replay 7a exists to prevent — every one of those jobs would be claimed ahead of the
bootstrap lane and outside the daily budget. Nothing spends until a worker runs, so
stopping here costs nothing.

Only then start the poller and unsuspend the two Neo4j housekeeping CronJobs that the
quiesced install suspended:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set sync.replicas=1 \
  --set cron.jobs.cleanup.suspend=false \
  --set cron.jobs.maintenance.suspend=false
kubectl -n graph-rag rollout status deploy/graph-rag-sync --timeout=180s
kubectl -n graph-rag logs deploy/graph-rag-sync
```

**Timing:** the sync app waits a full `POLL_INTERVAL_SECONDS` (6 h, per
`values.yaml`'s `config.POLL_INTERVAL_SECONDS`) before its *first* poll after every
start or rollout — it does not poll on boot. A poll that fails (DocExtractor or a store
unreachable) is logged with a traceback (`poll trigger failed; retrying in …`) and
retried at the next interval; it does not stop the loop. To pull again immediately, rerun
the `graph-rag-sync-once` pod above (the advisory lock keeps it single-flight with the
poller).

## 9. One-article smoke ingest (paid: one article)

**Do not do this by scaling `graph-rag-worker`.** It is a `Deployment`: the kubelet
restarts its pod the instant the container exits, so a `--max-batches 1` run inside it
would claim, and pay for, another article on *every* restart — indefinitely, on an empty
queue, or on an early crash — however fast anything watching the log reacts. Run it as a
one-off pod (step 5's pattern) instead; worker replicas stay at 0 throughout.

**First prove there is something to claim**, or the smoke proves nothing. A worker
claims `pending` jobs whose `next_attempt_at` has passed — the `incremental` lane always,
the `bootstrap` lane only while today's `token_ledger` total is below
`SEMANTIC_DAILY_TOKEN_BUDGET`:

```
kubectl -n graph-rag exec -i graph-rag-postgres-0 -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT count(*) AS done_before FROM semantic_jobs WHERE status = 'done';
SELECT lane, count(*) AS claimable FROM semantic_jobs
  WHERE status = 'pending' AND next_attempt_at <= now() GROUP BY 1 ORDER BY 1;
SELECT coalesce((SELECT tokens FROM token_ledger WHERE day = current_date), 0) AS tokens_today;
SQL
```

Record `done_before`. If there is no `claimable` row, or the only one is `bootstrap` and
`tokens_today` is at or over the budget, the worker would claim nothing: skip the smoke
(it would exit cleanly, log no batch line and change nothing — a pass that proved
nothing) and come back when there is work.

```
ovr=$(cat <<'JSON'
{
  "apiVersion": "v1",
  "spec": {
    "automountServiceAccountToken": false,
    "securityContext": {"runAsNonRoot": true, "runAsUser": 10001, "runAsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"}},
    "containers": [{
      "name": "graph-rag-smoke",
      "image": "ghcr.io/carev01/graph-rag:sha-abc1234",
      "command": ["python", "-m", "graph_sync.cli", "worker", "--batch", "1", "--max-batches", "1"],
      "envFrom": [{"secretRef": {"name": "graph-rag-secret"}},
                  {"configMapRef": {"name": "graph-rag-config"}}],
      "resources": {"requests": {"cpu": "100m", "memory": "384Mi"},
                    "limits": {"cpu": "1", "memory": "1Gi"}},
      "securityContext": {"allowPrivilegeEscalation": false, "capabilities": {"drop": ["ALL"]}}
    }]
  }
}
JSON
)
```

```
kubectl -n graph-rag run graph-rag-smoke --restart=Never \
  --dry-run=client -o yaml \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.cli worker --batch 1 --max-batches 1
```

```
kubectl -n graph-rag run graph-rag-smoke --restart=Never \
  --image=ghcr.io/carev01/graph-rag:sha-abc1234 \
  --overrides="$ovr" \
  --command -- python -m graph_sync.cli worker --batch 1 --max-batches 1
```

Wait with `pod=graph-rag-smoke`, `<tries>` = 240 (20 min; one article through the
extraction pipeline can take several minutes). Then prove the ingest actually engaged —
all three, not just a zero exit:

1. `phase=Succeeded`;
2. the log contains a `semantic batch: jobs=1 …` line (`kubectl -n graph-rag logs
   graph-rag-smoke | grep 'semantic batch:'`). The worker logs it only when it claimed
   a job, so its absence means nothing was claimed; a `semantic job … failed` line means
   the article was claimed and failed;
3. `SELECT count(*) FROM semantic_jobs WHERE status = 'done';` is `done_before + 1`.

Any of the three missing: stop and investigate before step 10. Then clean up — the pod
does not delete itself:

```
kubectl -n graph-rag delete pod graph-rag-smoke
```

## 10. Scale workers

Scale through Helm. `kubectl scale` edits the live Deployment behind Helm's back, and
`--reuse-values` re-applies the **release's stored values**, not the live state, so the
next `helm upgrade --reuse-values` for any reason silently undoes it. The one sanctioned
exception is the emergency stop in step 11, which is followed by a Helm reconcile.

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set worker.replicas=<N>
kubectl -n graph-rag rollout status deploy/graph-rag-worker --timeout=600s
kubectl -n graph-rag top pods
```

Watch actual usage against the requests in `values.yaml` (`worker.resources.requests`:
384Mi / 100m per replica) before raising further.

**Concurrency and `--batch` must move together.** `worker.args` sets `--batch 1`
(`values.yaml`), which must equal `INGEST_ARTICLE_CONCURRENCY` (an env var, default
1) whenever that is raised — the worker only checks for a stop signal *between*
batches (BACKLOG item 45), so a batch bigger than the dispatched concurrency makes
SIGTERM's grace period land mid-batch. Raise both together:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set-string worker.args='{--batch,<N>,--poll-seconds,5}' \
  --set-string config.INGEST_ARTICLE_CONCURRENCY=<N>
kubectl -n graph-rag rollout status deploy/graph-rag-worker --timeout=900s
```

(`--set-string`, not `--set` — a bare `--set worker.args={--batch,<N>,...}` renders
the batch number as an unquoted YAML integer, and the Deployment's `args` field is
`[]string`. Confirmed locally: `helm template deploy/helm/graph-rag --set
image.tag=sha-abc1234 --set worker.args='{--batch,5,--poll-seconds,5}'` renders
`- 5`; the same with `--set-string` renders `- "5"`. The rollout timeout covers old
pods finishing their in-flight article within the 600 s grace period.)

**Worker shutdown.** On SIGTERM a worker stops claiming new work but finishes its
in-flight batch; an article that outlasts `worker.terminationGracePeriodSeconds`
(600 s) is killed and its job is re-queued by the reaper after
`SEMANTIC_REAPER_LEASE_SECONDS` (2 h, per `values.config`) — re-running is safe (the
per-chunk `HAS_EPISODE` gate makes it idempotent), but prefer scaling to 0 between
batches where the schedule allows it.

## 11. Emergency stop and rollback

### Emergency stop — stop spending now

This must work whatever state the local checkout is in, so the first command does not
touch the chart:

```
kubectl -n graph-rag scale deploy/graph-rag-worker --replicas=0
kubectl -n graph-rag delete pod graph-rag-smoke --ignore-not-found
kubectl -n graph-rag get pods -l app=graph-rag-worker
```

This is the one sanctioned `kubectl scale` in this runbook. Workers finish their
in-flight article (up to the 600 s grace period) and exit; repeat the `get pods` until
none is listed. Then reconcile Helm's stored values, so the next `--reuse-values` upgrade
does not bring the old replica count back — **only from a checkout of the commit that is
deployed** (`helm upgrade` re-renders the local chart):

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set worker.replicas=0
```

If you cannot be sure the checkout matches, skip the reconcile and pass
`--set worker.replicas=0` (or the intended count) explicitly on the next upgrade.

### Rollback

First check what the target revision's values will bring back —
`helm get values graph-rag -n graph-rag --revision <revision>` — then roll back and
immediately re-apply the stop:

```
helm rollback graph-rag <revision> -n graph-rag
kubectl -n graph-rag scale deploy/graph-rag-worker --replicas=0
```

A rollback restores *that* revision's stored values, including its `worker.replicas`,
which may not be 0. The stop after it is `kubectl scale`, **not** `helm upgrade
--reuse-values --set worker.replicas=0`: a `helm upgrade` re-renders the chart in the
local checkout — the very templates you just rolled back away from — and would undo the
rollback in the same breath. The price is drift: the release now stores the rolled-back
replica count, so every later `helm upgrade` must pass `--set worker.replicas=…`
explicitly until the stored value is 0 again. If the target revision had workers above
0, run the emergency stop *before* the rollback too, so they are not started only to be
stopped.

The Postgres PVC survives a `helm uninstall` and is covered by the namespace's Kasten
backup policy (step 2).

**Data safety.** The PVC's StorageClass (`vsphere-csi-silver-sc`) reclaim policy is
`Delete` — deleting the PVC directly, or deleting the namespace, destroys the data
immediately; only Kasten's daily `production-apps-backup` export can recover it
after that. `helm uninstall` alone does not delete the PVC (StatefulSet
`volumeClaimTemplates` are left behind on purpose), so that path is safe by itself —
the danger is a manual `kubectl delete pvc`/`kubectl delete namespace`.

`postgres.size` and `postgres.storageClass` cannot be changed via `helm upgrade`:
`volumeClaimTemplates` on an existing StatefulSet are immutable. To grow the volume,
patch the PVC directly instead — the StorageClass allows expansion:

```
kubectl -n graph-rag patch pvc data-graph-rag-postgres-0 -p '{"spec":{"resources":{"requests":{"storage":"<new-size>"}}}}'
```

## 12. Theme-build

Leave `cron.jobs.theme-build.suspend=true` (the chart default) until after the
baseline bootstrap rebuild, then enable it:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set cron.jobs.theme-build.suspend=false
```

## 13. DocExtractor webhook

The cluster is **poll-only**, per the design: DocExtractor's webhook stays unregistered
for it, and `POLL_INTERVAL_SECONDS` (6 h) bounds the lag. The poll is always
authoritative anyway — a webhook only nudges an earlier pull, and the cursor comes from
Postgres, never the webhook's `watermark` (`CLIENT-USAGE-GUIDE.md` §6).

If a webhook is ever wanted, the route is `POST /webhooks/docextractor` on the sync
service (`src/graph_sync/webhook.py`), reachable in-cluster from DocExtractor at
`http://graph-rag-sync.graph-rag.svc:8000/webhooks/docextractor`. Add a
`WEBHOOK_SECRET` (e.g. `openssl rand -hex 32`, never echoed) to `graph-rag-secret`
and restart `deploy/graph-rag-sync` (step 3) — without it the route answers 401 to
everything. Then register it from a **trusted host** holding the admin key (it is not in
the cluster, step 3), with `DOCEXT_ADMIN_KEY`, `WEBHOOK_SECRET` (the same value) and
`WEBHOOK_PUBLIC_URL=http://graph-rag-sync.graph-rag.svc:8000/webhooks/docextractor`
set in that host's environment: `python -m graph_sync.cli register-webhook`. If the dev
host ever registered a webhook pointing at itself, delete that registration in
DocExtractor as part of step 14.

## 14. Retire the dev host

After cut-over, the dev host must not be able to act as a second production writer.

Rename its state container so nothing (a `docker start`, a compose file, a script)
resumes it by name, and keep it stopped — it is the pre-migration copy, retained only
as a fallback:

```
docker rename graphrag-postgres graphrag-postgres-retired
docker update --restart=no graphrag-postgres-retired
docker ps -a --filter name=graphrag-postgres-retired --format '{{.Names}} {{.Status}}'
```

The last line must show `Exited`.

Then, **before running any worker, `ingest`, bootstrap, theme-build or paid script from
the dev host again**, point the dev host's `.env` at a *scratch* Neo4j and a scratch
Postgres — never the production ones. Edit it yourself; do not paste it anywhere. Its
`NEO4J_*` values today are the production instance's, and its LLM keys are the paying
ones. A dev-host run against production Neo4j would, with its own (stale or empty)
Postgres queue, re-claim articles the cluster's workers are also processing: the same
article extracted twice (double spend), concurrent writes to the same entities outside
the cluster's warm-up advisory lock (which lives in the *cluster's* Postgres, so it does
not serialise the two), and — with an empty cursor — a full-corpus replay (step 7a).
