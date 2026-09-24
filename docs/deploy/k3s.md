# graph-rag on srv-k3s — cut-over runbook

Companion to `docs/superpowers/specs/2026-09-24-k3s-deployment-design.md` and the
chart at `deploy/helm/graph-rag`. Run each step in order and verify it before moving
to the next — nothing here is safe to run out of sequence.

**Never** paste `.env`, a secret file, or a real credential into a chat, ticket,
commit, or this document. Every credential below is either generated at runtime
into a shell variable that is never echoed (`$pw`, `$newpw`) or a placeholder
you fill in (`<N>`, `<revision>`, `<private-dir>`, `<new-size>`, `sha-abc1234`).

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
grep -o '^[A-Z_]*=' .env
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
the running database keeps the old one. Rotate the actual role first, entering the
new password interactively so it is never echoed, logged, or captured in shell
history or `kubectl`'s own command-line arguments:

```
kubectl -n graph-rag exec -it graph-rag-postgres-0 -- psql -U graphsync -d graphsync
```

At the `graphsync=#` prompt:

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

Nothing should spend tokens or touch Postgres until connectivity is proven and state
is migrated. Install with the poller at 0 replicas (workers already default to 0 and
`theme-build` is suspended by default in `values.yaml`):

```
helm upgrade --install graph-rag deploy/helm/graph-rag -n graph-rag \
  --set image.tag=sha-abc1234 \
  --set sync.replicas=0
```

Wait for the StatefulSet pod and the answer deployment to become Ready:

```
kubectl -n graph-rag rollout status statefulset/graph-rag-postgres
kubectl -n graph-rag rollout status deploy/graph-rag-answer
```

## 5. Connectivity

Run the in-image check before anything else touches Neo4j, Postgres, or the model
endpoints. It is read-only and free (no LLM tokens, no writes):

```
kubectl -n graph-rag exec deploy/graph-rag-answer -- python -m graph_sync.connectivity
```

Every line must read `OK` before continuing. A `FAIL` line names the dependency and a
redacted error — no credential in its `.env` is ever printed, including inside a
partially-quoted DSN. The optional tiers (`judge`, `report`, `map`, `rerank`,
`eval-judge`) are only probed when their `*_base_url` setting is non-empty; an unset
one reports `OK  <tier>  not configured` rather than failing.

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
until docker exec graphrag-postgres pg_isready -q; do sleep 1; done
```

Dump using the container's own `POSTGRES_USER`/`POSTGRES_DB` env vars rather than a
hardcoded user/db name, and confirm the dump is non-empty before going any further:

```
dumpfile=<private-dir>/graphsync.dump
docker exec graphrag-postgres sh -c 'pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB"' > "$dumpfile"
test -s "$dumpfile" || { echo "dump is empty -- stop, do not proceed" >&2; exit 1; }
```

Record every state table's contents to compare against after restore — all seven
tables `src/graph_sync/state_store.py`'s `CREATE TABLE` statements define
(`sync_cursor`, `bootstrap_progress`, `webhook_delivery`, `source_debounce`,
`dead_letter`, `semantic_jobs`, `token_ledger`). A row *count* is enough for the
five append-only/queue tables, but not for `sync_cursor` (always exactly one row,
by its own `CHECK (id = 1)`) or `bootstrap_progress` (a fixed row per shard whose
`watermark`/`last_id`/`status` still change) — hash their actual values instead:

```
docker exec -i graphrag-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT status, lane, count(*) FROM semantic_jobs GROUP BY 1, 2;
SELECT count(*) FROM webhook_delivery;
SELECT count(*) FROM source_debounce;
SELECT count(*) FROM dead_letter;
SELECT count(*) FROM token_ledger;
SELECT md5(cursor) FROM sync_cursor;
SELECT md5(string_agg(shard || watermark || coalesce(last_id, '') || status, ',' ORDER BY shard)) FROM bootstrap_progress;
SQL
docker stop graphrag-postgres
```

Copy the dump into the pod and restore it as a single all-or-nothing operation —
stop on the first error rather than leaving a half-restored database:

```
kubectl -n graph-rag cp "$dumpfile" graph-rag-postgres-0:/tmp/graphsync.dump
kubectl -n graph-rag exec graph-rag-postgres-0 -- \
  sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-acl --exit-on-error --single-transaction /tmp/graphsync.dump'
```

Re-run the exact same seven queries against the pod and compare every line to the
dev-host output — they must match exactly:

```
kubectl -n graph-rag exec -i graph-rag-postgres-0 -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT status, lane, count(*) FROM semantic_jobs GROUP BY 1, 2;
SELECT count(*) FROM webhook_delivery;
SELECT count(*) FROM source_debounce;
SELECT count(*) FROM dead_letter;
SELECT count(*) FROM token_ledger;
SELECT md5(cursor) FROM sync_cursor;
SELECT md5(string_agg(shard || watermark || coalesce(last_id, '') || status, ',' ORDER BY shard)) FROM bootstrap_progress;
SQL
```

Then delete the dump on both sides:

```
kubectl -n graph-rag exec graph-rag-postgres-0 -- rm /tmp/graphsync.dump
rm "$dumpfile"
```

## 7. Start the poller

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set sync.replicas=1
kubectl -n graph-rag logs deploy/graph-rag-sync
```

**Timing:** the sync app waits a full `POLL_INTERVAL_SECONDS` (6 h, per
`values.yaml`'s `config.POLL_INTERVAL_SECONDS`) before its *first* poll after every
start or rollout — it does not poll immediately on boot. To pull right away instead
of waiting:

```
kubectl -n graph-rag exec deploy/graph-rag-sync -- python -m graph_sync.cli sync-once
```

## 8. One-article smoke ingest (paid: one article)

Run this in an actual **worker** pod, not the sync pod: `graph-rag-sync` is sized
for the poll loop and its FastAPI app (512Mi limit, per `values.yaml`'s
`sync.resources`), not for driving `IngestDriver`'s extraction pipeline, which is
what the `worker` sized pods (`worker.resources`: 1Gi limit) exist for.

**The chosen approach, and why:** `graph-rag-worker` is a `Deployment`, not a `Job`
— a container that exits after `--max-batches 1` completes is restarted by the
kubelet immediately (no back-off on the first exit), so scaling to 0 "after you see
the summary line" is a race a human will lose. Script the reaction instead of
relying on reading the log fast enough: pipe the log through `grep -q -m1` for the
summary line and chain the scale-down as its immediate `&&`, so it fires the
instant the line appears rather than after a human reacts to it. The same command
also resets `worker.args` back to its non-`--max-batches` default — `--reuse-values`
carries whatever was last set via `--set`/`--set-string` forward into every
subsequent upgrade (see §9), so leaving `--max-batches` in place would silently
turn every future worker scale-up into another one-batch run:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set worker.replicas=1 \
  --set-string worker.args='{--batch,1,--poll-seconds,5,--max-batches,1}'

kubectl -n graph-rag rollout status deploy/graph-rag-worker

kubectl -n graph-rag logs -f deploy/graph-rag-worker | grep -q -m1 'semantic batch:' && \
  helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
    --set worker.replicas=0 \
    --set-string worker.args='{--batch,1,--poll-seconds,5}'
```

Even so, a restart that lands before the `grep`/`helm` pipeline completes could
claim a second article; treat "at most one, almost certainly exactly one" as the
guarantee here, not "exactly one," and check `queue-status` afterwards to see how
many jobs actually ran:

```
kubectl -n graph-rag exec deploy/graph-rag-sync -- python -m graph_sync.cli queue-status
```

## 9. Scale workers

Always scale through Helm, never `kubectl scale deploy/graph-rag-worker`:
`kubectl scale` edits the live Deployment directly, bypassing Helm's tracked
state entirely, and `--reuse-values` re-applies the **release's own last stored
values** (whatever the previous `helm install`/`upgrade` set — `values.yaml` only
if nothing has overridden `worker.replicas` since), not the live cluster state. The
next `helm upgrade --reuse-values`, for any reason, silently resets replicas back
to that stored value and undoes the `kubectl scale`.

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set worker.replicas=<N>
kubectl -n graph-rag top pods
```

Watch actual usage against the requests in `values.yaml` (`worker.resources.requests`:
384Mi / 100m per replica) before raising further.

**Concurrency and `--batch` must move together.** `worker.args` sets `--batch 1`
(`values.yaml`), which must equal `INGEST_ARTICLE_CONCURRENCY` (an env var, default
1) whenever that is raised — `run_concurrently` only checks for a stop signal
*between* batches, so a batch bigger than the dispatched concurrency makes SIGTERM's
grace period land mid-batch instead of between batches. Raise both together:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set-string worker.args='{--batch,<N>,--poll-seconds,5}' \
  --set-string config.INGEST_ARTICLE_CONCURRENCY=<N>
```

(`--set-string`, not `--set` — a bare `--set worker.args={--batch,<N>,...}` renders
the batch number as an unquoted YAML integer, and the Deployment's `args` field is
`[]string`. Confirmed locally: `helm template deploy/helm/graph-rag --set
image.tag=sha-abc1234 --set worker.args='{--batch,5,--poll-seconds,5}'` renders
`- 5` (a bare integer); the same command with `--set-string` in place of `--set`
renders `- "5"`.)

**Worker shutdown.** On SIGTERM a worker stops claiming new work but finishes its
in-flight batch; an article that outlasts `worker.terminationGracePeriodSeconds`
(600 s) is killed and its job is re-queued by the reaper after
`SEMANTIC_REAPER_LEASE_SECONDS` (2 h, per `values.config`) — re-running is safe (the
per-chunk `HAS_EPISODE` gate makes it idempotent), but prefer scaling to 0 between
batches where the schedule allows it, rather than relying on the reaper to clean up
an interrupted one.

## 10. Rollback

Stop spending immediately:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set worker.replicas=0
```

Roll the release back to a previous revision:

```
helm rollback graph-rag <revision> -n graph-rag
```

A rollback restores *that* revision's stored values, including whatever
`worker.replicas` it had at the time — which may not be 0. Immediately re-quiesce:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set worker.replicas=0
```

The Postgres PVC survives a `helm uninstall` and is covered by the namespace's Kasten
backup policy (step 2).

**Data safety.** The PVC's StorageClass (`vsphere-csi-silver-sc`) reclaim policy is
`Delete` — deleting the PVC directly, or deleting the namespace, destroys the data
immediately; only Kasten's daily `production-apps-backup` export can recover it
after that. `helm uninstall` alone does not delete the PVC (StatefulSet
`volumeClaimTemplates` are left behind on purpose), so that path is safe by itself —
the danger is a manual `kubectl delete pvc`/`kubectl delete namespace`.

`postgres.size` and `postgres.storageClass` cannot be changed via `helm upgrade`:
`volumeClaimTemplates` on an existing StatefulSet are immutable, and Helm will
reject (or Kubernetes will reject) an attempt to change them in place. To grow the
volume, patch the PVC directly instead — the StorageClass allows expansion:

```
kubectl -n graph-rag patch pvc data-graph-rag-postgres-0 -p '{"spec":{"resources":{"requests":{"storage":"<new-size>"}}}}'
```

## 11. Theme-build

Leave `cron.jobs.theme-build.suspend=true` (the chart default) until after the
baseline bootstrap rebuild, then enable it:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set cron.jobs.theme-build.suspend=false
```
