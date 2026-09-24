# graph-rag on srv-k3s — cut-over runbook

Companion to `docs/superpowers/specs/2026-09-24-k3s-deployment-design.md` and the
chart at `deploy/helm/graph-rag`. Run each step in order and verify it before moving
to the next — nothing here is safe to run out of sequence.

**Never** paste `.env`, a secret file, or a real credential into a chat, ticket,
commit, or this document. Every credential below is a placeholder (`<pw>`, `<user>`,
`<dump>`, ...).

## 1. Prerequisites

- `kubectl` context pointed at `srv-k3s.home.lan`.
- `helm` (v3).
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
one secret, `graph-rag-secret`, built once from the dev host's `.env`.

```
umask 077
tmpfile=$(mktemp /tmp/graph-rag-secret.XXXXXX.env)
# copy .env into $tmpfile, then strip surrounding quotes from every value:
#   --from-env-file keeps quotes literally, which breaks anything that parses
#   the value later (DSNs, URLs).
sed -i -E 's/^([A-Za-z_][A-Za-z0-9_]*)="?([^"]*)"?$/\1=\2/' "$tmpfile"
```

Add/override two keys the dev host's `.env` does not carry — a fresh Postgres
password for the in-cluster StatefulSet, and the DSN that points at it:

```
pw=$(openssl rand -hex 24)
echo "POSTGRES_PASSWORD=${pw}" >> "$tmpfile"
echo "POSTGRES_DSN=postgresql://graphsync:${pw}@graph-rag-postgres:5432/graphsync" >> "$tmpfile"
```

(`graphsync`/`graphsync` are the chart's `postgres.user` / `postgres.database`
defaults — see `deploy/helm/graph-rag/values.yaml`.)

```
kubectl -n graph-rag create secret generic graph-rag-secret --from-env-file="$tmpfile"
shred -u "$tmpfile"
```

Never paste `$tmpfile`'s contents or the secret's contents into a chat, ticket, or
commit.

**After changing `graph-rag-secret`** (rotation, or updating a key), every pod that
reads it must be restarted — `envFrom` is only read at container start, and the
chart's `checksum/config` pod-template annotation covers the ConfigMap, not this
out-of-band secret, so a plain `kubectl apply`/`helm upgrade` with no other change
will not roll these pods on its own:

```
kubectl -n graph-rag rollout restart deploy/graph-rag-sync deploy/graph-rag-answer deploy/graph-rag-worker
```

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
kubectl -n graph-rag get pod graph-rag-postgres-0 -w
kubectl -n graph-rag get deploy graph-rag-answer -w
```

## 5. Connectivity

Run the in-image check before anything else touches Neo4j, Postgres, or the model
endpoints. It is read-only and free (no LLM tokens, no writes):

```
kubectl -n graph-rag exec deploy/graph-rag-answer -- python -m graph_sync.connectivity
```

Every line must read `OK` before continuing. A `FAIL` line names the dependency and a
redacted error — no credential in its `.env` is ever printed, including inside a
partially-quoted DSN.

## 6. Migrate the state

On the dev host, the stopped `graphrag-postgres` container holds the sync cursor,
bootstrap progress, and token ledger built so far.

```
docker start graphrag-postgres
docker exec graphrag-postgres pg_dump -U <user> -Fc graphsync > <private-dir>/graphsync.dump
```

Record the row counts you will compare against after restore:

```
docker exec graphrag-postgres psql -U <user> -d graphsync -c \
  "SELECT status, lane, count(*) FROM semantic_jobs GROUP BY 1,2"
docker exec graphrag-postgres psql -U <user> -d graphsync -c "SELECT count(*) FROM sync_cursor"
docker exec graphrag-postgres psql -U <user> -d graphsync -c "SELECT count(*) FROM bootstrap_progress"
docker exec graphrag-postgres psql -U <user> -d graphsync -c "SELECT count(*) FROM token_ledger"
docker stop graphrag-postgres
```

Copy the dump into the pod and restore:

```
kubectl -n graph-rag cp <private-dir>/graphsync.dump graph-rag-postgres-0:/tmp/graphsync.dump
kubectl -n graph-rag exec graph-rag-postgres-0 -- \
  pg_restore -U graphsync -d graphsync --clean --if-exists --no-owner /tmp/graphsync.dump
```

Re-run the same four count queries against the pod (`kubectl -n graph-rag exec
graph-rag-postgres-0 -- psql -U graphsync -d graphsync -c "..."`) and compare — they
must match the dev-host counts exactly. Then delete the dump on both sides:

```
kubectl -n graph-rag exec graph-rag-postgres-0 -- rm /tmp/graphsync.dump
rm <private-dir>/graphsync.dump
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

```
kubectl -n graph-rag exec deploy/graph-rag-sync -- \
  python -m graph_sync.cli worker --max-batches 1 --batch 1
```

Confirm the `semantic batch: jobs=...` summary line in the output, then:

```
kubectl -n graph-rag exec deploy/graph-rag-sync -- python -m graph_sync.cli queue-status
```

## 9. Scale workers

Always scale through Helm, never `kubectl scale deploy/graph-rag-worker`: a later
`helm upgrade` (even an unrelated one, with `--reuse-values`) re-applies
`values.yaml`'s `worker.replicas` and silently resets any replica count set outside
Helm.

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
  --set worker.args='{--batch,<N>,--poll-seconds,5}' \
  --set-string config.INGEST_ARTICLE_CONCURRENCY=<N>
```

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
