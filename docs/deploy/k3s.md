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
  `ghcr.io/carev01/graph-rag:sha-<7>` (7-char short SHA of the commit on `main`).

Verify the image pulls anonymously — the repo and its GHCR package are public and the
image holds no secret:

```
docker logout ghcr.io
docker pull ghcr.io/carev01/graph-rag:sha-<7>
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

## 4. Install, quiesced

Nothing should spend tokens or touch Postgres until connectivity is proven and state
is migrated. Install with the poller at 0 replicas (workers already default to 0 and
`theme-build` is suspended by default in `values.yaml`):

```
helm upgrade --install graph-rag deploy/helm/graph-rag -n graph-rag \
  --set image.tag=sha-<7> \
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

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values --set worker.replicas=<N>
kubectl -n graph-rag top pods
```

Watch actual usage against the requests in `values.yaml` (`worker.resources.requests`:
384Mi / 100m per replica) before raising further.

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

## 11. Theme-build

Leave `cron.jobs.theme-build.suspend=true` (the chart default) until after the
baseline bootstrap rebuild, then enable it:

```
helm upgrade graph-rag deploy/helm/graph-rag -n graph-rag --reuse-values \
  --set cron.jobs.theme-build.suspend=false
```
