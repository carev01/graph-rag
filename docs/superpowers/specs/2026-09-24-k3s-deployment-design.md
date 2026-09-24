# graph-rag on srv-k3s — deployment design

**Date:** 2026-09-24. **Approved:** 2026-09-24 (design; decisions: migrate the Postgres
state by dump/restore; deploy answer-api now, internal only).
**Why:** the pre-bootstrap rehearsal must run on production-like infrastructure. Today
only Neo4j (VM `alpcirag01`) is production-grade; the state Postgres is a stopped Docker
container on the dev host and the services run from a shell.

## 1. Target

The existing single-node K3s cluster `srv-k3s.home.lan` (v1.35, 12 CPU, ~23.5 GiB
allocatable, **82% of memory already requested**), in the house style of the
`docextractor` release: Helm chart, GHCR images tagged `sha-<commit>`, plain Postgres
StatefulSet on `vsphere-csi-silver-sc`, Traefik ingress on `*.k3s.home.lan`, namespace
labelled `stack.stage=production` (which enrols it in Kasten's daily
`production-apps-backup` policy — backup + export).

Out of scope: moving Neo4j (stays on its VM); public exposure, auth, MCP (Phase F);
autoscaling; Prometheus metrics.

## 2. Image

One image for every role, built from a multi-stage `Dockerfile`: `uv sync --frozen
--no-dev --no-editable` in a builder stage on `python:3.12-slim`, copied venv in a slim
runtime stage, non-root user (uid 10001), `PYTHONUNBUFFERED=1`. `.dockerignore` excludes
`.env*`, `data/`, `.venv/`, caches, `tests/`, `docs/`, `.superpowers/`, `.git/`.
**The image must contain no secret** — the repo and its GHCR package are public.

**Packaging fix:** `pyproject.toml`'s wheel `packages` omits `src/theme_builder` (and
`src/compat`); a non-editable install would ship without `theme-build`. Add both, and a
unit test that every package under `src/` is listed, so a new package cannot silently
drop out of the image again.

Roles are selected by command:

| role | command |
|---|---|
| graph-sync | `uvicorn graph_sync.app:main --factory --host 0.0.0.0 --port 8000` |
| answer-api | `uvicorn answer_api.app:main --factory --host 0.0.0.0 --port 8000` |
| semantic worker | `python -m graph_sync.cli worker` |
| cleanup / maintenance | `python -m graph_extract.cli cleanup` / `maintenance` |
| theme-build | `python -m theme_builder.cli theme-build` |

CI: `.github/workflows/image.yml` builds on push to `main` (and manual dispatch) and
pushes `ghcr.io/carev01/graph-rag:sha-<short>` plus `:main`, using `GITHUB_TOKEN`
(`packages: write`). The chart pins `sha-<short>`, never a moving tag.

## 3. Chart — `deploy/helm/graph-rag`

| object | kind | notes |
|---|---|---|
| `graph-rag-postgres` | StatefulSet + headless Service | `postgres:16-alpine`, 10 Gi `vsphere-csi-silver-sc`, password from the secret |
| `graph-rag-sync` | Deployment ×1 + Service | graph-sync app: its own poll loop (`POLL_INTERVAL_SECONDS`, single-flight via advisory lock), `init_schema` on start, `/health` probes |
| `graph-rag-worker` | Deployment ×`worker.replicas` (default **0**) | semantic workers; `terminationGracePeriodSeconds` 600 |
| `graph-rag-answer` | Deployment ×1 + Service + Ingress `graphrag.k3s.home.lan` | `/health` probes; internal only |
| `graph-rag-cleanup` | CronJob, daily | `graph_extract.cli cleanup` |
| `graph-rag-maintenance` | CronJob, weekly | sweep + reconcile (`maintenance`) |
| `graph-rag-theme-build` | CronJob, weekly, **suspended** | incremental reports; enabled after the baseline rebuild |
| `graph-rag-config` | ConfigMap | deployment-specific overrides (below) |

No migration hook: every entry point runs `init_schema` (idempotent `IF NOT EXISTS`).
CronJobs use `concurrencyPolicy: Forbid`. Every pod `envFrom`s the secret first and the
ConfigMap second, so ConfigMap overrides win for the keys it sets.

**Config overrides** (ConfigMap, from `values.config`):
- `DOCEXT_BASE_URL=http://docextractor-backend.docextractor.svc:8000`, `DOCEXT_VERIFY_TLS=false`
  — in-cluster, no internal-CA hop;
- `SEMANTIC_REAPER_LEASE_SECONDS=7200` — the default 1,800 s is below the longest article
  measured (25 min at concurrency 1; a 47-episode Commvault page) and the lease has no
  heartbeat, so a live job could be reaped and processed twice;
- `POLL_INTERVAL_SECONDS=21600` (6 h — sources are re-extracted roughly quarterly).

**Secret** `graph-rag-secret` is **never in git and never in the chart**: created once from
`.env` by `kubectl create secret generic --from-env-file`, with `POSTGRES_DSN` rewritten to
`graph-rag-postgres:5432` and a `POSTGRES_PASSWORD` key added for the StatefulSet. The chart
only references it by name; `values.yaml` holds no credential.

**Resources** (requests / limits; measured and revised during the rehearsal):
postgres 256Mi / 1Gi; sync 256Mi / 512Mi; answer 512Mi / 1Gi; worker 384Mi / 1Gi each;
CronJobs 256Mi / 1Gi. ≈2.7 GiB requested at 4 workers, inside the ~4 GiB headroom.

## 4. Operations

- **Connectivity check** `scripts/connectivity_check.py` — run in a pod before anything
  spends: Neo4j (`RETURN 1`), Postgres (`SELECT 1`), DocExtractor `/api/health`, embedder
  and chunker reachability, the LLM gateway's model list. Read-only, no LLM tokens. Exits
  non-zero naming each failure.
- **Runbook** `docs/deploy/k3s.md`: namespace + label, secret from `.env`, state migration
  (start the stopped `graphrag-postgres` container once, `pg_dump -Fc`, `pg_restore` into
  the pod, verify table counts), `helm upgrade --install`, connectivity check, one-article
  smoke ingest, scaling workers, rollback (`helm rollback`; workers to 0), where Kasten
  backs it up.

## 5. Cut-over sequence (controller, each step verified before the next)

1. Build and push the image (CI). 2. Namespace, label, secret. 3. `helm install` with
workers 0 and theme-build suspended. 4. Connectivity check in-cluster. 5. Stop anything on
the dev host that could still write (nothing runs today). 6. Migrate state; compare row
counts. 7. One-article smoke ingest via a worker at `--max-batches 1` equivalent (replicas
1, then back to 0). 8. Hand over to the rehearsal.

## 6. Testing

- Unit: the packaging test (every `src/` package listed); chart render tests run
  `helm template` with test values and assert: the secret is referenced, never inlined;
  no value under `values.yaml` looks like a credential; envFrom order (secret, then
  ConfigMap); the lease override present; worker replicas default 0; the theme-build
  CronJob suspended (skipped when `helm` is absent, as in CI). The namespace and its
  `stack.stage=production` label are created by the runbook, not the chart — a chart
  cannot label a namespace `helm --create-namespace` makes.
- `helm lint`; `docker build` + an import/`--help` smoke of every role in the built image.
- The connectivity check has hermetic tests for its reporting (fakes), not the network.
