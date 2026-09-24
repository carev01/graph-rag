# graph-rag on srv-k3s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package graph-rag as one container image built by CI, a Helm chart for the `graph-rag` namespace on srv-k3s, an in-image connectivity check, and a cut-over runbook.

**Architecture:** One image, roles chosen by command. The chart mirrors the `docextractor` release (GHCR `sha-<short>` images, Postgres StatefulSet on `vsphere-csi-silver-sc`, Traefik ingress). Every app pod reads the out-of-band secret first and a ConfigMap of deployment overrides second. Nothing touches the cluster in this plan — cut-over is a controller step after it.

**Tech Stack:** Docker (multi-stage, uv 0.12.17), GitHub Actions (docker/build-push-action), Helm 4, Python 3.12, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-k3s-deployment-design.md`.

## Global Constraints

- **The image and the chart contain no secret.** `.dockerignore` excludes `.env*`; `values.yaml` has no credential; the chart never renders a `Secret` object. The secret `graph-rag-secret` is created out of band.
- Image: `ghcr.io/carev01/graph-rag`, tags `sha-<7-char sha>` and `main`; the chart requires an explicit `image.tag` and fails to render without one.
- Non-root runtime user uid `10001`; `PYTHONUNBUFFERED=1`.
- App pods `envFrom`: `secretRef graph-rag-secret` FIRST, `configMapRef graph-rag-config` SECOND.
- ConfigMap overrides exactly: `DOCEXT_BASE_URL=http://docextractor-backend.docextractor.svc:8000`, `DOCEXT_VERIFY_TLS=false`, `SEMANTIC_REAPER_LEASE_SECONDS=7200`, `POLL_INTERVAL_SECONDS=21600`.
- Worker replicas default `0`; worker `terminationGracePeriodSeconds: 600`; theme-build CronJob `suspend: true`; all CronJobs `concurrencyPolicy: Forbid`.
- Postgres: `postgres:16-alpine`, 10Gi `vsphere-csi-silver-sc`, password from secret key `POSTGRES_PASSWORD`, `PGDATA` in a subdirectory of the mount.
- Resources (requests/limits): postgres 256Mi/1Gi; sync 256Mi/512Mi; answer 512Mi/1Gi; worker 384Mi/1Gi; cronjobs 256Mi/1Gi.
- Ingress host `graphrag.k3s.home.lan`, class `traefik`, internal only.
- Deviation from the spec, ruled by the controller: the connectivity check lives at `src/graph_sync/connectivity.py` (`python -m graph_sync.connectivity`), not `scripts/`, so it ships in the image.
- Tests hermetic. Lint `uv run ruff check src tests`; types `uv run mypy src`. Chart tests skip when `helm` is absent.
- Subagents never run anything against the cluster (`kubectl apply/create/scale`, `helm install/upgrade`), never push images, never print `.env`.

---

### Task 1: Image, packaging fix, CI workflow

**Files:**
- Modify: `pyproject.toml` (`[tool.hatch.build.targets.wheel] packages`)
- Create: `Dockerfile`, `.dockerignore`, `.github/workflows/image.yml`
- Test: `tests/unit/test_packaging.py`

**Interfaces:**
- Produces: an image whose `/app/.venv` holds every `src/` package; roles run as `uvicorn graph_sync.app:main --factory …`, `uvicorn answer_api.app:main --factory …`, `python -m graph_sync.cli …`, `python -m graph_extract.cli …`, `python -m theme_builder.cli …`.

- [ ] **Step 1: Failing test** — `tests/unit/test_packaging.py`:

```python
"""Every package under src/ must be in the wheel -- the image installs the wheel
(non-editable), so a missing entry silently removes a whole service from it.
`theme_builder` was missing until 2026-09-24."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_every_src_package_is_in_the_wheel():
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text())
                 ["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    present = {f"src/{p.parent.name}" for p in (ROOT / "src").glob("*/__init__.py")}
    assert present - listed == set(), f"missing from the wheel: {sorted(present - listed)}"


def test_dockerignore_keeps_secrets_and_data_out():
    ignored = (ROOT / ".dockerignore").read_text().split()
    for entry in (".env", ".env.*", "data", ".venv", ".git", ".superpowers"):
        assert entry in ignored, f".dockerignore must exclude {entry}"
```

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_packaging.py -q` → both FAIL (theme_builder/compat missing; no `.dockerignore`).

- [ ] **Step 3: Implement.**

`pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/graph_sync", "src/graph_extract", "src/docext", "src/answer_api",
            "src/theme_builder", "src/compat"]
```

`.dockerignore`:
```
.env
.env.*
data
.venv
.git
.github
.superpowers
.mypy_cache
.pytest_cache
.ruff_cache
__pycache__
tests
docs
scripts
deploy
*.md
```

`Dockerfile`:
```dockerfile
# syntax=docker/dockerfile:1.7
# One image for every graph-rag role; the role is chosen by the container command.
# Contains NO secret: configuration arrives as env vars from the cluster.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/carev01/graph-rag"
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "answer_api.app:main", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

`.github/workflows/image.yml`:
```yaml
name: image
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
  packages: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - id: meta
        run: echo "short=${GITHUB_SHA::7}" >> "$GITHUB_OUTPUT"
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ghcr.io/carev01/graph-rag:sha-${{ steps.meta.outputs.short }}
            ghcr.io/carev01/graph-rag:main
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 4: Run to verify pass** — `uv run --extra dev pytest tests/unit/test_packaging.py -q` → PASS.

- [ ] **Step 5: Build and smoke the image locally (no push).**

```bash
docker build -t graph-rag:dev .
docker run --rm graph-rag:dev python -c "import graph_sync.app, graph_sync.cli, graph_extract.cli, theme_builder.cli, answer_api.app, docext.client; print('imports ok')"
docker run --rm graph-rag:dev python -m graph_sync.cli --help
docker run --rm graph-rag:dev python -m graph_extract.cli --help
docker run --rm graph-rag:dev python -m theme_builder.cli --help
docker run --rm graph-rag:dev id -u        # expect 10001
docker run --rm graph-rag:dev sh -c 'ls -a /app; test ! -e /app/.env && echo "no .env in image"'
docker image inspect graph-rag:dev --format '{{.Size}}'
```
Record all output and the image size in the report. If the build cannot reach the network (base image or PyPI), report BLOCKED with the error — do not work around it. (`compat` is deliberately not imported: it needs dev-only packages and is never run in the cluster.)

- [ ] **Step 6: Gate and commit** — `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest tests/unit -q`. Commit: `feat(deploy): container image, wheel packaging fix, GHCR build workflow`.

---

### Task 2: Helm chart

**Files:**
- Create: `deploy/helm/graph-rag/Chart.yaml`, `values.yaml`, `templates/_helpers.tpl`, `templates/configmap.yaml`, `templates/postgres.yaml`, `templates/sync.yaml`, `templates/answer.yaml`, `templates/worker.yaml`, `templates/cronjobs.yaml`
- Test: `tests/unit/test_helm_chart.py`

**Interfaces:**
- Consumes (Task 1): the image and role commands above.
- Produces: object names `graph-rag-postgres` (StatefulSet + Service), `graph-rag-sync`, `graph-rag-answer` (Deployment + Service + Ingress), `graph-rag-worker`, CronJobs `graph-rag-cleanup`, `graph-rag-maintenance`, `graph-rag-theme-build`, ConfigMap `graph-rag-config`; values `image.tag` (required), `sync.replicas` (default 1), `worker.replicas` (default 0).

- [ ] **Step 1: Failing tests** — `tests/unit/test_helm_chart.py`:

```python
"""Render the chart and assert the properties the spec makes binding. Skipped where
helm is not installed (CI); run locally before any cut-over."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).parents[2] / "deploy" / "helm" / "graph-rag"
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
IMAGE = "ghcr.io/carev01/graph-rag:sha-test"


def render(*sets: str) -> list[dict]:
    args = ["helm", "template", "t", str(CHART), "--set", "image.tag=sha-test"]
    for s in sets:
        args += ["--set", s]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def get(docs: list[dict], kind: str, name: str) -> dict:
    found = [d for d in docs if d["kind"] == kind and d["metadata"]["name"] == name]
    assert len(found) == 1, f"{kind}/{name} rendered {len(found)} times"
    return found[0]


def pod_spec(doc: dict) -> dict:
    if doc["kind"] == "CronJob":
        return doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return doc["spec"]["template"]["spec"]


APP = [("Deployment", "graph-rag-sync"), ("Deployment", "graph-rag-answer"),
       ("Deployment", "graph-rag-worker"), ("CronJob", "graph-rag-cleanup"),
       ("CronJob", "graph-rag-maintenance"), ("CronJob", "graph-rag-theme-build")]


def test_lint_passes():
    r = subprocess.run(["helm", "lint", str(CHART), "--set", "image.tag=sha-test"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_an_image_tag_is_required():
    r = subprocess.run(["helm", "template", "t", str(CHART)], capture_output=True, text=True)
    assert r.returncode != 0 and "image.tag" in r.stderr


def test_app_pods_take_the_secret_then_the_configmap_and_the_pinned_image():
    docs = render()
    for kind, name in APP:
        spec = pod_spec(get(docs, kind, name))
        (c,) = spec["containers"]
        assert c["image"] == IMAGE, name
        assert c["envFrom"] == [{"secretRef": {"name": "graph-rag-secret"}},
                                {"configMapRef": {"name": "graph-rag-config"}}], name
        assert spec["securityContext"]["runAsUser"] == 10001, name
        assert spec["securityContext"]["runAsNonRoot"] is True, name


def test_no_secret_is_rendered_or_stored_in_values():
    assert not [d for d in render() if d["kind"] == "Secret"]
    values = (CHART / "values.yaml").read_text()
    for line in values.splitlines():
        m = re.match(r"\s*([A-Za-z_]+)\s*:\s*(\S.*)?$", line)
        if not m or m.group(1) == "secretName" or not m.group(2):
            continue
        assert not re.search(r"(?i)password|api_?key|token|secret", m.group(1)), line


def test_configmap_overrides_are_exactly_the_spec():
    cm = get(render(), "ConfigMap", "graph-rag-config")
    assert cm["data"] == {
        "DOCEXT_BASE_URL": "http://docextractor-backend.docextractor.svc:8000",
        "DOCEXT_VERIFY_TLS": "false",
        "SEMANTIC_REAPER_LEASE_SECONDS": "7200",
        "POLL_INTERVAL_SECONDS": "21600",
    }


def test_worker_defaults_to_zero_replicas_with_a_long_grace_period():
    docs = render()
    w = get(docs, "Deployment", "graph-rag-worker")
    assert w["spec"]["replicas"] == 0
    assert pod_spec(w)["terminationGracePeriodSeconds"] == 600
    assert pod_spec(w)["containers"][0]["command"][:3] == ["python", "-m", "graph_sync.cli"]
    assert get(render("worker.replicas=4"), "Deployment", "graph-rag-worker")["spec"]["replicas"] == 4


def test_cronjobs_forbid_overlap_and_theme_build_starts_suspended():
    docs = render()
    for name in ("graph-rag-cleanup", "graph-rag-maintenance", "graph-rag-theme-build"):
        assert get(docs, "CronJob", name)["spec"]["concurrencyPolicy"] == "Forbid"
    assert get(docs, "CronJob", "graph-rag-theme-build")["spec"]["suspend"] is True
    assert get(docs, "CronJob", "graph-rag-cleanup")["spec"]["suspend"] is False


def test_postgres_takes_its_password_from_the_secret_and_keeps_pgdata_in_a_subdir():
    docs = render()
    sts = get(docs, "StatefulSet", "graph-rag-postgres")
    (c,) = pod_spec(sts)["containers"]
    env = {e["name"]: e for e in c["env"]}
    assert env["POSTGRES_PASSWORD"]["valueFrom"]["secretKeyRef"] == {
        "name": "graph-rag-secret", "key": "POSTGRES_PASSWORD"}
    assert env["PGDATA"]["value"].startswith("/var/lib/postgresql/data/")
    (vct,) = sts["spec"]["volumeClaimTemplates"]
    assert vct["spec"]["storageClassName"] == "vsphere-csi-silver-sc"
    assert vct["spec"]["resources"]["requests"]["storage"] == "10Gi"
    get(docs, "Service", "graph-rag-postgres")


def test_answer_api_is_exposed_internally_only():
    ing = get(render(), "Ingress", "graph-rag-answer")
    assert ing["spec"]["ingressClassName"] == "traefik"
    assert [r["host"] for r in ing["spec"]["rules"]] == ["graphrag.k3s.home.lan"]
    assert "tls" not in ing["spec"]


def test_sync_can_be_installed_scaled_down():
    assert get(render("sync.replicas=0"), "Deployment", "graph-rag-sync")["spec"]["replicas"] == 0
```

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_helm_chart.py -q` → FAIL (chart missing).

- [ ] **Step 3: Implement.**

`deploy/helm/graph-rag/Chart.yaml`:
```yaml
apiVersion: v2
name: graph-rag
description: Temporal GraphRAG over DocExtractor -- ingestion, workers, answer API
type: application
version: 0.1.0
appVersion: "0.1.0"
```

`deploy/helm/graph-rag/values.yaml`:
```yaml
# No credential belongs in this file. Credentials live in the out-of-band secret
# named below (created from .env; see docs/deploy/k3s.md).
image:
  repository: ghcr.io/carev01/graph-rag
  tag: ""            # required: sha-<7-char commit>
  pullPolicy: IfNotPresent

secretName: graph-rag-secret

# Deployment-specific overrides; applied AFTER the secret, so these win.
config:
  DOCEXT_BASE_URL: http://docextractor-backend.docextractor.svc:8000
  DOCEXT_VERIFY_TLS: "false"
  SEMANTIC_REAPER_LEASE_SECONDS: "7200"
  POLL_INTERVAL_SECONDS: "21600"

postgres:
  image: postgres:16-alpine
  storageClass: vsphere-csi-silver-sc
  size: 10Gi
  database: graphsync
  user: graphsync
  resources:
    requests: {cpu: 100m, memory: 256Mi}
    limits: {cpu: "1", memory: 1Gi}

sync:
  replicas: 1
  resources:
    requests: {cpu: 50m, memory: 256Mi}
    limits: {cpu: 500m, memory: 512Mi}

answer:
  host: graphrag.k3s.home.lan
  resources:
    requests: {cpu: 100m, memory: 512Mi}
    limits: {cpu: "1", memory: 1Gi}

worker:
  replicas: 0
  args: ["--batch", "10", "--poll-seconds", "5"]
  terminationGracePeriodSeconds: 600
  resources:
    requests: {cpu: 100m, memory: 384Mi}
    limits: {cpu: "1", memory: 1Gi}

cron:
  resources:
    requests: {cpu: 50m, memory: 256Mi}
    limits: {cpu: "1", memory: 1Gi}
  jobs:
    cleanup: {schedule: "30 3 * * *", suspend: false, command: ["python", "-m", "graph_extract.cli", "cleanup"]}
    maintenance: {schedule: "0 4 * * 0", suspend: false, command: ["python", "-m", "graph_extract.cli", "maintenance"]}
    theme-build: {schedule: "0 5 * * 0", suspend: true, command: ["python", "-m", "theme_builder.cli", "theme-build"]}
```

`templates/_helpers.tpl`:
```yaml
{{- define "graph-rag.labels" -}}
app.kubernetes.io/part-of: graph-rag
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "graph-rag.image" -}}
{{ .Values.image.repository }}:{{ required "image.tag is required (sha-<7-char commit>)" .Values.image.tag }}
{{- end }}

{{- define "graph-rag.envFrom" -}}
envFrom:
  - secretRef:
      name: {{ .Values.secretName }}
  - configMapRef:
      name: graph-rag-config
{{- end }}

{{- define "graph-rag.podSecurity" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
{{- end }}

{{- define "graph-rag.containerSecurity" -}}
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
{{- end }}
```

`templates/configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: graph-rag-config
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
data:
{{- range $k, $v := .Values.config }}
  {{ $k }}: {{ $v | quote }}
{{- end }}
```

`templates/postgres.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: graph-rag-postgres
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  clusterIP: None
  selector: {app: graph-rag-postgres}
  ports: [{name: postgres, port: 5432, targetPort: 5432}]
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: graph-rag-postgres
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  serviceName: graph-rag-postgres
  replicas: 1
  selector:
    matchLabels: {app: graph-rag-postgres}
  template:
    metadata:
      labels: {app: graph-rag-postgres}
    spec:
      securityContext:
        fsGroup: 70
      containers:
        - name: postgres
          image: {{ .Values.postgres.image }}
          ports: [{containerPort: 5432, name: postgres}]
          env:
            - {name: POSTGRES_DB, value: {{ .Values.postgres.database | quote }}}
            - {name: POSTGRES_USER, value: {{ .Values.postgres.user | quote }}}
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef: {name: {{ .Values.secretName }}, key: POSTGRES_PASSWORD}
            - {name: PGDATA, value: /var/lib/postgresql/data/pgdata}
          readinessProbe:
            exec: {command: ["pg_isready", "-U", {{ .Values.postgres.user | quote }}]}
            periodSeconds: 10
          resources: {{- toYaml .Values.postgres.resources | nindent 12 }}
          volumeMounts:
            - {name: data, mountPath: /var/lib/postgresql/data}
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: [ReadWriteOnce]
        storageClassName: {{ .Values.postgres.storageClass }}
        resources:
          requests:
            storage: {{ .Values.postgres.size }}
```

`templates/sync.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graph-rag-sync
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.sync.replicas }}
  strategy: {type: Recreate}   # single-flight poller: never two at once during a rollout
  selector:
    matchLabels: {app: graph-rag-sync}
  template:
    metadata:
      labels: {app: graph-rag-sync}
    spec:
      {{- include "graph-rag.podSecurity" . | nindent 6 }}
      containers:
        - name: sync
          image: {{ include "graph-rag.image" . }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command: ["uvicorn", "graph_sync.app:main", "--factory", "--host", "0.0.0.0", "--port", "8000"]
          {{- include "graph-rag.envFrom" . | nindent 10 }}
          {{- include "graph-rag.containerSecurity" . | nindent 10 }}
          ports: [{containerPort: 8000, name: http}]
          readinessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 15}
          livenessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 30, failureThreshold: 5}
          resources: {{- toYaml .Values.sync.resources | nindent 12 }}
---
apiVersion: v1
kind: Service
metadata:
  name: graph-rag-sync
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  selector: {app: graph-rag-sync}
  ports: [{name: http, port: 8000, targetPort: http}]
```

`templates/answer.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graph-rag-answer
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  replicas: 1
  selector:
    matchLabels: {app: graph-rag-answer}
  template:
    metadata:
      labels: {app: graph-rag-answer}
    spec:
      {{- include "graph-rag.podSecurity" . | nindent 6 }}
      containers:
        - name: answer
          image: {{ include "graph-rag.image" . }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command: ["uvicorn", "answer_api.app:main", "--factory", "--host", "0.0.0.0", "--port", "8000"]
          {{- include "graph-rag.envFrom" . | nindent 10 }}
          {{- include "graph-rag.containerSecurity" . | nindent 10 }}
          ports: [{containerPort: 8000, name: http}]
          # startup verifies the vector indexes (ensure_vector_indexes waits up to
          # VECTOR_INDEX_STARTUP_WAIT_SECONDS, default 60) before serving
          startupProbe: {httpGet: {path: /health, port: http}, periodSeconds: 10, failureThreshold: 18}
          readinessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 15}
          livenessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 30, failureThreshold: 5}
          resources: {{- toYaml .Values.answer.resources | nindent 12 }}
---
apiVersion: v1
kind: Service
metadata:
  name: graph-rag-answer
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  selector: {app: graph-rag-answer}
  ports: [{name: http, port: 8000, targetPort: http}]
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: graph-rag-answer
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  ingressClassName: traefik
  rules:
    - host: {{ .Values.answer.host }}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service: {name: graph-rag-answer, port: {name: http}}
```

`templates/worker.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graph-rag-worker
  labels: {{- include "graph-rag.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.worker.replicas }}
  selector:
    matchLabels: {app: graph-rag-worker}
  template:
    metadata:
      labels: {app: graph-rag-worker}
    spec:
      {{- include "graph-rag.podSecurity" . | nindent 6 }}
      # A worker finishes its in-flight batch on SIGTERM; a killed one's jobs are
      # re-queued by the lease reaper (SEMANTIC_REAPER_LEASE_SECONDS).
      terminationGracePeriodSeconds: {{ .Values.worker.terminationGracePeriodSeconds }}
      containers:
        - name: worker
          image: {{ include "graph-rag.image" . }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command: ["python", "-m", "graph_sync.cli", "worker"]
          args: {{- toYaml .Values.worker.args | nindent 12 }}
          {{- include "graph-rag.envFrom" . | nindent 10 }}
          {{- include "graph-rag.containerSecurity" . | nindent 10 }}
          resources: {{- toYaml .Values.worker.resources | nindent 12 }}
```

`templates/cronjobs.yaml`:
```yaml
{{- range $name, $job := .Values.cron.jobs }}
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: graph-rag-{{ $name }}
  labels: {{- include "graph-rag.labels" $ | nindent 4 }}
spec:
  schedule: {{ $job.schedule | quote }}
  suspend: {{ $job.suspend }}
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 0
      template:
        spec:
          restartPolicy: Never
          {{- include "graph-rag.podSecurity" $ | nindent 10 }}
          containers:
            - name: {{ $name }}
              image: {{ include "graph-rag.image" $ }}
              imagePullPolicy: {{ $.Values.image.pullPolicy }}
              command: {{- toYaml $job.command | nindent 16 }}
              {{- include "graph-rag.envFrom" $ | nindent 14 }}
              {{- include "graph-rag.containerSecurity" $ | nindent 14 }}
              resources: {{- toYaml $.Values.cron.resources | nindent 16 }}
{{- end }}
```

If an indentation in a template renders invalid YAML, fix the template (not the test); `helm template` output must parse with `yaml.safe_load_all`.

- [ ] **Step 4: Run to verify pass** — `uv run --extra dev pytest tests/unit/test_helm_chart.py -q` → PASS; also `helm lint deploy/helm/graph-rag --set image.tag=sha-test`.

- [ ] **Step 5: Mutation-test** (script asserting `old in s`; restore; never commit): (a) swap the two `envFrom` entries in `_helpers.tpl`; (b) set `worker.replicas: 1` in values; (c) remove `concurrencyPolicy: Forbid`; (d) add `API_KEY: "x"` under `config:` in values.yaml; (e) drop `required` from the image helper. Each must fail a test.

- [ ] **Step 6: Gate and commit** — `uv run ruff check src tests && uv run --extra dev pytest tests/unit -q`. Commit: `feat(deploy): Helm chart for the graph-rag namespace`.

---

### Task 3: Connectivity check and cut-over runbook

**Files:**
- Create: `src/graph_sync/connectivity.py`, `docs/deploy/k3s.md`
- Test: `tests/unit/test_connectivity.py`

**Interfaces:**
- Consumes: `graph_extract.config.get_extract_settings()` (`neo4j_uri`, `neo4j_user`, `neo4j_password`, `docext_base_url`, `docext_verify_tls`, `embed_base_url`, `chonkie_base_url`, `llm_base_url`, `cheap_llm_base_url`); `graph_sync.config.get_settings()` (`postgres_dsn`).
- Produces: `CheckResult(name: str, ok: bool, detail: str)`, `async run_checks(checks: list[tuple[str, Callable[[], Awaitable[str]]]]) -> list[CheckResult]`, `redact(text: str, secrets: list[str]) -> str`, `report(results: list[CheckResult]) -> tuple[str, int]`, `main()` (`python -m graph_sync.connectivity`).

- [ ] **Step 1: Failing tests** — `tests/unit/test_connectivity.py`:

```python
import pytest

from graph_sync import connectivity as c

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_real_settings(monkeypatch):
    """The unit conftest strips .env, so real settings cannot be built here."""
    monkeypatch.setattr(c, "_secret_values", lambda: [])


async def _ok() -> str:
    return "fine"


async def _boom() -> str:
    raise ConnectionError("refused at postgresql://u:hunter2@host/db")


async def test_run_checks_records_success_and_failure_without_raising():
    results = await c.run_checks([("a", _ok), ("b", _boom)])
    assert [(r.name, r.ok) for r in results] == [("a", True), ("b", False)]
    assert results[0].detail == "fine"
    assert "ConnectionError" in results[1].detail


def test_report_exit_code_is_nonzero_when_anything_failed():
    text, code = c.report([c.CheckResult("a", True, "x"), c.CheckResult("b", False, "y")])
    assert code == 1 and "FAIL" in text and "b" in text
    assert c.report([c.CheckResult("a", True, "x")])[1] == 0


def test_a_dsn_contributes_its_password_as_a_separate_secret():
    assert "hunter2" in c._dsn_passwords(["postgresql://u:hunter2@host:5432/db", "plain"])


def test_redact_removes_every_secret_value():
    assert c.redact("postgresql://u:hunter2@host/db key=sk-abc", ["hunter2", "sk-abc", ""]) \
        == "postgresql://u:***@host/db key=***"


async def test_failure_details_are_redacted_by_run_checks(monkeypatch):
    monkeypatch.setattr(c, "_secret_values", lambda: ["hunter2"])
    (r,) = await c.run_checks([("pg", _boom)])
    assert "hunter2" not in r.detail and "***" in r.detail
```

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_connectivity.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** — `src/graph_sync/connectivity.py`:

```python
"""Pre-flight connectivity check for a deployed graph-rag (docs/deploy/k3s.md).

    python -m graph_sync.connectivity

Read-only and free: no LLM tokens, no writes. Checks every dependency a worker or
the answer API needs and exits non-zero naming each failure. Error text is redacted
of every credential the settings hold before it is printed.
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import asyncpg
import httpx
from neo4j import AsyncGraphDatabase

from graph_extract.config import get_extract_settings
from graph_sync.config import get_settings

TIMEOUT = 15.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _secret_values() -> list[str]:
    """Every settings value whose field name marks it as a credential."""
    out: list[str] = []
    for model in (get_extract_settings(), get_settings()):
        for key, value in model.model_dump().items():
            if any(w in key.lower() for w in ("password", "key", "secret", "token", "dsn")):
                if isinstance(value, str) and value:
                    out.append(value)
    return out + _dsn_passwords(out)


def _dsn_passwords(values: list[str]) -> list[str]:
    """An error can quote a DSN partially; redact its password on its own too."""
    out = []
    for v in values:
        pw = urlparse(v).password if "://" in v else None
        if pw:
            out.append(pw)
    return out


def redact(text: str, secrets: list[str]) -> str:
    for s in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(s, "***")
    return text


async def run_checks(
        checks: list[tuple[str, Callable[[], Awaitable[str]]]]) -> list[CheckResult]:
    results: list[CheckResult] = []
    secrets = _secret_values()
    for name, check in checks:
        try:
            detail = await asyncio.wait_for(check(), TIMEOUT)
            results.append(CheckResult(name, True, redact(detail, secrets)))
        except Exception as exc:  # report every failure, never stop at the first
            results.append(CheckResult(
                name, False, redact(f"{type(exc).__name__}: {exc}", secrets)[:300]))
    return results


def report(results: list[CheckResult]) -> tuple[str, int]:
    lines = [f"{'OK  ' if r.ok else 'FAIL'}  {r.name:<12} {r.detail}" for r in results]
    return "\n".join(lines), 0 if all(r.ok for r in results) else 1


async def _http(url: str, verify: bool = True) -> str:
    """Reachable = any HTTP response below 500 (auth errors still prove the route)."""
    async with httpx.AsyncClient(timeout=TIMEOUT, verify=verify) as client:
        resp = await client.get(url)
    if resp.status_code >= 500:
        raise RuntimeError(f"HTTP {resp.status_code} from {url}")
    return f"HTTP {resp.status_code} {url}"


def _checks() -> list[tuple[str, Callable[[], Awaitable[str]]]]:
    s = get_extract_settings()
    g = get_settings()

    async def neo4j() -> str:
        driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        try:
            await driver.execute_query("RETURN 1")
        finally:
            await driver.close()
        return s.neo4j_uri

    async def postgres() -> str:
        conn = await asyncpg.connect(g.postgres_dsn, timeout=TIMEOUT)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()
        return "SELECT 1 ok"

    return [
        ("neo4j", neo4j),
        ("postgres", postgres),
        ("docextractor", lambda: _http(s.docext_base_url.rstrip("/") + "/api/health",
                                       verify=s.docext_verify_tls)),
        ("embedder", lambda: _http(s.embed_base_url.rstrip("/") + "/models")),
        ("chunker", lambda: _http(s.chonkie_base_url.rstrip("/") + "/")),
        ("llm", lambda: _http(s.llm_base_url.rstrip("/") + "/models")),
        ("cheap-llm", lambda: _http(s.cheap_llm_base_url.rstrip("/") + "/models")),
    ]


def main() -> None:
    text, code = report(asyncio.run(run_checks(_checks())))
    print(text)
    sys.exit(code)


if __name__ == "__main__":
    main()
```

Confirm the settings field names above exist (`grep` the two config modules) and use the real names if any differ.

- [ ] **Step 4: Run to verify pass** — `uv run --extra dev pytest tests/unit/test_connectivity.py -q`; `uv run ruff check src tests`; `uv run mypy src`.

- [ ] **Step 5: Runbook** — create `docs/deploy/k3s.md` with these sections, commands verbatim, adapting only names that differ from the chart:

1. **Prerequisites** — `kubectl` context on srv-k3s; `helm`; the image `ghcr.io/carev01/graph-rag:sha-<7>` built by the `image` workflow. Verify it pulls anonymously (`docker logout ghcr.io; docker pull …`); if not, set the GHCR package's visibility to public in GitHub → Packages → graph-rag → Settings (the repo is public and the image holds no secret).
2. **Namespace** — `kubectl create namespace graph-rag` and `kubectl label namespace graph-rag stack.stage=production` (enrols it in Kasten's daily `production-apps-backup` backup + export).
3. **Secret, from `.env`, never committed** — copy `.env` to a file under a private temp dir (`umask 077`); strip surrounding quotes from values (`--from-env-file` keeps them literally); set `POSTGRES_DSN=postgresql://graphsync:<pw>@graph-rag-postgres:5432/graphsync` and add `POSTGRES_PASSWORD=<pw>` (a new random password: `openssl rand -hex 24`); `kubectl -n graph-rag create secret generic graph-rag-secret --from-env-file=<file>`; `shred -u <file>`. Never paste the file or the secret into a chat, ticket or commit.
4. **Install, quiesced** — `helm upgrade --install graph-rag deploy/helm/graph-rag -n graph-rag --set image.tag=sha-<7> --set sync.replicas=0` (workers default to 0, theme-build suspended). Wait for `graph-rag-postgres-0` Ready and `graph-rag-answer` Ready.
5. **Connectivity** — `kubectl -n graph-rag exec deploy/graph-rag-answer -- python -m graph_sync.connectivity`; every line must be OK before continuing.
6. **Migrate the state** — on the dev host: `docker start graphrag-postgres`; `docker exec graphrag-postgres pg_dump -U <user> -Fc graphsync > <private-dir>/graphsync.dump`; record `SELECT status, lane, count(*) FROM semantic_jobs GROUP BY 1,2` and `SELECT count(*) FROM sync_cursor`, `bootstrap_progress`, `token_ledger`; `docker stop graphrag-postgres`. Then `kubectl -n graph-rag cp <dump> graph-rag-postgres-0:/tmp/graphsync.dump`; `kubectl -n graph-rag exec graph-rag-postgres-0 -- pg_restore -U graphsync -d graphsync --clean --if-exists --no-owner /tmp/graphsync.dump`; re-run the same count queries in the pod and compare — they must match exactly; delete the dump file on both sides.
7. **Start the poller** — `helm upgrade … --reuse-values --set sync.replicas=1`; check `kubectl -n graph-rag logs deploy/graph-rag-sync`.
8. **One-article smoke ingest** — `kubectl -n graph-rag exec deploy/graph-rag-sync -- python -m graph_sync.cli worker --max-batches 1 --batch 1`; confirm the batch summary line and `python -m graph_sync.cli queue-status`. (Paid: one article.)
9. **Scale workers** — `helm upgrade … --reuse-values --set worker.replicas=N`; watch `kubectl -n graph-rag top pods` against the requests above.
10. **Rollback** — `helm upgrade … --reuse-values --set worker.replicas=0` stops spending; `helm rollback graph-rag <rev> -n graph-rag`; the Postgres PVC survives uninstall and is in Kasten.
11. **Theme-build** — enable after the baseline rebuild: `--set cron.jobs.theme-build.suspend=false`.

- [ ] **Step 6: Gate and commit** — full CI gate: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live" -q` (foreground, ~15 min). Commit: `feat(deploy): in-image connectivity check and k3s cut-over runbook`.
