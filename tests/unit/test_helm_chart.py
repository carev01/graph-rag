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
