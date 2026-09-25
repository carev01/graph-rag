"""Static guards on docs/deploy/k3s.md: it is executed verbatim against production,
so the properties the cut-over review made binding are pinned here."""
from __future__ import annotations

import json
import re
from pathlib import Path

RUNBOOK = Path(__file__).parents[2] / "docs" / "deploy" / "k3s.md"
TEXT = RUNBOOK.read_text()
BLOCKS = re.findall(r"```\n(.*?)```", TEXT, re.S)


def _commands(prefix: str) -> list[str]:
    joined = [b.replace("\\\n", " ") for b in BLOCKS]
    return [line for b in joined for line in b.splitlines() if line.strip().startswith(prefix)]


def test_every_one_off_pod_has_a_dry_run_with_flags_before_the_separator():
    runs = _commands("kubectl -n graph-rag run ")
    names = {re.search(r"run (\S+)", r).group(1) for r in runs}
    # bootstrap-first rehearsal (§8): relane report, relane apply, and the
    # per-source structural bootstrap are each their own named one-off pod.
    assert names == {"graph-rag-connectivity", "graph-rag-crosscheck",
                     "graph-rag-sync-once", "graph-rag-smoke",
                     "graph-rag-relane", "graph-rag-relane-apply",
                     "graph-rag-bootstrap-src"}
    for name in names:
        mine = [r for r in runs if f"run {name} " in r]
        dry = [r for r in mine if "--dry-run=client" in r]
        assert len(dry) == 1 and len(mine) == 2, name
        head, _, _ = dry[0].partition(" -- ")
        assert "--dry-run=client -o yaml" in head, name
        assert "--restart=Never" in head, name


def test_every_pod_override_carries_resources_env_and_security():
    overrides = re.findall(r"ovr=\$\(cat <<'JSON'\n(.*?)\nJSON\n\)", TEXT, re.S)
    # 4 original (connectivity, crosscheck, sync-once, smoke) + 3 from the
    # bootstrap-first rehearsal (relane report, relane apply, bootstrap-src).
    assert len(overrides) == 7
    for raw in overrides:
        spec = json.loads(raw)["spec"]
        assert spec["automountServiceAccountToken"] is False
        (c,) = spec["containers"]
        assert c["image"] == "ghcr.io/carev01/graph-rag:sha-abc1234"
        assert c["envFrom"][0] == {"secretRef": {"name": "graph-rag-secret"}}
        assert set(c["resources"]) == {"requests", "limits"}
        assert c["securityContext"]["allowPrivilegeEscalation"] is False


def test_every_rollout_status_is_bounded():
    # Was 5 before the bootstrap-first rehearsal (§8): `sync.replicas=1` (and its
    # rollout-status wait) no longer fires here -- the poller stays suspended
    # until bootstrap completes, a later runbook step -- leaving postgres,
    # answer, and the two worker-scaling waits (§10).
    cmds = _commands("kubectl -n graph-rag rollout status")
    assert len(cmds) >= 4
    for cmd in cmds:
        assert "--timeout=" in cmd, cmd


def test_quiesced_install_stops_every_writer():
    (install,) = _commands("helm upgrade --install")
    for flag in ("sync.replicas=0", "answer.replicas=0",
                 "cron.jobs.cleanup.suspend=true", "cron.jobs.maintenance.suspend=true"):
        assert flag in install, flag


def test_no_exec_into_app_deployments_for_one_off_commands():
    assert "exec deploy/graph-rag-answer" not in TEXT
    assert "exec deploy/graph-rag-sync -- python -m graph_sync.cli sync-once" not in TEXT


def test_secret_build_drops_the_admin_key():
    assert "sed -i '/^DOCEXT_ADMIN_KEY=/d'" in TEXT


def test_semantic_claim_source_ids_commas_are_escaped_for_helm():
    """Helm's `--set`/`--set-string` value parser (`strvals`) splits a VALUE on
    a bare comma to start a new `key=value` assignment -- quoting the whole
    flag for the shell does not protect against this, since the shell has
    already handed Helm one argument by the time `strvals` parses it inside
    that argument. `--set-string config.SEMANTIC_CLAIM_SOURCE_IDS=a,b` fails
    outright (`key "b" has no value`), not with a silently-truncated
    single-id scope. Every assignment of this key in the runbook must escape
    its comma(s) with a backslash, or a multi-source rehearsal cannot even
    reach the cluster."""
    assignments = re.findall(r"SEMANTIC_CLAIM_SOURCE_IDS=(\S+)", TEXT)
    assert assignments, "expected at least one SEMANTIC_CLAIM_SOURCE_IDS= assignment in the runbook"
    for value in assignments:
        unescaped = value.replace("\\,", "")
        assert "," not in unescaped, f"unescaped comma in SEMANTIC_CLAIM_SOURCE_IDS={value!r}"
