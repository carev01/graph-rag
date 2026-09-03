"""Verdict computation and markdown rendering for the compatibility harness. Pure —
takes CheckResults, returns a string."""
from __future__ import annotations

from compat.model import COMPAT_GROUP_ID, CheckResult, Verdict

_STATUS_ICON = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell. Details carry Neo4j error text and
    Cypher fragments, where a literal | would break the row."""
    return text.replace("|", "\\|").replace("\n", " ")


def _scored(results: list[CheckResult]) -> list[CheckResult]:
    """Only non-informational, non-skipped results move the verdict."""
    return [r for r in results if not r.informational and r.status != "skip"]


def verdict(results: list[CheckResult]) -> Verdict:
    failures = [r for r in _scored(results) if r.status == "fail"]
    if not failures:
        return "GO"
    if all(f.cypher5_retry == "pass" for f in failures):
        return "GO_WITH_CONFIG"
    return "NO_GO"


def _retry_cell(r: CheckResult) -> str:
    if r.status != "fail":
        return "-"
    if r.cypher5_retry is None:
        return "n/a (procedural)"
    return _STATUS_ICON[r.cypher5_retry]


def _actions(results: list[CheckResult], v: Verdict) -> list[str]:
    out: list[str] = []
    if v == "GO_WITH_CONFIG":
        out.append(
            "Set `db.query.default_language=CYPHER_5` on the server (every failure "
            "passes under Cypher 5, so no code change is required).")
    for r in _scored(results):
        if r.status == "fail" and r.cypher5_retry != "pass":
            out.append(f"Fix `{r.group}` / **{r.name}** — fails under both language "
                       f"versions: {r.detail}")
    if v == "GO":
        out.append("No action required — the target is compatible.")
    return out


def render(results: list[CheckResult], *, target: dict[str, str],
           teardown_error: str | None = None) -> str:
    v = verdict(results)
    lines = ["# Neo4j / Graphiti Compatibility Report", ""]
    lines.append("## Target")
    lines.append("")
    for key, value in target.items():
        lines.append(f"- **{key}:** {value}")
    lines += ["", f"## Verdict: {v}", ""]

    for action in _actions(results, v):
        lines.append(f"- {action}")
    lines.append("")

    if teardown_error:
        lines += [
            "> **MANUAL CLEANUP REQUIRED.** Harness teardown failed, so test data may "
            f"remain on the target. Delete it with "
            f"`MATCH (n {{group_id:'{COMPAT_GROUP_ID}'}}) DETACH DELETE n` and drop any "
            "`compat_`-prefixed indexes.",
            "",
            f"> Teardown error: {teardown_error}",
            "",
        ]

    lines += ["## Results", "",
              "| group | check | status | Cypher 5 | detail |",
              "|---|---|---|---|---|"]
    for r in results:
        name = f"{r.name} (info)" if r.informational else r.name
        lines.append(f"| {r.group} | {_cell(name)} | {_STATUS_ICON[r.status]} | "
                     f"{_retry_cell(r)} | {_cell(r.detail)} |")

    skipped = [r for r in results if r.status == "skip"]
    lines += ["", "## Not verified", ""]
    if skipped:
        for r in skipped:
            lines.append(f"- **{r.name}** ({r.group}) — {r.detail}")
    else:
        lines.append("- Nothing skipped; every check ran.")

    lines += [
        "",
        "## Side effects",
        "",
        "Graphiti's own indexes (created by `build_indices_and_constraints()`) are "
        "left in place deliberately — the call is idempotent and those indexes are "
        "exactly what a real bootstrap needs. `graph_sync.neo4j_repo.init_schema()` "
        "(run by the `graph_sync structural schema` check) similarly leaves 5 global "
        "constraints (`vendor_id`, `product_id`, `source_id`, `article_id`, "
        "`chapter_id`) and 2 indexes (`article_source`, `chapter_source`) on the "
        "target — harmless and arguably desirable (a real bootstrap needs them too), "
        "but disclosed here since they are not `compat_`-prefixed and teardown does "
        "not drop them. All harness *data* "
        f"(`group_id='{COMPAT_GROUP_ID}'`) and all `compat_`-prefixed indexes are "
        "removed in teardown.",
        "",
    ]
    return "\n".join(lines)
