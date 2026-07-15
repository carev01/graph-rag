"""Cheap CliRunner smoke tests: every command's --help works without
constructing any live dependency (graphiti/neo4j/docext), since Typer
short-circuits on --help before the command body ever runs.
"""
from typer.testing import CliRunner

from graph_extract.cli import _render_quality_report_md, app

runner = CliRunner()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "probe" in result.stdout
    assert "eval" in result.stdout


def test_ingest_help():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--source-id" in result.stdout
    assert "--limit" in result.stdout


def test_probe_help():
    result = runner.invoke(app, ["probe", "--help"])
    assert result.exit_code == 0
    assert "--article-ids" in result.stdout
    assert "--n" in result.stdout


def test_eval_help():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    for sub in ("dedup", "provenance", "cost", "quality"):
        assert sub in result.stdout


def test_eval_dedup_help():
    result = runner.invoke(app, ["eval", "dedup", "--help"])
    assert result.exit_code == 0


def test_eval_provenance_help():
    result = runner.invoke(app, ["eval", "provenance", "--help"])
    assert result.exit_code == 0
    assert "--sample" in result.stdout


def test_eval_cost_help():
    result = runner.invoke(app, ["eval", "cost", "--help"])
    assert result.exit_code == 0
    assert "--episodes-processed" in result.stdout


def test_eval_quality_help():
    result = runner.invoke(app, ["eval", "quality", "--help"])
    assert result.exit_code == 0
    assert "--sample" in result.stdout


def test_quality_baseline_help():
    result = runner.invoke(app, ["quality-baseline", "--help"])
    assert result.exit_code == 0
    assert "--sample" in result.stdout


def test_quality_report_help():
    result = runner.invoke(app, ["quality-report", "--help"])
    assert result.exit_code == 0
    assert "--sample" in result.stdout


def test_render_quality_report_md_handles_aliased_should_distinct_pair():
    """Regression test: `dedup_report_v2` now emits canonical (str, str)
    pairs for `should_distinct` entries even when a `quality_labels`
    member is an alias list. Before the fix, `pair` could contain a raw
    list member, and `tuple(p["pair"])` in `_render_quality_report_md`
    would raise `TypeError: unhashable type: 'list'`.
    """
    baseline: dict = {
        "noise": {"entities": {}, "facts": {}},
        "dedup": {
            "should_merge": {},
            "should_distinct": [],
            "suspect_false_merge": {},
        },
        "type_precision": {"per_type": {}},
    }
    current: dict = {
        "noise": {
            "entities": {"rate": 0.1, "total": 10},
            "facts": {"rate": 0.2, "total": 20},
        },
        "dedup": {
            "should_merge": {},
            "should_distinct": [
                {
                    "pair": ["AWS Backup Vault Lock", "Azure immutable vault"],
                    "state": "distinct",
                    "collapsed": False,
                    "a_nodes": 1,
                    "b_nodes": 1,
                },
            ],
            "suspect_false_merge": {"count": 0},
        },
        "type_precision": {"precision": 0.9, "sampled": 5, "per_type": {}},
    }

    markdown = _render_quality_report_md(baseline, current)

    assert isinstance(markdown, str)
    assert "AWS Backup Vault Lock" in markdown
    assert "Azure immutable vault" in markdown
