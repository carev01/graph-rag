"""Cheap CliRunner smoke tests: every command's --help works without
constructing any live dependency (graphiti/neo4j/docext), since Typer
short-circuits on --help before the command body ever runs.
"""
from typer.testing import CliRunner

from graph_extract.cli import app

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
