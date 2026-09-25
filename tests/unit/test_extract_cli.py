"""Cheap CliRunner smoke tests: every command's --help works without
constructing any live dependency (graphiti/neo4j/docext), since Typer
short-circuits on --help before the command body ever runs.
"""
import re

import pytest
from typer.testing import CliRunner

from graph_extract.cli import _render_quality_report_md, app

runner = CliRunner()

# Typer renders help through rich, which forces a styled terminal when it sees
# GITHUB_ACTIONS/FORCE_COLOR (decided at import time, so a runner env cannot undo
# it): option names arrive wrapped in ANSI escapes and a plain substring check
# fails on CI only. Assert on the text, not the styling.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


class _FakeAsync:
    async def aclose(self): pass
    async def close(self): pass


class _FakeNeo:
    def driver(self, *a, **k): return _FakeAsync()


class _FakeLLM:
    async def generate_response(self, *a, **k): return {}


class _FakeGraphiti(_FakeAsync):
    """_build_ingest_driver wraps graphiti.llm_client (dedup guard), so the
    double must carry one; a bare string no longer suffices."""
    def __init__(self, name):
        self.name = name
        self.llm_client = _FakeLLM()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in _plain(result.stdout)
    assert "probe" in _plain(result.stdout)
    assert "eval" in _plain(result.stdout)


def test_ingest_help():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--source-id" in _plain(result.stdout)
    assert "--limit" in _plain(result.stdout)


def test_probe_help():
    result = runner.invoke(app, ["probe", "--help"])
    assert result.exit_code == 0
    assert "--article-ids" in _plain(result.stdout)
    assert "--n" in _plain(result.stdout)


def test_maintenance_help():
    result = runner.invoke(app, ["maintenance", "--help"])
    assert result.exit_code == 0
    assert "housekeeping" in _plain(result.stdout)
    assert "reconcile" in _plain(result.stdout)


def test_eval_help():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    for sub in ("dedup", "provenance", "cost", "quality"):
        assert sub in _plain(result.stdout)


def test_eval_dedup_help():
    result = runner.invoke(app, ["eval", "dedup", "--help"])
    assert result.exit_code == 0


def test_eval_provenance_help():
    result = runner.invoke(app, ["eval", "provenance", "--help"])
    assert result.exit_code == 0
    assert "--sample" in _plain(result.stdout)


def test_eval_cost_help():
    result = runner.invoke(app, ["eval", "cost", "--help"])
    assert result.exit_code == 0
    assert "--episodes-processed" in _plain(result.stdout)


def test_eval_quality_help():
    result = runner.invoke(app, ["eval", "quality", "--help"])
    assert result.exit_code == 0
    assert "--sample" in _plain(result.stdout)


def test_cleanup_help():
    result = runner.invoke(app, ["cleanup", "--help"])
    assert result.exit_code == 0


def test_merge_duplicates_help():
    """Report-only unless `--apply`, and the help itself says so: the operator
    reads this before the command ever prints its runtime warning."""
    result = runner.invoke(app, ["merge-duplicates", "--help"])
    assert result.exit_code == 0
    assert "--apply" in _plain(result.stdout)
    assert "Report-only by default" in _plain(result.stdout)


def test_quality_baseline_help():
    result = runner.invoke(app, ["quality-baseline", "--help"])
    assert result.exit_code == 0
    assert "--sample" in _plain(result.stdout)


def test_quality_report_help():
    result = runner.invoke(app, ["quality-report", "--help"])
    assert result.exit_code == 0
    assert "--sample" in _plain(result.stdout)


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


@pytest.mark.asyncio
async def test_build_ingest_driver_builds_cheap_tier_when_configured(monkeypatch):
    import graph_extract.cli as cli
    from graph_extract.config import ExtractSettings

    built = []
    monkeypatch.setattr(cli, "build_graphiti",
                        lambda s: built.append("strong") or _FakeGraphiti("SG"))
    monkeypatch.setattr(cli, "build_cheap_graphiti",
                        lambda s: built.append("cheap") or _FakeGraphiti("CG"))
    monkeypatch.setattr(cli, "make_docext_client", lambda **k: _FakeAsync())
    monkeypatch.setattr(cli, "AsyncGraphDatabase", _FakeNeo())
    async def _noop(g, **_kwargs): return None
    monkeypatch.setattr(cli, "init_indices", _noop)

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                        cheap_llm_api_key="or-key")   # routing on by default
    ingest, g, dx, drv = await cli._build_ingest_driver(s)
    assert "cheap" in built and ingest._cheap is not None
    assert ingest._cheap.instructions.endswith(cli.CHEAP_TIER_SALIENCE[-40:])
    assert ingest._cheap.max_chunk_tokens == s.cheap_max_chunk_tokens


@pytest.mark.asyncio
async def test_build_ingest_driver_strong_only_without_cheap_key(monkeypatch):
    import graph_extract.cli as cli
    from graph_extract.config import ExtractSettings

    monkeypatch.setattr(cli, "build_graphiti", lambda s: _FakeGraphiti("SG"))
    monkeypatch.setattr(cli, "build_cheap_graphiti", lambda s: (_ for _ in ()).throw(AssertionError("must not build cheap")))
    monkeypatch.setattr(cli, "make_docext_client", lambda **k: _FakeAsync())
    monkeypatch.setattr(cli, "AsyncGraphDatabase", _FakeNeo())
    async def _noop(g, **_kwargs): return None
    monkeypatch.setattr(cli, "init_indices", _noop)

    # cheap_llm_api_key set empty EXPLICITLY (init kwargs beat .env) so this "no
    # cheap key -> strong only" test is hermetic even when the deployment .env
    # supplies a real CHEAP_LLM_API_KEY.
    s = ExtractSettings(_env_file=None, cheap_llm_api_key="", docext_base_url="http://x",
                        docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u",
                        neo4j_password="p")
    ingest, g, dx, drv = await cli._build_ingest_driver(s)
    assert ingest._cheap is None
