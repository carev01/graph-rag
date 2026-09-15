"""The `ingest` command ends every run with the exact-name duplicate count.

Every constructor is stubbed (`_build_ingest_driver` is replaced whole), so
nothing connects and no extraction happens; what is pinned is the wiring: the
count is taken on the driver the run used, for the configured group, AFTER the
source was ingested (a count taken before it would describe the previous run),
and it is the last thing printed. The fake count is 7 and the group is not the
default, so a hard-coded zero or a default group cannot pass.
"""
from __future__ import annotations

from typer.testing import CliRunner

from graph_extract import cli
from graph_extract.config import ExtractSettings
from graph_extract.ingest_driver import IngestResult


class _Closable:
    async def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _Timings:
    def report(self) -> str:
        return "timings: n/a"


class _FakeIngest:
    _cheap = None

    def __init__(self, calls: list[str]):
        self.calls = calls
        self.timings = _Timings()

    async def ingest_source(self, source_id, limit):
        self.calls.append("ingest_source")
        return IngestResult(articles=3, episodes_added=2, episodes_skipped=1)


def test_ingest_prints_the_duplicate_count_last_and_after_the_run(monkeypatch):
    calls: list[str] = []
    counted: list[tuple[object, str]] = []
    driver = _Closable()
    settings = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                               neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                               group_id="g-ingest")

    async def _build(s):
        assert s is settings
        return _FakeIngest(calls), _Closable(), _Closable(), driver

    async def _cost(episodes):
        return {"episodes": episodes}

    async def _count(d, group_id):
        calls.append("count_duplicates")
        counted.append((d, group_id))
        return 7

    monkeypatch.setattr(cli, "get_extract_settings", lambda: settings)
    monkeypatch.setattr(cli, "_build_ingest_driver", _build)
    monkeypatch.setattr(cli, "cost_report", _cost)
    monkeypatch.setattr(cli, "count_duplicates", _count)

    result = CliRunner().invoke(cli.app, ["ingest", "--source-id", "src-1"])
    assert result.exit_code == 0, result.output
    assert calls == ["ingest_source", "count_duplicates"], "counted after the run, not before"
    assert counted == [(driver, "g-ingest")], "the run's own driver, the configured group"
    last = result.stdout.rstrip().splitlines()[-1]
    assert last.startswith("7 exact-name duplicate entities"), last
    assert "merge-duplicates" in last, "the line carries the remedy"
    assert result.stdout.index("ingest complete") < result.stdout.index("7 exact-name")
