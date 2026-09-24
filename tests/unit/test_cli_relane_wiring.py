"""`relane-jobs` cli wiring: report-only by default, `--apply` reaches both
steps, Neo4j/Postgres resources are always closed, and the printed line comes
from `graph_sync.relane.format_report` rather than being reassembled ad hoc in
the cli (which would drift from what the functions actually did)."""
from __future__ import annotations

import pytest

from graph_sync import cli
from graph_sync.config import Settings
from graph_sync.relane import BackfillReport, RelaneReport


def _sync_settings() -> Settings:
    return Settings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                    neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                    postgres_dsn="postgresql://x")


class _Repo:
    closed = False

    async def close(self) -> None:
        self.closed = True


class _Store:
    def __init__(self) -> None:
        self.closed = False
        self.schema_initialised = False

    async def init_schema(self) -> None:
        self.schema_initialised = True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("apply_flag", [False, True])
def test_relane_jobs_forwards_apply_to_both_steps(monkeypatch, apply_flag):
    received: dict = {}

    async def _fake_backfill(store, repo, *, apply):
        received["backfill_apply"] = apply
        return BackfillReport(candidates=1, backfilled=1, missing_in_graph=0)

    async def _fake_relane(store, repo, *, apply):
        received["relane_apply"] = apply
        return RelaneReport(checked=1, kept_incremental=1, moved_to_bootstrap=0)

    repo = _Repo()
    store = _Store()
    monkeypatch.setattr(cli, "get_settings", _sync_settings)
    monkeypatch.setattr(cli, "Neo4jRepo", lambda *a, **k: repo)
    monkeypatch.setattr(cli, "StateStore", lambda *a, **k: store)
    monkeypatch.setattr(cli, "backfill_source_ids", _fake_backfill)
    monkeypatch.setattr(cli, "relane_incremental_jobs", _fake_relane)

    cli.relane_jobs(apply=apply_flag)

    assert received == {"backfill_apply": apply_flag, "relane_apply": apply_flag}
    assert store.schema_initialised
    assert repo.closed and store.closed


def test_relane_jobs_default_flag_is_report_only():
    """The declared typer default for `--apply` must be `False`. Calling
    `cli.relane_jobs()` directly (as the other wiring tests do) does not
    exercise this -- bypassing typer's CLI parsing leaves the parameter bound
    to its raw `typer.Option(...)` sentinel, not the value a real invocation
    with no flag would resolve to -- so the default is asserted directly off
    the function signature instead."""
    import inspect

    default = inspect.signature(cli.relane_jobs).parameters["apply"].default
    assert default.default is False


def test_relane_jobs_prints_the_formatted_report(monkeypatch, capsys):
    async def _fake_backfill(store, repo, *, apply):
        return BackfillReport(candidates=3, backfilled=2, missing_in_graph=1)

    async def _fake_relane(store, repo, *, apply):
        return RelaneReport(checked=4, kept_incremental=1, moved_to_bootstrap=3)

    monkeypatch.setattr(cli, "get_settings", _sync_settings)
    monkeypatch.setattr(cli, "Neo4jRepo", lambda *a, **k: _Repo())
    monkeypatch.setattr(cli, "StateStore", lambda *a, **k: _Store())
    monkeypatch.setattr(cli, "backfill_source_ids", _fake_backfill)
    monkeypatch.setattr(cli, "relane_incremental_jobs", _fake_relane)

    cli.relane_jobs(apply=True)

    out = capsys.readouterr().out
    assert "2 backfilled" in out
    assert "3 moved to bootstrap" in out
    assert "APPLY" in out


def test_relane_jobs_closes_resources_even_if_a_step_raises(monkeypatch):
    repo = _Repo()
    store = _Store()

    async def _fake_backfill(store_, repo_, *, apply):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "get_settings", _sync_settings)
    monkeypatch.setattr(cli, "Neo4jRepo", lambda *a, **k: repo)
    monkeypatch.setattr(cli, "StateStore", lambda *a, **k: store)
    monkeypatch.setattr(cli, "backfill_source_ids", _fake_backfill)

    with pytest.raises(RuntimeError, match="boom"):
        cli.relane_jobs(apply=False)

    assert repo.closed and store.closed
