"""`graph_sync.cli._configure_logging`: without a handler, root logger's default
level (WARNING) silently drops the worker's INFO-level batch-summary/dedup/
vector-search/timing lines before they ever reach `kubectl logs`. See
docs/deploy/k3s.md (§8/§9) -- this is what makes the smoke-ingest and scale-worker
logs actually show anything in production."""
from __future__ import annotations

import logging

from graph_sync.cli import _configure_logging

# pytest's own `_pytest.logging` plugin re-attaches a fresh LogCaptureHandler to
# the root logger around every test phase, including a new one for `call` --
# separate from `setup`. So `monkeypatch.setattr(..., "handlers", ...)` must
# happen as the first line of the test body itself (running inside `call`,
# *after* pytest has already re-attached its own handler for this phase), not
# in an autouse fixture (which only runs during `setup`, before that handler
# exists) -- otherwise `_configure_logging`'s own "already configured" guard
# sees pytest's handler and no-ops, and the test would pass or fail for the
# wrong reason. `monkeypatch.setattr` restores the original `handlers`/`level`
# after the test either way, so no manual teardown is needed here.


def test_configure_logging_enables_info_for_the_worker_logger(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    _configure_logging()
    assert logging.getLogger("graph_sync.semantic_worker").isEnabledFor(logging.INFO)


def test_configure_logging_respects_log_level_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    _configure_logging()
    assert not logging.getLogger("graph_sync.semantic_worker").isEnabledFor(logging.INFO)


def test_configure_logging_quiets_httpx_and_httpcore_to_warning(monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    _configure_logging()
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
    assert not logging.getLogger("httpcore").isEnabledFor(logging.INFO)


def test_configure_logging_does_not_reconfigure_when_a_handler_already_exists(monkeypatch):
    sentinel = logging.NullHandler()
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [sentinel])
    monkeypatch.setattr(root, "level", logging.ERROR)
    _configure_logging()
    assert root.level == logging.ERROR
    assert root.handlers == [sentinel]
