"""`graph_sync.cli._configure_logging`: without a handler, root logger's default
level (WARNING) silently drops the worker's INFO-level batch-summary/dedup/
vector-search/timing lines before they ever reach `kubectl logs`. See
docs/deploy/k3s.md (§8/§9) -- this is what makes the smoke-ingest and scale-worker
logs actually show anything in production."""
from __future__ import annotations

import logging

import pytest

from graph_sync.cli import _configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """The root logger is process-global state; save/restore it so this test
    file cannot leak a handler or level change into any other test."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)


def _reset_root_logger() -> None:
    """Clear whatever handler is on the root logger *right now*, inside the
    test body. pytest's own log-capture plugin re-attaches a fresh
    LogCaptureHandler around every test phase, including a new one for `call`
    -- so clearing in an autouse fixture (which only runs during `setup`,
    before that handler is attached) is too early. Doing it here, as the first
    line of the test, is what actually gets `_configure_logging` a genuinely
    unconfigured root logger to configure."""
    logging.getLogger().handlers.clear()


def test_configure_logging_enables_info_for_the_worker_logger(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    _reset_root_logger()
    _configure_logging()
    assert logging.getLogger("graph_sync.semantic_worker").isEnabledFor(logging.INFO)


def test_configure_logging_respects_log_level_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    _reset_root_logger()
    _configure_logging()
    assert not logging.getLogger("graph_sync.semantic_worker").isEnabledFor(logging.INFO)


def test_configure_logging_does_not_reconfigure_when_a_handler_already_exists():
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.ERROR)
    _configure_logging()
    assert root.level == logging.ERROR
