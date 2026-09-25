"""Make every non-live test independent of the developer's .env FILE.

`tests/unit/conftest.py` strips settings variables from `os.environ`, but
`ExtractSettings()` -- which app lifespans and CLI commands reach through
`get_extract_settings()` -- still reads `.env` itself. Locally that file supplies
the five required connection fields, so those tests pass; CI has no `.env`, so the
same tests die in `ExtractSettings` validation before asserting anything. That
went unnoticed for a week because the local gate could not reproduce it.

For non-live tests: never read `.env`, give the required fields obvious
placeholders (no test may reach a real endpoint through them -- integration tests
point clients at their own testcontainers), and drop the cached settings object
on both sides so no test sees another's. `@live` tests are left alone: they exist
to use the real `.env`. `@real_endpoints` tests keep `.env` too (they exercise a real
LLM/embedder path when one is reachable and skip it when not) and only get
placeholders for the required fields they are missing.
"""
from __future__ import annotations

import os

import pytest

from graph_extract.config import ExtractSettings, get_extract_settings

_REQUIRED_PLACEHOLDERS = {
    "DOCEXT_BASE_URL": "http://docextractor.placeholder.invalid",
    "DOCEXT_READ_KEY": "placeholder-not-a-key",
    "NEO4J_URI": "bolt://neo4j.placeholder.invalid:7687",
    "NEO4J_USER": "placeholder",
    "NEO4J_PASSWORD": "placeholder",
}


@pytest.fixture(autouse=True)
def _no_dotenv_file(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        yield
        return
    if request.node.get_closest_marker("real_endpoints"):
        # Keep .env's endpoints (the test skips its endpoint checks when they are
        # unreachable, as on CI); only supply the required fields it lacks.
        for name, value in _REQUIRED_PLACEHOLDERS.items():
            if name not in os.environ:
                monkeypatch.setenv(name, value)
        get_extract_settings.cache_clear()
        yield
        get_extract_settings.cache_clear()
        return
    monkeypatch.setitem(ExtractSettings.model_config, "env_file", None)
    # Importing graphiti_core runs load_dotenv(), copying .env into os.environ,
    # which pydantic-settings reads regardless of env_file: strip every
    # settings-derived variable so defaults are defaults (this used to live in
    # tests/unit/conftest.py; it must run BEFORE the placeholders below).
    for field in ExtractSettings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)
    for name, value in _REQUIRED_PLACEHOLDERS.items():
        monkeypatch.setenv(name, value)
    get_extract_settings.cache_clear()
    yield
    get_extract_settings.cache_clear()
