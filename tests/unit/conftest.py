"""Make unit tests hermetic against the developer's .env.

`ExtractSettings(_env_file=None, ...)` is NOT enough on its own: importing
`graphiti_core` calls `load_dotenv()`, which copies .env into `os.environ`, and
pydantic-settings reads `os.environ` regardless of `_env_file`. So any unit test
asserting a field's DEFAULT silently picks up the local .env value instead --
and only when some earlier test in the run has already imported graphiti_core,
which makes it an order-dependent failure that passes in isolation.

This bit us for real: changing CHEAP_LLM_MODEL in .env while evaluating
extraction models broke three unrelated unit tests. Strip every settings-derived
variable from the environment for the whole unit suite so defaults are actually
defaults.
"""
from __future__ import annotations

import pytest

from graph_extract.config import ExtractSettings


@pytest.fixture(autouse=True)
def _hermetic_settings_env(monkeypatch):
    for field in ExtractSettings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)
