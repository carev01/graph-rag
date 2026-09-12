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
from graphiti_core.search.search import search as _defining_module_search
from graphiti_core.utils.maintenance import edge_operations

from graph_extract.config import ExtractSettings


@pytest.fixture(autouse=True)
def _hermetic_settings_env(monkeypatch):
    for field in ExtractSettings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)


@pytest.fixture(autouse=True)
def _ungated_edge_operations_search():
    """Make unit tests hermetic against the contradiction gate.

    `build_graphiti` installs the gate as a PROCESS-WIDE patch of
    `edge_operations.search` and nothing uninstalls it. The CI command
    (`pytest -m "not live"`) collects `tests/integration/` before `tests/unit/`
    in one process, and `test_compat_harness.py` calls `build_graphiti` in a
    non-live test -- so by the time the unit suite runs, the attribute is already
    the gated wrapper and two tripwires in `test_contradiction_gate.py`
    (`test_is_gate_installed_reports_state`,
    `test_edge_operations_imports_search_by_name`) fail. Running the two halves
    separately hid it. Pin the attribute to the defining module's function on
    both sides of every unit test so the suite is order-independent again.
    """
    edge_operations.search = _defining_module_search
    yield
    edge_operations.search = _defining_module_search
