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
from graphiti_core.search import search as _search_module
from graphiti_core.search.search import search as _defining_module_search
from graphiti_core.search import search_utils
from graphiti_core.utils.maintenance import edge_operations, node_operations

from graph_extract.vector_search import _ORIG_EDGE, _ORIG_NODE


# The settings-environment stripping that used to live here moved to
# tests/conftest.py (`_no_dotenv_file`), which also stops `.env` FILE reads and
# covers the integration suite.


# Every graphiti attribute this codebase patches PROCESS-WIDE, with the pristine
# value captured here at conftest import -- which pytest does during collection,
# before any test executes, so these are always the library's own functions.
#
# `build_graphiti` installs all of them and nothing uninstalls them. The CI
# command (`pytest -m "not live"`) collects `tests/integration/` before
# `tests/unit/` in ONE process, and `test_compat_harness.py` calls
# `build_graphiti` in a non-live test -- so without this, whichever unit tests
# assert on the pristine value fail depending on collection order. Running the
# two halves separately hides it, which is exactly how it reached CI twice.
#
# This has now bitten three times (settings env vars, `edge_operations.search`,
# `_extract_edge_timestamps`). Adding a patch site here is one line; writing a
# fourth bespoke fixture is not.
_PROCESS_WIDE_PATCHES = (
    (edge_operations, "search", _defining_module_search),
    (edge_operations, "_extract_edge_timestamps",
     edge_operations._extract_edge_timestamps),
    (search_utils, "get_entity_edge_return_query",
     search_utils.get_entity_edge_return_query),
    # graph_extract.vector_search.install_vector_search (called from
    # build_graphiti) patches these four process-wide too.
    (_search_module, "edge_similarity_search", _ORIG_EDGE),
    (_search_module, "node_similarity_search", _ORIG_NODE),
    (search_utils, "node_similarity_search", _ORIG_NODE),
    (node_operations, "node_similarity_search", _ORIG_NODE),
)


@pytest.fixture(autouse=True)
def _pristine_graphiti_attributes():
    """Pin every process-wide graphiti patch to the library's own function on both
    sides of each unit test, so the suite is order-independent."""
    for module, name, pristine in _PROCESS_WIDE_PATCHES:
        setattr(module, name, pristine)
    yield
    for module, name, pristine in _PROCESS_WIDE_PATCHES:
        setattr(module, name, pristine)
