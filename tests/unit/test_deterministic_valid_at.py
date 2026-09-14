"""`valid_at` should be the episode's reference time, not an LLM's reading of the
fact text.

We told DocExtractor that `content_changed_at` would become the fact's `valid_at`
(2026-09-13 reply 3). It did not: it set the EPISODE reference time, while
`valid_at` stayed the output of graphiti's `extract_timestamps` call. Measured on
the pilot re-ingest: 817 of 3,590 facts dated (23%), and that call cost 21.7% of
all LLM time -- 1.5 h of a 6.9 h run -- for a field three-quarters empty and one
we argued should not be the sort key.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from graphiti_core.utils.maintenance import edge_operations

from graph_extract import deterministic_valid_at as dva

_T = datetime(2026, 8, 1, 14, 3, 11, tzinfo=timezone.utc)


@pytest.fixture
def restore():
    original = edge_operations._extract_edge_timestamps
    yield
    edge_operations._extract_edge_timestamps = original


def _edge(**kw):
    base = dict(uuid="e1", valid_at=None, invalid_at=None)
    base.update(kw)
    return SimpleNamespace(**base)


class _LLM:
    def __init__(self):
        self.calls = 0

    async def generate_response(self, *a, **kw):
        self.calls += 1
        return {}


@pytest.mark.asyncio
async def test_valid_at_becomes_the_episode_reference_time_with_no_llm_call(restore):
    assert dva.install_deterministic_valid_at(enabled=True) is True
    llm, edge = _LLM(), _edge()
    await edge_operations._extract_edge_timestamps(
        llm, edge, SimpleNamespace(valid_at=_T))
    assert edge.valid_at == _T
    assert llm.calls == 0, "the whole point is not making the call"


@pytest.mark.asyncio
async def test_an_edge_that_already_has_a_date_is_left_alone(restore):
    """graphiti's combined-extraction path can set these from the extraction
    prompt; the original function returns early on that and so must we."""
    dva.install_deterministic_valid_at(enabled=True)
    earlier = datetime(2020, 1, 1, tzinfo=timezone.utc)
    edge = _edge(valid_at=earlier)
    await edge_operations._extract_edge_timestamps(
        _LLM(), edge, SimpleNamespace(valid_at=_T))
    assert edge.valid_at == earlier


@pytest.mark.asyncio
async def test_an_edge_with_only_invalid_at_is_also_left_alone(restore):
    dva.install_deterministic_valid_at(enabled=True)
    edge = _edge(invalid_at=_T)
    await edge_operations._extract_edge_timestamps(
        _LLM(), edge, SimpleNamespace(valid_at=_T))
    assert edge.valid_at is None, "the original returns early on either field"


@pytest.mark.asyncio
async def test_no_episode_reference_time_means_no_date_invented(restore):
    """An undated episode must yield an undated edge. Substituting `now()` here
    would be the crawl-clock defect all over again."""
    dva.install_deterministic_valid_at(enabled=True)
    edge = _edge()
    await edge_operations._extract_edge_timestamps(
        _LLM(), edge, SimpleNamespace(valid_at=None))
    assert edge.valid_at is None
    await edge_operations._extract_edge_timestamps(_LLM(), edge, None)
    assert edge.valid_at is None


@pytest.mark.asyncio
async def test_disabled_leaves_graphitis_own_function_in_place(restore):
    original = edge_operations._extract_edge_timestamps
    assert dva.install_deterministic_valid_at(enabled=False) is False
    assert edge_operations._extract_edge_timestamps is original


@pytest.mark.asyncio
async def test_installing_twice_does_not_double_wrap(restore):
    dva.install_deterministic_valid_at(enabled=True)
    first = edge_operations._extract_edge_timestamps
    dva.install_deterministic_valid_at(enabled=True)
    assert edge_operations._extract_edge_timestamps is first


def test_uninstall_restores_the_original(restore):
    original = edge_operations._extract_edge_timestamps
    dva.install_deterministic_valid_at(enabled=True)
    assert dva.is_installed() is True
    dva.install_deterministic_valid_at(enabled=False)
    assert dva.is_installed() is False
    assert edge_operations._extract_edge_timestamps is original


def test_the_library_still_calls_it_from_both_paths():
    """Tripwire: the patch is worthless if graphiti stops routing through this
    function. Both the no-candidate early return and the resolved-as-new path
    call it today."""
    import ast
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(edge_operations.resolve_extracted_edge))
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "_extract_edge_timestamps"]
    assert len(calls) == 2, f"expected 2 call sites, found {len(calls)}"


def test_the_default_is_on():
    """The behaviour it replaces is measured worse on every axis: coverage,
    cost and semantic coherence."""
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.valid_at_from_content_changed is True
