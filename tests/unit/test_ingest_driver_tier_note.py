"""BACKLOG 56: a failed article carries the extraction tier it failed on."""
from __future__ import annotations

import pytest

from graph_extract.ingest_driver import IngestDriver

pytestmark = pytest.mark.asyncio


class _Boom(IngestDriver):
    def __init__(self, routed: bool):   # no real dependencies needed
        self._routed = routed

    async def _ingest_article(self, article_id, res):
        if self._routed:
            res.tier, res.routed = "cheap", True
        raise RuntimeError("extraction failed")


@pytest.mark.parametrize("routed, note", [(True, "graph-rag tier=cheap"),
                                          (False, "graph-rag tier=unrouted")])
async def test_a_failure_is_noted_with_its_tier(routed, note):
    with pytest.raises(RuntimeError) as ei:
        await _Boom(routed).ingest_article("a1")
    assert note in getattr(ei.value, "__notes__", [])
