import pytest
from datetime import datetime, timezone
from graph_extract.graphiti_client import add_text_episode
from graph_extract.config import ExtractSettings
from graph_extract.ontology import EXTRACTION_INSTRUCTIONS

pytestmark = pytest.mark.asyncio


class _FakeG:
    async def add_episode(self, **kw):
        self._captured = kw
        return "ok"


async def test_add_text_episode_passes_custom_instructions():
    captured = {}

    class FakeG:
        async def add_episode(self, **kw):
            captured.update(kw)
            return "ok"

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    await add_text_episode(FakeG(), s, name="n", body="b", source_description="u",
                           reference_time=datetime.now(timezone.utc),
                           instructions="CUSTOM-INSTR")
    assert captured["custom_extraction_instructions"] == "CUSTOM-INSTR"
    assert captured["group_id"] == s.group_id


async def test_add_text_episode_defaults_to_global_instructions():
    captured = {}

    class FakeG:
        async def add_episode(self, **kw):
            captured.update(kw)
            return "ok"

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    await add_text_episode(FakeG(), s, name="n", body="b", source_description="u",
                           reference_time=datetime.now(timezone.utc))
    assert captured["custom_extraction_instructions"] == EXTRACTION_INSTRUCTIONS
