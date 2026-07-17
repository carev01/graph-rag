import pytest
from datetime import datetime, timezone
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti, init_indices, add_text_episode

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

async def test_add_one_episode_extracts_entities():
    s = get_extract_settings()
    g = build_graphiti(s)
    await init_indices(g)
    body = ("[AWS Backup › Backup vaults]\nAWS Backup supports immutable backups "
            "for Amazon S3 using Vault Lock in compliance mode, protecting against "
            "ransomware. Cross-region copy is supported.")
    res = await add_text_episode(g, s, name="live-test:0:deadbeef", body=body,
                                 source_description="AWS Backup docs",
                                 reference_time=datetime.now(timezone.utc))
    assert res.episode.uuid
    assert len(res.nodes) >= 2   # extracted some entities
    await g.close()


@pytest.mark.asyncio
async def test_add_text_episode_passes_custom_instructions():
    from datetime import datetime, timezone
    from graph_extract.graphiti_client import add_text_episode
    from graph_extract.config import ExtractSettings

    captured = {}

    class _FakeG:
        async def add_episode(self, **kw):
            captured.update(kw)
            return "ok"

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    await add_text_episode(_FakeG(), s, name="n", body="b", source_description="u",
                           reference_time=datetime.now(timezone.utc),
                           instructions="CUSTOM-INSTR")
    assert captured["custom_extraction_instructions"] == "CUSTOM-INSTR"
    assert captured["group_id"] == s.group_id


@pytest.mark.asyncio
async def test_add_text_episode_defaults_to_global_instructions():
    from datetime import datetime, timezone
    from graph_extract.graphiti_client import add_text_episode
    from graph_extract.ontology import EXTRACTION_INSTRUCTIONS
    from graph_extract.config import ExtractSettings

    captured = {}

    class _FakeG:
        async def add_episode(self, **kw):
            captured.update(kw)
            return "ok"

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    await add_text_episode(_FakeG(), s, name="n", body="b", source_description="u",
                           reference_time=datetime.now(timezone.utc))
    assert captured["custom_extraction_instructions"] == EXTRACTION_INSTRUCTIONS
