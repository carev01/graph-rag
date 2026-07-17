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
