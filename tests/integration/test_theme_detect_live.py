import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")   # match live_extract_driver's module loop


@pytest.mark.live
async def test_detect_communities_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from theme_builder.detect import detect_communities
    s = get_extract_settings()
    comms = await detect_communities(live_extract_driver, s.group_id,
                                     min_community_size=3, max_levels=3)
    assert comms, "expected >=1 community from the live GDS Leiden run"
    assert all(len(c.member_uuids) >= 3 for c in comms)
    ids = {c.community_id for c in comms}
    assert all(c.parent_id in ids for c in comms if c.parent_id is not None)
