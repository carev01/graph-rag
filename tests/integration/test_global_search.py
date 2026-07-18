import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _FakeEmbedder:
    def __init__(self, vec): self._vec = vec
    async def create_batch(self, texts): return [self._vec for _ in texts]


async def test_shortlist_ranks_and_filters_by_level(extract_driver):
    from answer_api.global_search import shortlist_communities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'near', title:'Near', "
                    "summary:'x', rating:6.0, cited_fact_uuids:['f1'], full_report:'[]', embedding:[1.0,0.0]})", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'far', title:'Far', "
                    "summary:'x', rating:9.0, cited_fact_uuids:['f2'], full_report:'[]', embedding:[0.0,1.0]})", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:0, community_id:'wronglvl', title:'L0', "
                    "summary:'x', rating:9.0, cited_fact_uuids:['f3'], full_report:'[]', embedding:[1.0,0.0]})", g=g)
    hits = await shortlist_communities(extract_driver, _FakeEmbedder([1.0, 0.0]), "q",
                                       level=1, k=5, group_id=g)
    assert [h.community_id for h in hits] == ["near", "far"]   # level-1 only, near first
