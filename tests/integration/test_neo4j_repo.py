import pytest

from graph_sync.models import ChapterRow, StructuralWrite, Tombstone, TocSnapshot

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _write(article_id="a1", source_id="s1", h="h1"):
    return StructuralWrite(
        vendor={"id": "v1", "name": "AWS", "website": "https://aws"},
        product={"id": "p1", "name": "AWS Backup", "version": None, "vendor_id": "v1"},
        source={"id": source_id, "name": "Dev Guide", "base_url": "b",
                "source_type": "web", "platform": "docusaurus", "last_extracted_at": None},
        article={"id": article_id, "title": "T", "source_url": "u", "topic_key": "tk",
                 "content_hash": h, "estimated_tokens": 10, "sort_order": 0,
                 "last_updated_at": None, "run_id": "r1", "seq": None,
                 "source_id": source_id, "removed": False},
    )


async def test_apply_structural_is_idempotent(neo4j_repo):
    await neo4j_repo.apply_structural(_write())
    await neo4j_repo.apply_structural(_write())  # second apply: no duplicates
    assert await neo4j_repo.article_count_by_source("s1") == 1
    assert await neo4j_repo.get_content_hash("a1") == "h1"


async def test_tombstone_keeps_node(neo4j_repo):
    await neo4j_repo.apply_structural(_write(article_id="a2", h="h2"))
    await neo4j_repo.tombstone_article(Tombstone("a2", "2026-07-11T00:00:00Z"))
    assert await neo4j_repo.get_content_hash("a2") == "h2"  # node still present


async def test_apply_toc_rewires_and_prunes(neo4j_repo):
    await neo4j_repo.apply_structural(_write(article_id="a3", source_id="s3"))
    snap1 = TocSnapshot(source_id="s3",
        chapters=[ChapterRow("cA", "s3", "A", "u", 0, 0),
                  ChapterRow("cB", "s3", "B", "u", 0, 1)],
        root_ids=["cA", "cB"], nesting=[], article_links=[("a3", "cB")])
    await neo4j_repo.apply_toc(snap1)
    # Rebuild TOC without cB -> cB pruned, a3 relinked to cA
    snap2 = TocSnapshot(source_id="s3",
        chapters=[ChapterRow("cA", "s3", "A", "u", 0, 0)],
        root_ids=["cA"], nesting=[], article_links=[("a3", "cA")])
    await neo4j_repo.apply_toc(snap2)
    assert await neo4j_repo.chapter_exists("cB") is False
    assert await neo4j_repo.article_chapter_id("a3") == "cA"
