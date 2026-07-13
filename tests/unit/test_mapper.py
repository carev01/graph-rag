from graph_sync.models import ContentRecord, TombstoneRecord
from graph_sync.catalog import SourceInfo
from graph_sync.mapper import map_content, map_tombstone

INFO = SourceInfo(
    source_id="s1", source_name="Developer Guide", base_url="https://b/",
    source_type="web", platform="docosaurus", last_extracted_at="2026-07-12T16:00:53Z",
    product_id="p1", product_name="AWS Backup", product_version=None,
    vendor_id="v1", vendor_name="AWS", vendor_website="https://aws.amazon.com",
)

def _rec(**over):
    base = dict(change_type="added", id="a1", topic_key="tk", source_id="s1",
               vendor="AWS", product="AWS Backup", title="What is AWS Backup?",
               source_url="https://x/whatis.html", content_hash="h1",
               estimated_tokens=4075, sort_order=0, seq=None, run_id="r1")
    base.update(over)
    return ContentRecord.model_validate(base)

def test_map_content_builds_structural_write():
    w = map_content(_rec(), INFO)
    assert w.vendor == {"id": "v1", "name": "AWS", "website": "https://aws.amazon.com"}
    assert w.product["id"] == "p1" and w.product["vendor_id"] == "v1"
    assert w.source["id"] == "s1" and w.source["platform"] == "docosaurus"
    assert w.article["id"] == "a1" and w.article["content_hash"] == "h1"
    assert w.article["source_id"] == "s1"
    assert "content_markdown" not in w.article

def test_map_tombstone():
    rec = TombstoneRecord.model_validate(
        {"change_type": "removed", "id": "a9", "source_id": "s1",
         "removed_at": "2026-07-11T18:41:55Z", "run_id": None})
    t = map_tombstone(rec)
    assert t.article_id == "a9" and t.removed_at == "2026-07-11T18:41:55Z"
