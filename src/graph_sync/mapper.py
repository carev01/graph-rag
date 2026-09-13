from __future__ import annotations
from graph_sync.catalog import SourceInfo
from graph_sync.models import ContentRecord, StructuralWrite, Tombstone, TombstoneRecord

def map_content(rec: ContentRecord, info: SourceInfo) -> StructuralWrite:
    return StructuralWrite(
        vendor={"id": info.vendor_id, "name": info.vendor_name,
                "website": info.vendor_website},
        product={"id": info.product_id, "name": info.product_name,
                 "version": info.product_version, "vendor_id": info.vendor_id},
        source={"id": info.source_id, "name": info.source_name,
                "base_url": info.base_url, "source_type": info.source_type,
                "platform": info.platform, "last_extracted_at": info.last_extracted_at},
        article={"id": rec.id, "title": rec.title, "source_url": rec.source_url,
                 "topic_key": rec.topic_key, "content_hash": rec.content_hash,
                 "estimated_tokens": rec.estimated_tokens, "sort_order": rec.sort_order,
                 "last_updated_at": rec.last_updated_at, "run_id": rec.run_id,
                 "last_updated_source": rec.last_updated_source,
                 "content_changed_at": rec.content_changed_at,
                 "content_changed_basis": rec.content_changed_basis,
                 "source_changed_at": rec.source_changed_at,
                 "seq": rec.seq, "source_id": rec.source_id, "removed": False},
    )

def map_tombstone(rec: TombstoneRecord) -> Tombstone:
    return Tombstone(article_id=rec.id, removed_at=rec.removed_at)
