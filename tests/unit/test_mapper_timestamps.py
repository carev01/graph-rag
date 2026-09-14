"""The upstream timestamp fields must reach the :Article node.

`content_changed_at` is the ordering axis and `content_changed_basis` is what the
Tier-1-only invalidation rule reads (BACKLOG 30); `source_changed_at` is the gate
that keeps an enrichment from ever expiring a fact. If the mapper drops them the
graph cannot answer any of that later, and nothing else would notice -- the write
is `SET a += $article`, so a missing key is silently just an absent property.
"""
from __future__ import annotations

from graph_sync.mapper import map_content
from graph_sync.catalog import SourceInfo
from graph_sync.models import ContentRecord

_INFO = SourceInfo(vendor_id="v", vendor_name="AWS", vendor_website=None,
                   product_id="p", product_name="Backup", product_version=None,
                   source_id="s", source_name="docs", base_url="https://x",
                   source_type="web", platform="docfx", last_extracted_at=None)


def _rec(**kw) -> ContentRecord:
    base = dict(change_type="updated", id="a1", topic_key="t", source_id="s",
                vendor="AWS", product="Backup", title="Backup Proxies",
                source_url="https://x/a", content_hash="h", estimated_tokens=10,
                sort_order=1)
    base.update(kw)
    return ContentRecord(**base)


def test_the_new_timestamp_fields_reach_the_article():
    art = map_content(_rec(
        content_changed_at="2026-08-01T14:03:11Z", content_changed_basis="exact",
        source_changed_at="2026-05-05T08:30:00Z",
        last_updated_at="2026-07-01T09:12:00Z", last_updated_source="vendor_meta",
    ), _INFO).article
    assert art["content_changed_at"] == "2026-08-01T14:03:11Z"
    assert art["content_changed_basis"] == "exact"
    assert art["source_changed_at"] == "2026-05-05T08:30:00Z"
    assert art["last_updated_at"] == "2026-07-01T09:12:00Z"
    assert art["last_updated_source"] == "vendor_meta"


def test_a_pre_deploy_record_still_validates_and_maps_nulls():
    """The feed predates the fields for as long as the rollout takes; a missing
    key must be None, not a validation error."""
    art = map_content(_rec(), _INFO).article
    assert art["content_changed_at"] is None
    assert art["content_changed_basis"] is None
    assert art["source_changed_at"] is None
    assert art["last_updated_source"] is None


def test_a_lower_bound_basis_survives_the_mapper():
    """Tier 2 may be early. Flattening or dropping the label would let a
    lower-bound article be compared as though it were exact."""
    art = map_content(_rec(content_changed_at="2026-08-01T00:00:00Z",
                           content_changed_basis="lower_bound"), _INFO).article
    assert art["content_changed_basis"] == "lower_bound"
