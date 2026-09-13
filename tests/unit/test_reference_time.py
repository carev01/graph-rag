"""The ordering axis. `content_changed_at` is when the SERVED markdown became
current -- the bytes `content_hash` covers and our re-ingest gate keys on -- so a
change we act on is always a change we can date.

It is preferred over `last_updated_at` even where the vendor declares a real date,
because only ~53 of 194 upstream sources carry one: mixing them would put "the
vendor's declared day" and "the served bytes became current" in one field,
switching by vendor. That incoherence is what produced this project's phantom
invalidations, where 48% of dated edges held an in-world date and 52% a crawl
timestamp.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from graph_extract.ingest_driver import CRAWL_FALLBACK, _reference_time


def _art(**kw):
    base = dict(id="a1", last_updated_at=None, extracted_at="2026-07-12T16:00:17Z",
                content_changed_at=None, content_changed_basis=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_content_changed_at_wins_and_carries_its_basis():
    at, basis = _reference_time(_art(content_changed_at="2026-08-01T14:03:11Z",
                                     content_changed_basis="exact"))
    assert at == datetime(2026, 8, 1, 14, 3, 11, tzinfo=timezone.utc)
    assert basis == "exact"


def test_it_beats_a_real_vendor_date():
    """The whole point: a vendor date is NOT preferred, however genuine. Uniform
    semantics across the corpus beat better semantics on a fifth of it."""
    at, basis = _reference_time(_art(content_changed_at="2026-08-01T14:03:11Z",
                                     content_changed_basis="exact",
                                     last_updated_at="2026-07-01T09:12:00Z"))
    assert at.month == 8, "sorted by content_changed_at, not the vendor's date"
    assert basis == "exact"


def test_a_lower_bound_basis_is_carried_not_flattened():
    """Tier 2 is 'changed no later than T' and may be early. Invalidation will be
    restricted to `exact`, so the label must survive to the caller."""
    _, basis = _reference_time(_art(content_changed_at="2026-08-01T00:00:00Z",
                                    content_changed_basis="lower_bound"))
    assert basis == "lower_bound"


def test_a_timestamp_with_no_basis_is_labelled_not_assumed_exact():
    _, basis = _reference_time(_art(content_changed_at="2026-08-01T00:00:00Z"))
    assert basis == "unlabelled"


def test_the_crawl_fallback_is_loud(caplog):
    """Silently reverting to crawl ordering is the original defect: 2,470 articles
    once shared a single crawl minute, so within such a block the order is not
    approximate, it is arbitrary."""
    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        at, basis = _reference_time(_art())
    assert basis == CRAWL_FALLBACK
    assert at == datetime(2026, 7, 12, 16, 0, 17, tzinfo=timezone.utc)
    assert "content_changed_at" in caplog.text and "crawl" in caplog.text


def test_a_malformed_content_changed_at_falls_back_rather_than_crashing(caplog):
    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        _, basis = _reference_time(_art(content_changed_at="not-a-date",
                                        content_changed_basis="exact"))
    assert basis == CRAWL_FALLBACK, "a bad value must not be trusted for its label"


def test_an_article_missing_the_field_entirely_still_works(caplog):
    """A pre-deploy upstream returns no such key at all."""
    art = SimpleNamespace(id="a1", last_updated_at=None,
                          extracted_at="2026-07-12T16:00:17Z")
    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        _, basis = _reference_time(art)
    assert basis == CRAWL_FALLBACK
