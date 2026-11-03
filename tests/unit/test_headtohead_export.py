from scripts.headtohead.export_corpus import CORPUS_QUERY, rows_to_corpus

KEYS = {"article_id", "vendor", "product", "source", "title", "url", "graph_content_hash"}


def _row(aid, vendor="V", product="P", source="S", **kw):
    base = {"article_id": aid, "vendor": vendor, "product": product, "source": source,
            "title": "t", "url": "u", "graph_content_hash": "h"}
    base.update(kw)
    return base


def test_rows_to_corpus_dedupes_and_sorts():
    rows = [_row("b"), _row("a", vendor="Z"), _row("a", vendor="A"), _row("a", vendor="M")]
    out = rows_to_corpus(rows)
    assert [r["article_id"] for r in out] == ["a", "b"]
    assert out[0]["vendor"] == "A"
    assert all(set(r) == KEYS for r in out)


def test_rows_to_corpus_keeps_missing_optional_fields_as_none():
    out = rows_to_corpus([{"article_id": "x", "vendor": "V", "title": "t"}])
    assert out[0]["product"] is None
    assert out[0]["source"] is None
    assert out[0]["url"] is None
    assert out[0]["graph_content_hash"] is None
    assert set(out[0]) == KEYS


def test_corpus_query_is_bounded():
    assert "[*1..2]" in CORPUS_QUERY
    assert "HAS_EPISODE" in CORPUS_QUERY
