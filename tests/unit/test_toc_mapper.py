import json
from pathlib import Path
from graph_sync.toc_mapper import map_toc

FIX = Path(__file__).parent.parent / "fixtures" / "aws_toc.json"

def test_map_toc_flattens_tree():
    toc = json.loads(FIX.read_text())
    snap = map_toc(toc)
    assert snap.source_id == toc["source_id"]
    ids = {c.id for c in snap.chapters}
    # Known root entry from the fixture:
    assert "6d93201e-6728-48cc-9f94-4cba18b5d2a5" in ids
    assert "6d93201e-6728-48cc-9f94-4cba18b5d2a5" in snap.root_ids
    # Nesting: the child chapter is linked under its parent
    assert ("6d93201e-6728-48cc-9f94-4cba18b5d2a5",
            "d57bd09b-bd96-40ec-a2dd-076b3a95ba85") in snap.nesting
    # An entry that carries an article_id links that article to its own chapter
    assert ("2c92266f-f84c-49c2-9e89-3b4376ec9043",
            "6d93201e-6728-48cc-9f94-4cba18b5d2a5") in snap.article_links

def test_map_toc_synthetic_both_article_and_section():
    toc = {"source_id": "s1", "entries": [
        {"id": "c0", "title": "Root", "url": "u", "level": 0, "sort_order": 0,
         "is_article": False, "article_id": "artRoot", "children": [
            {"id": "c1", "title": "Child", "url": "u2", "level": 1, "sort_order": 1,
             "is_article": True, "article_id": "artChild", "children": []}]}]}
    snap = map_toc(toc)
    assert ("artRoot", "c0") in snap.article_links
    assert ("artChild", "c1") in snap.article_links
    assert ("c0", "c1") in snap.nesting
    assert snap.root_ids == ["c0"]
