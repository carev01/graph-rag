from __future__ import annotations
from graph_sync.models import ChapterRow, TocSnapshot

def map_toc(toc: dict) -> TocSnapshot:
    source_id = toc["source_id"]
    snap = TocSnapshot(source_id=source_id)

    def walk(entry: dict, parent_id: str | None) -> None:
        cid = entry["id"]
        snap.chapters.append(ChapterRow(
            id=cid, source_id=source_id, title=entry.get("title", ""),
            url=entry.get("url"), level=entry.get("level", 0),
            sort_order=entry.get("sort_order", 0),
        ))
        if parent_id is None:
            snap.root_ids.append(cid)
        else:
            snap.nesting.append((parent_id, cid))
        if entry.get("article_id"):
            snap.article_links.append((entry["article_id"], cid))
        for child in entry.get("children", []) or []:
            walk(child, cid)

    for e in toc.get("entries", []):
        walk(e, None)
    return snap
