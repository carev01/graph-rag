"""Pure classifier for navigation / index / link-farm / changelog articles.

Single source of truth for "this article page is not durable knowledge
content" -- used to skip such pages at ingestion and to clean already-
extracted junk facts. Conservative: only clear non-content pages.
"""
from __future__ import annotations

import re

_NAV = re.compile(
    r"blogs?,?\s*videos"          # "Blogs, videos, tutorials, ..."
    r"|and other resources$"      # link-farm index tail
    r"|\brelease notes\b"         # "Archived release notes", "Release notes MABS"
    r"|what'?s new",              # changelog pages
    re.IGNORECASE,
)


def is_navigation_article(title: str) -> bool:
    """True for navigation/index/link-farm/changelog pages that carry no
    durable backup-domain knowledge (their extracted facts are retrieval noise)."""
    return bool(title and _NAV.search(title))
