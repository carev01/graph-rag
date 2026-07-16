import pytest
from graph_extract.article_filter import is_navigation_article

NAV = [
    "Blogs, videos, tutorials, and other resources",
    "Archived release notes",
    "Release notes MABS",
    "What's New in MABS",
    "What's new in Azure Backup",
]
KEEP = [
    "Overview of soft delete", "Encryption in Azure Backup", "Vault Lock",
    "FAQ-Soft Delete", "AWS Backup feature availability", "Requester tasks",
    "Cross-Region backup", "Amazon S3 backups",
]

@pytest.mark.parametrize("t", NAV)
def test_navigation_flagged(t): assert is_navigation_article(t) is True

@pytest.mark.parametrize("t", KEEP)
def test_content_kept(t): assert is_navigation_article(t) is False
