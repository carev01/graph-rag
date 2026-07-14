"""Acceptance criteria as data (reviewed at spec sign-off)."""

# Generic concepts that appear in BOTH vendors' docs and SHOULD resolve to one
# shared entity node (cross-vendor merge is correct here).
SHOULD_MERGE: list[str] = [
    "immutability",
    "cross-region copy",
    "Kubernetes",
    "RPO",
    "RTO",
    "encryption",
    "recovery point",
    "retention policy",
    "soft delete",
    "backup vault",
    "restore",
]

# Pairs that must stay DISTINCT (vendor-specific; collapsing them is a false merge).
SHOULD_DISTINCT: list[tuple[str, str]] = [
    ("Amazon S3", "Azure Blob Storage"),
    ("AWS Backup", "Azure Backup"),
    ("AWS Backup Vault Lock", "Azure immutable vault"),
]

# Name tokens that mark an entity as vendor-branded (used by the
# suspect-false-merge detector).
VENDOR_TOKENS: list[str] = ["aws", "amazon", "azure", "microsoft"]
