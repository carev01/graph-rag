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
# Each member is either a plain string (single form) or a list of forms
# (first = canonical, rest = aliases/spelling variants) -- a member resolves
# to the SET of nodes whose lowercased name matches any of its forms.
SHOULD_DISTINCT: list[tuple[str | list[str], str | list[str]]] = [
    ("Amazon S3", "Azure Blob Storage"),
    ("AWS Backup", "Azure Backup"),
    ("AWS Backup Vault Lock", ["Azure immutable vault", "Azure Backup Immutable vault"]),
]

# Name tokens that mark an entity as vendor-branded (used by the
# suspect-false-merge detector).
VENDOR_TOKENS: list[str] = ["aws", "amazon", "azure", "microsoft"]
