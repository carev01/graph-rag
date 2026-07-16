"""Pure classifier for high-confidence extraction noise.

Single source of truth used by both the extraction-quality metric and the
cleanup pass. No I/O; pattern-matching only.
"""

from __future__ import annotations

import re

_ARN = re.compile(r"^arn:aws:", re.IGNORECASE)

# Bare cloud resource ids, e.g. "snap-07ce8c3141d361233", "vol-00a422a05b9c6asd3".
# Alphanumeric body (not hex-only: real ids observed in 2a output contain
# non-hex letters), at least 6 chars, nothing else in the name.
_RESOURCE_ID = re.compile(
    r"^(?:snap|vol|i|ami|vpc|subnet|sg|eni)-[0-9a-z]{6,}$",
    re.IGNORECASE,
)

# AWS Organizations / root / OU ids ("o-a1b2c3d4e5", "r-f6g7h8i9j0example",
# "ou-1a2b-34cd56ef"). The single-letter o/r prefixes need a body that
# actually looks like an id -- REQUIRE at least one digit -- so real product
# names of the same shape ("R-Studio", "O-Ring") are kept.
_ORG_ID = re.compile(r"^(?:o|r|ou)-[0-9a-z-]*[0-9][0-9a-z-]*$", re.IGNORECASE)

# Pure-numeric identifiers, e.g. AWS 12-digit account ids ("112233445566").
# No backup concept is all digits; 6+ digits avoids catching versions/years.
_NUMERIC_ID = re.compile(r"^\d{6,}$")

# CamelCase "Invalid..." error/exception codes ("InvalidOrganizationBackupPlan").
_INVALID_CODE = re.compile(r"^Invalid[A-Z][A-Za-z]*$")

# CLI command lines, NOT product names. A name is a command only when it has
# actual command shape:
#   - a LOWERCASE "aws"/"az"/"kubectl"/"gcloud" (as typed in a shell) followed
#     by a lowercase subcommand ("aws backup list-vaults"). Case-sensitive on
#     purpose: prose like "AWS managed key" or "Azure paired region" uses a
#     Title-Case vendor prefix and must be kept, while "AWS Backup Vault Lock"
#     was already safe (uppercase subcommand);
#   - a PowerShell Verb-Noun cmdlet ("Install-Module", "Get-AzRecoveryPoint");
#   - a long flag ("--force"), a shell variable ("$VAULT"), or a backtick.
_CLI = re.compile(
    r"(?:^|\s)(?:aws|az|kubectl|gcloud)\s+[a-z][a-z0-9-]*"
    r"|(?:^|\s)(?:Install|Uninstall|Get|Set|New|Remove|Add|Import|Export|Enable|Disable)"
    r"-[A-Z][A-Za-z]*"
    r"|--[a-z]"
    r"|\$[A-Za-z_]"
    r"|`"
)

# Geographic region / availability-zone entity names. Low value for backup Q&A
# (they carry no cross-vendor semantics and churn constantly) and the extraction
# prompt cannot reliably suppress them, so drop names whose final word is
# "region"/"regions" ("AWS Regions", "Canada (Central) Region", "primary
# region", "Azure paired region"). "cross-region copy" and "Cross-Region backup"
# end in another word and are kept. NOTE: deliberate, reversible product
# decision — remove this pattern if region-scoped questions become in scope.
_REGION = re.compile(r"\bregions?$", re.IGNORECASE)

# CamelCase API/operation identifiers ending in a status/verb word or a bare
# request-field suffix. `Arn`/`Name` catch API parameter field names extracted
# as entities ("BackupVaultArn", "EncryptionKeyArn", "BackupVaultName"); the
# single-token guard in is_noise keeps real multi-word concepts safe.
_API_ERR = re.compile(
    r"^[A-Z][A-Za-z0-9]*(Failed|Error|Exception|RequestId|Id|Arn|Name)$"
)

# Service-action strings, e.g. "kms:GetKeyPolicy", "kms:put-key-policy",
# "s3:PutObject" — an IAM/API action, never a product/concept. A lowercase
# service prefix, a colon, then the action token (nothing else). `arn:...`
# is handled by _ARN above; prose with colons has a space and won't match.
_SERVICE_ACTION = re.compile(r"^[a-z][a-z0-9]{1,20}:[A-Za-z][A-Za-z0-9-]*$")

# Documentation / reference titles extracted as entities.
_DOC_TITLE = re.compile(
    r"\b(User Guide|Developer Guide|Getting Started Guide|Reference Guide|"
    r"Administration Guide|Administrator Guide|API Reference|Documentation)$")

# CamelCase API id / identifier field names ("AccountID", "DBInstanceIdentifier").
# Require a LOWERCASE char before the terminal ID/Identifier so real acronyms
# (RAID, GRID, UUID) are kept.
_API_ID = re.compile(r"^[A-Z][A-Za-z0-9]*[a-z](ID|Identifier)$")


def is_noise(name: str, type: str | None = None) -> bool:  # noqa: A002 - name fixed by contract
    """High-confidence, pattern-matchable noise. Conservative: borderline domain
    terms are kept (return False) to avoid dropping useful entities."""
    n = name.strip()
    if not n:
        return True
    if _ARN.search(n):
        return True
    if _RESOURCE_ID.match(n):
        return True
    if _ORG_ID.match(n):
        return True
    if _NUMERIC_ID.match(n):
        return True
    if _INVALID_CODE.match(n):
        return True
    if _CLI.search(n):
        return True
    if _REGION.search(n):
        return True
    if _SERVICE_ACTION.match(n):
        return True
    if _API_ERR.match(n) and " " not in n:
        return True
    if _DOC_TITLE.search(n):
        return True
    if _API_ID.match(n):
        return True
    return False
