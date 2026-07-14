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

# CLI command lines, NOT product names. A name is a command only when it has
# actual command shape:
#   - "aws"/"az"/"kubectl"/"gcloud" followed by a lowercase subcommand
#     ("aws backup list-vaults"), so Title-Case product names such as
#     "AWS Backup Vault Lock" or "Azure Backup" never match;
#   - a PowerShell Verb-Noun cmdlet ("Install-Module", "Get-AzRecoveryPoint");
#   - a long flag ("--force"), a shell variable ("$VAULT"), or a backtick.
_CLI = re.compile(
    r"(?:^|\s)(?i:aws|az|kubectl|gcloud)\s+[a-z][a-z0-9-]*"
    r"|(?:^|\s)(?:Install|Uninstall|Get|Set|New|Remove|Add|Import|Export|Enable|Disable)"
    r"-[A-Z][A-Za-z]*"
    r"|--[a-z]"
    r"|\$[A-Za-z_]"
    r"|`"
)

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
    if _CLI.search(n):
        return True
    if _SERVICE_ACTION.match(n):
        return True
    if _API_ERR.match(n) and " " not in n:
        return True
    return False
