import pytest

from graph_extract.noise_filter import is_noise

NOISE = [
    "arn:aws:backup:us-east-1:123456789012:recovery-point:2FC4C6F8-0FC1-546B-B91C-209C599C1D56",
    "arn:aws:ec2:us-east-1::snapshot/snap-00a129455bdbc9d99",
    "arn:aws:iam::123456789012:root",
    "CreatorRequestId",
    "AssociateBackupVaultMpaApprovalTeamFailed",
    "CreateRestoreAccessBackupVaultFailed",
    "Install-Module -Name Az.RecoveryServices -Force",
    "snap-07ce8c3141d361233",
    "vol-00a422a05b9c6asd3",
    # Service-action strings (IAM/API actions), never products/concepts.
    "kms:GetKeyPolicy",
    "kms:put-key-policy",
    "s3:PutObject",
    # API request-field names extracted as entities.
    "BackupVaultArn",
    "EncryptionKeyArn",
    "BackupVaultName",
    # Lowercase shell commands (as typed), distinct from Title-Case prose.
    "aws backup start-restore-job",
    "az backup vault list",
    # Geographic region / AZ names (low-value, prompt can't suppress).
    "AWS Regions",
    "Canada (Central) Region",
    "primary region",
    "Azure paired region",
    # Pure-numeric account ids, AWS org/root ids, Invalid-* error codes.
    "112233445566",
    "123456789012",
    "o-a1b2c3d4e5",
    "r-f6g7h8i9j0example",
    "InvalidOrganizationBackupPlan",
]
KEEP = [
    "immutability",
    "Amazon S3",
    "Azure Blob Storage",
    "RPO",
    "recovery point",
    "cross-region copy",
    "AWS Backup Vault Lock",
    "Kubernetes",
    "SEC 17a-4",
    "AWS Backup",
    "Azure Backup",
    "soft delete",
    "retention policy",
    # Title-Case prose with a vendor prefix — NOT shell commands.
    "AWS managed key",
    "AWS managed policy",
    "Azure managed disk",
    # A capability that contains "region" but doesn't end in it.
    "Cross-Region backup",
    # Short numbers (years/versions) are NOT ids — numeric pattern needs 6+ digits.
    "2024",
    "3-2-1 rule",
    # Real products shaped like an o-/r- id but with NO digit in the body — kept
    # (the org-id pattern requires a digit).
    "R-Studio",
    "O-Ring",
]


@pytest.mark.parametrize("name", NOISE)
def test_flags_noise(name: str) -> None:
    assert is_noise(name) is True


@pytest.mark.parametrize("name", KEEP)
def test_keeps_domain_terms(name: str) -> None:
    assert is_noise(name) is False
