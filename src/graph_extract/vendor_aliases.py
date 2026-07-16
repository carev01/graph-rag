"""Structural-vendor -> accepted semantic-name aliases (snapshot 2026-07).
Static, conservative: only high-confidence equivalences so reconciliation
never links a structural node to a wrong/noise semantic entity."""
from __future__ import annotations


def normalize(name: str) -> str:
    return " ".join(name.strip().split()).lower()


# canonical structural name -> extra accepted semantic surface forms (pre-normalised)
VENDOR_ALIASES: dict[str, frozenset[str]] = {
    "AWS": frozenset({"aws", "amazon web services", "amazon"}),
    "Microsoft": frozenset({"microsoft", "azure", "microsoft azure"}),
    # extend per vendor as the corpus widens (Veeam, Commvault, Rubrik, ...)
}


def accepted_forms(structural_name: str) -> set[str]:
    forms = {normalize(structural_name)}
    forms |= set(VENDOR_ALIASES.get(structural_name, frozenset()))
    return forms
