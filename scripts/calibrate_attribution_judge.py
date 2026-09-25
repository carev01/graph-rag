"""Calibrate the eval's attribution judge on answers with KNOWN misattribution counts.

    UV_OFFLINE=1 uv run --frozen --extra dev python scripts/calibrate_attribution_judge.py [--runs 3]

Paid but tiny (7 probes x runs, eval-judge tier). The judge's count is only evidence
once it scores these right: a probe it misses is a class of answer whose number in the
router eval means nothing. The first cut flagged deterministic timeline renders (it read
a product named INSIDE a correctly labelled fact as a misattribution) -- probe 2 and 7.
"""
from __future__ import annotations

import argparse
import asyncio

from answer_api.eval_router import _eval_judge_client_and_model, judge_attribution
from graph_extract.config import get_extract_settings

AWS = "(AWS · AWS Backup)"
AZ = "(Microsoft · Azure Backup)"

FACTS = "\n".join([
    f"[1] {AWS} AWS Backup encrypts backups in a vault with an AWS KMS key.",
    f"[2] {AZ} Azure Backup encrypts backup data with platform-managed keys by default.",
    f"[3] {AZ} Backup center gives a summarized view of replicated items for Azure Site "
    "Recovery.",
    f"[4] {AWS} AWS Backup Vault Lock in compliance mode prevents deletion of recovery points.",
    f"[5] {AZ} Azure Backup soft delete retains deleted backup data for 14 days.",
    f"[6] {AZ} Soft delete is not supported for operational backups of Azure Files shares.",
    f"[7] {AZ} Soft-delete operations can be performed with PowerShell and the Azure CLI.",
])

PROBES: list[tuple[str, str, int]] = [
    ("correct attribution",
     "AWS Backup encrypts backups with an AWS KMS key [1]. Azure Backup uses "
     "platform-managed keys by default [2].", 0),
    ("another product named inside a correctly labelled fact",
     "In Azure Backup, Backup center summarizes replicated items for Azure Site "
     "Recovery [3].", 0),
    ("one misattribution",
     "Azure Backup encrypts backups with an AWS KMS key [1]. Azure Backup soft delete "
     "keeps deleted data for 14 days [5].", 1),
    ("two misattributions",
     "Azure Backup's Vault Lock compliance mode prevents deletion of recovery points [4]. "
     "AWS Backup soft delete keeps deleted data for 14 days [5].", 2),
    ("correct comparison",
     "Both protect backups from deletion, differently: AWS Backup uses Vault Lock "
     "compliance mode [4], while Azure Backup keeps soft-deleted data for 14 days [5].", 0),
    ("over-generalisation to a second vendor",
     "Both AWS Backup and Azure Backup lock recovery points against deletion in compliance "
     "mode [4].", 1),
    ("workloads and tools named in correctly labelled facts (real-run failure)",
     "Soft delete does not cover operational backups of Azure Files shares [6], and "
     "soft-delete operations can be run from PowerShell or the Azure CLI [7].", 0),
    ("deterministic timeline render",
     f"- **Azure Backup soft delete retains deleted backup data for 14 days.** {AZ} — "
     "valid_at 2026-07-14 (current) [5]\n"
     f"- **Backup center gives a summarized view of replicated items for Azure Site "
     f"Recovery.** {AZ} — valid_at 2026-07-14 (current) [3]", 0),
]


async def main(runs: int) -> int:
    s = get_extract_settings()
    client, model = _eval_judge_client_and_model(s)
    misses = 0
    try:
        for name, answer, expected in PROBES:
            got = [await judge_attribution(client, model, "Compare AWS Backup and Azure "
                                           "Backup.", answer, FACTS) for _ in range(runs)]
            ok = all(g == expected for g in got)
            misses += 0 if ok else 1
            print(f"{'OK  ' if ok else 'MISS'} expected={expected} got={got}  {name}")
    finally:
        await client.close()
    print(f"{len(PROBES) - misses}/{len(PROBES)} probes scored right on every run (model {model})")
    return 1 if misses else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    raise SystemExit(asyncio.run(main(ap.parse_args().runs)))
