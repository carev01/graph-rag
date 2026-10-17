"""Item coverage for list questions (spec 2026-10-15 §7)."""
import json
from pathlib import Path

from answer_api.item_coverage import match_items

ITEMS = [{"item": "Hyper-V", "aliases": ["hyper-v"]},
         {"item": "Disk export", "aliases": ["vhd", "vhdx", "vmdk"]},
         {"item": "Amazon S3", "aliases": ["s3"]}]


def test_aliases_match_case_insensitively_on_word_boundaries():
    found, missing = match_items("Restore to Microsoft HYPER-V, or export as VHDX.", ITEMS)
    assert found == ["Hyper-V", "Disk export"] and missing == ["Amazon S3"]


def test_an_alias_inside_a_longer_word_does_not_match():
    _, missing = match_items("The s3cure option and vhdtool.", ITEMS)
    assert missing == ["Hyper-V", "Disk export", "Amazon S3"]


def test_golden_list_questions_are_well_formed():
    gold = json.loads((Path(__file__).resolve().parents[2] /
                       "src/answer_api/router_golden_tier1.json").read_text())
    lists = [q for q in gold if q.get("kind") == "single_list"]
    assert len(lists) == 5
    counts = {q["question"]: len(q["expected_items"]) for q in lists}
    assert counts == {
        "Which workloads does Veeam support immutability for?": 9,
        "Where can I restore a Windows agent backup?": 17,
        "Which hypervisors does Veeam Backup & Replication support?": 14,
        "Which resource types can AWS Backup protect?": 17,
        "What recovery options does Azure Backup offer for Azure VMs?": 10}
    for q in lists:
        assert q["expected_modes"] == ["local"] and q["expected_article_ids"] == []
        assert all(i["item"] and i["aliases"] for i in q["expected_items"])
