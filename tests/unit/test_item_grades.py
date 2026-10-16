"""Per-item evidence grades for list answers (spec 2026-10-15 §12.2)."""
import logging

import answer_api.item_grades as ig

Q_AGENT = "Where can I restore a Windows agent backup?"
T_AGENT = ["restore", "windows", "agent", "backup"]
SANGFOR = ("VBR supports backups created by Veeam Agent for Microsoft Windows for restore "
           "to Sangfor aSV.")
MORPHEUS = ("VBR can restore physical machines from backups created by Veeam Agents to the "
            "HPE Morpheus VM Essentials environment.")


def _fact(uuid, text, title=None):
    return {"fact": text, "fact_uuid": uuid, "valid_at": None, "invalid_at": None,
            "sources": [{"url": f"https://d/{uuid}", "title": title or uuid,
                         "vendor": "Veeam", "product": "Veeam Backup & Replication"}]}


def _grade(raw, facts, *, q=Q_AGENT, terms=T_AGENT, scores=None, floor=0.1):
    marker_map = {i: f for i, f in enumerate(facts, 1)}
    if scores is None:
        scores = {f["fact_uuid"]: 0.6 for f in facts}
    text, grades = ig.grade_items(raw, marker_map, q=q, terms=terms, scores=scores,
                                  floor=floor)
    return text, grades, marker_map


def test_a_fact_with_every_question_term_confirms_the_item():
    raw = "**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]"
    text, grades, mm = _grade(raw, [_fact("sf", SANGFOR)])
    assert text == raw
    assert [(g.item, g.grade, g.missing_terms, g.markers) for g in grades] == [
        ("Sangfor aSV", ig.CONFIRMED, [], [1])]
    assert grades[0].as_dict(mm) == {"item": "Sangfor aSV", "grade": "confirmed",
                                     "missing_terms": [], "markers": [1],
                                     "fact_uuids": ["sf"]}


def test_a_more_general_fact_flags_the_item_with_the_question_spelling():
    raw = "**Hypervisors**\n- **HPE Morpheus VM Essentials** — restore to Morpheus [1]"
    text, grades, _ = _grade(raw, [_fact("hpe", MORPHEUS)])
    assert grades[0].grade == ig.NEEDS_VALIDATION
    assert grades[0].missing_terms == ["windows"]
    assert text.splitlines()[1] == (
        "- **HPE Morpheus VM Essentials** — restore to Morpheus [1] "
        "*(needs validation: the cited source does not mention Windows)*")


def test_the_source_article_title_counts_as_evidence():
    fact = _fact("px", "VBR can restore Windows machine backups to Proxmox VE.",
                 title="Restoring Veeam Agent Backup to Proxmox VM")
    text, grades, _ = _grade("- **Proxmox VE** — restore as a VM [1]", [fact])
    assert grades[0].grade == ig.CONFIRMED and grades[0].missing_terms == []
    assert text == "- **Proxmox VE** — restore as a VM [1]"


def test_an_item_no_cited_fact_names_is_dropped():
    raw = "**Hypervisors**\n- **Sangfor aSV** — x [1]\n- **Oracle Linux KVM** — y [1]"
    text, grades, _ = _grade(raw, [_fact("sf", SANGFOR)])
    assert [g.grade for g in grades] == [ig.CONFIRMED, ig.DROPPED]
    assert "Oracle" not in text
    assert text == "**Hypervisors**\n- **Sangfor aSV** — x [1]"


def test_a_naming_fact_below_the_floor_is_dropped_and_an_unscored_one_kept():
    raw = "- **Sangfor aSV** — x [1]"
    text, grades, _ = _grade(raw, [_fact("sf", SANGFOR)], scores={"sf": 0.05}, floor=0.1)
    assert grades[0].grade == ig.DROPPED and text == ""
    text, grades, _ = _grade(raw, [_fact("sf", SANGFOR)], scores={"sf": None}, floor=0.1)
    assert grades[0].grade == ig.CONFIRMED and text == raw


def test_an_item_without_a_usable_marker_is_dropped():
    raw = "- **Sangfor aSV** — x [99]\n- **Sangfor aSV again** — no marker"
    text, grades, mm = _grade(raw, [_fact("sf", SANGFOR)])
    assert [g.grade for g in grades] == [ig.DROPPED, ig.DROPPED]
    assert [g.markers for g in grades] == [[], []]
    assert grades[0].as_dict(mm)["fact_uuids"] == []
    assert text == ""


def test_synonym_groups_match_restore_to_recovery_and_vm_to_vms():
    q = "What recovery options does Azure Backup offer for Azure VMs?"
    fact = _fact("alt", "Azure Backup can restore an Azure VM to an alternate location.")
    raw = "- **Alternate location restore** — to another place [1]"
    text, grades, _ = _grade(raw, [fact], q=q, terms=["recovery", "vms"])
    assert grades[0].grade == ig.CONFIRMED and text == raw


def test_the_limitations_section_is_untouched_and_emptied_headings_are_removed():
    raw = ("## Hypervisors\n- **Sangfor aSV** — x [1]\n\n"
           "## Other\n- **Oracle Linux KVM** — y [1]\n\n"
           "## Not supported / limitations\n- **Oracle Linux KVM** — not supported [1]\n"
           "## Kept heading\n- **Nothing** [99]")
    text, grades, _ = _grade(raw, [_fact("sf", SANGFOR)])
    assert [g.item for g in grades] == ["Sangfor aSV", "Oracle Linux KVM"]
    assert text == ("## Hypervisors\n- **Sangfor aSV** — x [1]\n\n"
                    "## Not supported / limitations\n- **Oracle Linux KVM** — not supported [1]\n"
                    "## Kept heading\n- **Nothing** [99]")


def test_a_bold_heading_left_empty_is_removed_and_others_stay():
    raw = ("**Hypervisors**\n- **Sangfor aSV** — x [1]\n\n"
           "**Other:**\n- **Oracle Linux KVM** — y [1]")
    text, _, _ = _grade(raw, [_fact("sf", SANGFOR)])
    assert text == "**Hypervisors**\n- **Sangfor aSV** — x [1]"


def test_comma_marker_lists_read_every_marker_and_plain_bullets_warn(caplog):
    facts = [_fact("o", "Oracle Linux KVM is a hypervisor."), _fact("sf", SANGFOR)]
    raw = "- **Sangfor aSV** — x [1, 2]\n- plain bullet without bold [1]"
    with caplog.at_level(logging.INFO, logger="answer_api.item_grades"):
        text, grades, mm = _grade(raw, facts)
    assert grades[0].markers == [1, 2] and grades[0].grade == ig.CONFIRMED
    assert grades[0].as_dict(mm)["fact_uuids"] == ["o", "sf"]
    assert "- plain bullet without bold [1]" in text
    assert "1 bullet line(s) not in item form; left ungraded" in caplog.text
    assert "list items: 1 confirmed, 0 needs validation, 0 dropped" in caplog.text


def test_the_first_supporting_facts_missing_terms_are_reported():
    other = _fact("o2", "Backups can be restored to the HPE Morpheus VM Essentials platform.")
    raw = "- **HPE Morpheus VM Essentials** — x [1] [2]"
    _, grades, _ = _grade(raw, [_fact("hpe", MORPHEUS), other])
    assert grades[0].grade == ig.NEEDS_VALIDATION and grades[0].missing_terms == ["windows"]


def test_any_supporting_fact_with_every_term_confirms():
    raw = "- **Sangfor aSV** — x [1] [2]"
    weak = _fact("w", "Sangfor aSV is a hypervisor.")
    _, grades, _ = _grade(raw, [weak, _fact("sf", SANGFOR)])
    assert grades[0].grade == ig.CONFIRMED


def test_a_missing_term_not_spelled_in_the_question_is_shown_as_is():
    raw = "- **HPE Morpheus VM Essentials** — x [1]"
    text, grades, _ = _grade(raw, [_fact("hpe", MORPHEUS)], terms=["windows", "linuxes"],
                             q="Where can I restore Windows backups?")
    assert grades[0].missing_terms == ["windows", "linuxes"]
    assert text.endswith("does not mention Windows, linuxes)*")
