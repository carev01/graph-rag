"""Per-item evidence grades for list answers (spec 2026-10-15 §12.2)."""
import logging

import pytest

import answer_api.item_grades as ig
import answer_api.list_sweep as ls
from answer_api.scope import Scope

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


def _grade(raw, facts, *, q=Q_AGENT, terms=T_AGENT, scores=None, floor=0.1,
          unnamed_floor=0.65, scope=None):
    marker_map = {i: f for i, f in enumerate(facts, 1)}
    if scores is None:
        scores = {f["fact_uuid"]: 0.6 for f in facts}
    got = ig.grade_items(raw, marker_map, q=q, terms=terms, scores=scores, floor=floor,
                         unnamed_floor=unnamed_floor, scope=scope)
    return got.text, got.grades, marker_map


def test_a_fact_with_every_question_term_confirms_the_item():
    raw = "**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]"
    text, grades, mm = _grade(raw, [_fact("sf", SANGFOR)])
    assert text == raw
    assert [(g.item, g.grade, g.missing_terms, g.markers) for g in grades] == [
        ("Sangfor aSV", ig.CONFIRMED, [], [1])]
    assert grades[0].as_dict(mm) == {"item": "Sangfor aSV", "grade": "confirmed",
                                     "missing_terms": [], "markers": [1], "reason": "named",
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


def test_named_support_at_or_above_the_floor_is_kept_below_it_falls_to_unnamed():
    raw = "- **Sangfor aSV** — x [1]"
    text, grades, _ = _grade(raw, [_fact("sf", SANGFOR)], scores={"sf": 0.45}, floor=0.4,
                             unnamed_floor=0.65)
    assert grades[0].grade == ig.CONFIRMED and grades[0].reason == "named" and text == raw

    # Below the named floor, the same (named) fact is re-checked against the unnamed
    # floor instead; still too low, so the item is dropped.
    text, grades, _ = _grade(raw, [_fact("sf", SANGFOR)], scores={"sf": 0.35}, floor=0.4,
                             unnamed_floor=0.65)
    assert grades[0].grade == ig.DROPPED and grades[0].reason == "below_floor" and text == ""


def test_an_unnamed_relevant_fact_flags_the_item_instead_of_dropping_it():
    other = _fact("rel", "VBR can restore Windows agent backups to many supported platforms.")
    raw = "- **Sangfor aSV** — x [1]"
    text, grades, _ = _grade(raw, [other], scores={"rel": 0.7}, floor=0.4, unnamed_floor=0.65)
    assert grades[0].grade == ig.NEEDS_VALIDATION and grades[0].reason == "unnamed"
    assert grades[0].missing_terms == []
    assert text == raw + " *(needs validation: the cited source does not name this item)*"

    # Relevant but below the unnamed floor: dropped.
    text, grades, _ = _grade(raw, [other], scores={"rel": 0.5}, floor=0.4, unnamed_floor=0.65)
    assert grades[0].grade == ig.DROPPED and grades[0].reason == "below_floor" and text == ""

    # Unscored and unnamed never keeps an item.
    text, grades, _ = _grade(raw, [other], scores={"rel": None}, floor=0.4, unnamed_floor=0.65)
    assert grades[0].grade == ig.DROPPED and text == ""


def test_an_item_without_a_usable_marker_is_dropped():
    raw = "- **Sangfor aSV** — x [99]\n- **Sangfor aSV again** — no marker"
    text, grades, mm = _grade(raw, [_fact("sf", SANGFOR)])
    assert [g.grade for g in grades] == [ig.DROPPED, ig.DROPPED]
    assert [g.reason for g in grades] == ["no_marker", "no_marker"]
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


NUTANIX = "Backups created by Veeam Agent for Microsoft Windows can be restored to Nutanix AHV."


def test_a_parent_heading_emptied_through_its_sub_heading_is_removed():
    raw = ("## A\n### A1\n- **Oracle KVM** — x [1]\n"
           "## B\n- **Nutanix AHV** — y [1]")
    text, _, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert text == "## B\n- **Nutanix AHV** — y [1]"


def test_a_parent_heading_with_a_surviving_sub_section_stays():
    raw = ("## A\n### A1\n- **Oracle KVM** — x [1]\n### A2\n- **Nutanix AHV** — y [1]\n"
           "**Bold**\n- **Oracle KVM** — z [1]")
    text, _, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert text == "## A\n### A2\n- **Nutanix AHV** — y [1]"


def test_a_heading_the_model_wrote_empty_stays():
    raw = "# Answer\n## A\n- **Oracle KVM** — x [1]\n## B\n- **Nutanix AHV** — y [1]"
    text, _, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert text == "# Answer\n## B\n- **Nutanix AHV** — y [1]"


def test_a_limitations_phrase_in_a_sentence_does_not_end_the_graded_region():
    raw = ("Items follow; see Not supported / limitations at the end.\n"
           "- **Oracle KVM** — x [1]\n- **Nutanix AHV** — y [1]\n\n"
           "**Not supported / limitations**\n- **Oracle KVM** — no [1]")
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.grade for g in grades] == [ig.DROPPED, ig.CONFIRMED]
    assert text == ("Items follow; see Not supported / limitations at the end.\n"
                    "- **Nutanix AHV** — y [1]\n\n"
                    "**Not supported / limitations**\n- **Oracle KVM** — no [1]")


def test_numbered_item_lines_are_graded_and_numbered_plain_lines_counted(caplog):
    raw = "1. **Nutanix AHV** — y [1]\n2) **Oracle KVM** — x [1]\n3. plain line [1]"
    with caplog.at_level(logging.WARNING, logger="answer_api.item_grades"):
        text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.grade for g in grades] == [ig.CONFIRMED, ig.DROPPED]
    assert text == "1. **Nutanix AHV** — y [1]\n3. plain line [1]"
    assert "1 bullet line(s) not in item form" in caplog.text


def test_a_cited_limitation_fact_never_supports_an_item():
    lim = _fact("lim", "Restore of Veeam Agent for Windows backups to Nutanix AHV is not "
                       "supported for encrypted backups.")
    _, grades, _ = _grade("- **Nutanix AHV** — y [1]", [lim])
    assert grades[0].grade == ig.DROPPED
    _, grades, _ = _grade("- **Nutanix AHV** — y [1] [2]", [lim, _fact("nx", NUTANIX)])
    assert grades[0].grade == ig.CONFIRMED


def test_short_terms_and_synonyms_match_whole_words_only():
    assert ig._present("vms", "restore an azure vm to a new place")
    assert ig._present("vms", "restore azure vms")
    assert not ig._present("vms", "restore a vmware machine from a vmdk")
    assert ig._present("aws", "copy to aws regions")
    assert not ig._present("aws", "awsome")
    assert ig._present("windows", "microsoft windows server")


def test_recover_and_copy_word_families_match():
    assert ig._present("recover", "restore an azure vm")
    assert ig._present("copies", "a copy job")
    assert ig._present("copy", "backup copies are kept")
    assert ig._present("policies", "a retention policy")


# --- Fix wave 2 -------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "Not supported / limitations:", "Not supported / limitations", "**Not supported / "
    "limitations:**", "**Not supported / limitations**:", "## Not supported / limitations:",
    "  not supported / limitations  "])
def test_the_limitations_phrase_alone_on_its_line_is_the_heading(line):
    assert ig._is_limits_heading(line)


def test_items_after_a_plain_limitations_line_are_untouched():
    raw = ("- **Nutanix AHV** — y [1]\n\nNot supported / limitations:\n"
           "- **Oracle KVM** — not supported [1]")
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.item for g in grades] == ["Nutanix AHV"]
    assert text == raw


def test_sub_bullets_of_a_dropped_item_go_with_it():
    raw = ("- **Oracle KVM** — x [1]\n  - restore as a VM\n    - with a nested note\n"
           "- **Nutanix AHV** — y [1]\n  - instant recovery")
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.grade for g in grades] == [ig.DROPPED, ig.CONFIRMED]
    assert text == "- **Nutanix AHV** — y [1]\n  - instant recovery"


def test_wrapped_continuation_lines_of_a_dropped_item_go_with_it():
    raw = ("**Hypervisors**\n- **Oracle KVM** — a long description that\n"
           "  wraps onto a second line\n\n  and a third after a blank [1]\n\n"
           "- **Nutanix AHV** — y [1]\nAn unindented closing line.")
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.grade for g in grades] == [ig.DROPPED, ig.CONFIRMED]
    # The blank separating the dropped item from the next one stays (one blank).
    assert text == ("**Hypervisors**\n\n- **Nutanix AHV** — y [1]\n"
                    "An unindented closing line.")


def test_children_stop_at_a_heading_or_a_blank_then_unindented_line():
    raw = ("- **Oracle KVM** — x [1]\n  wrapped\n\nPlain paragraph.\n"
           "- **Oracle KVM 2** — x [1]\n  ## Deeper heading\n- **Nutanix AHV** — y [1]")
    text, _, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert text == "Plain paragraph.\n  ## Deeper heading\n- **Nutanix AHV** — y [1]"


def test_the_ungraded_bullet_count_is_returned():
    raw = ("- **Nutanix AHV** — y [1]\n- plain bullet [1]\n* another plain [1]\n\n"
           "**Not supported / limitations**\n- plain limitation [1]")
    got = ig.grade_items(raw, {1: _fact("nx", NUTANIX)}, q=Q_AGENT, terms=T_AGENT,
                         scores={"nx": 0.6}, floor=0.1, unnamed_floor=0.65)
    assert got.ungraded == 2 and len(got.grades) == 1
    assert ig.grade_items("- **Nutanix AHV** — y [1]", {1: _fact("nx", NUTANIX)}, q=Q_AGENT,
                          terms=T_AGENT, scores={"nx": 0.6}, floor=0.1,
                          unnamed_floor=0.65).ungraded == 0


def test_scope_name_words_are_not_distinctive_item_words():
    excluded = ls.scope_words(Scope(vendors=("Veeam",)))
    assert excluded == {"veeam"}
    assert ig._item_words("Veeam Agent for Linux", set(), excluded) == ["agent", "linux"]
    assert ig._item_words("Veeam Agent for Linux", set(), set()) == ["veeam", "agent", "linux"]


def test_an_item_whose_only_overlap_is_the_vendor_word_is_not_named():
    fact = _fact("vw", "Veeam Agent for Microsoft Windows backups can be restored to Azure.")
    raw = "- **Veeam Kasten** — x [1]"
    # Unscoped, "veeam" is one of two distinctive words and tips the item to named.
    _, grades, _ = _grade(raw, [fact], scores={"vw": 0.5}, floor=0.4)
    assert grades[0].grade != ig.DROPPED
    _, grades, _ = _grade(raw, [fact], scores={"vw": 0.5}, floor=0.4,
                          scope=Scope(vendors=("Veeam",)))
    assert grades[0].grade == ig.DROPPED and grades[0].reason == "below_floor"


def test_each_dropped_item_is_logged_with_its_reason(caplog):
    long_item = "Oracle KVM " + "x" * 100
    raw = f"- **{long_item}** — x [1]\n  wrapped\n- **Sangfor aSV** — y [99]"
    with caplog.at_level(logging.INFO, logger="answer_api.item_grades"):
        _grade(raw, [_fact("nx", NUTANIX)])
    assert f"list item dropped (below_floor): {long_item[:80]} (+1 indented line(s))" \
        in caplog.text
    assert long_item[:81] not in caplog.text
    assert "list item dropped (no_marker): Sangfor aSV" in caplog.text


# --- Follow-up: bulleted group headings, distinctive-word fallback -----------------

_BULLETED = ("- **Hypervisors**\n"
             "  - **Nutanix AHV** — y [1]\n"
             "  - **Oracle KVM** — x [1]\n"
             "- **Clouds:**\n"
             "  - **Oracle Cloud** — z [1]\n"
             "- **Other**:\n"
             "  - plain nested note")


def test_bulleted_group_headings_are_not_items_and_keep_their_nested_items():
    got = ig.grade_items(_BULLETED, {1: _fact("nx", NUTANIX)}, q=Q_AGENT, terms=T_AGENT,
                         scores={"nx": 0.6}, floor=0.1, unnamed_floor=0.65)
    assert [(g.item, g.grade) for g in got.grades] == [
        ("Nutanix AHV", ig.CONFIRMED), ("Oracle KVM", ig.DROPPED),
        ("Oracle Cloud", ig.DROPPED)]
    assert sum(g.grade == ig.DROPPED for g in got.grades) == 2     # headings not counted
    assert got.ungraded == 1                                         # the plain nested note
    # "Clouds:" lost every nested item and goes; "Other" keeps its plain note.
    assert got.text == ("- **Hypervisors**\n  - **Nutanix AHV** — y [1]\n"
                        "- **Other**:\n  - plain nested note")


def test_a_bulleted_group_whose_items_all_drop_empties_the_answer():
    raw = "- **Clouds**\n  - **Oracle Cloud** — z [1]\n  - **Oracle KVM** — x [1]"
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.grade for g in grades] == [ig.DROPPED, ig.DROPPED] and text == ""


def test_a_bulleted_group_under_a_heading_and_a_top_level_item_after_it():
    raw = ("## Targets\n- **Clouds**\n  - **Oracle Cloud** — z [1]\n"
           "- **Nutanix AHV** — y [1]\n## Other\n- **Oracle KVM** — x [1]")
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.item for g in grades] == ["Oracle Cloud", "Nutanix AHV", "Oracle KVM"]
    assert text == "## Targets\n- **Nutanix AHV** — y [1]"


def test_a_dropped_item_keeps_the_nested_items_graded_on_their_own():
    raw = "- **Oracle KVM** — x [1]\n  - **Nutanix AHV** — y [1]\n  wrapped note"
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.grade for g in grades] == [ig.DROPPED, ig.CONFIRMED]
    assert text == "  - **Nutanix AHV** — y [1]\n  wrapped note"


def test_a_bulleted_limitations_heading_ends_the_graded_region():
    raw = ("- **Nutanix AHV** — y [1]\n- **Not supported / limitations**\n"
           "  - **Oracle KVM** — not supported [1]")
    text, grades, _ = _grade(raw, [_fact("nx", NUTANIX)])
    assert [g.item for g in grades] == ["Nutanix AHV"] and text == raw


def test_an_item_made_only_of_excluded_words_still_needs_its_words_named():
    scope = Scope(products=("Veeam Backup & Replication",))
    unrelated = _fact("ax", "Restore to Nutanix AHV is possible for Windows agents.")
    _, grades, _ = _grade("- **Veeam Backup** — x [1]", [unrelated], scores={"ax": 0.5},
                          floor=0.4, scope=scope)
    assert grades[0].grade == ig.DROPPED
    named = _fact("vb", "Veeam Backup restores Windows agent backups to Nutanix AHV.")
    _, grades, _ = _grade("- **Veeam Backup** — x [1]", [named], scores={"vb": 0.5},
                          floor=0.4, scope=scope)
    assert grades[0].grade == ig.CONFIRMED
