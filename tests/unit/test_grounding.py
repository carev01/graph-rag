"""Deterministic grounding gate (extraction-grounding spec section 2.3).

graphiti's node-extraction reply is filtered before graphiti sees it: an entity
survives only when the chunk text mentions it (verbatim, or by most of its
distinctive words) or it is the document's own product/vendor. The two texts
below are the real failure cases of 2026-10-17.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import pytest

from graph_extract import grounding
from graph_extract.ontology import (
    ENTITY_TYPES,
    EXTRACTION_INSTRUCTIONS,
    NAMING_ALIASES,
    NAMING_RULES,
)
from graph_extract.grounding import (
    CURRENT_GROUNDING,
    EXTRACT_TEXT_PROMPT_NAME,
    PRODUCT_ENTITY_TYPE_ID,
    GroundingContext,
    GroundingStats,
    filter_entities,
    is_grounded,
    note_unscoped,
)

# "Access verification for self-hosted connectors" (Veeam / Securiti.ai): a bare
# connector list that names no backup product. Extraction made 37 facts of the form
# "AWS Backup supports <connector>". Abridged to the connectors quoted in the trace.
SECURITI_TEXT = (
    "Access verification for self-hosted connectors\n"
    "* **DBaaS**\n"
    "  + Amazon DocumentDB\n"
    "  + Amazon DynamoDB\n"
    "  + Amazon Redshift\n"
    "  + AWS RDS MariaDB\n"
    "  + AWS RDS MySQL\n"
    "* **IaaS**\n"
    "  + Alibaba OSS\n"
    "  + Azure Files\n"
    "  + Azure Blob\n"
    "  + Google Cloud Storage\n"
    "  + HDFS\n"
    "  + LDAP\n"
    "  + NFS\n"
    "  + Oracle Object Storage\n"
    "  + S3\n"
    "  + SFTP\n"
    "  + SMB\n"
)

# "Controls and remediation" (AWS Backup): a short Security Hub control. Extraction
# invented Hyper-V, Azure VM, VMware vSphere, East US, LRS, SEC 17a-4 -- the
# instructions' own example vocabulary, none of it in this text.
AWS_CONTROL_TEXT = (
    "Controls and remediation\n"
    "## [Backup.1] AWS Backup: recovery points should be encrypted at rest\n"
    "**Category:** Protect > Data protection > Encryption of data-at-rest\n"
    "**Severity:** Medium\n"
    "**Resource type:** `AWS::Backup::RecoveryPoint`\n"
    "This Security Hub control checks if an AWS Backup recovery point is encrypted at rest. The "
    "control fails if the recovery point isn't encrypted at rest. Recovery points are "
    "created by a backup plan; use a backup plan that encrypts recovery points with an "
    "AWS KMS key.\n"
    "### Remediation\n"
    "To encrypt recovery points, see "
    "https://docs.aws.amazon.com/aws-backup/latest/devguide/encryption.html\n"
)


def test_the_fixtures_are_the_failure_shapes():
    assert "backup" not in SECURITI_TEXT.lower()
    assert "AWS RDS MySQL" in SECURITI_TEXT


# ------------------------------------------------------------------ is_grounded

@pytest.mark.parametrize("name, grounded", [
    ("AWS Backup", False),            # "aws" is there (AWS RDS), "backup" is not: 1 of 2
    ("Azure Blob Storage", True),     # "+ Azure Blob +": 2 of 3 distinctive words
    ("Oracle Object Storage", True),  # verbatim
    ("Amazon S3", True),              # "amazon" + "s3" both at word starts
    ("Kubernetes", False),
])
def test_securiti_connector_list(name, grounded):
    assert is_grounded(name, SECURITI_TEXT, ("Securiti.ai", "Veeam")) is grounded


@pytest.mark.parametrize("name, grounded", [
    ("Hyper-V", False),
    ("Azure VM", False),
    ("VMware vSphere", False),
    ("East US", False),       # "use" is not "us", "latest" is not "east"
    ("LRS", False),
    ("SEC 17a-4", False),     # neither "sec" nor "17a" is a whole word here
    ("recovery point", True),
    ("AWS Backup", True),     # named in the text, and the document's product
    ("backup plan", True),
])
def test_aws_control_text(name, grounded):
    assert is_grounded(name, AWS_CONTROL_TEXT, ("AWS Backup", "AWS")) is grounded


def test_inflection_counts_as_present():
    assert is_grounded("recovery point", "Delete old recovery points.", ())
    # the NAME inflected and the text not: the stem matches
    assert is_grounded("recovery points", "Delete an old recovery point.", ())
    assert is_grounded("Encrypted backups", "Backups are encrypting at rest.", ())


def test_a_token_must_start_a_word():
    # "east" inside "beasts", "vm" inside "kvm": neither starts a word
    assert not is_grounded("East US", "beasts used here", ())
    assert not is_grounded("Azure VM", "a kvm host on azure", ())


def test_the_verbatim_rule_needs_a_whole_word():
    """Rule (a) matches whole words only: a short name never rides on a longer word."""
    assert not is_grounded("VM", "a kvm host", ())
    assert is_grounded("SQL Server", "Microsoft SQL Server 2019", ())


@pytest.mark.parametrize("name, text", [
    ("VM", "load the kvm module"),
    ("AD", "load balancing"),
    ("S3", "the S3x appliance"),
    ("Hyper-V", "a type-1 hypervisor"),
    ("US", "use the console"),
    ("East US", "Use the eastern console"),
    ("NAS", "NASA archives"),
    ("Go", "Google Cloud"),
])
def test_imprecise_matches_are_dropped(name, text):
    assert not is_grounded(name, text, ())


@pytest.mark.parametrize("name, text", [
    ("air gap", "keep an air-gapped copy"),
    ("snapshot", "storage snapshotting is supported"),
    ("Azure VM", "protect Azure VMs"),
    ("Linux", "linux-based proxies"),
    ("Azure Archive tier", "Azure Archive tiering"),
    ("backup copy job", "create backup copy jobs"),
    ("recovery point objective", "recovery point objectives"),
    ("Veeam Agent for Microsoft Windows", "Veeam Agent for Windows"),
    ("SQL Server", "Microsoft SQL Server 2019"),
    ("HPE Morpheus VM Essentials", "restore to Morpheus VM Essentials"),
    ("Sangfor aSV", "Sangfor HCI (aSV)"),
    ("recovery point objectives", "the recovery point objective"),
])
def test_inflected_and_partial_names_are_kept(name, text):
    assert is_grounded(name, text, ())


def test_the_documented_product_or_vendor_is_always_grounded():
    text = "Configure the job to run daily."
    allowed = ("Veeam Backup & Replication", "Veeam")
    assert is_grounded("Veeam Backup & Replication", text, allowed)
    assert is_grounded("veeam backup & replication", text, allowed)
    assert is_grounded("VEEAM", text, allowed)
    assert not is_grounded("Veeam Agent for Linux", text, allowed)
    # compared by distinctive tokens: "and" vs "&" is the same name
    assert is_grounded("Veeam Backup and Replication", text, allowed)


def test_empty_allowed_names_are_ignored():
    assert not is_grounded("Commvault", "Configure the job.", ("", ""))


def test_naming_aliases_ground_the_canonical_name():
    assert is_grounded("Kubernetes", "Back up K8s clusters.", ())
    assert is_grounded("Proxmox VE", "Proxmox Virtual Environment hosts", ())
    assert is_grounded("Proxmox Virtual Environment", "a Proxmox VE cluster", ())
    # whole words only
    assert not is_grounded("Kubernetes", "the k8sx tool", ())


_RULE = re.compile(r"If (?:the text|it) says (.+?), name it (.+?)\.")


def _instruction_naming_rules() -> list[tuple[str, str]]:
    flat = " ".join(EXTRACTION_INSTRUCTIONS.split())
    return [(said, canonical) for said, canonical in _RULE.findall(flat) if said != canonical]


def test_every_instruction_naming_rule_is_in_the_alias_table():
    rules = _instruction_naming_rules()
    assert [c for _, c in rules] == [c for c, _, _ in NAMING_RULES]
    for said, canonical in rules:
        assert canonical.lower() in NAMING_ALIASES, canonical


def test_the_gate_keeps_every_instruction_canonical_name_for_its_alias():
    """The instructions tell the model to rename; the gate must not then drop the
    renamed entity because the chunk only has the alias. The two cannot drift."""
    for said, canonical in _instruction_naming_rules():
        assert is_grounded(canonical, f"Configure {said} for the job.", ()), (canonical, said)
        for alias in NAMING_ALIASES[canonical.lower()]:
            text = f"Configure {alias} for the job."
            assert is_grounded(canonical, text, ()), (canonical, alias)


def test_the_proxmox_pair_is_grounding_only():
    assert "proxmox" not in EXTRACTION_INSTRUCTIONS.lower()


def test_cbt_grounds_changed_block_tracking_and_is_grounding_only():
    assert NAMING_ALIASES["changed block tracking"] == ("cbt",)
    assert is_grounded("Changed Block Tracking", "Enable CBT on the VM.", ())
    assert not is_grounded("Changed Block Tracking", "the cbtx driver", ())
    flat = EXTRACTION_INSTRUCTIONS.lower()
    assert "changed block tracking" not in flat and "cbt" not in flat


@pytest.mark.parametrize("singular, plural", [
    ("storage policy", "storage policies"),
    ("protection policy", "protection policies"),
    ("backup proxy", "backup proxies"),
    ("tape library", "tape libraries"),
    ("repository", "repositories"),
    ("backup frequency", "backup frequencies"),
])
def test_ies_plurals_match_in_both_directions(singular, plural):
    assert is_grounded(singular, f"Configure the {plural} first.", ())
    assert is_grounded(plural, f"Configure the {singular} first.", ())


def test_ies_plurals_stay_whole_words():
    assert is_grounded("policy", "the policy was applied", ())
    assert is_grounded("copy", "files were copied", ())
    assert not is_grounded("policy", "the policiesx table", ())
    assert not is_grounded("proxy", "a proxiesque setup", ())


def test_the_instructions_forbid_vendor_expanded_names():
    flat = " ".join(EXTRACTION_INSTRUCTIONS.split())
    assert ("Write each entity's name as this chunk's text writes it, apart from the renames "
            "above; do not add a vendor prefix, product family or expansion the text does not use (if the text says "
            "vSphere, name it vSphere).") in flat


def test_the_first_unscoped_call_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(grounding, "_unscoped_warned", False)
    stats = GroundingStats()
    with caplog.at_level(logging.WARNING, logger="graph_extract.grounding"):
        note_unscoped(stats)
        note_unscoped(stats)
    assert stats.unscoped == 2
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ["grounding gate bypassed: extraction call without a grounding context"]


def test_a_name_with_no_distinctive_tokens_needs_a_verbatim_match():
    assert not is_grounded("The A", "nothing here", ())
    assert is_grounded("The A", "see the a list", ())


# --------------------------------------------------------------- filter_entities

def _reply(*names: str) -> dict:
    return {"extracted_entities": [{"name": n, "entity_type_id": 1} for n in names]}


def _ctx(text: str, allowed: tuple[str, ...] = ()) -> GroundingContext:
    return GroundingContext(text=text, allowed=allowed, stats=GroundingStats())


def test_filter_drops_ungrounded_entities_and_counts_them():
    ctx = _ctx(SECURITI_TEXT, ("Securiti.ai", "Veeam"))
    out = filter_entities(
        _reply("AWS Backup", "Azure Blob Storage", "Oracle Object Storage", "Hyper-V"), ctx)
    assert [e["name"] for e in out["extracted_entities"]] == [
        "Azure Blob Storage", "Oracle Object Storage"]
    assert (ctx.stats.calls, ctx.stats.dropped) == (1, 2)
    assert ctx.stats.dropped_names == ["AWS Backup", "Hyper-V"]


def test_filter_keeps_other_reply_keys_and_entity_fields():
    ctx = _ctx("Azure Blob")
    reply = {"extracted_entities": [{"name": "Azure Blob", "entity_type_id": 3,
                                     "episode_indices": [0]}], "extra": 1}
    out = filter_entities(reply, ctx)
    assert out == reply


def test_dropped_names_are_capped_at_50():
    ctx = _ctx("nothing")
    filter_entities(_reply(*[f"Thing{i}" for i in range(80)]), ctx)
    assert ctx.stats.dropped == 80
    assert len(ctx.stats.dropped_names) == 50


@pytest.mark.parametrize("reply", [
    None, "text", [], {"entities": []}, {"extracted_entities": "x"},
])
def test_a_malformed_reply_passes_through_unchanged_and_is_counted(reply, caplog):
    ctx = _ctx("anything")
    with caplog.at_level(logging.WARNING, logger="graph_extract.grounding"):
        out = filter_entities(reply, ctx)
    assert out is reply
    assert (ctx.stats.parse_failures, ctx.stats.dropped) == (1, 0)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_an_entity_without_a_string_name_is_kept():
    """A failure never reads as absence: graphiti's own validation decides."""
    ctx = _ctx("nothing")
    reply = {"extracted_entities": [{"entity_type_id": 1}, {"name": " ", "entity_type_id": 1}]}
    assert filter_entities(reply, ctx)["extracted_entities"] == reply["extracted_entities"]
    assert ctx.stats.dropped == 0


def test_stats_merge_sums_and_keeps_the_name_cap():
    a = GroundingStats(calls=1, dropped=30, parse_failures=1, unscoped=2,
                       dropped_names=[f"a{i}" for i in range(30)])
    b = GroundingStats(calls=2, dropped=30, parse_failures=0, unscoped=1,
                       dropped_names=[f"b{i}" for i in range(30)])
    a.merge(b)
    assert (a.calls, a.dropped, a.parse_failures, a.unscoped) == (3, 60, 1, 3)
    assert len(a.dropped_names) == 50 and a.dropped_names[30] == "b0"


def test_the_context_var_defaults_to_none():
    assert CURRENT_GROUNDING.get() is None


async def test_pin_graphiti_extracts_text_episodes_under_this_prompt_name():
    """Tripwire: the gate keys on graphiti's prompt name and reply shape."""
    from graphiti_core.nodes import EpisodeType, EpisodicNode
    from graphiti_core.prompts.extract_nodes import ExtractedEntities
    from graphiti_core.utils.maintenance import node_operations

    seen: dict = {}

    class _LLM:
        async def generate_response(self, messages, *args, **kwargs):
            seen.update(kwargs)
            return _reply("x")

    ep = EpisodicNode(name="ep", group_id="g", source=EpisodeType.text,
                      source_description="d", content="c",
                      valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    ctx = {"episode_content": "c", "episode_timestamp": "t", "previous_episodes": [],
           "custom_extraction_instructions": "", "entity_types": [],
           "source_description": "d"}
    await node_operations._call_extraction_llm(_LLM(), ep, ctx)
    assert seen["prompt_name"] == EXTRACT_TEXT_PROMPT_NAME
    assert seen["response_model"] is ExtractedEntities


# ------------------------------------------- documented-product injection (follow-up)

def _pctx(text: str, product: str) -> GroundingContext:
    return GroundingContext(text=text, allowed=tuple(n for n in (product, "Veeam") if n),
                            stats=GroundingStats(), product=product)


def test_an_omitted_product_is_appended_once_with_the_product_type_id():
    ctx = _pctx(SECURITI_TEXT, "Securiti.ai")
    out = filter_entities(_reply("Azure Blob Storage", "AWS Backup"), ctx)
    assert out["extracted_entities"] == [
        {"name": "Azure Blob Storage", "entity_type_id": 1},
        {"name": "Securiti.ai", "entity_type_id": PRODUCT_ENTITY_TYPE_ID},
    ]
    assert (ctx.stats.injected, ctx.stats.dropped) == (1, 1)


def test_the_product_is_appended_even_when_nothing_was_dropped():
    ctx = _pctx(SECURITI_TEXT, "Securiti.ai")
    out = filter_entities(_reply("Azure Blob Storage"), ctx)
    assert [e["name"] for e in out["extracted_entities"]] == ["Azure Blob Storage", "Securiti.ai"]
    assert (ctx.stats.injected, ctx.stats.dropped) == (1, 0)


def test_the_product_is_appended_to_an_empty_reply():
    ctx = _pctx("nothing named", "Securiti.ai")
    out = filter_entities(_reply(), ctx)
    assert out["extracted_entities"] == [
        {"name": "Securiti.ai", "entity_type_id": PRODUCT_ENTITY_TYPE_ID}]


@pytest.mark.parametrize("extracted", [
    "Veeam Backup & Replication", "veeam backup & replication", "Veeam Backup and Replication",
])
def test_a_present_product_is_not_duplicated(extracted):
    ctx = _pctx("Configure the job.", "Veeam Backup & Replication")
    reply = _reply(extracted, "job")
    out = filter_entities(reply, ctx)
    assert [e["name"] for e in out["extracted_entities"]] == [extracted, "job"]
    assert ctx.stats.injected == 0
    assert out is reply, "nothing dropped, nothing injected: the reply is untouched"


def test_an_unknown_product_injects_nothing():
    ctx = _pctx("Azure Blob", "")
    reply = _reply("Azure Blob")
    assert filter_entities(reply, ctx) is reply
    assert ctx.stats.injected == 0


@pytest.mark.parametrize("reply", [None, {"entities": []}, {"extracted_entities": "x"}])
def test_a_malformed_reply_gets_no_product(reply):
    ctx = _pctx("anything", "Securiti.ai")
    assert filter_entities(reply, ctx) is reply
    assert (ctx.stats.parse_failures, ctx.stats.injected) == (1, 0)


def test_the_injected_entity_validates_as_graphiti_s_extracted_entities():
    from graphiti_core.prompts.extract_nodes import ExtractedEntities

    ctx = _pctx("x", "Securiti.ai")
    parsed = ExtractedEntities(**filter_entities(_reply(), ctx))
    assert parsed.extracted_entities[0].name == "Securiti.ai"


def test_tripwire_graphiti_numbers_entity_types_as_assumed():
    """PRODUCT_ENTITY_TYPE_ID must be the id graphiti itself shows the model for
    Product, and maps back to the Product label (node_operations, 0.30.1)."""
    from graphiti_core.utils.maintenance.node_operations import _build_entity_types_context

    context = _build_entity_types_context(ENTITY_TYPES)
    assert context[0]["entity_type_name"] == "Entity"
    assert context[PRODUCT_ENTITY_TYPE_ID]["entity_type_name"] == "Product"
    assert context[PRODUCT_ENTITY_TYPE_ID]["entity_type_id"] == PRODUCT_ENTITY_TYPE_ID


def test_injected_counts_merge():
    a = GroundingStats(injected=2)
    a.merge(GroundingStats(injected=3))
    assert a.injected == 5
