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
    PRODUCT_ALIASES,
    PRODUCT_ENTITY_TYPE_ID,
    GroundingContext,
    GroundingStats,
    filter_entities,
    is_grounded,
    is_product_alias,
    load_product_aliases,
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
    renamed entity because the chunk only has the alias. The two cannot drift --
    whatever type the model gives it, Product included (no rule (c) there)."""
    for said, canonical in _instruction_naming_rules():
        for product_typed in (False, True):
            assert is_grounded(canonical, f"Configure {said} for the job.", (),
                               product_typed=product_typed), (canonical, said, product_typed)
            for alias in NAMING_ALIASES[canonical.lower()]:
                text = f"Configure {alias} for the job."
                assert is_grounded(canonical, text, (), product_typed=product_typed), (
                    canonical, alias, product_typed)


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
    assert ("Write each entity's name as this chunk's text writes it, apart from the naming "
            "rules above and the documented product's exact name; do not add a vendor prefix, "
            "product family or expansion the text does not use.") in flat
    # The old "(if the text says vSphere, name it vSphere)" example contradicts the
    # vSphere naming rule and must not come back.
    assert "name it vSphere." not in flat


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


# ------------------------- final-review fix wave: foreign products need a phrase

# The Securiti connector list plus the connectors the 2026-10-17 proof saw riding on
# rule (c): "AWS S3", "Azure SQL", and a stray "Backup" heading.
SECURITI_PLUS_TEXT = SECURITI_TEXT + "  + AWS S3\n  + Azure SQL\n* **Backup**\n"

VBR_TEXT = (
    "Restoring to Microsoft Azure\n"
    "You can restore backups of machines to Microsoft Azure. Before you restore, add a "
    "Microsoft Azure compute account and configure the backup infrastructure.\n"
)


def _typed(name: str, type_id: int) -> dict:
    return {"name": name, "entity_type_id": type_id}


@pytest.mark.parametrize("name", [
    "AWS Backup", "Azure Backup", "Google Cloud Backup", "Veeam Backup for AWS",
])
def test_a_foreign_product_riding_on_generic_words_is_dropped(name):
    # rule (c) alone would keep every one of these: each token is somewhere in the list
    assert is_grounded(name, SECURITI_PLUS_TEXT, ("Securiti.ai", "Veeam"))
    assert not is_grounded(name, SECURITI_PLUS_TEXT, ("Securiti.ai", "Veeam"),
                           product_typed=True)
    ctx = _ctx(SECURITI_PLUS_TEXT, ("Securiti.ai", "Veeam"))
    out = filter_entities({"extracted_entities": [_typed(name, PRODUCT_ENTITY_TYPE_ID)]}, ctx)
    assert out["extracted_entities"] == []
    assert ctx.stats.dropped_names == [name]


def test_the_same_names_typed_otherwise_keep_rule_c():
    ctx = _ctx(SECURITI_PLUS_TEXT, ("Securiti.ai", "Veeam"))
    concept = list(ENTITY_TYPES).index("Concept") + 1
    reply = {"extracted_entities": [_typed("AWS Backup", concept),
                                    _typed("Google Cloud Backup", 1)]}
    assert filter_entities(reply, ctx) is reply
    assert ctx.stats.dropped == 0


def test_a_product_named_as_a_phrase_or_alias_is_kept():
    ctx = _ctx(SECURITI_PLUS_TEXT + "Oracle RAC\n", ("Securiti.ai", "Veeam"))
    reply = {"extracted_entities": [
        _typed("Oracle Object Storage", PRODUCT_ENTITY_TYPE_ID),     # rule (a)
        _typed("aws s3", PRODUCT_ENTITY_TYPE_ID),                    # rule (a), any case
        _typed("Oracle Real Application Clusters", PRODUCT_ENTITY_TYPE_ID),  # rule (d)
    ]}
    assert filter_entities(reply, ctx) is reply


def test_on_a_vbr_chunk_a_foreign_veeam_product_needs_its_phrase():
    allowed = ("Veeam Backup & Replication", "Veeam")
    vba = "Veeam Backup for Microsoft Azure"
    assert is_grounded(vba, VBR_TEXT, allowed), "rule (c) alone keeps it today"
    ctx = GroundingContext(text=VBR_TEXT, allowed=allowed, stats=GroundingStats(),
                           product="Veeam Backup & Replication")
    out = filter_entities({"extracted_entities": [
        _typed(vba, PRODUCT_ENTITY_TYPE_ID),
        _typed("Veeam Backup & Replication", PRODUCT_ENTITY_TYPE_ID)]}, ctx)
    assert [e["name"] for e in out["extracted_entities"]] == ["Veeam Backup & Replication"]
    with_phrase = VBR_TEXT + "Veeam Backup for Microsoft Azure appliances can also be used.\n"
    assert is_grounded(vba, with_phrase, allowed, product_typed=True)


def test_the_documented_product_and_vendor_stay_grounded_when_product_typed():
    allowed = ("Veeam Backup & Replication", "Veeam")
    for name in ("Veeam Backup and Replication", "Veeam"):
        assert is_grounded(name, "Configure the job.", allowed, product_typed=True)


# --------------------- final-review fix wave: abbreviations fold into the product

VBR = "Veeam Backup & Replication"


def test_the_alias_table_is_read_from_answer_api_data():
    assert PRODUCT_ALIASES["vbr"] == (VBR,)
    # list targets are kept whole
    assert "Veeam Agent for Linux" in PRODUCT_ALIASES["veeam agent"]


def test_graph_extract_does_not_import_answer_api_code():
    import ast
    from pathlib import Path

    src = Path(grounding.__file__).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("answer_api")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("answer_api") for a in node.names)


@pytest.mark.parametrize("raw", ['[]', '{"VBR": 3}', '{"VBR": null}', '{"VBR": [3]}',
                                 '{"VBR": {"a": 1}}'])
def test_a_malformed_alias_file_fails_loudly(tmp_path, raw):
    path = tmp_path / "aliases.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_product_aliases(path)


def test_the_alias_file_is_read_as_scope_resolver_reads_it(tmp_path):
    """Both consumers must agree: an empty target list is skipped, a blank name is
    ignored -- neither is an error (ScopeResolver warns and moves on)."""
    from answer_api.scope import ScopeResolver

    raw = ('{"VBR": "Veeam Backup & Replication", "Empty": [], "Blank": [""], '
           '"Mixed": ["", "Veeam Backup & Replication"]}')
    path = tmp_path / "aliases.json"
    path.write_text(raw)
    table = load_product_aliases(path)
    assert table == {"vbr": (VBR,), "mixed": (VBR,)}
    import json
    resolver = ScopeResolver(["Veeam"], [(VBR, "Veeam")], json.loads(raw))
    assert resolver.detect("VBR jobs").products == (VBR,)
    assert resolver.detect("Mixed jobs").products == (VBR,)
    assert resolver.detect("Empty jobs").is_empty()
    assert resolver.detect("Blank jobs").is_empty()


def test_a_missing_alias_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_product_aliases(tmp_path / "absent.json")


def test_vbr_becomes_the_documented_product_and_nothing_is_injected():
    ctx = GroundingContext(text="Open the VBR console and run the job.",
                           allowed=(VBR, "Veeam"), stats=GroundingStats(), product=VBR)
    out = filter_entities({"extracted_entities": [_typed("VBR", PRODUCT_ENTITY_TYPE_ID),
                                                  _typed("job", 1)]}, ctx)
    assert out["extracted_entities"] == [_typed(VBR, PRODUCT_ENTITY_TYPE_ID), _typed("job", 1)]
    assert (ctx.stats.injected, ctx.stats.renamed, ctx.stats.dropped) == (0, 1, 0)


def test_the_alias_match_is_case_insensitive_and_takes_the_product_type():
    ctx = GroundingContext(text="vbr settings", allowed=(VBR, "Veeam"),
                           stats=GroundingStats(), product=VBR)
    out = filter_entities({"extracted_entities": [_typed("vbr", 1)]}, ctx)
    assert out["extracted_entities"] == [_typed(VBR, PRODUCT_ENTITY_TYPE_ID)]
    assert ctx.stats.injected == 0


def test_an_alias_next_to_the_product_is_folded_not_duplicated():
    ctx = GroundingContext(text="VBR is Veeam Backup & Replication.", allowed=(VBR, "Veeam"),
                           stats=GroundingStats(), product=VBR)
    out = filter_entities({"extracted_entities": [_typed(VBR, PRODUCT_ENTITY_TYPE_ID),
                                                  _typed("VBR", PRODUCT_ENTITY_TYPE_ID)]}, ctx)
    assert out["extracted_entities"] == [_typed(VBR, PRODUCT_ENTITY_TYPE_ID)]
    assert (ctx.stats.injected, ctx.stats.renamed) == (0, 1)


def test_an_alias_of_a_different_product_is_left_alone():
    # VB365 is in the table, but it names Veeam Backup for Microsoft 365, not VBR
    ctx = GroundingContext(text="VB365 and VBR both write to the repository.",
                           allowed=(VBR, "Veeam"), stats=GroundingStats(), product=VBR)
    reply = {"extracted_entities": [_typed("VB365", PRODUCT_ENTITY_TYPE_ID),
                                    _typed(VBR, PRODUCT_ENTITY_TYPE_ID)]}
    assert filter_entities(reply, ctx) is reply
    assert ctx.stats.renamed == 0
    assert not is_product_alias("VB365", VBR)


def test_no_table_entry_means_no_rename():
    # "B&R" is a plausible short form but the table does not list it; initials are
    # never derived either
    ctx = GroundingContext(text="B&R and VB&R settings", allowed=(VBR, "Veeam"),
                           stats=GroundingStats(), product=VBR)
    reply = {"extracted_entities": [_typed("B&R", 1), _typed("VB&R", 1)]}
    out = filter_entities(reply, ctx)
    assert [e["name"] for e in out["extracted_entities"]] == ["B&R", "VB&R", VBR]
    assert (ctx.stats.renamed, ctx.stats.injected) == (0, 1)
    assert not is_product_alias("VBR", VBR, aliases={})


def test_the_product_itself_is_not_its_own_alias():
    assert not is_product_alias(VBR, VBR, aliases={"veeam backup & replication": (VBR,)})
    assert not is_product_alias("VBR", "", aliases={"vbr": (VBR,)})


def test_an_ungrounded_alias_is_dropped_and_the_product_injected():
    ctx = GroundingContext(text="Configure the job.", allowed=(VBR, "Veeam"),
                           stats=GroundingStats(), product=VBR)
    out = filter_entities({"extracted_entities": [_typed("VBR", PRODUCT_ENTITY_TYPE_ID)]}, ctx)
    assert out["extracted_entities"] == [_typed(VBR, PRODUCT_ENTITY_TYPE_ID)]
    assert (ctx.stats.dropped, ctx.stats.renamed, ctx.stats.injected) == (1, 0, 1)


def test_renamed_counts_merge():
    a = GroundingStats(renamed=2)
    a.merge(GroundingStats(renamed=1))
    assert a.renamed == 3


def test_the_instructions_fold_abbreviations_into_the_documented_product():
    flat = " ".join(EXTRACTION_INSTRUCTIONS.split())
    assert ("If the text names the documented product by an abbreviation or short form, "
            "use the documented product's exact name.") in flat


# -------------------- final-review fix wave: cross-vendor conditional naming rules

@pytest.mark.parametrize("canonical, text", [
    ("VMware vSphere", "Back up vSphere hosts."),
    ("Azure VM", "Restore to an Azure virtual machine."),
    ("Azure VM", "Restore to Azure virtual machines."),
    ("Amazon EC2", "Restore to EC2 instances."),
    ("SQL Server", "Protect MSSQL databases."),
    ("Microsoft 365", "Back up M365 mailboxes."),
    ("Microsoft 365", "Back up Office 365 mailboxes."),
    ("Oracle Real Application Clusters", "Protect Oracle RAC databases."),
])
def test_cross_vendor_canonical_names_ground_on_their_alias(canonical, text):
    assert is_grounded(canonical, text, ())
    assert is_grounded(canonical, text, (), product_typed=True)


@pytest.mark.parametrize("canonical", [
    "VMware vSphere", "Azure VM", "Amazon EC2", "SQL Server", "Microsoft 365",
    "Oracle Real Application Clusters",
])
def test_cross_vendor_naming_rules_are_conditional(canonical):
    # the rule existing grounds nothing: a text without the alias drops the name
    assert canonical in [c for c, _, _ in NAMING_RULES]
    assert not is_grounded(canonical, "Configure the backup job schedule.", ())


# ------------- follow-up: '&'-normalised phrases and single-target scope aliases

@pytest.mark.parametrize("name, text", [
    ("Veeam Backup & Replication", "Install Veeam Backup and Replication first."),
    ("Veeam Backup and Replication", "Install Veeam Backup & Replication first."),
    ("Hyper-V", "a Hyper V host"),
    ("Backup and Archive (B&A)", "see Backup and Archive (B and A) docs"),
    ("Veeam Backup & Replication", "**Veeam Backup&Replication** console"),
])
def test_rule_a_matches_on_normalised_phrases(name, text):
    assert is_grounded(name, text, (), product_typed=True)


@pytest.mark.parametrize("name, text", [
    ("AWS Backup", "+ AWS\n+ Backup\n"),                  # two list items
    ("AWS Backup", "| AWS | Backup |"),                     # two table cells
    ("AWS Backup", "Supported: AWS, Backup Exec"),         # a comma list
    ("Azure Backup", "Restore to Azure. Backup jobs run nightly."),   # sentence end
])
def test_a_normalised_phrase_never_crosses_a_boundary(name, text):
    assert not is_grounded(name, text, (), product_typed=True)


@pytest.mark.parametrize("name", [
    "AWS Backup", "Azure Backup", "Google Cloud Backup", "Veeam Backup for AWS",
])
def test_the_securiti_names_stay_dropped_after_normalisation(name):
    assert not is_grounded(name, SECURITI_PLUS_TEXT, ("Securiti.ai", "Veeam"),
                           product_typed=True)
    assert not is_grounded(name, SECURITI_TEXT, ("Securiti.ai", "Veeam"),
                           product_typed=True)


@pytest.mark.parametrize("name, text", [
    ("Veeam Backup & Replication", "Back up VB365 data with VBR."),
    ("Veeam Backup and Replication", "Add the VBR server."),
    ("Veeam Backup for Microsoft 365", "Configure VB365 proxies."),
    ("Veeam Service Provider Console", "Manage tenants in VSPC."),
    ("Veeam Kasten for Kubernetes", "Kasten K10 policies export restore points."),
    ("Veeam Agent for Microsoft Windows", "Deploy Veeam Agent for Windows on servers."),
])
def test_a_product_grounds_on_a_scope_alias_that_names_it_alone(name, text):
    assert is_grounded(name, text, ("Veeam Data Cloud", "Veeam"), product_typed=True)


def test_scope_aliases_ground_only_product_typed_names():
    # a Concept named after a product is not grounded by the product's alias
    assert not is_grounded("Veeam Service Provider Console", "Manage tenants in VSPC.", ())


def test_a_multi_target_alias_grounds_none_of_its_targets():
    text = "Install the Veeam Agent on each machine."
    for target in PRODUCT_ALIASES["veeam agent"]:
        assert not is_grounded(target, text, ("Veeam Backup & Replication", "Veeam"),
                               product_typed=True), target
    # an injected table: two targets -> neither is grounded by the alias
    table = {"ab": ("Alpha Backup", "Beta Backup"), "ga": ("Gamma Archive",)}
    assert not is_grounded("Alpha Backup", "AB jobs", (), product_typed=True,
                           scope_aliases=table)
    assert is_grounded("Gamma Archive", "GA jobs", (), product_typed=True,
                       scope_aliases=table)


def test_a_scope_alias_must_be_a_whole_word():
    assert not is_grounded("Veeam Backup & Replication", "the VBRx module", (),
                           product_typed=True)


def test_a_vbr_chunk_keeps_its_cross_product_mentions_through_filter():
    vb365 = "Veeam Backup for Microsoft 365"
    ctx = GroundingContext(
        text="VB365 can use Veeam Backup and Replication repositories. Kasten K10 too.",
        allowed=(vb365, "Veeam"), stats=GroundingStats(), product=vb365)
    reply = {"extracted_entities": [
        _typed("Veeam Backup & Replication", PRODUCT_ENTITY_TYPE_ID),
        _typed("Veeam Kasten for Kubernetes", PRODUCT_ENTITY_TYPE_ID),
        _typed(vb365, PRODUCT_ENTITY_TYPE_ID)]}
    assert filter_entities(reply, ctx) is reply
    assert ctx.stats.dropped == 0


def test_entra_id_grounds_microsoft_entra_id():
    assert is_grounded("Microsoft Entra ID", "Sign in with Entra ID.", (), product_typed=True)
    assert not is_grounded("Microsoft Entra ID", "Sign in with Entra.", (), product_typed=True)
