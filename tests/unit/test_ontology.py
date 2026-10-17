from pydantic import BaseModel
from graph_extract.ontology import (
    ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP, EXTRACTION_INSTRUCTIONS,
)

def test_entity_types_are_the_nine():
    assert set(ENTITY_TYPES) == {
        "Vendor", "Product", "Tool", "Workload", "Capability", "Platform",
        "Concept", "Region", "Requirement"}
    assert all(issubclass(t, BaseModel) for t in ENTITY_TYPES.values())

def test_tool_type_registered():
    from graph_extract.ontology import Tool
    assert ENTITY_TYPES["Tool"] is Tool
    doc = Tool.__doc__ or ""
    assert "NOT the product" in doc
    assert "SDK" in doc and "console" in doc.lower()

def test_tool_edge_wired():
    assert "Operates" in EDGE_TYPES
    assert ("Tool", "Product") in EDGE_TYPE_MAP
    assert "Operates" in EDGE_TYPE_MAP[("Tool", "Product")]

def test_limits_edge_broadened_to_capability_and_platform():
    assert "Limits" in EDGE_TYPE_MAP[("Product", "Workload")]
    assert "Limits" in EDGE_TYPE_MAP[("Product", "Capability")]
    assert "Limits" in EDGE_TYPE_MAP[("Product", "Platform")]
    # Broadening is additive: existing edges remain.
    assert "Provides" in EDGE_TYPE_MAP[("Product", "Capability")]
    assert "IntegratesWith" in EDGE_TYPE_MAP[("Product", "Platform")]

def test_no_entity_type_has_attributes():
    # `version` was dropped from Product (low value for backup products, and it
    # made the cheap extraction model balloon/truncate during graphiti's
    # attribute-extraction step). With NO entity type carrying attributes,
    # graphiti skips that step entirely.
    for name, model in ENTITY_TYPES.items():
        assert model.model_fields == {}, f"{name} unexpectedly declares attributes"


def test_instructions_are_grounded_in_chunk_text():
    # 2026-10-17 extraction-grounding: a mandatory-subject rule plus leaky
    # examples caused extraction to invent "AWS Backup" as the subject of
    # subjectless chunks. The instructions must now forbid inventing entities
    # and defer the documented product's name to the per-chunk DOCUMENT CONTEXT
    # (graphiti_client.document_context) rather than naming it here.
    instr = EXTRACTION_INSTRUCTIONS
    flat = " ".join(instr.split())
    assert "Extract ONLY entities and facts that THIS chunk's text states" in flat
    assert "DOCUMENT CONTEXT" in flat
    lowered = flat.lower()
    assert "never add an entity" in lowered
    assert "never make any other product the subject of a fact" in lowered


def test_instructions_leak_no_example_products():
    # The instructions' own example vocabulary was being emitted as fabricated
    # facts on thin chunks (BACKLOG extraction-grounding). None of the example
    # products that used to appear here may appear again.
    for leaked in ("AWS Backup", "Azure Backup", "Veeam Backup & Replication"):
        assert leaked not in EXTRACTION_INSTRUCTIONS


def test_instructions_capture_limitation_statements():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "not supported" in lowered
    assert "limits" in lowered

def test_instructions_exclude_ui_elements():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "ui noise" in lowered
    assert "panes" in lowered and "wizards" in lowered
    assert "action labels" in lowered

def test_tool_docstring_has_negative_ui_examples():
    from graph_extract.ontology import Tool
    doc = Tool.__doc__ or ""
    assert "pane" in doc.lower()
    assert "button" in doc.lower()
    assert "wizard" in doc.lower()
    assert "Restore pane" in doc
    assert "NOT an action" in doc


def test_region_docstring_has_negative_examples():
    from graph_extract.ontology import Region
    doc = Region.__doc__ or ""
    assert "Availability Zone" in doc
    assert "API operation" in doc
    assert "PII" in doc
    assert "policy/permission/role name" in doc

def test_concept_and_platform_cover_storage_classes():
    # extraction-grounding (2026-10-17): docstrings name no concrete storage
    # class/tier (no "S3 Standard"), only the generic boundary.
    from graph_extract.ontology import Concept, Platform
    assert "storage classes" in (Concept.__doc__ or "")
    assert "storage classes" in (Platform.__doc__ or "")

def test_edge_type_map_references_defined_types():
    for (src, tgt), edges in EDGE_TYPE_MAP.items():
        assert src in ENTITY_TYPES and tgt in ENTITY_TYPES
        assert all(e in EDGE_TYPES for e in edges)

def test_instructions_mention_canonical_names():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "kubernetes" in lowered and "immutability" in lowered

def test_instructions_exclude_noise_entities():
    # The concrete examples (ARNs, resource IDs, specific error codes, CLI
    # invocations) were leaking into extraction as fabricated facts on thin
    # chunks, so the rule is now stated generically, with no example to leak.
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "do not extract" in lowered
    assert "arns" in lowered
    assert "resource ids" in lowered
    assert "error or exception codes" in lowered
    assert "cli commands" in lowered
    assert "example or placeholder values" in lowered
    assert "extract the concept, not the example" in lowered

def test_instructions_exclude_timezones_but_not_regions():
    # v5: regions are first-class (Region entity), no longer excluded as noise.
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "time zones" in lowered
    assert "a specific region or jurisdiction is a region" in lowered
    assert "specific geographic regions" not in lowered

def test_instructions_state_type_boundaries():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "storage-redundancy tiers" in lowered
    assert "configuration settings" in lowered
    assert "concepts, never platforms" in lowered
    assert "general domain nouns are concepts, not requirements" in lowered

def test_instructions_have_vendor_scoping_rule():
    # No example vendor/branded-feature names here: those leaked into
    # extraction as fabricated entities (test_instructions_leak_no_example_products).
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "vendor-branded features distinct" in lowered
    assert "keep different vendors' products" in lowered
    assert "do not fold it into immutability" in lowered

def test_docstrings_state_type_boundaries():
    from graph_extract.ontology import (
        Capability, Concept, Platform, Product, Requirement,
    )
    assert "NOT" in (Platform.__doc__ or "")
    assert "CLI" in (Platform.__doc__ or "")
    assert "region" in (Platform.__doc__ or "").lower()
    assert "NOT" in (Capability.__doc__ or "")
    assert "regulation" in (Concept.__doc__ or "").lower()
    # extraction-grounding (2026-10-17): no concrete tier name ("LRS"), just
    # the generic boundary.
    assert "storage-redundancy tier" in (Concept.__doc__ or "")
    assert "CLI" in (Product.__doc__ or "")
    assert "Tool" in (Product.__doc__ or "")
    assert "NOT" in (Requirement.__doc__ or "")
    assert "Concept" in (Requirement.__doc__ or "")

def test_region_type_and_edge():
    from graph_extract import ontology as o
    assert "Region" in o.ENTITY_TYPES
    assert "AvailableIn" in o.EDGE_TYPES
    for pair in [("Product", "Region"), ("Capability", "Region"), ("Workload", "Region")]:
        assert "AvailableIn" in o.EDGE_TYPE_MAP[pair]
    for (a, b), edges in o.EDGE_TYPE_MAP.items():
        assert a in o.ENTITY_TYPES and b in o.ENTITY_TYPES
        assert all(e in o.EDGE_TYPES for e in edges)
    instr = o.EXTRACTION_INSTRUCTIONS.lower()
    assert "available in" in instr
    assert "specific geographic regions" not in instr  # regions are now extracted


def test_cheap_tier_salience_present_and_not_global():
    from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS
    assert "BE SELECTIVE" in CHEAP_TIER_SALIENCE
    assert "AvailableIn" in CHEAP_TIER_SALIENCE and "Limits" in CHEAP_TIER_SALIENCE
    # cheap-tier-only: the salience text must NOT be in the global instructions
    assert "BE SELECTIVE" not in EXTRACTION_INSTRUCTIONS


# -- extraction-grounding (2026-10-17), task 2: RestoresTo relation and
# example-free type descriptions. A traced AWS article produced "Azure Archive
# tier", "SEC 17a-4", "LRS", "ZRS" -- exactly the old docstrings' own example
# vocabulary -- so every type docstring must now name no concrete entity, and
# restore/recovery/migration statements get a dedicated relation instead of
# being squeezed into Supports edges pointing at the backup source.

def test_restores_to_registered():
    from graph_extract.ontology import EDGE_TYPES, RestoresTo
    assert EDGE_TYPES["RestoresTo"] is RestoresTo


def test_restores_to_wired_for_every_listed_pair():
    from graph_extract.ontology import EDGE_TYPE_MAP
    pairs = [
        ("Product", "Platform"), ("Product", "Workload"),
        ("Capability", "Platform"), ("Capability", "Workload"),
        ("Workload", "Platform"), ("Workload", "Workload"),
    ]
    for pair in pairs:
        assert "RestoresTo" in EDGE_TYPE_MAP[pair], pair


def test_restores_to_is_additive_existing_types_kept():
    from graph_extract.ontology import EDGE_TYPE_MAP
    assert "Supports" in EDGE_TYPE_MAP[("Product", "Workload")]
    assert "Limits" in EDGE_TYPE_MAP[("Product", "Workload")]
    assert "IntegratesWith" in EDGE_TYPE_MAP[("Product", "Platform")]
    assert "Limits" in EDGE_TYPE_MAP[("Product", "Platform")]
    assert "AppliesTo" in EDGE_TYPE_MAP[("Capability", "Workload")]


_LEAKY_EXAMPLE_NAMES = (
    "AWS", "Azure", "Veeam", "NetBackup", "Amazon", "S3", "LRS", "ZRS",
    "SEC 17a-4", "East US", "Germany", "Kubernetes", "SQL Server", "Oracle",
    "Hyper-V", "VMware", "Windows", "Linux", "RPO", "GDPR",
)


def test_no_type_docstring_names_a_concrete_example_entity():
    from graph_extract.ontology import ENTITY_TYPES, EDGE_TYPES
    for name, model in {**ENTITY_TYPES, **EDGE_TYPES}.items():
        doc = model.__doc__ or ""
        for leaked in _LEAKY_EXAMPLE_NAMES:
            assert leaked not in doc, f"{name} docstring leaks {leaked!r}"


def test_cheap_tier_salience_names_no_concrete_example():
    from graph_extract.ontology import CHEAP_TIER_SALIENCE
    for leaked in ("Amazon S3", "cross-Region", "China"):
        assert leaked not in CHEAP_TIER_SALIENCE


def test_instructions_have_restores_to_rule():
    from graph_extract.ontology import EXTRACTION_INSTRUCTIONS
    flat = " ".join(EXTRACTION_INSTRUCTIONS.split())
    assert ("the DESTINATION is the target entity and the relation is "
            "RestoresTo") in flat
    assert "use the product that performs the restore as the subject" in flat


def test_instructions_extract_the_documented_product_as_an_entity():
    # 2026-10-17 proof smoke: a connector list with no product named lost 28 of its
    # facts because the implied subject was never extracted as an entity.
    text = " ".join(EXTRACTION_INSTRUCTIONS.split()).lower()
    assert "extract the documented product as an entity in every chunk that states facts about it" in text


def test_requirement_docstring_keeps_only_generic_categories():
    # final-review fix wave: the concrete examples leaked like every other example
    from graph_extract.ontology import Requirement
    doc = Requirement.__doc__ or ""
    for leaked in ("Backup Operator", "TCP 443", "e.g."):
        assert leaked not in doc
    for category in ("permission", "minimum version", "open port", "license",
                     "required role/account"):
        assert category in doc


def test_cheap_tier_salience_defers_to_the_naming_rules():
    from graph_extract.ontology import CHEAP_TIER_SALIENCE
    flat = " ".join(CHEAP_TIER_SALIENCE.split())
    assert "Use the naming rules above and short names, never descriptive phrases" in flat
    assert "canonical entity names" not in flat
