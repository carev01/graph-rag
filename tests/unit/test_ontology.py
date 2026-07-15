from pydantic import BaseModel
from graph_extract.ontology import (
    ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP, EXTRACTION_INSTRUCTIONS,
)

def test_entity_types_are_the_eight():
    assert set(ENTITY_TYPES) == {
        "Vendor", "Product", "Tool", "Workload", "Capability", "Platform",
        "Concept", "Requirement"}
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

def test_instructions_capture_limitation_statements():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "not supported" in lowered
    assert "limits" in lowered

def test_instructions_exclude_ui_elements():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "ui elements" in lowered
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

def test_concept_and_platform_cover_storage_classes():
    from graph_extract.ontology import Concept, Platform
    assert "S3 Standard" in (Concept.__doc__ or "")
    assert "storage classes" in (Platform.__doc__ or "")

def test_edge_type_map_references_defined_types():
    for (src, tgt), edges in EDGE_TYPE_MAP.items():
        assert src in ENTITY_TYPES and tgt in ENTITY_TYPES
        assert all(e in EDGE_TYPES for e in edges)

def test_instructions_mention_canonical_names():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "kubernetes" in lowered and "immutability" in lowered

def test_instructions_exclude_noise_entities():
    lowered = EXTRACTION_INSTRUCTIONS.lower()
    assert "do not extract" in lowered
    assert "arn:" in lowered
    assert "snap-" in lowered and "vol-" in lowered
    assert "install-module" in lowered

def test_instructions_exclude_regions_and_timezones():
    assert "Australia East" in EXTRACTION_INSTRUCTIONS
    assert "us-east-1" in EXTRACTION_INSTRUCTIONS
    assert "UTC" in EXTRACTION_INSTRUCTIONS

def test_instructions_state_type_boundaries():
    assert "LRS" in EXTRACTION_INSTRUCTIONS
    assert "backup frequency" in EXTRACTION_INSTRUCTIONS.lower()
    assert "recovery point" in EXTRACTION_INSTRUCTIONS

def test_instructions_have_vendor_scoping_rule():
    assert "DISTINCT" in EXTRACTION_INSTRUCTIONS
    assert "GENERIC" in EXTRACTION_INSTRUCTIONS
    assert "AWS Backup Vault Lock" in EXTRACTION_INSTRUCTIONS
    assert "Azure immutable vault" in EXTRACTION_INSTRUCTIONS
    assert "do not merge" in EXTRACTION_INSTRUCTIONS.lower()

def test_docstrings_state_type_boundaries():
    from graph_extract.ontology import (
        Capability, Concept, Platform, Product, Requirement,
    )
    assert "NOT" in (Platform.__doc__ or "")
    assert "CLI" in (Platform.__doc__ or "")
    assert "region" in (Platform.__doc__ or "").lower()
    assert "NOT" in (Capability.__doc__ or "")
    assert "regulation" in (Concept.__doc__ or "").lower()
    assert "LRS" in (Concept.__doc__ or "")
    assert "CLI" in (Product.__doc__ or "")
    assert "Tool" in (Product.__doc__ or "")
    assert "NOT" in (Requirement.__doc__ or "")
    assert "Concept" in (Requirement.__doc__ or "")
