from pydantic import BaseModel
from graph_extract.ontology import (
    ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP, EXTRACTION_INSTRUCTIONS,
)

def test_entity_types_are_the_seven():
    assert set(ENTITY_TYPES) == {
        "Vendor", "Product", "Workload", "Capability", "Platform",
        "Concept", "Requirement"}
    assert all(issubclass(t, BaseModel) for t in ENTITY_TYPES.values())

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

def test_instructions_have_vendor_scoping_rule():
    assert "DISTINCT" in EXTRACTION_INSTRUCTIONS
    assert "GENERIC" in EXTRACTION_INSTRUCTIONS
    assert "AWS Backup Vault Lock" in EXTRACTION_INSTRUCTIONS
    assert "Azure immutable vault" in EXTRACTION_INSTRUCTIONS
    assert "do not merge" in EXTRACTION_INSTRUCTIONS.lower()

def test_docstrings_state_type_boundaries():
    from graph_extract.ontology import Capability, Concept, Platform, Product
    assert "NOT" in (Platform.__doc__ or "")
    assert "CLI" in (Platform.__doc__ or "")
    assert "NOT" in (Capability.__doc__ or "")
    assert "regulation" in (Concept.__doc__ or "").lower()
    assert "CLI" in (Product.__doc__ or "")
