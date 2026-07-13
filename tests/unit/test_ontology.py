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
