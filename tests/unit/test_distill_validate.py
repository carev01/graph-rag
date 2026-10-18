"""`validate_answer`: schema, index-range (`validate_capture`) and grounding
checks on a teacher answer, in that order (task-4 brief)."""
from __future__ import annotations

from typing import Any

from graphiti_core.prompts.dedupe_nodes import nodes as prompt_dedupe_nodes
from graphiti_core.prompts.extract_nodes import extract_text as prompt_extract_text
from graphiti_core.utils.maintenance.node_operations import _build_entity_types_context

from graph_extract.grounding import PRODUCT_ENTITY_TYPE_ID
from graph_extract.ontology import ENTITY_TYPES
from scripts.distill import validate

EXTRACT_TEXT = "extract_nodes.extract_text"
DEDUPE_NODES = "dedupe_nodes.nodes"


def _extract_text_messages(text: str) -> list[dict[str, str]]:
    """The real graphiti `extract_text` prompt over the real ontology types, so
    `PRODUCT_ENTITY_TYPE_ID` lands in range exactly as production renders it."""
    context = {
        "entity_types": _build_entity_types_context(ENTITY_TYPES),
        "episode_content": text,
        "custom_extraction_instructions": "",
    }
    return [{"role": m.role, "content": m.content} for m in prompt_extract_text(context)]


def _dedupe_nodes_messages(extracted: list[dict[str, Any]],
                           existing: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = {"previous_episodes": [], "episode_content": "irrelevant",
              "extracted_nodes": extracted, "existing_nodes": existing}
    return [{"role": m.role, "content": m.content} for m in prompt_dedupe_nodes(context)]


def test_valid_extract_text_answer_passes():
    messages = _extract_text_messages("Veeam Backup protects workloads.")
    answer = {"extracted_entities": [
        {"name": "Veeam Backup", "entity_type_id": 0, "episode_indices": [0]}]}
    ctx = validate.build_grounding_context(EXTRACT_TEXT, messages,
                                           product="Veeam Backup", vendor="Veeam")
    assert validate.validate_answer(EXTRACT_TEXT, messages, answer, grounding_ctx=ctx) == []


def test_ungrounded_entity_name_is_rejected_with_the_name():
    messages = _extract_text_messages("Veeam Backup protects workloads.")
    answer = {"extracted_entities": [
        {"name": "Veeam Backup", "entity_type_id": 0, "episode_indices": [0]},
        {"name": "Microsoft Azure", "entity_type_id": 0, "episode_indices": [0]}]}
    ctx = validate.build_grounding_context(EXTRACT_TEXT, messages,
                                           product="Veeam Backup", vendor="Veeam")
    reasons = validate.validate_answer(EXTRACT_TEXT, messages, answer, grounding_ctx=ctx)
    assert reasons and any("Microsoft Azure" in r for r in reasons)


def test_missing_documented_product_is_rejected_as_injection():
    messages = _extract_text_messages("Veeam Backup protects workloads with immutability.")
    answer = {"extracted_entities": [
        {"name": "immutability", "entity_type_id": 0, "episode_indices": [0]}]}
    ctx = validate.build_grounding_context(EXTRACT_TEXT, messages,
                                           product="Veeam Backup", vendor="Veeam")
    reasons = validate.validate_answer(EXTRACT_TEXT, messages, answer, grounding_ctx=ctx)
    assert reasons and any("Veeam Backup" in r for r in reasons)


def test_vbr_alias_of_the_documented_product_is_rejected_as_a_rename():
    messages = _extract_text_messages("VBR protects workloads.")
    answer = {"extracted_entities": [
        {"name": "VBR", "entity_type_id": PRODUCT_ENTITY_TYPE_ID, "episode_indices": [0]}]}
    ctx = validate.build_grounding_context(
        EXTRACT_TEXT, messages, product="Veeam Backup & Replication", vendor="Veeam")
    reasons = validate.validate_answer(EXTRACT_TEXT, messages, answer, grounding_ctx=ctx)
    assert any("VBR" in r for r in reasons)
    assert any("Veeam Backup & Replication" in r for r in reasons)


def test_dedupe_nodes_out_of_range_candidate_id_is_rejected():
    extracted = [{"id": 0, "name": "Foo", "entity_types": ["Entity"]}]
    existing = [{"candidate_id": 0, "name": "Bar", "entity_types": ["Entity"]}]
    messages = _dedupe_nodes_messages(extracted, existing)
    answer = {"entity_resolutions": [
        {"id": 0, "name": "Foo", "duplicate_candidate_id": 5}]}
    reasons = validate.validate_answer(DEDUPE_NODES, messages, answer, grounding_ctx=None)
    assert "candidate_id_out_of_range" in reasons


def test_schema_invalid_answer_is_rejected():
    messages = _dedupe_nodes_messages(
        [{"id": 0, "name": "Foo", "entity_types": ["Entity"]}], [])
    answer: dict[str, Any] = {"entity_resolutions": [{"id": 0}]}   # missing required fields
    reasons = validate.validate_answer(DEDUPE_NODES, messages, answer, grounding_ctx=None)
    assert reasons and reasons[0].startswith("schema_invalid")


def test_build_grounding_context_is_none_for_non_extract_text_prompts():
    messages = _dedupe_nodes_messages([], [])
    assert validate.build_grounding_context(DEDUPE_NODES, messages,
                                            product="x", vendor="y") is None
