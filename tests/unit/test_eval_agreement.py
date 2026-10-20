"""Agreement of a student's answers with the teacher's reference, per prompt type."""
from __future__ import annotations

import json

from scripts import eval_agreement as ea


def _row(prompt: str, output: dict | None, reference: dict) -> dict:
    return {"prompt": prompt, "output": None if output is None else json.dumps(output),
            "reference": json.dumps(reference)}


def test_extract_text_name_f1_is_case_insensitive_and_scores_types():
    ref = {"extracted_entities": [{"name": "Azure Backup", "entity_type_id": 2},
                                  {"name": "vault", "entity_type_id": 4}]}
    out = {"extracted_entities": [{"name": "azure backup", "entity_type_id": 2},
                                  {"name": "policy", "entity_type_id": 4}]}
    m = ea.score([_row("extract_nodes.extract_text", out, ref)])["extract_nodes.extract_text"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["type_agree"] == 1.0


def test_extract_edges_pairs_and_relation_agreement():
    ref = {"edges": [{"source_entity_name": "A", "target_entity_name": "B", "relation_type": "Supports"},
                     {"source_entity_name": "A", "target_entity_name": "C", "relation_type": "Requires"}]}
    out = {"edges": [{"source_entity_name": "a", "target_entity_name": "b", "relation_type": "Provides"}]}
    m = ea.score([_row("extract_edges.edge", out, ref)])["extract_edges.edge"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)
    assert m["type_agree"] == 0.0
    assert m["count_ratio"] == 0.5


def test_dedupe_nodes_per_entity_accuracy_and_merge_errors():
    ref = {"entity_resolutions": [{"id": 0, "duplicate_candidate_id": -1},
                                  {"id": 1, "duplicate_candidate_id": 3}]}
    out = {"entity_resolutions": [{"id": 0, "duplicate_candidate_id": 2},
                                  {"id": 1, "duplicate_candidate_id": 3}]}
    m = ea.score([_row("dedupe_nodes.nodes", out, ref)])["dedupe_nodes.nodes"]
    assert m["accuracy"] == 0.5
    assert m["false_merge"] == 1 and m["missed_merge"] == 0


def test_resolve_edge_exact_set_and_flag():
    ref = {"duplicate_facts": [0], "contradicted_facts": []}
    rows = [_row("dedupe_edges.resolve_edge", {"duplicate_facts": [0], "contradicted_facts": []}, ref),
            _row("dedupe_edges.resolve_edge", {"duplicate_facts": [], "contradicted_facts": []}, ref)]
    m = ea.score(rows)["dedupe_edges.resolve_edge"]
    assert m["exact"] == 0.5
    assert m["dup_flag_agree"] == 0.5


def test_summaries_name_match_and_word_overlap():
    ref = {"summaries": [{"name": "X", "summary": "x backs up vms daily"}]}
    out = {"summaries": [{"name": "X", "summary": "x backs up vms weekly"}]}
    m = ea.score([_row("extract_nodes.extract_summaries_batch", out, ref)])[
        "extract_nodes.extract_summaries_batch"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)
    assert abs(m["word_jaccard"] - 4 / 6) < 1e-9


def test_unusable_output_counts_as_all_missed():
    ref = {"extracted_entities": [{"name": "A", "entity_type_id": 1}]}
    m = ea.score([_row("extract_nodes.extract_text", None, ref)])["extract_nodes.extract_text"]
    assert m["unusable"] == 1 and m["fn"] == 1
