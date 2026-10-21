"""`dpo_pairs`: only the student's terminal edge replies are judged, labels map back to
edges through the per-pass shuffle, a fact is dropped only when both passes call it
unsupported, and `chosen` differs from `rejected` by exactly those edges."""
from __future__ import annotations

import json

from scripts.distill import dpo_pairs as dp

CTX = "DOCUMENT CONTEXT: This text documents Veeam Backup & Replication."


def _user(cur: str = "[Restore a VM › Steps]\n\nClick Restore.") -> str:
    return (f"\n<PREVIOUS_MESSAGES>\n[]\n</PREVIOUS_MESSAGES>\n<CURRENT_MESSAGE>\n{cur}\n"
            "</CURRENT_MESSAGE>\n")


def _edge(i: int) -> dict:
    return {"source_entity_name": f"S{i}", "target_entity_name": f"T{i}",
            "relation_type": "R", "fact": f"fact {i}", "valid_at": None,
            "invalid_at": None, "episode_indices": [0]}


def _capture(**over) -> str:
    rec = {"call_id": "c1", "tier": "cheap", "prompt_name": dp.PROMPT,
           "model": dp.STUDENT_MODEL, "finish_reason": "stop",
           "messages": [{"role": "system", "content": "sys"},
                        {"role": "user", "content": _user()}],
           "response_format": {"type": "json_schema"},
           "completion": json.dumps({"edges": [_edge(0), _edge(1), _edge(2)]})}
    rec.update(over)
    return json.dumps(rec)


def test_edge_records_keep_only_judgeable_student_replies(monkeypatch) -> None:
    monkeypatch.setattr(dp, "document_context", lambda msgs: CTX)
    lines = [_capture(),
             _capture(call_id="x1", model="solar-pro4"),
             _capture(call_id="x2", finish_reason="length"),
             _capture(call_id="x3", completion="{not json"),
             _capture(call_id="x4", completion=json.dumps({"edges": []})),
             _capture(call_id="x5", prompt_name="extract_nodes.extract_text"),
             _capture(call_id="x6", tier="strong")]
    recs, skipped = dp.edge_records(lines)
    assert [r["id"] for r in recs] == ["c1"]
    assert skipped == {"not_student": 1, "nonterminal": 1, "unparseable": 1, "no_edges": 1}
    assert recs[0]["group"] == f"{CTX}|Restore a VM"


def test_labels_follow_the_shuffle_back_to_their_edges(monkeypatch) -> None:
    monkeypatch.setattr(dp, "document_context", lambda msgs: CTX)
    rec = dp.edge_records([_capture()])[0][0]
    prompt, order = dp.judge_prompt(rec, 1)
    assert sorted(order) == [0, 1, 2] and CTX in prompt and "Click Restore." in prompt
    shown = [line.split(": ", 1)[1] for line in prompt.split("FACTS:\n")[1].splitlines()]
    assert shown == [f"fact {i}" for i in order]
    obj = {"labels": [{"n": n, "verdict": "unsupported" if i == 1 else "stated",
                       "category": "over_inference" if i == 1 else None, "why": "w"}
                      for n, i in enumerate(order, 1)]}
    labels = dp.labels_by_edge(obj, order)
    assert [x["verdict"] for x in labels] == ["stated", "unsupported", "stated"]


def test_unusable_judge_replies_are_rejected() -> None:
    assert dp.labels_by_edge({"labels": []}, [0]) is None
    assert dp.labels_by_edge({"labels": [{"verdict": "unsupported"}]}, [0]) is None
    assert dp.labels_by_edge({"labels": [{"verdict": "maybe"}]}, [0]) is None


def test_pair_removes_only_edges_both_passes_reject(monkeypatch) -> None:
    monkeypatch.setattr(dp, "document_context", lambda msgs: CTX)
    rec = dp.edge_records([_capture()])[0][0]
    st = {"verdict": "stated", "category": None, "why": None}
    un = {"verdict": "unsupported", "category": "wrong_detail", "why": "w"}
    a, b = [st, un, un], [st, un, st]
    bad = dp.unsupported_both(a, b)
    assert bad == [1]
    pair = dp.build_pair(rec, bad, a, b)
    assert json.loads(pair["rejected"]) == rec["payload"]
    assert json.loads(pair["chosen"])["edges"] == [_edge(0), _edge(2)]
    assert pair["messages"] == rec["messages"]
    assert pair["meta"]["removed"][0]["fact"] == "S1 -> T1: fact 1"
    assert dp.build_pair(rec, [], a, b) is None
    assert dp.build_pair(rec, [0, 1, 2], a, b) is None


def test_split_is_stable_per_group() -> None:
    assert dp.split_of("g", 0.0) == "train" and dp.split_of("g", 1.0) == "val"
    assert len({dp.split_of("same", 0.5) for _ in range(5)}) == 1


def test_document_context_is_found_in_any_message() -> None:
    from graph_extract.graphiti_client import document_context
    block = document_context("Veeam", "Veeam Backup & Replication", "User Guide")
    msgs = [{"role": "system", "content": "static"},
            {"role": "user", "content": f"rules{block}\n<CURRENT_MESSAGE>x</CURRENT_MESSAGE>"}]
    assert dp.document_context(msgs) == block.strip()
    assert dp.document_context(msgs[:1]) == ""
