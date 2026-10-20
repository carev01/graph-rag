"""`eval_finetune` request/reply plumbing: `--guided` sends a real json_schema
payload (a dataset record names its schema, it does not carry it), and an error
reply surfaces its cause instead of (None, None)."""
from __future__ import annotations

from scripts import eval_finetune as ef
from scripts.build_extraction_dataset import response_schema


def test_named_schema_is_expanded_to_the_served_payload() -> None:
    rec = {"prompt_name": "extract_edges.edge", "response_schema": "ExtractedEdges"}
    fmt = ef.response_format_for(rec)
    assert fmt == response_schema("extract_edges.edge")
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "ExtractedEdges"


def test_full_payload_is_sent_as_is() -> None:
    payload = {"type": "json_schema", "json_schema": {"name": "X", "schema": {}}}
    rec = {"prompt_name": "extract_edges.edge", "response_schema": payload}
    assert ef.response_format_for(rec) is payload


def test_ok_reply_returns_content_and_finish() -> None:
    out = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
    assert ef.reply_or_error(200, out) == ("{}", "stop")


def test_error_body_surfaces_status_and_message() -> None:
    txt, why = ef.reply_or_error(400, {"error": {"message": "bad response_format"}})
    assert txt is None
    assert why is not None and why.startswith("HTTP 400")
    assert "bad response_format" in why


def test_200_without_choices_is_an_error_not_silence() -> None:
    txt, why = ef.reply_or_error(200, {"object": "error"})
    assert txt is None and why is not None and "HTTP 200" in why
