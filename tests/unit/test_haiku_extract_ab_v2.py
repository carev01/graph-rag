"""`haiku_extract_ab` v2 arm helpers: routed arms price each tier at their own rate,
distillation train articles are excluded, and vLLM's success counter is summed."""
from __future__ import annotations

import pytest

from scripts import haiku_extract_ab as ab


def test_v2_cheap_tier_is_free_and_dense_tables_pay_strong() -> None:
    assert ab.rate_key("v2", "cheap") == "local"
    assert ab.rate_key("v2", "strong") == "strong"
    assert ab.article_cost("local", 10_000, 0, 5_000) == 0.0
    assert ab.rate_key("routed", "cheap") == "cheap"
    assert ab.rate_key("haiku", "haiku") == "haiku"


def test_train_articles_only_are_excluded() -> None:
    sample = {"articles": [{"article_id": "a", "split": "train"},
                           {"article_id": "b", "split": "heldout"},
                           {"article_id": "c", "split": "train"}]}
    assert ab.train_article_ids(sample) == {"a", "c"}


def test_vllm_success_total_sums_every_label_set() -> None:
    text = ("# HELP vllm:request_success_total Count\n"
            "# TYPE vllm:request_success_total counter\n"
            'vllm:request_success_total{finished_reason="stop",model_name="m"} 12.0\n'
            'vllm:request_success_total{finished_reason="length",model_name="m"} 3.0\n'
            'vllm:request_success_created{finished_reason="stop",model_name="m"} 1.7e9\n'
            'vllm:num_requests_running{model_name="m"} 2.0\n')
    assert ab.vllm_success_total(text) == 15.0


def test_v2_arm_needs_its_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("V2_BASE_URL", raising=False)
    monkeypatch.delenv("V2_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="V2_BASE_URL"):
        ab.arm_settings("v2", None, None)  # type: ignore[arg-type]


def test_v2_arm_points_only_the_cheap_tier_at_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V2_BASE_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setenv("V2_API_KEY", "k\n")
    s = ab.arm_settings("v2", None, None)  # type: ignore[arg-type]
    assert s["cheap_llm_base_url"] == "http://127.0.0.1:18080/v1"
    assert s["cheap_llm_api_key"] == "k"
    assert s["cheap_llm_model"] == "qwen35-graphrag"
    assert s["cheap_llm_cache_layout"] is True
    assert "extraction_routing" not in s and "llm_base_url" not in s


def test_chain_comes_from_docext_vendor_product_and_the_sample_source() -> None:
    art = {"id": "a1", "source_id": "s1", "vendor": {"id": "v1", "name": "Veeam"},
           "product": {"id": "p1", "name": "Veeam Backup & Replication"}}
    assert ab.chain_from_docext(art, "User Guide") == {
        "article": "a1", "vendor": {"id": "v1", "name": "Veeam"},
        "product": {"id": "p1", "name": "Veeam Backup & Replication"},
        "source": {"id": "s1", "name": "User Guide"}}


@pytest.mark.parametrize("missing", ["vendor", "product", "source_id"])
def test_chain_refuses_an_incomplete_article(missing: str) -> None:
    art = {"id": "a1", "source_id": "s1", "vendor": {"id": "v1", "name": "Veeam"},
           "product": {"id": "p1", "name": "VBR"}}
    art[missing] = None
    with pytest.raises(SystemExit, match="a1"):
        ab.chain_from_docext(art, "User Guide")


def test_reused_rows_must_match_the_document_context_setting() -> None:
    old = [{"arm": "routed", "article_id": "a"}]            # pre-chain rows lack the key
    new = [{"arm": "routed", "article_id": "a", "document_context": True}]
    ab.check_reuse_context(new, chain=True)
    ab.check_reuse_context(old, chain=False)
    with pytest.raises(SystemExit, match="document context"):
        ab.check_reuse_context(old, chain=True)
    with pytest.raises(SystemExit, match="document context"):
        ab.check_reuse_context(new, chain=False)


def test_capture_writes_each_arm_to_its_own_file(tmp_path) -> None:
    assert ab.capture_settings("v2", tmp_path, True) == {
        "llm_capture_path": str(tmp_path / "capture-v2.jsonl")}
    assert ab.capture_settings("v2", tmp_path, False) == {}


@pytest.mark.parametrize(("value", "arms"), [("none", []), ("", []), ("routed,v2", ["routed", "v2"]),
                                             ("v2,", ["v2"])])
def test_arm_list_reads_none_as_no_arm(value: str, arms: list[str]) -> None:
    assert ab.arm_list(value) == arms


def test_adapter_arms_name_the_lora_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V2_BASE_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setenv("V2_API_KEY", "k\n")
    a = ab.arm_settings("v2a", None, None)  # type: ignore[arg-type]
    av = ab.arm_settings("v2av", None, None)  # type: ignore[arg-type]
    assert a["cheap_llm_model"] == av["cheap_llm_model"] == "extractor"
    assert "edge_verifier_enabled" not in a
    assert av["edge_verifier_enabled"] is True and av["edge_verifier_model"] == "verifier"
    assert av["edge_verifier_base_url"] == "http://127.0.0.1:18080/v1"
    assert av["edge_verifier_api_key"] == "k"
    assert {k: v for k, v in av.items() if not k.startswith("edge_verifier")} == a
    assert ab.rate_key("v2a", "cheap") == ab.rate_key("v2av", "cheap") == "local"


def _row(tier: str, calls: int, vcalls: int = 0, dropped: int = 0, failed: int = 0,
         edges: int | None = None, empty: int = 0, retried: int = 0) -> dict:
    return {"tier": tier, "llm_calls": calls, "dedup": {
        "verifier_calls": vcalls, "verifier_dropped": dropped, "verifier_failed_open": failed,
        "verifier_empty": empty, "edge_replies": vcalls + empty if edges is None else edges,
        "unusable_retried": retried}}


def test_vllm_gate_accounts_for_verifier_requests() -> None:
    rows = [_row("cheap", 10, 3, 1), _row("strong", 7)]
    assert ab.vllm_gate_problems("v2av", rows, 13.0) == []
    assert ab.vllm_gate_problems("v2av", rows, 10.0)                       # verifier not on vLLM
    assert ab.vllm_gate_problems("v2av", [_row("cheap", 10)], 10.0)        # never fired
    assert ab.vllm_gate_problems("v2av", [_row("cheap", 10, 3, 0)], 13.0)  # fired, dropped nothing
    assert ab.vllm_gate_problems("v2av", [_row("cheap", 10, 3, 1, 1)], 12.0)  # failed open
    assert ab.vllm_gate_problems("v2a", [_row("cheap", 10)], 10.0) == []
    assert ab.vllm_gate_problems("v2a", [_row("cheap", 10, 2, 1)], 12.0)   # verifier on a plain arm
    assert ab.vllm_gate_problems("v2", [_row("strong", 5)], 0.0)           # no cheap article
    assert ab.vllm_gate_problems("v2", [_row("cheap", 4), {"tier": "cheap", "llm_calls": 0,
                                                            "dedup": None}], 4.0) == []


def test_vllm_gate_checks_every_edge_reply_was_verified() -> None:
    assert ab.vllm_gate_problems("v2av", [_row("cheap", 10, 3, 1, edges=4)], 13.0)  # one skipped
    assert ab.vllm_gate_problems("v2av", [_row("cheap", 10, 3, 1, edges=4, empty=1)], 13.0) == []


def test_vllm_gate_discounts_truncation_retries_that_went_to_the_strong_tier() -> None:
    assert ab.vllm_gate_problems("v2av", [_row("cheap", 11, 3, 1, retried=1)], 13.0) == []
    assert ab.vllm_gate_problems("v2a", [_row("cheap", 11, retried=1)], 10.0) == []
