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
