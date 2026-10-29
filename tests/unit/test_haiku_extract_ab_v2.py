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


class _Resp:
    def __init__(self, status: int, body: dict | None = None) -> None:
        self.status_code, self.body = status, body or {}

    def json(self) -> dict:
        return self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Docext:
    def __init__(self, gone: set[str]) -> None:
        self.gone = gone

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, path: str) -> _Resp:
        aid = path.rsplit("/", 1)[-1]
        if aid in self.gone:
            return _Resp(404)
        return _Resp(200, {"id": aid, "vendor": {"id": "v", "name": "V"},
                           "product": {"id": "p", "name": "P"}, "source_id": "s"})


def test_articles_removed_upstream_are_dropped_and_reported(monkeypatch) -> None:
    import asyncio
    monkeypatch.setattr(ab, "make_docext_client", lambda **k: _Docext({"b"}))
    monkeypatch.setattr(ab, "get_extract_settings", lambda: type("S", (), {
        "docext_base_url": "x", "docext_read_key": "k", "docext_admin_key": "",
        "docext_verify_tls": True})())
    arts = [{"id": i, "source": "Guide"} for i in "abc"]
    kept, gone = asyncio.run(ab.fetch_chains(arts))
    assert [a["id"] for a in kept] == ["a", "c"] and gone == ["b"]
    assert kept[0]["chain"]["vendor"]["id"] == "v"


def test_too_many_articles_removed_upstream_stops_the_run(monkeypatch) -> None:
    import asyncio
    monkeypatch.setattr(ab, "make_docext_client", lambda **k: _Docext(set("abcdefg")))
    monkeypatch.setattr(ab, "get_extract_settings", lambda: type("S", (), {
        "docext_base_url": "x", "docext_read_key": "k", "docext_admin_key": "",
        "docext_verify_tls": True})())
    with pytest.raises(SystemExit, match="removed upstream"):
        asyncio.run(ab.fetch_chains([{"id": i, "source": "G"} for i in "abcdefgh"]))


def _http_error(status: int, url: str):
    import httpx
    req = httpx.Request("GET", url)
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(status, request=req))


def test_ingest_one_retries_docextractor_server_errors(tmp_path, monkeypatch) -> None:
    import asyncio
    from types import SimpleNamespace

    from graph_extract.dedup_guard import DedupIndexStats

    async def graph(driver, aid):
        return {"facts": 3, "entities": 2, "fact_texts": ["f"]}

    monkeypatch.setattr(ab, "_article_graph", graph)
    monkeypatch.setattr(ab, "DOCEXT_RETRY_SECONDS", 0)
    calls = []

    class Ingest:
        async def ingest_article(self, aid):
            calls.append(aid)
            if len(calls) < 3:
                raise _http_error(502, f"https://docextractor.x/api/articles/{aid}")
            return SimpleNamespace(tier="cheap", episodes_added=1, dedup=DedupIndexStats())

    row = asyncio.run(ab.ingest_one("v2", Ingest(), None, {"id": "a1"}, ab.Spend(5), tmp_path))
    assert row["error"] is None and len(calls) == 3 and row["docext_retries"] == 2


def test_ingest_one_does_not_retry_client_errors_or_give_up_late(tmp_path, monkeypatch) -> None:
    import asyncio
    monkeypatch.setattr(ab, "DOCEXT_RETRY_SECONDS", 0)
    calls = []

    class Ingest:
        def __init__(self, status: int) -> None:
            self.status = status

        async def ingest_article(self, aid):
            calls.append(aid)
            raise _http_error(self.status, f"https://docextractor.x/api/articles/{aid}")

    row = asyncio.run(ab.ingest_one("v2", Ingest(404), None, {"id": "a"}, ab.Spend(5), tmp_path))
    assert row["error"] and len(calls) == 1
    calls.clear()
    row = asyncio.run(ab.ingest_one("v2", Ingest(503), None, {"id": "b"}, ab.Spend(5), tmp_path))
    assert row["error"] and len(calls) == ab.DOCEXT_ATTEMPTS


def test_judge_fetch_retries_docextractor_and_reports_a_lasting_failure(monkeypatch) -> None:
    import asyncio
    from types import SimpleNamespace
    monkeypatch.setattr(ab, "DOCEXT_RETRY_SECONDS", 0)
    calls = []

    async def flaky(dx, aid):
        calls.append(aid)
        if len(calls) < 2:
            raise _http_error(502, f"https://docextractor.x/api/articles/{aid}")
        return SimpleNamespace(content_markdown="md")

    monkeypatch.setattr(ab, "fetch_article", flaky)
    assert asyncio.run(ab.fetch_markdown(None, "a1")) == "md" and len(calls) == 2

    async def down(dx, aid):
        raise _http_error(503, f"https://docextractor.x/api/articles/{aid}")

    monkeypatch.setattr(ab, "fetch_article", down)
    assert asyncio.run(ab.fetch_markdown(None, "a2")) is None
