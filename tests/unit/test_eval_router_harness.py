import pytest

import answer_api.eval_router as er
from answer_api.scope import ScopeResolver
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_RESOLVER = ScopeResolver([], [], {})


def _env(mode, article_id):
    cites = [{"marker": 1, "fact_uuid": "f1",
              "sources": [{"article_id": article_id, "url": "u", "title": "t"}]}] if article_id else []
    return {"mode": mode, "answer": "ans", "citations": cites,
            "routing": {"chosen": mode, "via": "llm", "fallback_from": None}}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    async def _fake_router(*a, q, mode_override, scope, settings):
        if mode_override is not None:
            return _env(mode_override, "A1")          # comparative forced mode
        return _env("local" if q == "loc" else "drift", "A1")
    async def _fake_judge(client, model, q, answer, facts):
        return 4
    async def _fake_facts(driver, g, uuids):
        return ["fact"]
    async def _fake_texts_by_uuid(driver, g, uuids):
        return {u: "fact" for u in uuids}
    async def _fake_attribution(client, model, q, answer, labelled_facts):
        return 0
    monkeypatch.setattr(er.router_mod, "answer_router", _fake_router)
    monkeypatch.setattr(er, "_faithfulness_judge", _fake_judge)
    monkeypatch.setattr(er, "_cited_fact_texts", _fake_facts)
    monkeypatch.setattr(er, "_cited_fact_texts_by_uuid", _fake_texts_by_uuid)
    monkeypatch.setattr(er, "judge_attribution", _fake_attribution)


async def test_run_eval_aggregates_and_comparative():
    questions = [
        {"question": "loc", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["A1"]},
        {"question": "broad", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": []},
    ]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    assert summary["n"] == 2
    assert summary["routing_accuracy"] == 1.0              # loc->local, broad->drift both expected
    assert summary["grounding_precision"] == 1.0           # only 'loc' scored (broad is [])
    assert summary["comparative"] is not None              # the broad question ran the comparative pass
    report = er.format_report(summary)
    assert "Routing accuracy" in report and "drift_wins" in report.lower()


async def test_per_question_records_cited_count_and_surviving_ranges(monkeypatch):
    """BACKLOG 0d follow-through: the harness never persisted answer text, so a
    range-shorthand answer (judged against its two endpoints) was
    indistinguishable from a well-cited one in the report. Record both counts."""
    async def _range_router(*a, q, mode_override, scope, settings):
        env = _env("global", "A1")
        env["answer"] = "Both encrypt at rest [1]-[26]."
        env["citations"] = [dict(env["citations"][0], marker=m) for m in (1, 26)]
        return env
    monkeypatch.setattr(er.router_mod, "answer_router", _range_router)
    questions = [{"question": "enc", "intent": "global", "expected_modes": ["global"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    rec = summary["per_question"][0]
    assert rec["cited"] == 2
    assert rec["ranges"] == 1
    row = [ln for ln in er.format_report(summary).splitlines() if "| enc |" in ln][0]
    assert "| 2 | 1 |" in row                      # cited | ranges columns


async def test_per_question_records_bag_share(monkeypatch):
    """Threshold the bag-pasting: `mps` shows a bag exists, `bag` shows how much
    of the answer's citation mass is bag. Here 8 of 9 markers."""
    bag = " ".join(f"[{i}]" for i in range(1, 9))
    async def _bag_router(*a, q, mode_override, scope, settings):
        env = _env("global", "A1")
        env["answer"] = f"Both vendors encrypt {bag}. AES [9]."
        env["citations"] = [dict(env["citations"][0], marker=m) for m in range(1, 10)]
        return env
    monkeypatch.setattr(er.router_mod, "answer_router", _bag_router)
    questions = [{"question": "enc", "intent": "global", "expected_modes": ["global"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    assert summary["per_question"][0]["bag_share"] == round(8 / 9, 2)
    report = er.format_report(summary)
    assert "| bag |" in report
    assert "| 0.89 |" in [ln for ln in report.splitlines() if "| enc |" in ln][0]


async def test_a_refusal_records_no_bag_share(monkeypatch):
    """0.0 is the best score on this scale; an answer that cited nothing must
    not earn it."""
    async def _refusal_router(*a, q, mode_override, scope, settings):
        env = _env("global", None)
        env["answer"] = "I don't have enough thematic coverage to answer that."
        return env
    monkeypatch.setattr(er.router_mod, "answer_router", _refusal_router)
    questions = [{"question": "enc", "intent": "global", "expected_modes": ["global"],
                  "expected_article_ids": []}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    assert summary["per_question"][0]["bag_share"] is None
    assert summary["bag_share_by_mode"] == {}
    assert "| - |" in [ln for ln in er.format_report(summary).splitlines()
                       if "| enc |" in ln][0]


async def test_the_run_persists_answer_text_and_cited_facts(monkeypatch, tmp_path):
    """Every metric this harness has gained -- cited, ranges, mps, bag -- cost a
    paid full eval run to observe, because the harness discarded the answers it
    scored. Persist the raw material so the next metric is a re-read, not a
    re-run."""
    questions = [{"question": "loc", "intent": "local", "expected_modes": ["local"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    raw = summary["raw"]
    assert raw[0]["question"] == "loc"
    assert raw[0]["answer"] == "ans"
    assert raw[0]["cited_facts"] == ["fact"]
    assert raw[0]["chosen"] == "local"
    # It is raw material, not a scored column: it must stay out of the report.
    assert "cited_facts" not in er.format_report(summary)


async def test_per_question_records_markers_per_sentence(monkeypatch):
    """BACKLOG 0b: 19 markers pasted on one sentence must be distinguishable
    from one marker per claim. Record the per-sentence distribution's mean and
    max and print them as a `mps` column."""
    async def _bag_router(*a, q, mode_override, scope, settings):
        env = _env("global", "A1")
        env["answer"] = "KMS keys [1] [2] [3]. AES [4]."
        env["citations"] = [dict(env["citations"][0], marker=m) for m in (1, 2, 3, 4)]
        return env
    monkeypatch.setattr(er.router_mod, "answer_router", _bag_router)
    questions = [{"question": "enc", "intent": "global", "expected_modes": ["global"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    rec = summary["per_question"][0]
    assert rec["mps_mean"] == 2.0 and rec["mps_max"] == 3
    row = [ln for ln in er.format_report(summary).splitlines() if "| enc |" in ln][0]
    assert "| 4 | 0 | 2.0/3 |" in row              # cited | ranges | mps columns
    assert "| mps |" in er.format_report(summary)
