import pytest

import answer_api.eval_router as er
from answer_api.scope import ScopeResolver
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_RESOLVER = ScopeResolver([], [], {})
_REAL_ROUTER = er.router_mod.answer_router


def _env(mode, article_id):
    cites = [{"marker": 1, "fact_uuid": "f1",
              "sources": [{"article_id": article_id, "url": "u", "title": "t"}]}] if article_id else []
    return {"mode": mode, "answer": "ans", "citations": cites,
            "routing": {"chosen": mode, "via": "llm", "fallback_from": None}}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    async def _fake_router(*a, q, mode_override, scope, settings, **kw):
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
    async def _range_router(*a, q, mode_override, scope, settings, **kw):
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
    async def _bag_router(*a, q, mode_override, scope, settings, **kw):
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
    async def _refusal_router(*a, q, mode_override, scope, settings, **kw):
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
    async def _bag_router(*a, q, mode_override, scope, settings, **kw):
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


async def test_a_forced_mode_cross_vendor_question_does_not_enumerate(monkeypatch):
    """The comparative pass forces each mode in turn; an enumeration-shaped
    question must run that mode, not the same enumeration three times."""
    enumerated = []
    dispatched = []

    async def _enum(*a, **kw):
        enumerated.append(kw["q"])
        return {"answer": "E", "citations": [], "applies_to": []}

    async def _local(*a, facets=None, **kw):
        dispatched.append(("local", facets))
        return {"answer": "L", "retrieved": 1, "applies_to": [],
                "citations": _env("local", "A1")["citations"]}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(er.router_mod, "answer_router", _REAL_ROUTER)
    monkeypatch.setattr(er.router_mod.enum_mod, "enumerate_vendors", _enum)
    monkeypatch.setattr(er.router_mod.synth_mod, "answer_local", _local)
    monkeypatch.setattr(er.router_mod.freshness_mod, "freshness", _fresh)
    resolver = ScopeResolver(["Veeam", "AWS"], [], {}, content_vendors=("Veeam", "AWS"))
    q = {"question": "Which vendors support immutable backups?", "intent": "global",
         "expected_modes": ["global"], "expected_article_ids": ["A1"]}
    assert er.classify_shape(q["question"], resolver,
                             resolver.resolve(q["question"])).kind == "enumeration"
    env, *_ = await er._score_one((None,) * 11, q, "local", _S, resolver)
    assert enumerated == []
    assert dispatched == [("local", None)]
    assert env["mode"] == "local" and env["shape"] == "single"


async def test_facet_coverage_is_scored_from_the_requested_shape_after_relaxation(
        monkeypatch):
    """The router relaxed a comparison to single unscoped and the answer cites
    one vendor: facet coverage must be False (not None, which drops the
    question from the metric)."""
    async def _relaxed_router(*a, q, mode_override, scope, settings, **kw):
        env = _env("local", "A1")
        env["citations"][0]["sources"][0]["vendor"] = "Veeam"
        env.update(shape="single", shape_requested="comparison", facets=[])
        return env

    monkeypatch.setattr(er.router_mod, "answer_router", _relaxed_router)
    resolver = ScopeResolver(["Veeam", "AWS"], [], {}, content_vendors=("Veeam", "AWS"))
    questions = [{"question": "How do Veeam and AWS lock backups?", "intent": "local",
                  "expected_modes": ["local"], "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, resolver)
    assert summary["per_question"][0]["facet_coverage"] is False
    assert summary["facet_coverage"] == 0.0 and summary["facet_coverage_n"] == 1


async def test_report_and_progress_show_facet_and_enumeration_columns(monkeypatch, capsys):
    questions = [{"question": "loc", "intent": "local", "expected_modes": ["local"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    line = capsys.readouterr().out
    assert "facet=- enum=- items=-" in line
    summary["per_question"][0].update(facet_coverage=True, enumeration_complete=False)
    report = er.format_report(summary).splitlines()
    header = next(ln for ln in report if ln.startswith("| intent |"))
    assert ("| misattributed | facet_coverage | enumeration_complete | item_coverage "
            "| flagged/dropped | question |" in header)
    assert header.count("|") == next(ln for ln in report if ln.startswith("|---")).count("|")
    row = next(ln for ln in report if ln.endswith("| loc |"))
    assert "| True | False | - | - | loc |" in row


async def test_score_one_retrieves_a_named_subject_question_with_its_facet_scope(
        monkeypatch):
    seen = {}

    async def _router(*a, q, mode_override, scope, settings, shape=None, **kw):
        seen.update(scope=scope, shape=shape)
        return _env("local", "A1")

    monkeypatch.setattr(er.router_mod, "answer_router", _router)
    resolver = ScopeResolver(["Veeam", "AWS"], [], {}, content_vendors=("Veeam", "AWS"))
    q = {"question": "Which vendors does Veeam integrate with?", "intent": "local",
         "expected_modes": ["local"], "expected_article_ids": ["A1"]}
    await er._score_one((None,) * 11, q, None, _S, resolver)
    assert seen["shape"].kind == "single"
    assert seen["scope"].vendors == ("Veeam",) and seen["scope"].source == "detected"


async def test_classifier_hit_scores_the_classified_mode_when_present(monkeypatch):
    """A list answer is labelled local, but the classifier picked drift: the
    classifier metric must score the classifier's pick (routing.classified)."""
    async def _list_router(*a, q, mode_override, scope, settings, **kw):
        env = _env("local", "A1")
        env["routing"]["classified"] = "drift"
        return env
    monkeypatch.setattr(er.router_mod, "answer_router", _list_router)
    questions = [{"question": "lst", "intent": "drift", "expected_modes": ["drift"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    rec = summary["per_question"][0]
    assert rec["routing_hit"] is False
    assert rec["classifier_hit"] is True


async def test_item_grade_counts_are_recorded_summed_and_reported(monkeypatch):
    def _graded(*grades):
        return [{"item": f"i{n}", "grade": g, "missing_terms": [], "markers": [1],
                 "fact_uuids": ["f1"]} for n, g in enumerate(grades)]

    async def _router(*a, q, mode_override, scope, settings, **kw):
        env = _env("local", "A1")
        if q == "list1":
            env["item_grades"] = _graded("confirmed", "needs_validation", "dropped", "dropped")
            env["items_ungraded"] = 3
        elif q == "list2":
            env["item_grades"] = _graded("needs_validation")
        return env

    monkeypatch.setattr(er.router_mod, "answer_router", _router)
    questions = [{"question": q, "intent": "local", "expected_modes": ["local"],
                  "expected_article_ids": ["A1"]} for q in ("list1", "list2", "loc")]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    recs = {r["question"]: r for r in summary["per_question"]}
    assert (recs["list1"]["items_flagged"], recs["list1"]["items_dropped"]) == (1, 2)
    assert (recs["list2"]["items_flagged"], recs["list2"]["items_dropped"]) == (1, 0)
    assert recs["loc"]["items_flagged"] is None and recs["loc"]["items_dropped"] is None
    assert (recs["list1"]["items_graded"], recs["loc"]["items_graded"]) == (4, None)
    assert (recs["list1"]["items_ungraded"], recs["loc"]["items_ungraded"]) == (3, None)
    assert (summary["items_flagged"], summary["items_dropped"],
            summary["items_graded_n"]) == (2, 2, 2)
    report = er.format_report(summary).splitlines()
    header = next(ln for ln in report if ln.startswith("| intent |"))
    assert "| item_coverage | flagged/dropped | question |" in header
    assert header.count("|") == next(ln for ln in report if ln.startswith("|---")).count("|")
    assert next(ln for ln in report if ln.endswith("| list1 |")).endswith("| 1/2 | list1 |")
    assert next(ln for ln in report if ln.endswith("| loc |")).endswith("| - | loc |")


async def test_a_list_answer_with_no_item_lines_renders_no_grade_cell(monkeypatch):
    async def _router(*a, q, mode_override, scope, settings, **kw):
        return {**_env("local", "A1"), "item_grades": []}

    monkeypatch.setattr(er.router_mod, "answer_router", _router)
    questions = [{"question": "loc", "intent": "local", "expected_modes": ["local"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    rec = summary["per_question"][0]
    assert (rec["items_graded"], rec["items_flagged"], rec["items_dropped"]) == (0, 0, 0)
    row = next(ln for ln in er.format_report(summary).splitlines() if ln.endswith("| loc |"))
    assert row.endswith("| - | loc |")


# ---------------------------------------------------------------------------
# Task 18 (spec §14.3, after paid run 13): the faithfulness judge must see
# labelled fact lines and must be told that a disclosed gap ("needs
# validation" / "†") is not unsupported.
# ---------------------------------------------------------------------------

_DISCLOSED_GAP_SENTENCE = (
    "Each cited fact is labelled (Vendor · Product) with the documentation that "
    "states it. A line carrying a '(needs validation: ...)' note or a '†' mark "
    "discloses that its cited fact does not state the noted qualifier: judge such "
    "a line only on whether its cited facts support the item apart from that "
    "disclosed qualifier. "
)


async def test_judge_prompt_carries_the_disclosed_gap_sentence_verbatim_before_the_reply_rule():
    assert _DISCLOSED_GAP_SENTENCE in er._JUDGE_PROMPT
    assert (er._JUDGE_PROMPT.index(_DISCLOSED_GAP_SENTENCE)
            < er._JUDGE_PROMPT.index("Reply with ONLY the integer"))


# ---------------------------------------------------------------------------
# Chunked faithfulness judge for long answers (spec §13.4, after run 12)
# ---------------------------------------------------------------------------

async def test_split_blocks_splits_at_headings_and_blank_line_paragraphs():
    text = ("# Heading\nLine A.\n\nPara two line 1.\nPara two line 2.\n\n"
            "## Sub\nPara three.")
    assert er._split_blocks(text) == [
        "# Heading",
        "Line A.",
        "Para two line 1.\nPara two line 2.",
        "## Sub",
        "Para three.",
    ]


async def test_chunk_answer_groups_consecutive_blocks_up_to_the_limit():
    text = "AAAAAAAAAA\n\nBBBBBBBBBB\n\nCCCCCCCCCC"
    # Each pair of 10-char blocks is 20 chars once joined -- over a 15-char
    # limit each block is its own chunk, under a 25-char limit the first two
    # group together.
    assert er._chunk_answer(text, limit=15) == ["AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"]
    assert er._chunk_answer(text, limit=25) == [
        "AAAAAAAAAA\n\nBBBBBBBBBB", "CCCCCCCCCC"]


async def test_chunked_faithfulness_judge_takes_the_minimum(monkeypatch):
    citations = [{"marker": 1, "fact_uuid": "f1"}, {"marker": 2, "fact_uuid": "f2"}]
    texts = {"f1": "fact one", "f2": "fact two"}
    monkeypatch.setattr(er, "_chunk_answer",
                         lambda text, limit=er._CHUNK_SIZE: ["First claim [1].", "Second claim [2]."])

    async def fake_judge(client, model, q, answer, cited_facts):
        return {"[1] fact one": 5, "[2] fact two": 2}[cited_facts[0]]
    monkeypatch.setattr(er, "_faithfulness_judge", fake_judge)
    score, judged = await er._chunked_faithfulness_judge(
        None, None, "q", "whole answer", citations, texts)
    assert score == 2                           # the minimum of 5 and 2, not their average
    assert judged == 2


async def test_chunked_faithfulness_judge_all_none_gives_none_not_zero(monkeypatch):
    citations = [{"marker": 1, "fact_uuid": "f1"}, {"marker": 2, "fact_uuid": "f2"}]
    texts = {"f1": "fact one", "f2": "fact two"}
    monkeypatch.setattr(er, "_chunk_answer",
                         lambda text, limit=er._CHUNK_SIZE: ["First claim [1].", "Second claim [2]."])

    async def fake_judge_none(client, model, q, answer, cited_facts):
        return None
    monkeypatch.setattr(er, "_faithfulness_judge", fake_judge_none)
    score, judged = await er._chunked_faithfulness_judge(
        None, None, "q", "whole answer", citations, texts)
    assert score is None
    assert judged == 2                          # both chunks were sent to the judge


async def test_chunked_faithfulness_judge_gives_each_chunk_only_its_own_labelled_facts(
        monkeypatch):
    """Task 18 (spec §14.3): each chunk gets the labelled line(s) of its OWN
    markers, not the plain fact text and not another chunk's facts."""
    citations = [
        {"marker": 1, "fact_uuid": "f1", "sources": [{"vendor": "V1", "product": "P1"}]},
        {"marker": 2, "fact_uuid": "f2", "sources": [{"vendor": "V2", "product": "P2"}]},
    ]
    texts = {"f1": "fact one", "f2": "fact two"}
    monkeypatch.setattr(er, "_chunk_answer", lambda text, limit=er._CHUNK_SIZE: [
        "No citation at all in this chunk.", "First claim [1].", "Second claim [2]."])
    calls: list[tuple] = []

    async def fake_judge(client, model, q, answer, cited_facts):
        calls.append((answer, list(cited_facts)))
        return 5
    monkeypatch.setattr(er, "_faithfulness_judge", fake_judge)
    score, judged = await er._chunked_faithfulness_judge(
        None, None, "q", "whole answer", citations, texts)
    assert judged == 2                          # the no-citation chunk is skipped, never judged
    assert calls == [
        ("First claim [1].", ["[1] (V1 · P1) fact one"]),
        ("Second claim [2].", ["[2] (V2 · P2) fact two"]),
    ]
    assert score == 5


async def test_judge_faithfulness_short_answer_uses_the_single_call_path(monkeypatch):
    calls: list[list[str]] = []

    async def fake_single(client, model, q, answer, cited_facts):
        calls.append(cited_facts)
        return 4
    monkeypatch.setattr(er, "_faithfulness_judge", fake_single)
    citations = [{"marker": 1, "fact_uuid": "f1"}]
    faith, chunks = await er._judge_faithfulness(
        None, None, "q", "short answer [1].", citations, ["fact one"], {"f1": "fact one"})
    assert faith == 4
    assert chunks is None                       # never chunked, not "chunked zero times"
    assert calls == [["fact one"]]


async def test_judge_faithfulness_dispatches_to_chunking_above_either_threshold(monkeypatch):
    chunked_calls: list[tuple] = []

    async def fake_chunked(client, model, q, answer, citations_arg, texts_arg):
        chunked_calls.append((answer, citations_arg, texts_arg))
        return 3, 5
    monkeypatch.setattr(er, "_chunked_faithfulness_judge", fake_chunked)

    # Over the fact-count threshold (61 citations), short text.
    many_citations = [{"marker": i, "fact_uuid": f"f{i}"} for i in range(1, 62)]
    faith, chunks = await er._judge_faithfulness(
        None, None, "q", "short answer.", many_citations, [], {})
    assert (faith, chunks) == (3, 5)

    # Over the character threshold, few citations.
    long_answer = "x" * 6001
    one_citation = [{"marker": 1, "fact_uuid": "f1"}]
    faith, chunks = await er._judge_faithfulness(
        None, None, "q", long_answer, one_citation, ["fact"], {"f1": "fact"})
    assert (faith, chunks) == (3, 5)
    assert len(chunked_calls) == 2


async def test_run_eval_records_faith_chunks_for_a_long_answer(monkeypatch):
    """End to end through `_score_one`/`run_eval`: a long answer's per-question
    record carries `faith_chunks` (the number of chunks judged) and the
    minimum chunk score, not the single-call score."""
    para1 = "Para one covers topic A at some length repeatedly. " * 60 + "[1]."
    para2 = "Para two covers topic B at some length repeatedly. " * 60 + "[2]."
    long_answer = para1 + "\n\n" + para2
    # Two real citations plus padding ones (never mentioned in the text) to
    # cross the fact-count threshold -- mirrors run 12's 110-254-fact answers.
    citations = (
        [{"marker": 1, "fact_uuid": "f1",
          "sources": [{"article_id": "A1", "url": "u", "title": "t"}]},
         {"marker": 2, "fact_uuid": "f2",
          "sources": [{"article_id": "A1", "url": "u", "title": "t"}]}]
        + [{"marker": n, "fact_uuid": f"f{n}",
            "sources": [{"article_id": "A1", "url": "u", "title": "t"}]}
           for n in range(3, 62)])

    async def _router(*a, q, mode_override, scope, settings, **kw):
        env = _env("local", "A1")
        env["citations"] = citations
        env["answer"] = long_answer
        return env
    monkeypatch.setattr(er.router_mod, "answer_router", _router)

    async def fake_judge(client, model, q, answer, cited_facts):
        return 2 if "[1]" in answer else 5
    monkeypatch.setattr(er, "_faithfulness_judge", fake_judge)

    questions = [{"question": "loc", "intent": "local", "expected_modes": ["local"],
                  "expected_article_ids": ["A1"]}]
    summary = await er.run_eval((None,) * 11, questions, _S, _RESOLVER)
    rec = summary["per_question"][0]
    assert rec["faith_chunks"] == 2
    assert rec["faithfulness"] == 2             # the minimum of the two chunk scores (2, 5)
