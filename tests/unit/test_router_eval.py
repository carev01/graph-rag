from answer_api.router_eval import (
    aggregate, markers_per_sentence, _parse_judge_score, routing_hit,
)


# --- BACKLOG 0b: markers per sentence ----------------------------------------
# Pasting 19 markers onto one sentence was indistinguishable from good citation
# in every measure the eval had (cited, ranges, grounding, the judge). This is
# the distribution that separates "one marker per claim" from bag-pasting.

def test_markers_per_sentence_counts_each_cited_sentence():
    text = "AWS encrypts with KMS [1] [2]. Azure uses AES [3].\nVault Lock is WORM [4] [5] [6]."
    assert markers_per_sentence(text) == [2, 1, 3]


def test_markers_per_sentence_skips_uncited_sentences_and_headings():
    text = "## AWS Backup\nAWS encrypts with KMS [1]. It is enabled by default.\n\nAzure [2]."
    assert markers_per_sentence(text) == [1, 1]


def test_markers_per_sentence_folds_a_trailing_marker_only_segment_into_its_sentence():
    """Models sometimes write `Claim. [1] [2]` -- the markers after the full
    stop belong to the sentence before them, not to a new empty one."""
    assert markers_per_sentence("Claim one. [1] [2] Claim two [3].") == [2, 1]


def test_markers_per_sentence_bag_pasting_is_visible():
    bag = " ".join(f"[{i}]" for i in range(1, 20))
    assert markers_per_sentence(f"Both vendors encrypt at rest {bag}.") == [19]


def test_markers_per_sentence_empty_or_refusal_is_empty():
    assert markers_per_sentence("") == []
    assert markers_per_sentence("I don't have enough thematic coverage.") == []


def test_aggregate_reports_markers_per_sentence_by_mode():
    pq = [
        {"question": "a", "intent": "global", "chosen": "global", "routing_hit": True,
         "grounding_hit": None, "faithfulness": 2, "mps_mean": 1.0, "mps_max": 1},
        {"question": "b", "intent": "global", "chosen": "global", "routing_hit": True,
         "grounding_hit": None, "faithfulness": 5, "mps_mean": 3.0, "mps_max": 19},
        {"question": "c", "intent": "local", "chosen": "local", "routing_hit": True,
         "grounding_hit": None, "faithfulness": 5},              # older record: no mps keys
    ]
    s = aggregate(pq)
    assert s["mps_by_mode"] == {"global": {"mean": 2.0, "max": 19}}


def test_routing_hit():
    assert routing_hit("global", ["global", "drift"]) is True
    assert routing_hit("local", ["global", "drift"]) is False


def test_parse_judge_score():
    assert _parse_judge_score("5") == 5
    assert _parse_judge_score("The score is 4.") == 4
    assert _parse_judge_score("```3```") == 3
    assert _parse_judge_score("banana") == 0
    assert _parse_judge_score("7") == 5        # clamp high
    assert _parse_judge_score("-2") == 0       # clamp low


def test_aggregate_core_metrics():
    pq = [
        {"question": "a", "intent": "local", "chosen": "local",
         "routing_hit": True, "grounding_hit": True, "faithfulness": 5},
        {"question": "b", "intent": "local", "chosen": "drift",
         "routing_hit": False, "grounding_hit": False, "faithfulness": 2},
        {"question": "c", "intent": "global", "chosen": "global",
         "routing_hit": True, "grounding_hit": None, "faithfulness": 4},
    ]
    s = aggregate(pq)
    assert s["n"] == 3
    assert abs(s["routing_accuracy"] - 2 / 3) < 1e-9
    assert s["routing_by_intent"]["local"] == 0.5 and s["routing_by_intent"]["global"] == 1.0
    assert s["grounding_precision"] == 0.5          # only a,b scored (c is None)
    assert abs(s["faithfulness_mean"] - 11 / 3) < 1e-9
    assert s["comparative"] is None                 # no comparative blocks


def test_aggregate_comparative_drift_wins():
    def broad(qid, comp):
        return {"question": qid, "intent": "drift", "chosen": "drift",
                "routing_hit": True, "grounding_hit": None, "faithfulness": 4,
                "comparative": comp}
    win = [broad("q", {
        "local": {"grounding_hit": None, "faithfulness": 2},
        "global": {"grounding_hit": None, "faithfulness": 3},
        "drift": {"grounding_hit": None, "faithfulness": 5}})]
    assert aggregate(win)["drift_wins"] is True
    lose = [broad("q", {
        "local": {"grounding_hit": None, "faithfulness": 2},
        "global": {"grounding_hit": None, "faithfulness": 5},
        "drift": {"grounding_hit": None, "faithfulness": 3}})]
    assert aggregate(lose)["drift_wins"] is False
