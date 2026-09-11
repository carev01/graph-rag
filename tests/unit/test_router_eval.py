from answer_api.router_eval import (
    aggregate, bag_share, markers_per_sentence, _parse_judge_score, routing_hit,
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


# --- BACKLOG: threshold the bag-pasting --------------------------------------
# `mps` exposes that a 19-marker sentence exists; it cannot say how much of the
# answer's citation mass sits in such sentences. An answer with one bag and
# thirty well-cited claims and an answer that is nothing but a bag share the
# same mps_max. bag_share weights by markers, not by sentences.

def test_bag_share_is_zero_when_every_sentence_cites_sparsely():
    assert bag_share("KMS [1] [2]. AES [3]. WORM [4] [5] [6].") == 0.0


def test_bag_share_counts_markers_not_sentences():
    """One 8-marker bag beside four single-marker claims: 4 of 5 sentences are
    clean, but 8 of 12 markers -- two thirds of the citation mass -- are bag."""
    bag = " ".join(f"[{i}]" for i in range(1, 9))
    text = f"Both vendors encrypt {bag}. A [9]. B [10]. C [11]. D [12]."
    assert bag_share(text) == 8 / 12


def test_bag_share_threshold_is_eight_inclusive():
    seven = " ".join(f"[{i}]" for i in range(1, 8))
    eight = " ".join(f"[{i}]" for i in range(1, 9))
    assert bag_share(f"Claim {seven}.") == 0.0
    assert bag_share(f"Claim {eight}.") == 1.0


def test_bag_share_is_none_when_there_is_nothing_to_measure():
    """A refusal or an uncited answer has no citation mass. Returning 0.0 would
    read as 'no bag-pasting', the best possible score, for an answer that cited
    nothing at all."""
    assert bag_share("I don't have enough thematic coverage.") is None
    assert bag_share("") is None


def test_aggregate_reports_bag_share_by_mode():
    pq = [
        {"question": "a", "intent": "global", "chosen": "global", "routing_hit": True,
         "grounding_hit": None, "faithfulness": 5, "bag_share": 0.0},
        {"question": "b", "intent": "global", "chosen": "global", "routing_hit": True,
         "grounding_hit": None, "faithfulness": 5, "bag_share": 0.6},
        {"question": "c", "intent": "local", "chosen": "local", "routing_hit": True,
         "grounding_hit": None, "faithfulness": 5},          # older record: no key
        {"question": "d", "intent": "global", "chosen": "global", "routing_hit": True,
         "grounding_hit": None, "faithfulness": None, "bag_share": None},   # refusal
    ]
    s = aggregate(pq)
    assert s["bag_share_by_mode"] == {"global": 0.3}


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
