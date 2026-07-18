from answer_api.router_eval import routing_hit, _parse_judge_score, aggregate


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
