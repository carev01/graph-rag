"""A failed eval question must be visible and must not be scored as a routing miss."""
from answer_api.eval_router import format_report
from answer_api.router_eval import aggregate


def _rec(intent, chosen, hit, *, failed=False, faith=5):
    return {"question": f"q-{intent}-{chosen}", "intent": intent, "chosen": chosen,
            "routing_hit": hit, "grounding_hit": True, "faithfulness": faith,
            "failed": failed}


def test_counts_failed_questions():
    s = aggregate([_rec("local", "local", True),
                   _rec("global", "error", False, failed=True, faith=None)])
    assert s["questions_failed"] == 1


def test_routing_accuracy_excludes_failed_questions():
    """A network error is not a routing miss. Scoring it as one silently depresses
    the metric -- 14 connection errors did exactly that on 2026-09-10."""
    s = aggregate([_rec("local", "local", True),
                   _rec("global", "error", False, failed=True, faith=None)])
    assert s["routing_accuracy"] == 1.0


def test_routing_accuracy_is_zero_when_every_question_failed():
    s = aggregate([_rec("local", "error", False, failed=True, faith=None)])
    assert s["questions_failed"] == 1
    assert s["routing_accuracy"] == 0.0


def test_a_genuine_miss_still_counts_against_routing():
    s = aggregate([_rec("local", "timeline", False), _rec("global", "global", True)])
    assert s["questions_failed"] == 0
    assert s["routing_accuracy"] == 0.5


def test_report_shows_the_failed_count():
    s = aggregate([_rec("local", "local", True),
                   _rec("global", "error", False, failed=True, faith=None)])
    assert "failed: 1" in format_report(s)


def test_failed_row_renders_as_dash_not_false():
    s = aggregate([_rec("global", "error", False, failed=True, faith=None)])
    row = [ln for ln in format_report(s).splitlines() if "q-global-error" in ln][0]
    assert "| - |" in row


def test_records_without_a_failed_key_are_treated_as_ran():
    """Backward compatibility: older records have no `failed` key."""
    s = aggregate([{"question": "q", "intent": "local", "chosen": "local",
                    "routing_hit": True, "grounding_hit": True, "faithfulness": 5}])
    assert s["questions_failed"] == 0 and s["routing_accuracy"] == 1.0
