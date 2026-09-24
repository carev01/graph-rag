from datetime import datetime, timezone

from answer_api.timeline import _fact_status

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def test_current(): assert _fact_status(None, None) == "current"
def test_current_ignores_sweep_flag(): assert _fact_status(None, True) == "current"   # invalid_at null wins
def test_expired(): assert _fact_status("2019-01-10T13:45:24Z", True) == "expired"
def test_superseded(): assert _fact_status("2019-01-10T13:45:24Z", None) == "superseded"
def test_superseded_false_flag(): assert _fact_status("2019-01-10T13:45:24Z", False) == "superseded"


def test_future_invalid_at_is_current():
    """A fact with a future end date is current."""
    assert _fact_status(datetime(2028, 9, 1, tzinfo=timezone.utc), False) == "current"


def test_past_invalid_at_is_superseded():
    """A fact with a past end date is superseded."""
    assert _fact_status(datetime(2026, 3, 31, tzinfo=timezone.utc), False) == "superseded"
