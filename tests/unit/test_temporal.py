from datetime import datetime, timedelta, timezone

import pytest

from answer_api.temporal import is_current

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def test_no_end_date_is_current():
    assert is_current(None, NOW) is True


def test_a_past_end_date_is_not_current():
    assert is_current(NOW - timedelta(days=1), NOW) is False


def test_a_future_end_date_is_current():
    assert is_current(datetime(2028, 9, 1, tzinfo=timezone.utc), NOW) is True


def test_naive_datetimes_are_read_as_utc():
    assert is_current(datetime(2028, 9, 1), NOW) is True
    assert is_current(datetime(2026, 3, 31), NOW) is False


def test_now_defaults_to_the_clock():
    assert is_current(datetime(2999, 1, 1, tzinfo=timezone.utc)) is True


def test_a_string_is_rejected_not_parsed():
    """M1: the input is graphiti's datetime | None. Neo4j's toString(datetime)
    form ("...Z[UTC]") is not ISO, so a str branch would raise on real values;
    there is none, and a string must fail loudly rather than be half-parsed."""
    with pytest.raises((TypeError, AttributeError)):
        is_current("2019-01-10T13:45:24Z", NOW)  # type: ignore[arg-type]
