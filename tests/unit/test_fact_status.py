from answer_api.timeline import _fact_status


def test_current(): assert _fact_status(None, None) == "current"
def test_current_ignores_sweep_flag(): assert _fact_status(None, True) == "current"   # invalid_at null wins
def test_expired(): assert _fact_status("2019-01-10T13:45:24Z", True) == "expired"
def test_superseded(): assert _fact_status("2019-01-10T13:45:24Z", None) == "superseded"
def test_superseded_false_flag(): assert _fact_status("2019-01-10T13:45:24Z", False) == "superseded"
