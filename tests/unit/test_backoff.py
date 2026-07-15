from graph_sync.semantic_worker import exp_backoff


def test_exp_backoff_grows_and_caps():
    assert exp_backoff(1, base=30, cap=3600) == 30
    assert exp_backoff(2, base=30, cap=3600) == 60
    assert exp_backoff(3, base=30, cap=3600) == 120
    assert exp_backoff(20, base=30, cap=3600) == 3600
