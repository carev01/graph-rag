from graph_extract.usage import UsageTally


def test_usage_tally_accumulates():
    t = UsageTally()
    t.add("entity", prompt=100, completion=40)
    t.add("entity", prompt=50, completion=10)
    t.add("edge", prompt=30, completion=5)
    assert t.prompt_tokens == 180 and t.completion_tokens == 55
    assert t.calls == 3
    assert t.by_call["entity"]["calls"] == 2
