from graph_extract.eval import _parse_yes_no


def test_parse_yes_no_plain_yes():
    assert _parse_yes_no("yes") is True


def test_parse_yes_no_plain_no():
    assert _parse_yes_no("no") is False


def test_parse_yes_no_reasoning_then_yes():
    content = "The fact is clearly stated in the text. Therefore, yes."
    assert _parse_yes_no(content) is True
    # The bug this fix addresses: the old `.startswith("yes")` parser
    # misreads reasoning-then-verdict output as "no".
    old_buggy_verdict = content.strip().lower().startswith("yes")
    assert old_buggy_verdict is False


def test_parse_yes_no_think_tag_then_no():
    content = "<think>hmm the text says X but the fact says Y</think>\nno"
    assert _parse_yes_no(content) is False


def test_parse_yes_no_unparseable():
    assert _parse_yes_no("I cannot determine this.") is None


def test_parse_yes_no_last_token_wins():
    assert _parse_yes_no("It's not a no... yes") is True


def test_parse_yes_no_none_content():
    assert _parse_yes_no(None) is None
