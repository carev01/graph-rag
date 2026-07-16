from answer_api.synthesize import _finalize_answer

MM = {1: {"fact_uuid": "f1"}, 2: {"fact_uuid": "f2"}}


def test_strips_url():
    ans, cited = _finalize_answer("See https://evil/x for details [1].", MM)
    assert "http" not in ans and cited == [1]


def test_strips_url_case_insensitive():
    # LLMs sometimes capitalize the scheme; a URL must NEVER survive.
    ans, cited = _finalize_answer("See HTTPS://EVIL.COM/x and Http://e/y [1].", MM)
    assert "EVIL.COM" not in ans and "Http:" not in ans and cited == [1]


def test_keeps_valid_drops_invented_markers():
    ans, cited = _finalize_answer("Immutable [1], and cross-region [2], and made-up [9].", MM)
    assert cited == [1, 2]        # 9 not in marker_map -> dropped


def test_ordered_unique():
    _, cited = _finalize_answer("[2] then [1] then [2] again.", MM)
    assert cited == [2, 1]        # first-seen order, deduped


def test_no_markers():
    ans, cited = _finalize_answer("I don't have enough information.", MM)
    assert cited == []
