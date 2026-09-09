from answer_api.synthesize import _finalize_answer

MM = {1: {"fact_uuid": "f1"}, 2: {"fact_uuid": "f2"}}


def test_strips_url():
    ans, cited = _finalize_answer("See https://evil/x for details [1].", MM)
    assert "http" not in ans and cited == [1]


def test_strips_url_case_insensitive():
    # LLMs sometimes capitalize the scheme; a URL must NEVER survive.
    ans, cited = _finalize_answer("See HTTPS://EVIL.COM/x and Http://e/y [1].", MM)
    assert "EVIL.COM" not in ans and "Http:" not in ans and cited == [1]


def test_url_flush_against_marker_keeps_marker():
    # A URL written with no space before the marker must be stripped WITHOUT
    # swallowing the legitimate [1] citation.
    ans, cited = _finalize_answer("Immutability https://evil/x[1] and more.", MM)
    assert "http" not in ans and cited == [1]


def test_keeps_valid_drops_invented_markers():
    ans, cited = _finalize_answer(
        "Immutable [1], and cross-region [2], and made-up [9].", MM)
    assert cited == [1, 2]        # 9 not in marker_map -> dropped
    assert "[9]" not in ans       # ...and removed from the prose, not just the list


def test_ordered_unique():
    _, cited = _finalize_answer("[2] then [1] then [2] again.", MM)
    assert cited == [2, 1]        # first-seen order, deduped


def test_no_markers():
    ans, cited = _finalize_answer("I don't have enough information.", MM)
    assert cited == []


def test_removes_invented_marker_from_the_text_not_just_the_list():
    """The defect this slice fixes: an unresolvable marker was filtered out of
    `cited` but LEFT IN THE PROSE, so readers saw citations the envelope did not
    have."""
    ans, cited = _finalize_answer("Immutable [1], and made-up [9].", MM)
    assert cited == [1]
    assert "[9]" not in ans
    assert "[1]" in ans


def test_range_with_neither_end_resolving_disappears_entirely():
    """Observed in a real global answer: '[31]-[60]'. Both markers and the
    separator must go, leaving no orphaned dash."""
    ans, cited = _finalize_answer("The reports do not compare them [31]-[60].", MM)
    assert cited == []
    assert "[31]" not in ans and "[60]" not in ans
    assert "-" not in ans


def test_range_with_one_end_resolving_keeps_that_end():
    ans, cited = _finalize_answer("Encryption differs [1]-[60].", MM)
    assert cited == [1]
    assert "[1]" in ans and "[60]" not in ans
    assert "-" not in ans


def test_legitimate_range_of_two_valid_markers_is_preserved():
    """Both ends resolve, so this is a real citation span -- leave it alone."""
    ans, cited = _finalize_answer("Both vendors encrypt [1]-[2].", MM)
    assert cited == [1, 2]
    assert "[1]-[2]" in ans


def test_no_whitespace_artefacts_after_removal():
    """This runs on EVERY answer, so sloppy removal would damage correct ones."""
    ans, _ = _finalize_answer("Immutability [9] is supported [1] .", MM)
    assert "  " not in ans
    assert " ." not in ans


def test_url_removal_no_longer_leaves_a_double_space():
    """Pre-existing wart the whitespace repair also fixes."""
    ans, cited = _finalize_answer("See https://evil/x for details [1].", MM)
    assert "  " not in ans and cited == [1]


def test_paragraph_structure_is_preserved():
    """Repair spaces and tabs only -- never line breaks."""
    ans, _ = _finalize_answer("First para [9].\n\nSecond para [1].", MM)
    assert "\n\n" in ans


def test_refusal_string_passes_through_untouched():
    refusal = "I don't have enough information to answer that from the available sources."
    ans, cited = _finalize_answer(refusal, MM)
    assert ans == refusal and cited == []


def test_marker_zero_is_stripped():
    """[0] is falsy -- guard against an `if n:` style membership bug."""
    ans, cited = _finalize_answer("Claimed [0] and real [1].", MM)
    assert cited == [1]
    assert "[0]" not in ans and "[1]" in ans


def test_url_and_unresolvable_markers_together():
    """URL stripping runs first, then marker stripping -- both must land."""
    ans, cited = _finalize_answer("See https://evil/x [9] but really [1].", MM)
    assert "http" not in ans
    assert "[9]" not in ans and "[1]" in ans
    assert "  " not in ans and cited == [1]


def test_chained_range_with_middle_marker_unresolvable_does_not_fabricate_a_span():
    """'[1]-[2]-[3]' with only [2] unresolvable must drop the middle marker
    without fusing the surviving outer markers into a fabricated '[1]-[3]'
    span nobody asserted -- the old single-pass range regex did exactly that."""
    mm = {1: {"fact_uuid": "f1"}, 3: {"fact_uuid": "f3"}}  # [2] absent on purpose
    ans, cited = _finalize_answer("Claim [1]-[2]-[3].", mm)
    assert cited == [1, 3]
    assert "[1]-[3]" not in ans
    assert "[1] [3]" in ans


def test_comma_list_marker_is_split_and_both_resolve():
    """'[1, 2]' is a form models emit constantly; without normalisation the
    marker regex (single markers only) never matches it, so `cited` stays []
    while the reader still sees a bracketed list -- a marker visible in the
    answer with no corresponding citation entry."""
    ans, cited = _finalize_answer("Both [1, 2] agree.", MM)
    assert cited == [1, 2]
    assert "[1]" in ans and "[2]" in ans
    assert "[1, 2]" not in ans


def test_comma_list_marker_with_inner_spaces_is_normalised():
    ans, cited = _finalize_answer("Only [ 9 ] mentions it.", MM)
    assert cited == []          # 9 not in marker_map
    assert "[9]" not in ans and "[ 9 ]" not in ans


def test_comma_list_with_one_unresolvable_marker_drops_only_that_one():
    ans, cited = _finalize_answer("Immutable [1, 9] is claimed.", MM)
    assert cited == [1]
    assert "[1]" in ans and "[9]" not in ans and "[1, 9]" not in ans


def test_single_marker_comma_list_form_is_unchanged():
    """[1] has no comma -- the normalisation pass must be a no-op on it."""
    ans, cited = _finalize_answer("Immutable [1].", MM)
    assert cited == [1]
    assert ans == "Immutable [1]."


def test_em_dash_sentence_punctuation_survives_an_adjacent_removed_marker():
    """Spec 3.3 only licenses dropping a separator 'where a marker was removed
    from each side' -- here the em-dash is ordinary sentence punctuation, not
    a range separator, because only ONE side is a marker."""
    ans, cited = _finalize_answer(
        "enforced [9] — and cannot be disabled [1].", MM)
    assert cited == [1]
    assert "[9]" not in ans
    assert "—" in ans          # the em-dash must survive
    assert "and cannot be disabled [1]." in ans


def test_hyphen_in_compound_word_survives_an_adjacent_removed_marker():
    """The hyphen belongs to 'recovery', not to a range with [9] -- only one
    side of it is a marker, so it must not be eaten."""
    ans, cited = _finalize_answer("Point-in-time [9]-recovery", MM)
    assert cited == []
    assert "[9]" not in ans
    assert "-recovery" in ans        # the hyphen must survive
    assert "Point-in-time" in ans


def test_trailing_space_before_newline_is_removed():
    """A marker removed right before a line break used to leave a lone trailing
    space abutting the '\\n' -- neither the double-space nor the
    space-before-punctuation repair catches that, only a dedicated one does."""
    ans, cited = _finalize_answer("Line one [9]\nLine two.", MM)
    assert cited == []
    assert " \n" not in ans
    assert ans == "Line one\nLine two."
