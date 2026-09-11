"""The global reduce prompt must bind the answer to its evidence.

A real run produced 3,516 characters of prose on 4 citations, opening with
meta-commentary about what the community reports did NOT contain. The prompt
permitted all of it. These tests pin the constraints that forbid it -- they are
prompt-shape tests, so the real verification is the eval re-run in Task 3.
"""
# NOTE: `_REFUSAL` is module-local. global_search.py:22 defines its OWN
# ("I don't have enough thematic coverage to answer that from the community
# reports."), which is NOT synthesize.py's. Import it from global_search.
from answer_api.global_search import _REDUCE_PROMPT, _REFUSAL


def test_still_has_its_format_fields():
    assert "{q}" in _REDUCE_PROMPT and "{blocks}" in _REDUCE_PROMPT


def test_forbids_commentary_about_absent_evidence():
    """THE new constraint. The traced failure opened with exactly this."""
    low = _REDUCE_PROMPT.lower()
    assert "do not comment on what the findings do not contain" in low
    assert "absence of evidence is not a finding" in low


def test_requires_every_claim_to_carry_a_marker():
    """The old prompt already said 'cite every claim'; what is NEW is making an
    uncited sentence explicitly disallowed."""
    assert "a sentence with no marker is not allowed" in _REDUCE_PROMPT.lower()


def test_refusal_is_offered_for_thin_evidence_not_only_irrelevance():
    """The broadened trigger: the old prompt refused only when NOTHING was
    relevant, so on thin-but-relevant evidence the model padded instead. The
    _REFUSAL string itself is unchanged -- only when it applies."""
    assert _REFUSAL in _REDUCE_PROMPT
    assert "if the findings do not support an answer" in _REDUCE_PROMPT.lower()
    assert "if nothing is relevant" not in _REDUCE_PROMPT.lower()


def test_still_forbids_urls():
    """Design decision #2 -- the model must never write a URL."""
    assert "url" in _REDUCE_PROMPT.lower()


def _reduce_rules() -> str:
    """The Rules block only. `_MAP_PROMPT` shares phrases with `_REDUCE_PROMPT`
    ("Absence of evidence is not a finding"), and this module deliberately
    imports only the reduce prompt -- scoping further to its Rules block makes
    a match against any other prompt text impossible."""
    after_rules = _REDUCE_PROMPT.split("Rules:", 1)[1]
    return after_rules.split("QUESTION:", 1)[0]


def test_forbids_range_shorthand_and_shows_the_individual_form():
    """BACKLOG 0d. The reducer wrote `[1]-[26]` for prose resting on 26 facts and
    `_finalize_answer` kept two citations. Expanding ranges stays rejected (a
    30-marker span is a guess, not a citation), so the fix is to stop the model
    writing one: the prompt must show the individual form AND name the range
    form as forbidden. Before this slice none of "range", "span", "individually"
    or the `[1] [2] [3]` example appeared anywhere in the prompt."""
    rules = _reduce_rules()
    low = rules.lower()
    assert "[1] [2] [3]" in rules                 # the required form, shown literally
    assert "[1]-[3]" in rules                     # the forbidden form, shown literally
    assert "individually" in low
    assert "never" in low and "range" in low and "span" in low


def _rule_count() -> int:
    """Lines that START with `- `. `rules.count("- ")` counted the substring
    anywhere, so a rule containing `--` would have failed the test below with
    a docstring claiming the tuned rules were disturbed (BACKLOG 0d follow-up)."""
    return sum(1 for ln in _reduce_rules().splitlines() if ln.startswith("- "))


def test_range_rule_did_not_disturb_the_tuned_rules():
    """Changing as little as possible: the refusal trigger and the
    meta-commentary ban must be exactly where they were."""
    rules = _reduce_rules().lower()
    assert "do not comment on what the findings do not contain" in rules
    assert "if the findings do not support an answer" in rules
    assert _rule_count() == 7                     # six tuned rules + the 0d range rule


def test_explains_marker_bound_fact_lines_and_binds_the_citation_to_them():
    """BACKLOG 0b. The FINDINGS are now `[N] <fact>` lines, so the prompt must
    say what a marker IS (the fact printed beside it) and that a claim drawn
    from that fact cites that marker. Before this slice neither phrase existed
    anywhere in the prompt: it spoke of "the [N] fact markers shown" with the
    facts themselves never shown at all."""
    preamble = _REDUCE_PROMPT.split("Rules:", 1)[0].lower()
    assert "followed by the fact" in preamble
    rules = _reduce_rules().lower()
    assert "must cite that [n]" in rules
    # the 0d rule, the refusal trigger, the meta-commentary ban and the URL ban
    # are all still there and still individually listed
    assert "never write a range" in rules
    assert "do not write any url" in rules
    assert _rule_count() == 7
