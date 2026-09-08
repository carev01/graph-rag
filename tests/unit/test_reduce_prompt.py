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
