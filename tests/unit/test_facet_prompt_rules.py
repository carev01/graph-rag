"""Claim and comparison rules (run 9): every synthesis prompt keeps facts' qualifiers
and caps markers per sentence; multi-side prompts write contrasts, never joint claims."""
import pytest

from answer_api import drift, enumeration, global_search, synthesize
from answer_api.attribution import CLAIM_RULES, COMPARISON_RULES

_FACET = [synthesize._FACET_PROMPT, global_search._REDUCE_FACETS_PROMPT,
          drift._SYNTH_FACETS_PROMPT, enumeration._ENUM_PROMPT]
_FACET_IDS = ["local", "global", "drift", "enumeration"]
_SINGLE = [synthesize._PROMPT, global_search._REDUCE_PROMPT, drift._SYNTH_PROMPT]
_SINGLE_IDS = ["local", "global", "drift"]


@pytest.mark.parametrize("prompt", _FACET + _SINGLE,
                         ids=[f"facet-{i}" for i in _FACET_IDS]
                         + [f"single-{i}" for i in _SINGLE_IDS])
def test_every_synthesis_prompt_carries_the_claim_rules(prompt):
    assert CLAIM_RULES in prompt


def test_claim_rules_name_the_run_9_defects():
    assert "keeping its qualifiers" in CLAIM_RULES
    assert "Do not infer names, capabilities or relationships that no fact states." \
        in CLAIM_RULES
    assert "A sentence never carries more than three markers" in CLAIM_RULES
    assert "never as a range such as [1]-[3]" in CLAIM_RULES


@pytest.mark.parametrize("prompt", _FACET, ids=_FACET_IDS)
def test_every_facet_prompt_writes_contrasts_not_joint_claims(prompt):
    assert COMPARISON_RULES in prompt
    assert "Never make one claim about several sides together" in COMPARISON_RULES
    assert "introduce no point that the side sections did not already state" \
        in COMPARISON_RULES.replace("\n", "")


@pytest.mark.parametrize("prompt", _SINGLE, ids=_SINGLE_IDS)
def test_single_prompts_do_not_carry_the_comparison_rules(prompt):
    assert COMPARISON_RULES not in prompt


@pytest.mark.parametrize("prompt", _FACET + _SINGLE)
def test_no_prompt_keeps_the_superseded_marker_wording(prompt):
    assert "one or two markers of the facts it rests on" not in prompt
    assert "give each side's claim in its own clause" not in prompt
