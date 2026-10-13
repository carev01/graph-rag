"""Run 7 prompt rules: per-side clauses when comparing, few markers per claim."""
import pytest

from answer_api import drift, enumeration, global_search, synthesize

_COMPARE = ("If you compare the sides, give each side's claim in its own clause with "
            "only that side's markers.")
_FEW = ("Cite each claim with the one or two markers of the facts it rests on; never "
        "attach a long list of markers to one sentence.")


@pytest.mark.parametrize("prompt", [synthesize._FACET_PROMPT,
                                    global_search._REDUCE_FACETS_PROMPT,
                                    drift._SYNTH_FACETS_PROMPT,
                                    enumeration._ENUM_PROMPT],
                         ids=["local", "global", "drift", "enumeration"])
def test_every_facet_prompt_keeps_comparisons_per_side(prompt):
    assert _COMPARE in prompt


def test_enumeration_prompt_limits_markers_per_claim():
    assert _FEW in enumeration._ENUM_PROMPT


@pytest.mark.parametrize("prompt", [synthesize._PROMPT, global_search._REDUCE_PROMPT,
                                    drift._SYNTH_PROMPT],
                         ids=["local", "global", "drift"])
def test_single_prompts_do_not_carry_the_facet_rules(prompt):
    assert _COMPARE not in prompt and _FEW not in prompt
