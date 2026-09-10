"""generate_report's verify -> retry -> drop flow."""
import json
from types import SimpleNamespace

import pytest

from theme_builder.context import ContextResult
from theme_builder.report import ReportStats, VerifyResult, generate_report

GOOD = {"title": "T", "summary": "S", "rating": 5, "rating_explanation": "r",
        "tags": ["aws"],
        "full_report": [{"finding": "F1", "fact_ids": ["u1"]},
                        {"finding": "F2", "fact_ids": ["u2"]}]}


def _ctx():
    return ContextResult(text="ctx", fact_uuids={"u1", "u2"},
                         fact_texts={"u1": "fact one", "u2": "fact two"})


class _Client:
    def __init__(self, payloads):
        self._payloads = [json.dumps(p) if isinstance(p, dict) else p for p in payloads]
        self.calls = []

        async def _create(**kw):
            self.calls.append(kw)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._payloads.pop(0)))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def _verifier(*results):
    seq = list(results)
    calls = []

    async def _v(findings, summary, fact_texts):
        calls.append((findings, summary, fact_texts))
        return seq.pop(0)

    _v.calls = calls
    return _v


@pytest.mark.asyncio
async def test_supported_report_passes_through_with_one_verify_call():
    v = _verifier(VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD, "A regenerated summary."])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None
    assert len(json.loads(rep.full_report)) == 2
    # 1 generation + 1 summary regeneration; still exactly one verify call.
    assert len(c.calls) == 2 and len(v.calls) == 1
    assert st == ReportStats()


@pytest.mark.asyncio
async def test_unsupported_finding_triggers_one_regeneration():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD, GOOD, "A regenerated summary."])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None and len(json.loads(rep.full_report)) == 2
    # 2 report generations (should regenerate exactly once) + 1 summary regeneration.
    assert len(c.calls) == 3
    assert st.reverified is True and st.findings_dropped == 0


@pytest.mark.asyncio
async def test_finding_still_unsupported_after_retry_is_dropped():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported={2}, summary_supported=True))
    c = _Client([GOOD, GOOD, "A regenerated summary."])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None
    kept = json.loads(rep.full_report)
    assert [f["finding"] for f in kept] == ["F1"]
    assert st.findings_dropped == 1
    assert rep.cited_fact_uuids == ["u1"], "dropped finding's fact must not stay cited"


@pytest.mark.asyncio
async def test_unverifiable_report_is_skipped_not_written():
    """None from the verifier means 'could not check'. It must NEVER be treated as
    supported."""
    v = _verifier(None)
    st = ReportStats()
    rep = await generate_report(_Client([GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None and st.unverified is True


@pytest.mark.asyncio
async def test_unverifiable_on_the_retry_is_also_skipped():
    v = _verifier(VerifyResult(unsupported={1}, summary_supported=True), None)
    st = ReportStats()
    rep = await generate_report(_Client([GOOD, GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None and st.unverified is True


@pytest.mark.asyncio
async def test_all_findings_dropped_skips_the_report():
    v = _verifier(VerifyResult(unsupported={1, 2}, summary_supported=True),
                  VerifyResult(unsupported={1, 2}, summary_supported=True))
    st = ReportStats()
    rep = await generate_report(_Client([GOOD, GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None, "an empty report must not be written"
    assert st.findings_dropped == 2


@pytest.mark.asyncio
async def test_retry_returning_zero_findings_is_skipped_not_written():
    """IMPORTANT 3: the retry can come back with `full_report: []`. Re-verifying an
    empty findings list yields `unsupported=set()`, so the `if not kept: return None`
    guard is never reached and a citation-free report -- empty full_report, empty
    cited_fact_uuids, a summary generated from an empty STATEMENTS block -- would be
    embedded and made retrievable."""
    empty = dict(GOOD, full_report=[])
    v = _verifier(VerifyResult(unsupported={1}, summary_supported=True),
                  VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD, empty, "Free prose about a title."])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is None, "a report with no findings must never be written"
    assert st.staged_report is None, "this is not a verification outage"
    # no wasted re-verify and no wasted summary call on an empty findings list
    assert len(v.calls) == 1
    assert len(c.calls) == 2


@pytest.mark.asyncio
async def test_no_verifier_keeps_todays_behaviour_exactly():
    """Ten existing call sites pass no verifier; they must be unaffected."""
    rep = await generate_report(_Client([GOOD]), "m", _ctx())
    assert rep is not None and len(json.loads(rep.full_report)) == 2


@pytest.mark.asyncio
async def test_report_with_no_findings_does_not_call_the_verifier():
    """Nothing to verify; spending a request on it is waste."""
    empty = dict(GOOD, full_report=[])
    v = _verifier()      # would IndexError if called
    rep = await generate_report(_Client([empty]), "m", _ctx(), verifier=v,
                                stats=ReportStats())
    assert rep is not None and json.loads(rep.full_report) == []
    assert v.calls == []


@pytest.mark.asyncio
async def test_the_retry_names_the_offending_findings():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD, GOOD, "A regenerated summary."])
    await generate_report(c, "m", _ctx(), verifier=v, stats=ReportStats())
    retry_prompt = c.calls[1]["messages"][0]["content"]
    # Split on text that ACTUALLY appears in _RETRY_NOTE. "NOT supported" does not,
    # so splitting on it returned the whole prompt and the assertion held by
    # accident (F1 is in the original context, which the retry prompt repeats).
    marker = "Your previous answer was REJECTED."
    assert marker in retry_prompt
    offender_block = retry_prompt.split(marker)[-1]
    assert "F2" in offender_block
    assert "F1" not in offender_block.split("Rewrite the report")[0]


@pytest.mark.asyncio
async def test_retry_generation_failure_degrades_to_first_verified():
    """When retry generation returns unparseable reply (None), the code skips
    re-verification and degrades to 'first-pass verified': it keeps findings
    the original verdict didn't flag, drops those it did. This is correct because
    report and findings were bound together from the same first generation, so
    indices stay consistent and we don't discard the whole report just because
    the retry failed to generate valid JSON."""
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True))
    # First generation succeeds, retry fails (needs two unparseable to exhaust _generate_once's retry)
    c = _Client([GOOD, "not json 1", "not json 2", "A regenerated summary."])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None
    kept = json.loads(rep.full_report)
    assert [f["finding"] for f in kept] == ["F1"], "only F1 should be kept (F2 was flagged unsupported)"
    assert len(v.calls) == 1, "verifier should be called exactly once (no re-verification)"
    assert st.reverified is True
    assert st.findings_dropped == 1
