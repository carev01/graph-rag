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
    c = _Client([GOOD])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None
    assert len(json.loads(rep.full_report)) == 2
    assert len(c.calls) == 1 and len(v.calls) == 1
    assert st == ReportStats()


@pytest.mark.asyncio
async def test_unsupported_finding_triggers_one_regeneration():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported=set(), summary_supported=True))
    c = _Client([GOOD, GOOD])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None and len(json.loads(rep.full_report)) == 2
    assert len(c.calls) == 2, "should regenerate exactly once"
    assert st.reverified is True and st.findings_dropped == 0


@pytest.mark.asyncio
async def test_finding_still_unsupported_after_retry_is_dropped():
    v = _verifier(VerifyResult(unsupported={2}, summary_supported=True),
                  VerifyResult(unsupported={2}, summary_supported=True))
    c = _Client([GOOD, GOOD])
    st = ReportStats()
    rep = await generate_report(c, "m", _ctx(), verifier=v, stats=st)
    assert rep is not None
    kept = json.loads(rep.full_report)
    assert [f["finding"] for f in kept] == ["F1"]
    assert st.findings_dropped == 1
    assert rep.cited_fact_uuids == ["u1"], "dropped finding's fact must not stay cited"


@pytest.mark.asyncio
async def test_unsupported_summary_skips_the_whole_report():
    """A one-paragraph summary cannot be partially salvaged, and map_report reads
    it, so an ungrounded summary would leak straight through."""
    v = _verifier(VerifyResult(unsupported=set(), summary_supported=False),
                  VerifyResult(unsupported=set(), summary_supported=False))
    st = ReportStats()
    rep = await generate_report(_Client([GOOD, GOOD]), "m", _ctx(), verifier=v, stats=st)
    assert rep is None and st.summary_unsupported is True


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
    c = _Client([GOOD, GOOD])
    await generate_report(c, "m", _ctx(), verifier=v, stats=ReportStats())
    retry_prompt = c.calls[1]["messages"][0]["content"]
    assert "F2" in retry_prompt
    assert "F1" not in retry_prompt.split("NOT supported")[-1]
