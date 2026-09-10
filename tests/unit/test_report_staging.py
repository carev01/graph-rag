"""A report whose verification could not complete is STAGED, not destroyed.

The first live rebuild lost 7 of 41 communities to transient verifier errors. The
report generation had already been paid for; only the check failed."""
import json
from types import SimpleNamespace

import pytest

from theme_builder.context import ContextResult
from theme_builder.report import CommunityReport, ReportStats, VerifyResult, generate_report

GOOD = {"title": "T", "summary": "S", "rating": 5, "rating_explanation": "r",
        "tags": ["aws"],
        "full_report": [{"finding": "F1", "fact_ids": ["u1"]}]}


def _ctx():
    return ContextResult(text="ctx", fact_uuids={"u1"}, fact_texts={"u1": "fact one"})


class _Client:
    def __init__(self, payloads):
        self._payloads = [json.dumps(p) if isinstance(p, dict) else p for p in payloads]

        async def _create(**kw):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._payloads.pop(0)))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def _verifier(*results):
    seq = list(results)

    async def _v(findings, summary, fact_texts):
        return seq.pop(0)

    return _v


@pytest.mark.asyncio
async def test_unverifiable_report_is_staged_not_destroyed():
    st = ReportStats()
    rep = await generate_report(_Client([GOOD]), "m", _ctx(),
                                verifier=_verifier(None), stats=st)
    assert rep is None, "must not be returned as a usable report"
    assert st.unverified is True
    assert isinstance(st.staged_report, CommunityReport)
    assert st.staged_report.summary == "S"
    assert json.loads(st.staged_report.full_report)[0]["finding"] == "F1"


@pytest.mark.asyncio
async def test_staged_report_carries_the_retry_generation_when_there_was_one():
    """If the retry regenerated the report and THEN the verifier died, the newer
    report is the one worth keeping."""
    st = ReportStats()
    newer = {**GOOD, "summary": "S2"}
    rep = await generate_report(
        _Client([GOOD, newer]), "m", _ctx(),
        verifier=_verifier(VerifyResult(unsupported={1}, summary_supported=True), None),
        stats=st)
    assert rep is None and st.unverified is True
    assert st.staged_report is not None and st.staged_report.summary == "S2"


@pytest.mark.asyncio
async def test_a_genuinely_unsupported_report_is_not_staged():
    """Staging is for 'could not check', never for 'checked and it failed'.

    NOTE: an unsupported SUMMARY no longer fails a report (Task 8) -- the summary is
    regenerated from kept findings. The genuine-failure case is now "every finding
    dropped", which is what this exercises. GOOD carries one finding, so dropping it
    drops them all."""
    st = ReportStats()
    rep = await generate_report(
        _Client([GOOD, GOOD]), "m", _ctx(),
        verifier=_verifier(VerifyResult(unsupported={1}, summary_supported=True),
                           VerifyResult(unsupported={1}, summary_supported=True)),
        stats=st)
    assert rep is None and st.findings_dropped == 1
    assert st.staged_report is None


@pytest.mark.asyncio
async def test_a_verified_report_stages_nothing():
    """The second payload is the summary regeneration added in Task 8."""
    st = ReportStats()
    rep = await generate_report(_Client([GOOD, "A regenerated summary."]), "m", _ctx(),
                                verifier=_verifier(VerifyResult(set(), True)), stats=st)
    assert rep is not None and st.staged_report is None
