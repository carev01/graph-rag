"""The summary is regenerated from kept findings, never a reason to lose a report."""
import json
from types import SimpleNamespace

import pytest

from theme_builder.context import ContextResult
from theme_builder.report import ReportStats, VerifyResult, generate_report

GOOD = {"title": "T", "summary": "ORIGINAL", "rating": 5, "rating_explanation": "r",
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

    async def _v(findings, summary, fact_texts):
        return seq.pop(0)

    return _v


@pytest.mark.asyncio
async def test_unsupported_summary_no_longer_loses_the_report():
    """The old rule skipped the whole report here. It destroyed 15 of 41 communities."""
    st = ReportStats()
    rep = await generate_report(
        _Client([GOOD, "A regenerated summary."]), "m", _ctx(),
        verifier=_verifier(VerifyResult(unsupported=set(), summary_supported=False)),
        stats=st)
    assert rep is not None, "a failing summary must never lose a report"
    assert len(json.loads(rep.full_report)) == 2


@pytest.mark.asyncio
async def test_summary_is_replaced_by_the_regenerated_one():
    rep = await generate_report(
        _Client([GOOD, "A regenerated summary."]), "m", _ctx(),
        verifier=_verifier(VerifyResult(set(), True)), stats=ReportStats())
    assert rep is not None and rep.summary == "A regenerated summary."


@pytest.mark.asyncio
async def test_regeneration_sees_only_the_kept_findings():
    """F2 was dropped, so the summariser must not be shown it."""
    c = _Client([GOOD, GOOD, "Summary of F1 only."])
    rep = await generate_report(
        c, "m", _ctx(),
        verifier=_verifier(VerifyResult(unsupported={2}, summary_supported=True),
                           VerifyResult(unsupported={2}, summary_supported=True)),
        stats=ReportStats())
    assert rep is not None
    prompt = c.calls[-1]["messages"][0]["content"]
    assert "F1" in prompt and "F2" not in prompt


@pytest.mark.asyncio
async def test_all_findings_dropped_still_skips_without_summarising():
    """Nothing to summarise -- and no wasted call."""
    c = _Client([GOOD, GOOD])
    st = ReportStats()
    rep = await generate_report(
        c, "m", _ctx(),
        verifier=_verifier(VerifyResult(unsupported={1, 2}, summary_supported=True),
                           VerifyResult(unsupported={1, 2}, summary_supported=True)),
        stats=st)
    assert rep is None and st.findings_dropped == 2
    assert len(c.calls) == 2, "must not call the summariser with no findings"


@pytest.mark.asyncio
async def test_unusable_summary_reply_keeps_the_original_summary():
    """A summariser hiccup must not lose the report either."""
    rep = await generate_report(
        _Client([GOOD, ""]), "m", _ctx(),
        verifier=_verifier(VerifyResult(set(), True)), stats=ReportStats())
    assert rep is not None and rep.summary == "ORIGINAL"


@pytest.mark.asyncio
async def test_no_verifier_skips_regeneration_entirely():
    rep = await generate_report(_Client([GOOD]), "m", _ctx())
    assert rep is not None and rep.summary == "ORIGINAL"
