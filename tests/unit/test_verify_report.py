"""The verifier decides whether a finding is supported by the facts it cites."""
import json
from types import SimpleNamespace

import pytest

from theme_builder.report import VerifyResult, verify_report

FACTS = {
    "u1": "AWS Backup provides continuous backups for Amazon Aurora.",
    "u2": "AWS Backup continuously backs up transaction logs for SAP HANA databases.",
}
FINDINGS = [
    {"finding": "AWS Backup provides continuous backups for Aurora.", "fact_ids": ["u1"]},
    {"finding": "PITR has 1-second precision up to 35 days.", "fact_ids": ["u2"]},
]


class _FakeClient:
    """Records the calls it received so tests can assert on retry behaviour."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

        async def _create(**kw):
            self.calls.append(kw)
            reply = self._replies.pop(0)
            if reply is None:                      # empty choices list
                return SimpleNamespace(choices=[])
            return SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=reply))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.mark.asyncio
async def test_parses_unsupported_indices_and_summary_verdict():
    payload = json.dumps({"unsupported": [2], "summary_supported": True})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "sum", FACTS)
    assert got == VerifyResult(unsupported={2}, summary_supported=True)


@pytest.mark.asyncio
async def test_all_supported_is_an_empty_set_not_none():
    """Empty set means 'checked, all fine'. None means 'could not check'. These
    must never be confused."""
    payload = json.dumps({"unsupported": [], "summary_supported": True})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "sum", FACTS)
    assert got is not None and got.unsupported == set()


@pytest.mark.asyncio
async def test_unsupported_summary_is_reported():
    payload = json.dumps({"unsupported": [], "summary_supported": False})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "sum", FACTS)
    assert got is not None and got.summary_supported is False


@pytest.mark.asyncio
async def test_retries_once_with_a_larger_budget_then_succeeds():
    payload = json.dumps({"unsupported": [1], "summary_supported": True})
    c = _FakeClient(["not json at all", payload])
    got = await verify_report(c, "m", FINDINGS, "sum", FACTS, max_tokens=1000)
    assert got == VerifyResult(unsupported={1}, summary_supported=True)
    assert len(c.calls) == 2
    assert c.calls[1]["max_tokens"] > c.calls[0]["max_tokens"]


@pytest.mark.asyncio
async def test_returns_none_when_both_attempts_are_unusable():
    """MUST be None, never a 'supported' verdict. An unusable reply reading as
    'supported' is exactly the defect fixed in the faithfulness judge."""
    got = await verify_report(_FakeClient(["", None]), "m", FINDINGS, "sum", FACTS)
    assert got is None


@pytest.mark.asyncio
async def test_empty_choices_list_does_not_raise():
    got = await verify_report(_FakeClient([None, None]), "m", FINDINGS, "sum", FACTS)
    assert got is None


@pytest.mark.asyncio
async def test_prompt_shows_each_finding_with_the_text_of_its_cited_facts():
    payload = json.dumps({"unsupported": [], "summary_supported": True})
    c = _FakeClient([payload])
    await verify_report(c, "m", FINDINGS, "the summary", FACTS)
    sent = c.calls[0]["messages"][0]["content"]
    assert "AWS Backup provides continuous backups for Amazon Aurora." in sent
    assert "transaction logs for SAP HANA" in sent
    assert "the summary" in sent
    assert "FINDING 1" in sent and "FINDING 2" in sent


@pytest.mark.asyncio
async def test_finding_citing_no_facts_is_shown_as_having_none():
    payload = json.dumps({"unsupported": [1], "summary_supported": True})
    c = _FakeClient([payload])
    await verify_report(c, "m", [{"finding": "x", "fact_ids": []}], "s", FACTS)
    assert "(no facts cited)" in c.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_garbage_indices_are_ignored_not_crashed_on():
    payload = json.dumps({"unsupported": ["2", "nope", None], "summary_supported": True})
    got = await verify_report(_FakeClient([payload]), "m", FINDINGS, "s", FACTS)
    assert got is not None and got.unsupported == {2}
