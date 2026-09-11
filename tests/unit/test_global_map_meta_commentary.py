"""BACKLOG 0c: the map step must never emit meta-commentary about what a community
report LACKS ("The provided report contains no information about Azure Backup").
Handed to the reduce step as a "finding", such a sentence reads as "the findings do
not support an answer" and trips the refusal rule written into _REDUCE_PROMPT --
while the real, cited facts sit right beside it. The ban was written into
_REDUCE_PROMPT in the citation-integrity slice and never applied to _MAP_PROMPT."""
from __future__ import annotations

import json

import pytest

from answer_api import global_search as gs

_S = gs.ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


class _RecordingClient:
    """Serves canned replies and records every prompt it was sent."""

    def __init__(self, contents):
        self._c = list(contents)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        self.calls.append(kw)
        c = self._c.pop(0)
        choice = type("C", (), {"message": type("M", (), {"content": c})()})()
        return type("R", (), {"choices": [choice]})()


def _hit(cid="c1", title="T", cited=("f1", "f2")):
    return gs.CommunityHit(cid, title, "sum", 1, 7.0, list(cited), "[]", 0.9, None)


@pytest.mark.asyncio
async def test_map_prompt_forbids_commentary_on_what_the_report_lacks():
    """The rendered map prompt must (a) forbid key points about what the report
    does NOT contain and (b) tell the model that a community with nothing
    relevant returns an EMPTY key_points list -- the only shape the reduce step
    can safely ignore."""
    client = _RecordingClient([json.dumps({"key_points": [], "fact_ids": []})])
    await gs.map_report(client, "mm", "compare X and Y", _hit())
    prompt = client.calls[0]["messages"][0]["content"]
    assert "does not contain" in prompt and "Absence of evidence" in prompt
    assert "empty" in prompt and "key_points" in prompt
