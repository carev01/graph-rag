"""Free-gate calibration output for list questions (spec 2026-10-15 §7, task 11)."""
import pytest

import answer_api.eval_retrieval as er_mod
from answer_api import list_mode
from answer_api.eval_retrieval import check_list_items
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape
from graph_extract.config import ExtractSettings



def _settings():
    # Built per test, never at collection time (hermetic-test rule).
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


_SHAPE = QuestionShape("single", (Facet("", Scope()),))
Q = {"question": "Where can I restore a Windows agent backup?",
    "expected_items": [{"item": "Sangfor aSV", "aliases": ["Sangfor aSV"]},
                       {"item": "Oracle Linux KVM", "aliases": ["Oracle Linux KVM"]}]}


def _fact(uuid, text):
    return {"fact": text, "fact_uuid": uuid, "valid_at": None, "invalid_at": None,
            "sources": [{"url": f"https://d/{uuid}", "title": uuid,
                         "vendor": "Veeam", "product": "VBR"}]}

F1 = _fact("f1", "VBR supports Veeam Agent restore of Windows backup to Sangfor aSV.")
F2 = _fact("f2", "VBR can restore a Windows agent backup to Sangfor aSV as a VM.")
F3 = _fact("f3", "VBR supports Veeam Agent restore of Windows backup to Proxmox VE.")


def _patch_retrieve_list(monkeypatch, got):
    async def _fake(graphiti, driver, map_client, map_model, *, q, scope, group_id, settings):
        return got

    monkeypatch.setattr(list_mode, "retrieve_list", _fake)


@pytest.mark.asyncio
async def test_item_scores_pick_the_best_matching_fact(monkeypatch):
    """Two facts name the expected item at different scores; the item's score is
    the highest of the two, not the first or the last."""
    got = list_mode.ListRetrieval(results=[F1, F3], limitations=[F2],
                                  scores={"f1": 0.5, "f2": 0.9, "f3": 0.3})
    _patch_retrieve_list(monkeypatch, got)
    out = await check_list_items(None, None, None, None, _settings(), Q, _SHAPE)
    assert out["item_scores"]["Sangfor aSV"] == 0.9
    assert out["item_grades"]["Sangfor aSV"] == "confirmed"


@pytest.mark.asyncio
async def test_an_unmatched_item_scores_none_and_grades_unretrieved(monkeypatch):
    got = list_mode.ListRetrieval(results=[F1, F3], limitations=[],
                                  scores={"f1": 0.5, "f3": 0.3})
    _patch_retrieve_list(monkeypatch, got)
    out = await check_list_items(None, None, None, None, _settings(), Q, _SHAPE)
    assert out["item_scores"]["Oracle Linux KVM"] is None
    assert out["item_grades"]["Oracle Linux KVM"] == "unretrieved"


@pytest.mark.asyncio
async def test_a_matching_fact_with_no_score_still_scores_none(monkeypatch):
    """A matching fact that was never scored (degraded rerank) leaves the item's
    score None rather than crashing or picking an arbitrary fallback."""
    got = list_mode.ListRetrieval(results=[F1], limitations=[], scores={})
    _patch_retrieve_list(monkeypatch, got)
    out = await check_list_items(None, None, None, None, _settings(), Q, _SHAPE)
    assert out["item_scores"]["Sangfor aSV"] is None


@pytest.mark.asyncio
async def test_retrieve_list_raising_still_fails_with_the_error(monkeypatch):
    """Closes the Task 5 deferred minor: every retrieval attempt failing must read
    as FAIL with the error, never as a pass with zero items found, and the
    calibration fields must still be present (empty) for the printer."""
    async def _raise(*a, **kw):
        raise RuntimeError("every list retrieval failed")

    monkeypatch.setattr(list_mode, "retrieve_list", _raise)
    out = await check_list_items(None, None, None, None, _settings(), Q, _SHAPE)
    assert out["pass"] is False
    assert out["found"] == 0 and out["total"] == 2
    assert sorted(out["missing"]) == ["Oracle Linux KVM", "Sangfor aSV"]
    assert out["error"] == "every list retrieval failed"
    assert out["item_scores"] == {}
    assert out["item_grades"] == {}


@pytest.mark.asyncio
async def test_the_grader_simulation_gets_the_questions_scope(monkeypatch):
    got = list_mode.ListRetrieval(results=[F1], limitations=[], scores={"f1": 0.9})
    _patch_retrieve_list(monkeypatch, got)
    seen = []
    real = er_mod.grade_items

    def _spy(*a, **kw):
        seen.append(kw.get("scope"))
        return real(*a, **kw)

    monkeypatch.setattr(er_mod, "grade_items", _spy)
    scope = Scope(vendors=("Veeam",))
    await check_list_items(None, None, None, None, _settings(), Q,
                           QuestionShape("single", (Facet("", scope),)))
    assert seen == [scope]
