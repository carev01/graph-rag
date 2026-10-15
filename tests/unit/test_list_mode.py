"""List-mode detection, limitation classifier and expansion (spec 2026-10-15 §3-§4)."""
import logging
from types import SimpleNamespace

import pytest

import answer_api.list_mode as lm
from answer_api.scope import Scope
from graph_extract.config import ExtractSettings

LISTS = [
    "Which workloads does Veeam support immutability for?",
    "Where can I restore a Windows agent backup?",
    "Which hypervisors does Veeam Backup & Replication support?",
    "Which resource types can AWS Backup protect?",
    "What recovery options does Azure Backup offer for Azure VMs?",
    "What are the ways to restore a Veeam Agent backup?",
    "List the supported databases for Veeam plug-ins",
    "What destinations can AWS Backup copy recovery points to?",
]
NOT_LISTS = [
    "What is a hardened repository?",
    "How does Veeam SureBackup verify backups?",
    "How do you halt and resume plan testing in Veeam Recovery Orchestrator?",
    "Does AWS Backup support Amazon S3?",
    "What should I consider when enabling backup immutability for Veeam Agent backups?",
    "Why does a backup copy job fail?",
]


@pytest.mark.parametrize("q", LISTS)
def test_list_questions_are_detected(q):
    assert lm.is_list_question(q)


@pytest.mark.parametrize("q", NOT_LISTS)
def test_other_questions_are_not_lists(q):
    assert not lm.is_list_question(q)


@pytest.mark.parametrize("fact,lim", [
    ("Veeam Agent does not support the backup immutability feature for Google Cloud.", True),
    ("A Windows host is not supported as a hardened repository.", True),
    ("Immutability cannot be disabled once set.", True),
    ("Selective disk restore is only available for managed disks.", True),
    ("Veeam Agent provides instant restore capability to Hyper-V VMs", False),
    ("AWS Backup supports Amazon EFS file systems.", False),
])
def test_limitation_classifier(fact, lim):
    assert lm.is_limitation(fact) is lim


class _Client:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.prompts = content, exc, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        if self.exc:
            raise self.exc
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


async def test_expansion_returns_item_kind_queries_capped():
    c = _Client('["restore Veeam Agent backup to VMware vSphere", "to Hyper-V", "to Amazon EC2", '
                '"x", "a", "b", "c", "d", "e", "f", "g", "h"]')
    out = await lm.expand_queries(c, "m", "Where can I restore a Windows agent backup?", 10)
    assert out[0] == "restore Veeam Agent backup to VMware vSphere"
    assert len(out) == 10
    assert "kinds of item" in c.prompts[0] and "Where can I restore" in c.prompts[0]


@pytest.mark.parametrize("client", [_Client(None), _Client("not json"), _Client("{}"),
                                    _Client(exc=RuntimeError("down"))])
async def test_failed_expansion_returns_empty_and_warns(client, caplog):
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        assert await lm.expand_queries(client, "m", "Which X?", 10) == []
    assert "list expansion" in caplog.text


async def test_expansion_drops_blank_and_duplicate_queries():
    c = _Client('["a", " ", "a", "b"]')
    assert await lm.expand_queries(c, "m", "Which X?", 10) == ["a", "b"]


def _settings(**kw):
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p", **kw)


def _fact(uuid, text):
    return {"fact": text, "fact_uuid": uuid, "valid_at": None, "invalid_at": None,
            "sources": [{"url": f"https://d/{uuid}", "title": uuid,
                         "vendor": "Veeam", "product": "Veeam Backup & Replication"}]}


@pytest.fixture
def patched(monkeypatch):
    calls = []

    def install(per_query, expansion='["q-hyperv", "q-ec2"]', fail=()):
        async def _search(graphiti, driver, *, q, k, scope=None, group_id, **kw):
            calls.append({"q": q, "k": k, "pool": kw.get("rerank_pool")})
            if q in fail:
                raise RuntimeError(f"search failed for {q}")
            return {"results": per_query.get(q, []), "count": 0}

        async def _collapse(driver, results, *, cosine):
            seen, out = set(), []
            for r in results:
                if r["fact_uuid"] not in seen:
                    seen.add(r["fact_uuid"])
                    out.append(r)
            return out

        async def _rerank_top(q, cands, *, k, settings, what, order_name):
            return list(cands)[:k]

        monkeypatch.setattr(lm, "search_local", _search)
        monkeypatch.setattr(lm, "collapse_near_duplicates", _collapse)
        monkeypatch.setattr(lm, "_rerank_top", _rerank_top)
        return _Client(expansion)
    return install, calls


async def test_retrieve_list_merges_every_query_and_budgets_limitations(patched):
    install, calls = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({
        q: [_fact("az", "Restore to Microsoft Azure is supported."),
            _fact("lim", "Restore to Google Cloud is not supported for encrypted disks.")],
        "q-hyperv": [_fact("hv", "Instant Recovery to a Hyper-V VM is supported.")],
        "q-ec2": [_fact("ec2", "Restore to Amazon EC2 is supported."),
                  _fact("az", "Restore to Microsoft Azure is supported.")]})
    s = _settings(list_supported_k=50, list_limitations_k=12)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=s)
    assert [c["q"] for c in calls] == [q, "q-hyperv", "q-ec2"]
    assert all(c["k"] == 20 and c["pool"] == 40 for c in calls)
    assert [r["fact_uuid"] for r in got.results] == ["az", "hv", "ec2"]
    assert [r["fact_uuid"] for r in got.limitations] == ["lim"]
    assert got.expansion_queries == ["q-hyperv", "q-ec2"] and got.degraded == []


async def test_supported_budget_is_never_taken_by_limitations(patched):
    install, _ = patched
    q = "Which workloads does Veeam support immutability for?"
    facts = [_fact(f"l{i}", f"Immutability is not supported for case {i}.") for i in range(20)]
    facts += [_fact(f"s{i}", f"Immutability is supported for workload {i}.") for i in range(5)]
    client = install({q: facts}, expansion="[]")
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings(list_supported_k=50, list_limitations_k=12))
    assert len(got.results) == 5 and len(got.limitations) == 12


async def test_a_failed_query_degrades_and_all_failed_raises(patched):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, fail=("q-ec2",))
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert got.degraded == ["retrieval failed for: q-ec2"]
    client = install({}, fail=(q, "q-hyperv", "q-ec2"))
    with pytest.raises(RuntimeError, match="every list retrieval failed"):
        await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                               settings=_settings())


async def test_answer_list_groups_supported_then_limitations_and_cites_by_marker(patched, monkeypatch):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    map_client = install({q: [_fact("az", "Restore to Microsoft Azure is supported."),
                              _fact("lim", "Restore is not supported to Xen.")]}, expansion="[]")
    synth = _Client("**Clouds**\n- Microsoft Azure [1]\n\n**Not supported / limitations**\n- Xen [2]")
    out = await lm.answer_list(None, None, map_client, "m", synth, "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings())
    prompt = synth.prompts[0]
    assert "SUPPORTED FACTS:\n[1] " in prompt and "LIMITATION FACTS:\n[2] " in prompt
    assert "every item" in prompt.lower() and "Not supported / limitations" in prompt
    assert out["list_mode"] is True and out["expansion_queries"] == []
    assert [c["fact_uuid"] for c in out["citations"]] == ["az", "lim"]
    assert out["citations"][0]["sources"][0]["url"] == "https://d/az"
    assert "degraded" not in out


async def test_answer_list_with_no_facts_refuses_without_an_llm_call(patched):
    install, _ = patched
    map_client = install({}, expansion="[]")
    synth = _Client("unused")
    out = await lm.answer_list(None, None, map_client, "m", synth, "s", q="Which X?",
                               scope=Scope(), group_id="g", settings=_settings())
    assert out["citations"] == [] and synth.prompts == []
