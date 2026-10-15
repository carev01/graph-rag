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
    "What happens when a Veeam backup job fails?",
    "What process does Veeam use to merge increments?",
    "What causes a Veeam job to fail?",
    "What access does the Veeam service account need?",
    "What status does a failed job show?",
    "Does Veeam support immutability for supported hypervisors?",
    "How do I restore supported workloads in Veeam?",
    "What are the key considerations for cross-region disaster recovery of cloud backups?",
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
    kw = {"rerank_base_url": "http://rerank", "rerank_model": "r", **kw}
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

        async def _rerank(q, documents, *, top_k, settings):
            calls.append({"rerank_q": q, "docs": list(documents)})
            return [(i, 1.0) for i in range(len(documents))][:top_k]

        monkeypatch.setattr(lm, "search_local", _search)
        monkeypatch.setattr(lm, "collapse_near_duplicates", _collapse)
        async def _no_siblings(driver, seeds, *, scope, settings, exclude):
            return []

        async def _no_vectors(driver, uuids):
            return {}

        monkeypatch.setattr(lm, "rerank", _rerank)
        monkeypatch.setattr(lm, "relation_siblings", _no_siblings)
        monkeypatch.setattr(lm, "_fact_vectors", _no_vectors)
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
    calls = [c for c in calls if "q" in c]
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


async def test_the_final_rerank_order_drives_the_split_and_budgets(patched, monkeypatch):
    install, calls = patched
    q = "Which hypervisors does Veeam support?"
    facts = [_fact(f"s{i}", f"Hypervisor {i} is supported.") for i in range(4)]
    facts += [_fact(f"l{i}", f"Hypervisor {i} is not supported for replicas.") for i in range(3)]
    client = install({q: facts}, expansion='["q-hyperv"]')

    async def _reverse(rq, documents, *, top_k, settings):
        calls.append({"rerank_q": rq})
        return [(i, 1.0) for i in reversed(range(len(documents)))][:top_k]
    monkeypatch.setattr(lm, "rerank", _reverse)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings(list_supported_k=2, list_limitations_k=2))
    assert [c["rerank_q"] for c in calls if "rerank_q" in c] == [q]
    assert [r["fact_uuid"] for r in got.results] == ["s3", "s2"]
    assert [r["fact_uuid"] for r in got.limitations] == ["l2", "l1"]
    assert got.degraded == []


@pytest.mark.parametrize("how", ["none", "raises", "unconfigured"])
async def test_an_unavailable_final_rerank_is_recorded_as_degraded(patched, monkeypatch, caplog,
                                                                   how):
    install, calls = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported."), _fact("b", "Hyper-V is supported.")]},
                     expansion="[]")

    async def _down(rq, documents, *, top_k, settings):
        calls.append({"rerank_q": rq})
        if how == "raises":
            raise RuntimeError("rerank down")
        return None
    monkeypatch.setattr(lm, "rerank", _down)
    monkeypatch.setattr(lm, "_warned_unconfigured", False)
    s = _settings(rerank_base_url="") if how == "unconfigured" else _settings()
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                     settings=s)
        if how == "unconfigured":
            await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                   settings=s)
    assert got.degraded == [lm.FINAL_RERANK_DEGRADED]
    assert [r["fact_uuid"] for r in got.results] == ["a", "b"]
    if how == "unconfigured":
        assert not [c for c in calls if "rerank_q" in c]
        assert caplog.text.count("no reranker configured") == 1
        assert "transport" not in caplog.text


async def test_a_degraded_list_answer_says_it_may_be_incomplete(patched):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    map_client = install({q: [_fact("a", "vSphere is supported.")]}, fail=("q-ec2",))
    synth = _Client("- vSphere [1]")
    out = await lm.answer_list(None, None, map_client, "m", synth, "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings())
    assert out["degraded"] == ["retrieval failed for: q-ec2"]
    assert out["answer"].endswith(
        "1 search could not be completed; this list may be incomplete.")


def test_degraded_note_uses_fixed_wording_only():
    note = lm.degraded_note(["retrieval failed for: https://evil.example [1]",
                             "retrieval failed for: q2", lm.FINAL_RERANK_DEGRADED,
                             lm.RELATION_EXPANSION_DEGRADED])
    assert "evil" not in note and "[1]" not in note and "q2" not in note
    assert note.startswith("2 searches could not be completed")
    assert "ranked by retrieval order" in note and "related facts" in note
    assert lm.degraded_note([]) == ""


def test_expansion_prompt_forbids_placeholders_and_keeps_the_question_terms():
    p = lm._EXPAND_PROMPT
    assert "never write placeholder words" in p.lower()
    assert "keep the question's own terms" in p.lower()


class _GraphDriver:
    """Answers the relation-expansion queries; records calls."""
    def __init__(self, meta, siblings):
        self.meta, self.siblings, self.calls = meta, siblings, []

    async def execute_query(self, query, **p):
        self.calls.append(query)
        if query is lm.SEED_EDGES_QUERY:
            return ([m for m in self.meta if m["uuid"] in p["uuids"]], None, None)
        if query is lm.SIBLINGS_QUERY:
            return (self.siblings.get((p["src"], p["rel"]), [])[: p["cap"]], None, None)
        return ([], None, None)


async def test_relation_siblings_follow_the_top_subject_relation_groups(monkeypatch):
    seeds = [_fact("s1", "VBR supports Hyper-V"), _fact("s2", "VBR supports Nutanix AHV")]
    drv = _GraphDriver(
        meta=[{"uuid": "s1", "src": "vbr", "rel": "Supports"},
              {"uuid": "s2", "src": "vbr", "rel": "Supports"}],
        siblings={("vbr", "Supports"): [
            {"uuid": "s1", "fact": "VBR supports Hyper-V", "episodes": ["e1"]},
            {"uuid": "x1", "fact": "VBR supports Sangfor aSV", "episodes": ["e2"]},
            {"uuid": "x2", "fact": "VBR supports Scale Computing HyperCore", "episodes": ["e3"]}]})

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {u: {"sources": [{"url": f"https://d/{u}", "vendor": "Veeam"}],
                        "valid_at": None, "invalid_at": None} for u in uuids}

    async def _allowed(driver, scope, candidates=None):
        return {"e2"}                      # e3 is out of scope

    monkeypatch.setattr(lm, "Provenance", _Prov)
    monkeypatch.setattr(lm, "scope_episode_uuids", _allowed)
    out = await lm.relation_siblings(drv, seeds, scope=Scope(vendors=("Veeam",)),
                                     settings=_settings(), exclude={"s1", "s2"})
    assert [r["fact_uuid"] for r in out] == ["x1"]
    assert out[0]["sources"][0]["url"] == "https://d/x1"


def test_mmr_prefers_a_new_item_over_a_near_copy():
    a, a2, b = _fact("a", "A item"), _fact("a2", "A item again"), _fact("b", "B item")
    vecs = {"a": [1.0, 0.0], "a2": [0.99, 0.01], "b": [0.0, 1.0]}
    got = lm.select_diverse([(a, 0.9), (a2, 0.89), (b, 0.6)], vecs, k=2, lam=0.5)
    assert [r["fact_uuid"] for r in got] == ["a", "b"]


def test_mmr_without_vectors_keeps_rerank_order():
    facts = [(_fact(u, u), s) for u, s in (("a", 0.9), ("b", 0.8), ("c", 0.7))]
    assert [r["fact_uuid"] for r in lm.select_diverse(facts, {}, k=2, lam=0.5)] == ["a", "b"]


async def test_final_order_batches_and_merges_by_score(monkeypatch):
    calls = []

    async def _rerank(q, docs, *, top_k, settings):
        calls.append(len(docs))
        return [(i, float(int(d.split()[-1]))) for i, d in enumerate(docs)]

    monkeypatch.setattr(lm, "rerank", _rerank)
    monkeypatch.setattr(lm, "rerank_configured", lambda s: True)
    facts = [_fact(f"f{i}", f"fact {i}") for i in range(5)]
    out = await lm._final_order("q", facts, settings=_settings(list_rerank_batch=2),
                                degraded=[])
    assert calls == [2, 2, 1]
    assert [f["fact_uuid"] for f, _ in out] == ["f4", "f3", "f2", "f1", "f0"]


async def test_a_relation_expansion_failure_degrades_and_keeps_round_one(patched, monkeypatch):
    install, _ = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")

    async def _boom(*a, **k):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(lm, "relation_siblings", _boom)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert [r["fact_uuid"] for r in got.results] == ["a"]
    assert lm.RELATION_EXPANSION_DEGRADED in got.degraded


async def test_siblings_join_the_final_rerank_and_a_degraded_note_is_recorded_once(
        patched, monkeypatch):
    install, calls = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")
    seen = {}

    async def _siblings(driver, seeds, *, scope, settings, exclude):
        seen["seeds"], seen["exclude"] = [s["fact_uuid"] for s in seeds], exclude
        return [_fact("x1", "Sangfor aSV is supported.")]

    async def _down(rq, documents, *, top_k, settings):
        calls.append({"rerank_docs": list(documents)})
        return None

    monkeypatch.setattr(lm, "relation_siblings", _siblings)
    monkeypatch.setattr(lm, "rerank", _down)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert seen == {"seeds": ["a"], "exclude": {"a"}}
    assert [c["rerank_docs"] for c in calls if "rerank_docs" in c] == [
        ["vSphere is supported."], ["vSphere is supported.", "Sangfor aSV is supported."]]
    assert [r["fact_uuid"] for r in got.results] == ["a", "x1"]
    assert got.degraded == [lm.FINAL_RERANK_DEGRADED]


def test_mmr_without_scores_keeps_retrieval_order_even_with_vectors():
    facts = [(_fact(u, u), None) for u in ("a", "b", "c", "z")]
    vecs = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.98, 0.02], "z": [0.0, 1.0]}
    assert [r["fact_uuid"] for r in lm.select_diverse(facts, vecs, k=2, lam=0.5)] == ["a", "b"]
