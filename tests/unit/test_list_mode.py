"""List-mode detection, limitation classifier and expansion (spec 2026-10-15 §3-§4)."""
import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest

import answer_api.list_mode as lm
import answer_api.router as router_mod
import answer_api.synthesize as synth_mod
from answer_api.fact_dedup import EMBEDDINGS_QUERY
from answer_api.scope import _EPISODES_AMONG, Scope
from answer_api.shape import Facet, QuestionShape
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


# --- List prompt and completeness pass share the faithfulness/subject rules (spec §14) --

def test_the_shared_rules_have_their_verbatim_wording():
    assert lm._FAITHFUL_DESCRIPTION_RULE == (
        "Describe each item only as its cited facts state it, and never change what a "
        "fact says it does: a fact about backing up is not a restore option, and "
        "supported-for is not recommended. ")
    assert lm._SUBJECT_RULE == (
        "List only items that answer the question for the subject it names: skip items "
        "about a different workload or product, but keep items a fact states for a "
        "broader category that contains the subject (for example, all agents when the "
        "question asks about a Windows agent). ")


def test_both_prompts_contain_both_shared_rules():
    for prompt in (lm._LIST_PROMPT, lm._COMPLETE_PROMPT):
        assert lm._FAITHFUL_DESCRIPTION_RULE in prompt
        assert lm._SUBJECT_RULE in prompt


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


_REAL_FACT_VECTORS = lm._fact_vectors


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

        async def _collapse(driver, results, *, cosine, vectors=None):
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
        async def _no_siblings(driver, seeds, *, scope, settings, exclude, degraded):
            return []

        async def _no_vectors(driver, uuids, *, cache, degraded):
            return {}

        async def _no_sweep(driver, q, *, scope, group_id, settings):
            return []

        monkeypatch.setattr(lm, "rerank", _rerank)
        monkeypatch.setattr(lm, "relation_siblings", _no_siblings)
        monkeypatch.setattr(lm, "_fact_vectors", _no_vectors)
        monkeypatch.setattr(lm, "keyword_sweep", _no_sweep)
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
    assert ("Name each item the way the facts name it (the product, platform or feature "
            "the fact names), not by a broader category.") in prompt
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
    """Answers the relation-expansion queries; records calls. The siblings query
    returns edges UNORDERED (reverse uuid order here) up to `fetch`, as the real
    unordered LIMIT may; the scope query keeps the candidate episodes in
    `in_scope`; `fail` makes one (src, rel) group raise, `fail_scope` the scope query."""
    def __init__(self, meta, siblings, in_scope=(), fail=(), fail_scope=False):
        self.meta, self.siblings, self.calls = meta, siblings, []
        self.in_scope, self.fail, self.fail_scope = set(in_scope), set(fail), fail_scope

    async def execute_query(self, query, **p):
        self.calls.append((query, p))
        if query is lm.SEED_EDGES_QUERY:
            return ([m for m in self.meta if m["uuid"] in p["uuids"]], None, None)
        if query is lm.SIBLINGS_QUERY:
            if (p["src"], p["rel"]) in self.fail:
                raise RuntimeError("group query failed")
            rows = sorted(self.siblings.get((p["src"], p["rel"]), []),
                          key=lambda r: r["uuid"], reverse=True)
            return (rows[: p["fetch"]], None, None)
        if query is _EPISODES_AMONG:
            if self.fail_scope:
                raise RuntimeError("scope query failed")
            return ([{"u": [e for e in p["eps"] if e in self.in_scope]}], None, None)
        return ([], None, None)


class _Prov:
    def __init__(self, driver):
        pass

    async def resolve_citations(self, uuids):
        return {u: {"sources": [{"url": f"https://d/{u}", "vendor": "Veeam"}],
                    "valid_at": None, "invalid_at": None} for u in uuids}


_SEEDS = [_fact("s1", "VBR supports Hyper-V"), _fact("s2", "VBR supports Nutanix AHV")]
_META = [{"uuid": "s1", "src": "vbr", "rel": "Supports"},
         {"uuid": "s2", "src": "vbr", "rel": "Supports"}]
_SIBS = {("vbr", "Supports"): [
    {"uuid": "s1", "fact": "VBR supports Hyper-V", "episodes": ["e1"]},
    {"uuid": "x1", "fact": "VBR supports Sangfor aSV", "episodes": ["e2"]},
    {"uuid": "x0", "fact": "VBR supports Scale Computing HyperCore", "episodes": ["e3"]}]}


async def test_relation_siblings_follow_the_top_subject_relation_groups(monkeypatch):
    drv = _GraphDriver(meta=_META, siblings=_SIBS, in_scope={"e1", "e2"})   # e3 out of scope
    monkeypatch.setattr(lm, "Provenance", _Prov)
    degraded: list[str] = []
    out = await lm.relation_siblings(drv, _SEEDS, scope=Scope(vendors=("Veeam",),
                                                              products=("VBR",)),
                                     settings=_settings(list_relation_group_cap=2),
                                     exclude={"s1", "s2"}, degraded=degraded)
    # The scope filter runs before the cap: out-of-scope x0 sorts first but never
    # takes one of the two slots, so in-scope x1 survives.
    assert [r["fact_uuid"] for r in out] == ["x1"]
    assert out[0]["sources"][0]["url"] == "https://d/x1"
    [(sq, sp)] = [c for c in drv.calls if c[0] is lm.SIBLINGS_QUERY]
    assert sp["fetch"] == 8                               # group cap x 4, unordered
    [(_, p)] = [c for c in drv.calls if c[0] is _EPISODES_AMONG]
    assert (p["vendors"], p["products"], p["sources"]) == (["veeam"], ["vbr"], [])
    assert sorted(p["eps"]) == ["e1", "e2", "e3"]         # bounded by the fetched edges
    assert degraded == []


def test_siblings_query_is_an_unordered_bounded_fetch_without_a_scope_filter():
    q = lm.SIBLINGS_QUERY
    assert "LIMIT $fetch" in q
    for absent in ("ORDER BY", "EXISTS", "$vendors", "$products", "$sources"):
        assert absent not in q


async def test_sibling_groups_are_sorted_and_capped_per_group_and_in_total(monkeypatch):
    meta = [{"uuid": "s1", "src": "vbr", "rel": "A"}, {"uuid": "s2", "src": "vbr", "rel": "A"},
            {"uuid": "s3", "src": "vbr", "rel": "B"}]
    sibs = {("vbr", "A"): [{"uuid": f"a{i:02d}", "fact": f"A {i}", "episodes": ["e"]}
                           for i in range(20)],
            ("vbr", "B"): [{"uuid": f"b{i}", "fact": f"B {i}", "episodes": ["e"]}
                           for i in range(5)]}
    monkeypatch.setattr(lm, "Provenance", _Prov)
    seeds = _SEEDS + [_fact("s3", "VBR B")]
    drv = _GraphDriver(meta=meta, siblings=sibs)
    s = _settings(list_relation_group_cap=3, list_relation_total_cap=5)
    out = await lm.relation_siblings(drv, seeds, scope=Scope(), settings=s,
                                     exclude=set(), degraded=[])
    # Group A (most frequent) first. Its fetch stops at 12 of 20 edges (a19..a08, in
    # the fake's reverse order); sorted by uuid, its cap of 3 keeps a08-a10. The total
    # cap of 5 then cuts group B after b0, b1.
    assert [r["fact_uuid"] for r in out] == ["a08", "a09", "a10", "b0", "b1"]
    assert all(c[1]["fetch"] == 12 for c in drv.calls if c[0] is lm.SIBLINGS_QUERY)


def test_total_sibling_cap_default():
    assert _settings().list_relation_total_cap == 600


async def test_a_failed_scope_filter_degrades_once(monkeypatch):
    drv = _GraphDriver(meta=_META, siblings=_SIBS, fail_scope=True)
    monkeypatch.setattr(lm, "Provenance", _Prov)
    degraded: list[str] = []
    out = await lm.relation_siblings(drv, _SEEDS, scope=Scope(vendors=("Veeam",)),
                                     settings=_settings(), exclude={"s1", "s2"},
                                     degraded=degraded)
    assert out == [] and degraded == [lm.RELATION_EXPANSION_DEGRADED]


async def test_unscoped_relation_siblings_keep_every_sibling(monkeypatch):
    drv = _GraphDriver(meta=_META, siblings=_SIBS)
    monkeypatch.setattr(lm, "Provenance", _Prov)
    out = await lm.relation_siblings(drv, _SEEDS, scope=Scope(), settings=_settings(),
                                     exclude={"s1", "s2"}, degraded=[])
    assert [r["fact_uuid"] for r in out] == ["x0", "x1"]
    assert all(c[0] is not _EPISODES_AMONG for c in drv.calls)


async def test_one_failed_group_query_degrades_once_and_keeps_the_others(monkeypatch):
    meta = _META + [{"uuid": "s3", "src": "vbr", "rel": "Backs up"},
                    {"uuid": "s4", "src": "vbr", "rel": "Restores to"}]
    sibs = {**_SIBS, ("vbr", "Backs up"): [{"uuid": "y1", "fact": "VBR backs up Y",
                                            "episodes": ["e9"]}]}
    drv = _GraphDriver(meta=meta, siblings=sibs, fail={("vbr", "Supports"),
                                                       ("vbr", "Restores to")})
    monkeypatch.setattr(lm, "Provenance", _Prov)
    degraded: list[str] = []
    seeds = _SEEDS + [_fact("s3", "VBR backs up X"), _fact("s4", "VBR restores to Z")]
    out = await lm.relation_siblings(drv, seeds, scope=Scope(), settings=_settings(),
                                     exclude={"s1", "s2", "s3", "s4"}, degraded=degraded)
    assert [r["fact_uuid"] for r in out] == ["y1"]
    assert degraded == [lm.RELATION_EXPANSION_DEGRADED]


async def test_group_queries_run_concurrently_and_cancellation_propagates(monkeypatch):
    started = []
    gate = asyncio.Event()

    class _Slow(_GraphDriver):
        async def execute_query(self, query, **p):
            if query is lm.SIBLINGS_QUERY:
                started.append(p["rel"])
                if len(started) == 2:
                    gate.set()
                await gate.wait()             # deadlocks unless both run at once
                raise asyncio.CancelledError() if p["rel"] == "B" else RuntimeError("x")
            return await super().execute_query(query, **p)

    meta = [{"uuid": "s1", "src": "vbr", "rel": "A"}, {"uuid": "s2", "src": "vbr", "rel": "B"}]
    drv = _Slow(meta=meta, siblings={})
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(lm.relation_siblings(drv, _SEEDS, scope=Scope(),
                                                    settings=_settings(), exclude=set(),
                                                    degraded=[]), timeout=2)
    assert sorted(started) == ["A", "B"]


def test_mmr_prefers_a_new_item_over_a_near_copy():
    a, a2, b = _fact("a", "A item"), _fact("a2", "A item again"), _fact("b", "B item")
    vecs = {"a": [1.0, 0.0], "a2": [0.99, 0.01], "b": [0.0, 1.0]}
    got = lm.select_diverse([(a, 0.9), (a2, 0.89), (b, 0.6)], vecs, k=2, lam=0.5)
    assert [r["fact_uuid"] for r in got] == ["a", "b"]


def test_mmr_accepts_decoded_float32_arrays():
    a, a2, b = _fact("a", "A item"), _fact("a2", "A item again"), _fact("b", "B item")
    vecs = {u: np.asarray(v, dtype=np.float32)
            for u, v in {"a": [127, 0], "a2": [126, 1], "b": [0, 127]}.items()}
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

    async def _siblings(driver, seeds, *, scope, settings, exclude, degraded):
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
    # Round one failed (no scores at all): the siblings follow in retrieval order
    # without a second rerank call.
    assert [c["rerank_docs"] for c in calls if "rerank_docs" in c] == [
        ["vSphere is supported."]]
    assert [r["fact_uuid"] for r in got.results] == ["a", "x1"]
    assert got.degraded == [lm.FINAL_RERANK_DEGRADED]


def test_mmr_without_scores_keeps_retrieval_order_even_with_vectors():
    facts = [(_fact(u, u), None) for u in ("a", "b", "c", "z")]
    vecs = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.98, 0.02], "z": [0.0, 1.0]}
    assert [r["fact_uuid"] for r in lm.select_diverse(facts, vecs, k=2, lam=0.5)] == ["a", "b"]


async def test_round_two_reranks_only_the_siblings_and_merges_by_score(patched, monkeypatch):
    install, calls = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported."),
                          _fact("b", "Hyper-V is supported.")]}, expansion="[]")
    scores = {"vSphere is supported.": 0.9, "Hyper-V is supported.": 0.3,
              "Sangfor aSV is supported.": 0.5, "Morpheus is supported.": 0.1}

    async def _siblings(driver, seeds, *, scope, settings, exclude, degraded):
        return [_fact("x1", "Sangfor aSV is supported."), _fact("x2", "Morpheus is supported.")]

    async def _scored(rq, documents, *, top_k, settings):
        calls.append({"rerank_docs": list(documents)})
        return [(i, scores[d]) for i, d in enumerate(documents)]

    monkeypatch.setattr(lm, "relation_siblings", _siblings)
    monkeypatch.setattr(lm, "rerank", _scored)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert [c["rerank_docs"] for c in calls if "rerank_docs" in c] == [
        ["vSphere is supported.", "Hyper-V is supported."],
        ["Sangfor aSV is supported.", "Morpheus is supported."]]
    assert [r["fact_uuid"] for r in got.results] == ["a", "x1", "b", "x2"]
    assert got.degraded == []


async def test_a_failed_round_two_keeps_round_one_scores_and_puts_siblings_last(
        patched, monkeypatch):
    install, calls = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported."),
                          _fact("b", "Hyper-V is supported.")]}, expansion="[]")

    async def _siblings(driver, seeds, *, scope, settings, exclude, degraded):
        return [_fact("x1", "Sangfor aSV is supported.")]

    async def _first_only(rq, documents, *, top_k, settings):
        if "Sangfor aSV is supported." in documents:
            return None
        return [(1, 0.8), (0, 0.4)]

    monkeypatch.setattr(lm, "relation_siblings", _siblings)
    monkeypatch.setattr(lm, "rerank", _first_only)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert [r["fact_uuid"] for r in got.results] == ["b", "a", "x1"]
    assert got.degraded == [lm.FINAL_RERANK_DEGRADED]


async def test_a_failed_embedding_fetch_is_recorded_and_noted(patched, monkeypatch):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")
    monkeypatch.setattr(lm, "_fact_vectors", _REAL_FACT_VECTORS)

    class _Broken:
        async def execute_query(self, query, **p):
            raise RuntimeError("neo4j down")

    got = await lm.retrieve_list(None, _Broken(), client, "m", q=q, scope=Scope(),
                                 group_id="g", settings=_settings())
    assert [r["fact_uuid"] for r in got.results] == ["a"]
    assert got.degraded == [lm.DIVERSITY_DEGRADED]
    note = lm.degraded_note(got.degraded)
    assert note == "Repeated items could not be filtered out; this list may be incomplete."


async def test_single_answers_collapse_at_their_own_threshold(monkeypatch):
    seen = {}

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, **kw):
        return {"results": [_fact("a", "vSphere is supported.")], "count": 1}

    async def _collapse(driver, results, *, cosine, vectors=None):
        seen["cosine"], seen["vectors"] = cosine, vectors
        return results

    monkeypatch.setattr(synth_mod.search_mod, "search_local", _search)
    monkeypatch.setattr(synth_mod, "collapse_near_duplicates", _collapse)
    s = _settings(local_single_dedup_cosine=0.92, list_dedup_cosine=0.96)
    await synth_mod.answer_local(None, None, _Client("- vSphere [1]"), "s", q="q",
                                 group_id="g", settings=s)
    assert seen["cosine"] == 0.92 and seen["vectors"] is None     # no cache on this path


class _EmbeddingDriver:
    """Counts every uuid asked of EMBEDDINGS_QUERY; vectors keep all facts distinct."""
    def __init__(self, uuids):
        self.vectors = {u: [1.0 if j == i else 0.0 for j in range(len(uuids))]
                        for i, u in enumerate(uuids)}
        self.asked: list[str] = []
        self.queries = 0

    async def execute_query(self, query, **p):
        assert query is EMBEDDINGS_QUERY
        self.queries += 1
        self.asked += list(p["uuids"])
        return ([{"uuid": u, "embedding": self.vectors.get(u)} for u in p["uuids"]],
                None, None)


async def test_retrieve_list_fetches_each_embedding_at_most_once(monkeypatch, caplog):
    q = "Which hypervisors does Veeam Backup & Replication support?"
    per_query = {q: [_fact("a", "vSphere is supported."), _fact("b", "Hyper-V is supported.")],
                 "q-ahv": [_fact("b", "Hyper-V is supported."),
                           _fact("c", "Nutanix AHV is supported.")]}

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, **kw):
        return {"results": per_query.get(q, []), "count": 0}

    async def _rerank(rq, documents, *, top_k, settings):
        return [(i, 1.0 - i / 100) for i in range(len(documents))]

    async def _siblings(driver, seeds, *, scope, settings, exclude, degraded):
        return [_fact("x1", "Sangfor aSV is supported."), _fact("a", "vSphere is supported.")]

    async def _no_sweep(driver, q, *, scope, group_id, settings):
        return []

    monkeypatch.setattr(lm, "search_local", _search)
    monkeypatch.setattr(lm, "rerank", _rerank)
    monkeypatch.setattr(lm, "relation_siblings", _siblings)
    monkeypatch.setattr(lm, "keyword_sweep", _no_sweep)
    drv = _EmbeddingDriver(["a", "b", "c", "x1"])
    with caplog.at_level(logging.INFO, logger="answer_api.list_mode"):
        got = await lm.retrieve_list(None, drv, _Client('["q-ahv"]'), "m", q=q, scope=Scope(),
                                     group_id="g", settings=_settings())
    assert sorted(drv.asked) == ["a", "b", "c", "x1"]      # once each, across both
    assert drv.queries == 2                                # collapses and the MMR selection
    assert {r["fact_uuid"] for r in got.results} == {"a", "b", "c", "x1"}
    assert got.degraded == []
    [line] = [r.getMessage() for r in caplog.records if "list stages" in r.getMessage()]
    for stage in ("expansion=", "searches=", "collapse=", "rerank=", "relations=",
                  "selection="):
        assert stage in line


async def test_swept_facts_join_round_one(patched, monkeypatch):
    install, calls = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")

    async def _sweep(driver, qq, *, scope, group_id, settings):
        return [_fact("sw", "Restore to Sangfor aSV is supported.")]

    monkeypatch.setattr(lm, "keyword_sweep", _sweep)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert "sw" in [r["fact_uuid"] for r in got.results]
    rerank_calls = [c["docs"] for c in calls if "docs" in c]
    assert rerank_calls and "Restore to Sangfor aSV is supported." in rerank_calls[0]


async def test_a_failed_sweep_is_degraded_once_and_keeps_the_per_query_results(
        patched, monkeypatch):
    install, _ = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")

    async def _boom(driver, qq, *, scope, group_id, settings):
        raise RuntimeError("fulltext index down")

    monkeypatch.setattr(lm, "keyword_sweep", _boom)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert got.degraded.count(lm.SWEEP_DEGRADED) == 1
    assert [r["fact_uuid"] for r in got.results] == ["a"]
    assert "keyword search could not be completed" in lm.degraded_note([lm.SWEEP_DEGRADED])


async def test_a_failed_sweep_alone_never_triggers_every_retrieval_failed(patched, monkeypatch):
    install, _ = patched
    q = "Which hypervisors does Veeam Backup & Replication support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")

    async def _boom(driver, qq, *, scope, group_id, settings):
        raise RuntimeError("fulltext index down")

    monkeypatch.setattr(lm, "keyword_sweep", _boom)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert [r["fact_uuid"] for r in got.results] == ["a"]
    assert got.degraded == [lm.SWEEP_DEGRADED]


_GRADED_FACTS = {
    "VBR supports backups created by Veeam Agent for Microsoft Windows for restore to "
    "Sangfor aSV.": ("sf", 0.9),
    "VBR can restore physical machines from backups created by Veeam Agents to the HPE "
    "Morpheus VM Essentials environment.": ("hpe", 0.5),
    "Restore of Windows agent backups to Oracle Linux KVM is possible.": ("low", 0.05),
}


async def _scored_rerank(rq, documents, *, top_k, settings):
    return [(i, _GRADED_FACTS[d][1]) for i, d in enumerate(documents)][:top_k]


async def test_list_retrieval_carries_the_final_rerank_scores(patched, monkeypatch):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({q: [_fact(u, text) for text, (u, _s) in _GRADED_FACTS.items()]},
                     expansion="[]")
    monkeypatch.setattr(lm, "rerank", _scored_rerank)
    got = await lm.retrieve_list(None, None, client, "m", q=q, scope=Scope(), group_id="g",
                                 settings=_settings())
    assert got.scores == {"sf": 0.9, "hpe": 0.5, "low": 0.05}


async def test_answer_list_grades_bold_items_before_citing(patched, monkeypatch):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({q: [_fact(u, text) for text, (u, _s) in _GRADED_FACTS.items()]},
                     expansion="[]")
    monkeypatch.setattr(lm, "rerank", _scored_rerank)
    synth = _Client("**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]\n"
                    "- **HPE Morpheus VM Essentials** — restore as a VM [2]\n"
                    "- **Oracle Linux KVM** — restore as a VM [3]")
    out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings(list_item_floor=0.1))
    assert "- **<item name>** —" in synth.prompts[0]
    assert [(g["item"], g["grade"], g["missing_terms"], g["fact_uuids"])
            for g in out["item_grades"]] == [
        ("Sangfor aSV", "confirmed", [], ["sf"]),
        ("HPE Morpheus VM Essentials", "needs_validation", ["windows"], ["hpe"]),
        ("Oracle Linux KVM", "dropped", [], ["low"])]
    assert "Oracle" not in out["answer"]
    assert ("restore as a VM [2] *(needs validation: the cited source does not mention "
            "Windows)*") in out["answer"]
    assert [c["fact_uuid"] for c in out["citations"]] == ["sf", "hpe"]


async def test_the_refusal_paths_carry_empty_item_grades(patched):
    install, _ = patched
    out = await lm.answer_list(None, None, install({}, expansion="[]"), "m", _Client("x"), "s",
                               q="Which X?", scope=Scope(), group_id="g", settings=_settings())
    assert out["item_grades"] == []
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")
    out = await lm.answer_list(None, None, client, "m", _Client(""), "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings())
    assert out["citations"] == [] and out["item_grades"] == []


# --- Fix wave 2 -------------------------------------------------------------------

async def test_an_answer_whose_every_item_is_dropped_refuses(patched, monkeypatch, caplog):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({q: [_fact(u, text) for text, (u, _s) in _GRADED_FACTS.items()]},
                     expansion="[]", fail=())
    monkeypatch.setattr(lm, "rerank", _scored_rerank)
    synth = _Client("**Hypervisors**\n- **Oracle Linux KVM** — restore as a VM [3]\n"
                    "- plain bullet [1]")
    with caplog.at_level(logging.INFO, logger="answer_api.list_mode"):
        out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                                   group_id="g", settings=_settings(list_item_floor=0.1))
    assert out["answer"] == synth_mod._REFUSAL
    assert out["citations"] == [] and out["cited"] == 0 and out["applies_to"] == []
    assert [g["grade"] for g in out["item_grades"]] == ["dropped"]
    assert out["items_ungraded"] == 1
    assert out["list_mode"] is True and out["expansion_queries"] == []
    assert "every item dropped by grading; taking the normal route" in caplog.text


async def test_an_all_dropped_answer_keeps_the_degraded_entries(patched, monkeypatch):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({q: [_fact(u, text) for text, (u, _s) in _GRADED_FACTS.items()]},
                     fail=("q-ec2",))
    monkeypatch.setattr(lm, "rerank", _scored_rerank)
    synth = _Client("- **Oracle Linux KVM** — restore as a VM [3]")
    out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings(list_item_floor=0.1))
    assert out["citations"] == [] and out["degraded"] == ["retrieval failed for: q-ec2"]


class _SeqClient:
    """Replies with (content, finish_reason) in turn; records each max_tokens."""

    def __init__(self, replies):
        self.replies, self.max_tokens = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.max_tokens.append(kw["max_tokens"])
        content, finish = self.replies.pop(0)
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


async def test_a_list_cut_short_once_is_retried_wider_without_a_note(patched, caplog):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")
    synth = _SeqClient([("- vSph", "length"), ("- vSphere [1]", "stop")])
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                                   group_id="g", settings=_settings())
    assert synth.max_tokens == [10000, 16000]
    assert out["answer"] == "- vSphere [1]" and "degraded" not in out
    assert "max_tokens=10000" in caplog.text and "chars" in caplog.text


async def test_a_list_cut_short_twice_keeps_the_text_and_says_so(patched, caplog):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")
    synth = _SeqClient([("- vSph", "length"), ("- vSphere [1]\n- Hyp", "length")])
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                                   group_id="g", settings=_settings())
    assert synth.max_tokens == [10000, 16000]
    assert out["degraded"] == [lm.LIST_TRUNCATED]
    assert out["answer"].startswith("- vSphere [1]\n- Hyp")
    assert out["answer"].endswith(
        "The answer was cut short; this list may be incomplete.")
    assert "still cut short at max_tokens=16000" in caplog.text


def test_the_truncation_note_is_fixed_wording():
    assert lm.LIST_TRUNCATED == "list answer cut short"
    assert lm.degraded_note([lm.LIST_TRUNCATED]) == (
        "The answer was cut short; this list may be incomplete.")


async def test_complete_with_finish_reason_returns_the_reply_finish_reason_and_budget():
    got = await synth_mod._complete_with_finish_reason(
        _SeqClient([("text", "length")]), "m", "p", max_tokens=10)
    assert tuple(got) == ("text", "length", 10)
    seq = _SeqClient([(None, "length"), ("  ", "stop")])
    got = await synth_mod._complete_with_finish_reason(seq, "m", "p", max_tokens=10)
    assert tuple(got) == (None, "stop", 30) and seq.max_tokens == [10, 30]
    seq = _SeqClient([(None, "length"), ("wide", "length")])
    got = await synth_mod._complete_with_finish_reason(seq, "m", "p", max_tokens=10)
    assert tuple(got) == ("wide", "length", 30)


async def test_completion_helpers_accept_replies_without_a_finish_reason():
    class _Bare:
        def __init__(self, content):
            self.content = content
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **kw):
            choice = SimpleNamespace(message=SimpleNamespace(content=self.content))
            return SimpleNamespace(choices=[choice])

    assert await synth_mod._complete_or_none(_Bare("ok"), "m", "p", max_tokens=5) == "ok"
    assert await synth_mod._complete_or_none(_Bare(None), "m", "p", max_tokens=5) is None
    got = await synth_mod._complete_with_finish_reason(_Bare("ok"), "m", "p", max_tokens=5)
    assert tuple(got) == ("ok", None, 5)


async def test_a_list_cut_short_at_a_budget_already_wide_is_not_retried(patched, caplog):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")
    # Empty at 10000 -> the completion's own retry at 30000 -> cut short there.
    synth = _SeqClient([(None, "length"), ("- vSphere [1]\n- Hyp", "length")])
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                                   group_id="g", settings=_settings())
    assert synth.max_tokens == [10000, 30000]
    assert out["degraded"] == [lm.LIST_TRUNCATED]
    assert out["answer"].startswith("- vSphere [1]\n- Hyp")
    assert "max_tokens=30000" in caplog.text and "no wider retry" in caplog.text


@pytest.mark.parametrize("retry_reply", [("- vS", "length"), (None, "length")])
async def test_a_list_cut_short_twice_keeps_the_longer_partial_text(patched, retry_reply):
    install, _ = patched
    q = "Which hypervisors does Veeam support?"
    client = install({q: [_fact("a", "vSphere is supported.")]}, expansion="[]")
    replies = [("- vSphere [1]\n- Hyp", "length"), retry_reply]
    if retry_reply[0] is None:            # unusable at 16000 -> its own retry at 48000
        replies.append((None, "length"))
    synth = _SeqClient(replies)
    out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings())
    assert synth.max_tokens[:2] == [10000, 16000]
    assert out["degraded"] == [lm.LIST_TRUNCATED]
    assert out["answer"].startswith("- vSphere [1]\n- Hyp")


async def test_final_order_batches_run_concurrently(monkeypatch):
    started: list[int] = []
    both = asyncio.Event()

    async def _rerank(q, docs, *, top_k, settings):
        started.append(len(docs))
        if len(started) == 2:
            both.set()
        # Each batch waits until BOTH have started: sequential calls would hang here.
        await asyncio.wait_for(both.wait(), timeout=2)
        return [(i, float(int(d.split()[-1]))) for i, d in enumerate(docs)]

    monkeypatch.setattr(lm, "rerank", _rerank)
    monkeypatch.setattr(lm, "rerank_configured", lambda s: True)
    facts = [_fact(f"f{i}", f"fact {i}") for i in range(4)]
    degraded: list[str] = []
    out = await lm._final_order("q", facts, settings=_settings(list_rerank_batch=2),
                                degraded=degraded)
    assert sorted(started) == [2, 2] and degraded == []
    assert [f["fact_uuid"] for f, _ in out] == ["f3", "f2", "f1", "f0"]


@pytest.mark.parametrize("how", ["raises", "none"])
async def test_one_failed_rerank_batch_degrades_the_whole_order(monkeypatch, how):
    async def _rerank(q, docs, *, top_k, settings):
        if docs[0] == "fact 2":
            if how == "raises":
                raise RuntimeError("batch down")
            return None
        return [(i, 1.0) for i in range(len(docs))]

    monkeypatch.setattr(lm, "rerank", _rerank)
    monkeypatch.setattr(lm, "rerank_configured", lambda s: True)
    facts = [_fact(f"f{i}", f"fact {i}") for i in range(4)]
    degraded: list[str] = []
    out = await lm._final_order("q", facts, settings=_settings(list_rerank_batch=2),
                                degraded=degraded)
    assert degraded == [lm.FINAL_RERANK_DEGRADED]
    assert [(f["fact_uuid"], s) for f, s in out] == [(f"f{i}", None) for i in range(4)]


@pytest.mark.parametrize("reply", [[], [(7, 0.9), (-1, 0.8)]])
async def test_a_rerank_reply_with_no_valid_index_is_a_failed_rerank(monkeypatch, caplog, reply):
    async def _rerank(q, docs, *, top_k, settings):
        return list(reply)

    monkeypatch.setattr(lm, "rerank", _rerank)
    monkeypatch.setattr(lm, "rerank_configured", lambda s: True)
    facts = [_fact(f"f{i}", f"fact {i}") for i in range(3)]
    degraded: list[str] = []
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        out = await lm._final_order("q", facts, settings=_settings(), degraded=degraded)
    assert degraded == [lm.FINAL_RERANK_DEGRADED]
    assert [s for _, s in out] == [None, None, None]
    assert "no valid index" in caplog.text


async def test_answer_list_reports_ungraded_bullets_and_passes_the_scope(patched, monkeypatch):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({q: [_fact(u, text) for text, (u, _s) in _GRADED_FACTS.items()]},
                     expansion="[]")
    monkeypatch.setattr(lm, "rerank", _scored_rerank)
    seen = {}
    real = lm.grade_items

    def _spy(*a, **kw):
        seen["scope"] = kw.get("scope")
        return real(*a, **kw)

    monkeypatch.setattr(lm, "grade_items", _spy)
    synth = _Client("- **Sangfor aSV** — restore as a VM [1]\n- plain bullet [2]")
    scope = Scope(vendors=("Veeam",))
    out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=scope,
                               group_id="g", settings=_settings(list_item_floor=0.1))
    assert out["items_ungraded"] == 1 and seen["scope"] is scope
    out = await lm.answer_list(None, None, install({}, expansion="[]"), "m", _Client("x"),
                               "s", q="Which X?", scope=Scope(), group_id="g",
                               settings=_settings())
    assert out["items_ungraded"] == 0


async def test_a_bulleted_group_layout_is_answered_not_refused(patched, monkeypatch):
    install, _ = patched
    q = "Where can I restore a Windows agent backup?"
    client = install({q: [_fact(u, text) for text, (u, _s) in _GRADED_FACTS.items()]},
                     expansion="[]")
    monkeypatch.setattr(lm, "rerank", _scored_rerank)
    synth = _Client("- **Hypervisors**\n  - **Sangfor aSV** — restore as a VM [1]\n"
                    "  - **Oracle Linux KVM** — restore as a VM [3]\n"
                    "- **Clouds:**\n  - **Oracle Linux KVM** — again [3]")
    out = await lm.answer_list(None, None, client, "m", synth, "s", q=q, scope=Scope(),
                               group_id="g", settings=_settings(list_item_floor=0.1))
    assert [(g["item"], g["grade"]) for g in out["item_grades"]] == [
        ("Sangfor aSV", "confirmed"), ("Oracle Linux KVM", "dropped"),
        ("Oracle Linux KVM", "dropped")]
    # (_finalize_answer collapses space runs, indentation included: compare stripped.)
    assert [ln.strip() for ln in out["answer"].splitlines()] == [
        "- **Hypervisors**", "- **Sangfor aSV** — restore as a VM [1]"]
    assert [c["fact_uuid"] for c in out["citations"]] == ["sf"]


# --- Completeness pass (spec §13.3) -----------------------------------------------

class _Replies:
    """Replies with each content in turn (an exception instance is raised); a
    (content, finish_reason) tuple reply reports that finish reason instead of "stop".
    Records every call's keyword arguments."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        content, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


_COMPLETE_FACTS = {
    "sf": ("VBR supports backups created by Veeam Agent for Microsoft Windows for restore "
           "to Sangfor aSV.", 0.9),
    "olvm": ("VBR can restore backups created by Veeam Agent for Microsoft Windows to "
             "Oracle Linux Virtualization Manager.", 0.8),
    "hpe": ("VBR can restore physical machines from backups created by Veeam Agents to the "
            "HPE Morpheus VM Essentials environment.", 0.7),
    "lim": ("Restore of Windows agent backups to Xen is not supported.", 0.3),
}
_COMPLETE_Q = "Where can I restore a Windows agent backup?"
_COMPLETE_SYNTH = ("**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]\n\n"
                   "**Not supported / limitations**\n- Xen is not supported [4]")


async def _complete_rerank(rq, documents, *, top_k, settings):
    score = {text: s for text, s in _COMPLETE_FACTS.values()}
    return [(i, score[d]) for i, d in enumerate(documents)][:top_k]


async def _answer_with_completion(patched, monkeypatch, reply, *, synth=_COMPLETE_SYNTH,
                                  **settings):
    """Supported markers: [1] sf, [2] olvm, [3] hpe; limitation [4] lim. The map tier
    answers the expansion with [] and the completeness pass with `reply` -- a list of
    replies when the pass itself makes more than one call (e.g. the internal retry on
    no usable content)."""
    install, _ = patched
    install({_COMPLETE_Q: [_fact(u, text) for u, (text, _s) in _COMPLETE_FACTS.items()]})
    monkeypatch.setattr(lm, "rerank", _complete_rerank)
    replies = reply if isinstance(reply, list) else [reply]
    map_client = _Replies("[]", *replies)
    out = await lm.answer_list(None, None, map_client, "m", _Client(synth), "s",
                               q=_COMPLETE_Q, scope=Scope(), group_id="g",
                               settings=_settings(list_item_floor=0.1, **settings))
    return out, map_client


async def test_missing_items_are_added_before_the_limitations_and_graded(
        patched, monkeypatch, caplog):
    reply = ('Sure: [{"item": "Oracle Linux Virtualization Manager", "description": '
             '"restore agent backups as VMs", "markers": [2]}, '
             '{"item": "HPE Morpheus VM Essentials", "description": "restore physical '
             'machines", "markers": [3]}]')
    with caplog.at_level(logging.INFO, logger="answer_api.list_mode"):
        out, map_client = await _answer_with_completion(patched, monkeypatch, reply)
    call = map_client.calls[1]
    assert call["model"] == "m" and call["temperature"] == 0 and call["max_tokens"] == 8000
    prompt = call["messages"][0]["content"]
    assert _COMPLETE_Q in prompt and "Sangfor aSV" in prompt
    assert "[2] " in prompt and "[3] " in prompt
    assert "[1] " not in prompt and "[4] " not in prompt
    assert "JSON" in prompt and "URL" in prompt and "15 words" in prompt
    lines = [ln.strip() for ln in out["answer"].splitlines()]
    also = lines.index("### Also documented")
    assert lines.index("- **Sangfor aSV** — restore as a VM [1]") < also
    assert lines[also + 1] == ("- **Oracle Linux Virtualization Manager** — restore agent "
                               "backups as VMs [2]")
    assert lines[also + 2].startswith("- **HPE Morpheus VM Essentials** — restore physical "
                                      "machines [3] *(needs validation")
    assert also < lines.index("**Not supported / limitations**")
    assert [(g["item"], g["grade"]) for g in out["item_grades"]] == [
        ("Sangfor aSV", "confirmed"),
        ("Oracle Linux Virtualization Manager", "confirmed"),
        ("HPE Morpheus VM Essentials", "needs_validation")]
    assert [c["fact_uuid"] for c in out["citations"]] == ["sf", "olvm", "hpe", "lim"]
    assert "degraded" not in out
    assert "list completeness pass: 2 uncited facts, 2 items added" in caplog.text


async def test_added_items_without_a_limitations_section_are_appended(patched, monkeypatch):
    reply = '[{"item": "OLVM", "description": "restore as a VM", "markers": [2]}]'
    out, _ = await _answer_with_completion(
        patched, monkeypatch, reply, synth="- **Sangfor aSV** — restore as a VM [1]")
    lines = [ln.strip() for ln in out["answer"].splitlines() if ln.strip()]
    assert lines[-2:] == ["### Also documented", "- **OLVM** — restore as a VM [2]"]


async def test_a_completion_naming_an_existing_item_is_ignored(patched, monkeypatch):
    reply = ('[{"item": " sangfor ASV ", "description": "again", "markers": [2]}, '
             '{"item": "Sangfor aSV", "description": "again", "markers": [3]}]')
    out, map_client = await _answer_with_completion(patched, monkeypatch, reply)
    assert len(map_client.calls) == 2
    assert "Also documented" not in out["answer"]
    assert [c["fact_uuid"] for c in out["citations"]] == ["sf", "lim"]
    assert "degraded" not in out


async def test_completion_markers_outside_the_uncited_set_are_dropped(patched, monkeypatch):
    reply = ('[{"item": "Oracle Linux Virtualization Manager", "description": "restore '
             'as a VM [1]", "markers": [1, 4, 99, true, "3", 2]}, '
             '{"item": "Cited Already", "description": "x", "markers": [1, 4]}, '
             '{"item": "", "description": "x", "markers": [3]}, '
             '{"item": "No Description", "markers": [3]}, "plain string", '
             '{"item": "Many", "description": "x", "markers": [3, 2, 3, 2]}]')
    out, _ = await _answer_with_completion(patched, monkeypatch, reply)
    lines = [ln.strip() for ln in out["answer"].splitlines()]
    also = lines.index("### Also documented")
    assert lines[also + 1] == ("- **Oracle Linux Virtualization Manager** — restore as a "
                               "VM [2]")
    assert lines[also + 2].startswith("- **Many** — x [3][2]")
    assert "Cited Already" not in out["answer"] and "No Description" not in out["answer"]


@pytest.mark.parametrize("reply", ["not json", '{"item": "x"}', "[{]",
                                   RuntimeError("map tier down")])
async def test_a_failed_completeness_pass_degrades_and_keeps_the_first_answer(
        patched, monkeypatch, caplog, reply):
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        out, map_client = await _answer_with_completion(patched, monkeypatch, reply)
    assert len(map_client.calls) == 2
    assert map_client.calls[1]["max_tokens"] == 8000
    assert out["degraded"] == [lm.COMPLETENESS_DEGRADED]
    assert "Also documented" not in out["answer"]
    assert out["answer"].startswith("**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]")
    assert out["answer"].endswith(
        "Some documented items may be missing; this list may be incomplete.")
    assert "list completeness pass" in caplog.text


async def test_a_completeness_pass_with_no_usable_content_retries_then_degrades(
        patched, monkeypatch, caplog):
    """`_complete_with_finish_reason`'s own retry on no usable content (None, None)
    still ends in a degraded pass that keeps the first answer."""
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        out, map_client = await _answer_with_completion(patched, monkeypatch, [None, None])
    assert len(map_client.calls) == 3
    assert map_client.calls[1]["max_tokens"] == 8000
    assert map_client.calls[2]["max_tokens"] == 24000
    assert out["degraded"] == [lm.COMPLETENESS_DEGRADED]
    assert "Also documented" not in out["answer"]
    assert "list completeness pass failed" in caplog.text


async def test_a_completeness_pass_cut_short_degrades_and_keeps_the_first_answer(
        patched, monkeypatch, caplog):
    partial = '[{"item": "Oracle Linux Virtualization Manager", "description": "x"'
    with caplog.at_level(logging.WARNING, logger="answer_api.list_mode"):
        out, map_client = await _answer_with_completion(
            patched, monkeypatch, (partial, "length"))
    assert len(map_client.calls) == 2
    assert map_client.calls[1]["max_tokens"] == 8000
    assert out["degraded"] == [lm.COMPLETENESS_DEGRADED]
    assert "Also documented" not in out["answer"]
    assert out["answer"].startswith("**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]")
    assert out["answer"].endswith(
        "Some documented items may be missing; this list may be incomplete.")
    assert (f"list completeness pass cut short at 8000 tokens ({len(partial)} chars); "
           "keeping the first answer") in caplog.text


async def test_the_completeness_pass_is_skipped_when_disabled(patched, monkeypatch):
    out, map_client = await _answer_with_completion(
        patched, monkeypatch, "unused", list_completeness_pass=False)
    assert len(map_client.calls) == 1
    assert "Also documented" not in out["answer"] and "degraded" not in out


async def test_the_completeness_pass_is_skipped_without_uncited_facts(patched, monkeypatch):
    synth = ("- **Sangfor aSV** — restore as a VM [1]\n- **OLVM** — restore [2]\n"
             "- **HPE Morpheus VM Essentials** — restore [3]")
    out, map_client = await _answer_with_completion(patched, monkeypatch, "unused",
                                                    synth=synth)
    assert len(map_client.calls) == 1
    assert "degraded" not in out


def test_the_completeness_note_is_fixed_wording():
    assert lm.COMPLETENESS_DEGRADED == "completeness check unavailable"
    assert lm.degraded_note([lm.COMPLETENESS_DEGRADED]) == (
        "Some documented items may be missing; this list may be incomplete.")


# --- Completeness pass, fix round 1 ------------------------------------------------

async def test_a_refused_synthesis_skips_the_pass_and_the_router_falls_through(
        patched, monkeypatch):
    out, map_client = await _answer_with_completion(
        patched, monkeypatch, '[{"item": "OLVM", "description": "x", "markers": [2]}]',
        synth=synth_mod._REFUSAL)
    assert len(map_client.calls) == 1                 # the expansion only
    assert out["citations"] == [] and out["cited"] == 0
    assert "Also documented" not in out["answer"]

    # Through the router: the real answer_list's refusal falls to the normal route.
    modes = []

    async def _run_mode(mode, *a, **kw):
        modes.append(mode)
        return mode, None, {"answer": "normal", "citations": [
            {"marker": 1, "fact_uuid": "x", "sources": []}]}

    async def _classify(q, **kw):
        return "drift", "llm"

    monkeypatch.setattr(router_mod, "_run_mode", _run_mode)
    monkeypatch.setattr(router_mod, "classify", _classify)
    router_map = _Replies("[]", '[{"item": "OLVM", "description": "x", "markers": [2]}]')
    env = await router_mod.answer_router(
        None, None, None, _Client(synth_mod._REFUSAL), "s", router_map, "m", None, "c",
        q=_COMPLETE_Q, mode_override=None, scope=Scope(),
        settings=_settings(list_item_floor=0.1),
        shape=QuestionShape("single", (Facet("", Scope()),)))
    assert len(router_map.calls) == 1
    assert modes == ["drift"] and env["answer"] == "normal"
    assert env["routing"].get("path") != "list"


async def test_the_duplicate_check_normalises_both_sides(patched, monkeypatch):
    synth = ("**Hypervisors**\n- **Sangfor  aSV:** — restore as a VM [1]\n\n"
             "**Not supported / limitations**\n- Xen is not supported [4]")
    reply = ('[{"item": "sangfor aSV", "description": "again", "markers": [2]}, '
             '{"item": "**Sangfor   aSV**:", "description": "again", "markers": [3]}, '
             '{"item": "OLVM:", "description": "**restore** as a *VM*", "markers": [2]}]')
    out, _ = await _answer_with_completion(patched, monkeypatch, reply, synth=synth)
    lines = [ln.strip() for ln in out["answer"].splitlines()]
    also = lines.index("### Also documented")
    assert lines[also + 1].startswith("- **OLVM** — restore as a VM [2]")
    assert lines[also + 2] == ""
    assert "again" not in out["answer"]


async def test_a_truncated_answer_still_gets_the_pass_and_both_notes(patched, monkeypatch):
    install, _ = patched
    install({_COMPLETE_Q: [_fact(u, text) for u, (text, _s) in _COMPLETE_FACTS.items()]})
    monkeypatch.setattr(lm, "rerank", _complete_rerank)
    partial = "**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]\n- **Ora"
    synth = _SeqClient([(partial[:-4], "length"), (partial, "length")])
    map_client = _Replies("[]", '[{"item": "Oracle Linux Virtualization Manager", '
                          '"description": "restore agent backups as VMs", "markers": [2]}]')
    out = await lm.answer_list(None, None, map_client, "m", synth, "s", q=_COMPLETE_Q,
                               scope=Scope(), group_id="g",
                               settings=_settings(list_item_floor=0.1))
    assert synth.max_tokens == [10000, 16000] and len(map_client.calls) == 2
    assert out["degraded"] == [lm.LIST_TRUNCATED]
    lines = [ln.strip() for ln in out["answer"].splitlines() if ln.strip()]
    assert lines[:3] == ["**Hypervisors**", "- **Sangfor aSV** — restore as a VM [1]",
                         "- **Ora"]
    assert lines[3:5] == ["### Also documented", "- **Oracle Linux Virtualization "
                          "Manager** — restore agent backups as VMs [2]"]
    assert [g["item"] for g in out["item_grades"]] == [
        "Sangfor aSV", "Oracle Linux Virtualization Manager"]
    assert [c["fact_uuid"] for c in out["citations"]] == ["sf", "olvm"]
    assert out["answer"].endswith("The answer was cut short; this list may be incomplete.")
    combined = lm.degraded_note([lm.LIST_TRUNCATED, lm.COMPLETENESS_DEGRADED])
    assert combined == ("The answer was cut short and some documented items may be "
                        "missing; this list may be incomplete.")


async def test_a_truncated_answer_with_a_failed_pass_combines_both_notes(patched, monkeypatch):
    install, _ = patched
    install({_COMPLETE_Q: [_fact(u, text) for u, (text, _s) in _COMPLETE_FACTS.items()]})
    monkeypatch.setattr(lm, "rerank", _complete_rerank)
    partial = "**Hypervisors**\n- **Sangfor aSV** — restore as a VM [1]\n- **Ora"
    synth = _SeqClient([(partial[:-4], "length"), (partial, "length")])
    out = await lm.answer_list(None, None, _Replies("[]", "not json"), "m", synth, "s",
                               q=_COMPLETE_Q, scope=Scope(), group_id="g",
                               settings=_settings(list_item_floor=0.1))
    assert out["degraded"] == [lm.LIST_TRUNCATED, lm.COMPLETENESS_DEGRADED]
    assert out["answer"].endswith("The answer was cut short and some documented items may "
                                  "be missing; this list may be incomplete.")
