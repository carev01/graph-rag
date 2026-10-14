"""Facet searches rerank a larger candidate pool; side questions are vendor-subject.

Gate run 2 (amendment 2, task 14): large scopes (Veeam, Microsoft) filled k from the
corpus-wide hybrid search with poor ranking, and the cheap model kept cross-vendor
wording in the side questions. A cross-encoder rerank of a wider pool and a
vendor-subject rewrite prompt fixed both when measured."""
import logging
from types import SimpleNamespace

import pytest

import answer_api.drift as drift
import answer_api.enumeration as en
import answer_api.eval_retrieval as er
import answer_api.search as search_mod
import answer_api.synthesize as synth
from answer_api.scope import Scope
from answer_api.shape import _FACET_QUERIES_PROMPT, Facet, QuestionShape
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_RERANK = dict(rerank_base_url="http://rr", rerank_model="rerank-x")
_V = Facet("Veeam", Scope(vendors=("Veeam",)))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",)))


def _settings(**kw):
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


def _edge(uuid):
    return SimpleNamespace(uuid=uuid, fact=f"fact {uuid}", invalid_at=None, valid_at=None,
                           episodes=["in"])


class _Graph:
    def __init__(self, n):
        self.edges = [_edge(f"e{i}") for i in range(n)]
        self.limits = []

    async def _search(self, q, config, group_ids, center_node_uuid=None):
        self.limits.append(config.limit)
        return SimpleNamespace(edges=self.edges[:config.limit])


@pytest.fixture(autouse=True)
def _prov(monkeypatch):
    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {}

    async def _allowed(driver, scope, candidates=None):
        return set(candidates or [])

    monkeypatch.setattr(search_mod, "Provenance", _Prov)
    monkeypatch.setattr(search_mod, "scope_episode_uuids", _allowed)


def _uuids(res):
    return [r["fact_uuid"] for r in res["results"]]


# --- the side-question prompt --------------------------------------------------------

async def test_side_prompt_makes_each_side_the_subject_and_drops_cross_vendor_wording():
    p = _FACET_QUERIES_PROMPT
    assert p.startswith("Each SIDE below is a vendor or product whose OWN documentation")
    assert "start with the side's name as the subject" in p
    assert '"How does <side> ...", "Does <side> ..."' in p
    assert ('Drop cross-vendor wording such as "which vendors", "compare", "differ" '
            'or "each"') in p
    assert "do not mention any other side or vendor" in p
    out = p.format(sides="- Veeam", q="Q?")
    assert out.endswith("SIDES:\n- Veeam\n\nQUESTION: Q?\nJSON:")
    assert '{"<label>": "<question>", ...}' in out


# --- search_local(rerank_pool=...) ---------------------------------------------------

async def test_pool_reranks_a_wider_candidate_set_and_returns_k(monkeypatch):
    seen = {}

    async def _rerank(query, documents, *, top_k, settings):
        seen.update(query=query, n=len(documents), top_k=top_k)
        return [(30, 0.9), (2, 0.8), (17, 0.7)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    g = _Graph(100)
    res = await search_mod.search_local(g, None, q="ransomware", k=3, group_id="g",
                                        rerank_pool=40, settings=_settings(**_RERANK))
    assert g.limits == [120]                       # over-fetch scales with the pool
    assert seen == {"query": "ransomware", "n": 40, "top_k": 3}
    assert _uuids(res) == ["e30", "e2", "e17"]
    assert res["count"] == 3


async def test_scoped_pool_raises_the_adaptive_target_to_the_pool(monkeypatch):
    async def _rerank(query, documents, *, top_k, settings):
        return [(i, 1.0) for i in reversed(range(len(documents)))][:top_k]

    async def _scan(*a, **kw):
        return []

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    monkeypatch.setattr(search_mod, "_scoped_scan", _scan)
    g = _Graph(500)
    res = await search_mod.search_local(g, None, q="q", k=2, group_id="g",
                                        scope=_V.scope, rerank_pool=40,
                                        settings=_settings(**_RERANK))
    assert g.limits == [120]
    assert _uuids(res) == ["e39", "e38"]


async def test_pool_pads_a_short_rerank_from_the_original_order(monkeypatch):
    async def _rerank(query, documents, *, top_k, settings):
        return [(5, 0.9)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    res = await search_mod.search_local(_Graph(50), None, q="q", k=3, group_id="g",
                                        rerank_pool=40, settings=_settings(**_RERANK))
    assert _uuids(res) == ["e5", "e0", "e1"]


@pytest.mark.parametrize("outcome", ["none", "raise", "empty"])
async def test_pool_rerank_failure_returns_the_original_top_k(monkeypatch, caplog, outcome):
    async def _rerank(*a, **kw):
        if outcome == "raise":
            raise RuntimeError("voyage down")
        return None if outcome == "none" else []

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    with caplog.at_level(logging.WARNING, logger="answer_api.search"):
        res = await search_mod.search_local(_Graph(50), None, q="q", k=3, group_id="g",
                                            rerank_pool=40, settings=_settings(**_RERANK))
    assert _uuids(res) == ["e0", "e1", "e2"]
    if outcome != "empty":
        assert "original order" in caplog.text


# Settings are built inside the test, never at collection: graphiti's load_dotenv()
# has copied a developer .env into os.environ by then, and the conftest fixture that
# strips it only runs per test -- a collection-time _settings() saw .env's reranker.
@pytest.mark.parametrize("make_kw", [
    lambda: {},                                                    # pool 0 (default)
    lambda: {"rerank_pool": 0, "settings": _settings(**_RERANK)},  # explicit 0
    lambda: {"rerank_pool": 40, "settings": _settings()},          # no reranker configured
    lambda: {"rerank_pool": 40},                                   # no settings
], ids=["default", "explicit-0", "unconfigured", "no-settings"])
async def test_no_pool_leaves_the_call_path_unchanged(monkeypatch, make_kw):
    async def _rerank(*a, **k):
        raise AssertionError("rerank must not be called")

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    g = _Graph(100)
    res = await search_mod.search_local(g, None, q="q", k=5, group_id="g", **make_kw())
    assert g.limits == [15]
    assert _uuids(res) == [f"e{i}" for i in range(5)]


async def test_pool_scan_fallback_targets_the_pool(monkeypatch):
    scans = []

    async def _scan(graphiti, driver, *, q, scope, k, group_id, exclude,
                    include_invalid=False, max_episodes=6000, settings=None,
                    rerank=True, episode_count=None):
        scans.append(k)
        return [_edge(f"s{i}") for i in range(k)]

    async def _rerank(query, documents, *, top_k, settings):
        return [(len(documents) - 1, 1.0)]

    monkeypatch.setattr(search_mod, "_scoped_scan", _scan)
    monkeypatch.setattr(search_mod, "rerank", _rerank)
    res = await search_mod.search_local(_Graph(2), None, q="q", k=3, group_id="g",
                                        scope=_V.scope, rerank_pool=10,
                                        settings=_settings(**_RERANK,
                                                           local_scoped_scan_candidates=8))
    assert scans == [8]                    # local_scoped_scan_candidates (task 17)
    assert _uuids(res) == ["s7", "e0", "e1"]


class _ScanGraph(_Graph):
    """A hybrid search returning `n` edges, plus an embedder for the scoped scan."""

    def __init__(self, n):
        super().__init__(n)
        self.embedder = SimpleNamespace(create=self._embed)

    async def _embed(self, input_data):
        return [0.0]


class _ScanDriver:
    """The scoped scan's query: up to `k` scope edges, cosine order.

    `_scoped_scan` now runs the (vec-free) episode-count/cap pre-check before
    the full scan query (task 17, item 2): that first call is answered with a
    small `n`, under any cap, so the scan proceeds as before."""

    def __init__(self):
        self.ks = []

    async def execute_query(self, query, **kw):
        if query is search_mod._EPISODE_COUNT:
            return [{"n": 1}], None, None
        self.ks.append(kw["k"])
        recs = [{"n": 1, "uuid": f"s{i}", "fact": f"fact s{i}", "episodes": ["in"],
                 "valid_at": None, "invalid_at": None} for i in range(kw["k"])]
        return recs, None, None


async def test_a_pooled_search_that_falls_back_to_the_scan_reranks_once(monkeypatch):
    # Final review: the scan's own 200-candidate rerank was thrown away by the
    # pool's rerank of the merged candidates -- one cross-encoder call suffices.
    calls = []

    async def _rerank(query, documents, *, top_k, settings):
        calls.append((len(documents), top_k))
        return [(len(documents) - 1, 1.0)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    driver = _ScanDriver()
    res = await search_mod.search_local(_ScanGraph(2), driver, q="q", k=3, group_id="g",
                                        scope=_V.scope, rerank_pool=10,
                                        settings=_settings(**_RERANK))
    assert calls == [(202, 3)]         # the pool's single rerank over 2 + 200 merged
    assert driver.ks == [200]          # cosine top local_scoped_scan_candidates, once
    assert _uuids(res) == ["s199", "e0", "e1"]


async def test_a_non_pooled_scoped_search_keeps_the_scans_rerank(monkeypatch):
    calls = []

    async def _rerank(query, documents, *, top_k, settings):
        calls.append((len(documents), top_k))
        return [(len(documents) - 1, 1.0)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    driver = _ScanDriver()
    res = await search_mod.search_local(_ScanGraph(1), driver, q="q", k=3, group_id="g",
                                        scope=_V.scope, settings=_settings(**_RERANK))
    assert driver.ks == [200]          # local_scoped_scan_candidates
    assert calls == [(200, 2)]         # the scan's own rerank, once
    assert _uuids(res)[:2] == ["e0", "s199"]


# --- facet callers pass the pool; single callers do not -----------------------------

async def test_default_facet_rerank_pool():
    assert _settings().local_facet_rerank_pool == 40


class _Synth:
    def __init__(self, reply="X [1]."):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.reply = reply

    async def _create(self, **kw):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.reply), finish_reason="stop")])


async def test_local_facets_pass_the_pool_and_single_does_not(monkeypatch):
    calls = []

    async def _search(graphiti, driver, **kw):
        calls.append(kw)
        return {"query": kw["q"], "count": 0, "results": []}

    monkeypatch.setattr(synth.search_mod, "search_local", _search)
    s = _settings(local_facet_rerank_pool=33)
    await synth.answer_local(None, None, _Synth(), "m", q="q", group_id="g",
                             facets=[_V, _A], settings=s)
    assert [c.get("rerank_pool") for c in calls] == [33, 33]
    assert all(c["settings"] is s for c in calls)
    calls.clear()
    await synth.answer_local(None, None, _Synth(), "m", q="q", group_id="g", settings=s)
    assert len(calls) == 1 and "rerank_pool" not in calls[0]


async def test_drift_followup_forwards_the_pool_only_when_given(monkeypatch):
    calls = []

    async def _search(graphiti, driver, **kw):
        calls.append(kw)
        return {"results": []}

    monkeypatch.setattr(drift, "search_local", _search)
    fu = drift.FollowUp(query="q", community_id=None, iteration=1)
    await drift._run_followup(None, None, fu, k=5, group_id="g", rerank_pool=12)
    await drift._run_followup(None, None, fu, k=5, group_id="g")
    assert calls[0]["rerank_pool"] == 12
    assert "rerank_pool" not in calls[1]


async def test_drift_facet_scoped_followups_pass_the_pool(monkeypatch):
    import answer_api.global_search as gs

    ran = []

    def _hit(cid):
        return gs.CommunityHit(community_id=cid, title=cid, summary="s", level=1,
                               rating=5, cited_fact_uuids=[], full_report="[]",
                               similarity=0.9)

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("cv")] if scope == _V.scope else [_hit("ca")]

    async def _followup(graphiti, driver, fu, *, k, group_id, scope=None, **kw):
        ran.append((fu.query, scope, kw.get("rerank_pool")))
        return []

    class _PrimerSynth(_Synth):
        async def _create(self, **kw):
            p = kw["messages"][0]["content"]
            text = ('{"preliminary_answer": "d", "follow_ups": [{"query": "veeam detail",'
                    ' "community_id": "cv", "relevance": 9}]}'
                    if "follow-up questions" in p else "x")
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason="stop")])

    monkeypatch.setattr(drift, "shortlist_communities", _shortlist)
    monkeypatch.setattr(drift, "_run_followup", _followup)
    s = _settings(local_facet_rerank_pool=21)
    await drift.drift_search(None, None, None, _PrimerSynth(), "m", q="compare", level=1,
                             iterations=1, primer_k=5, max_followups=4, followup_k=8,
                             group_id="g", settings=s, facets=[_V, _A])
    # The Veeam-community follow-up and the AWS coverage follow-up are both facet-scoped.
    assert ("veeam detail", _V.scope, 21) in ran
    assert ("compare", _A.scope, 21) in ran
    ran.clear()
    await drift.drift_search(None, None, None, _PrimerSynth(), "m", q="single", level=1,
                             iterations=1, primer_k=5, max_followups=4, followup_k=8,
                             group_id="g", settings=s)
    assert ran and all(pool is None for _, _, pool in ran)


async def test_enumeration_gather_passes_the_pool(monkeypatch):
    calls = []

    async def _search(graphiti, driver, **kw):
        calls.append(kw)
        return {"results": []}

    async def _shortlist(*a, **kw):
        return []

    monkeypatch.setattr(en, "search_local", _search)
    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    s = _settings(local_facet_rerank_pool=17)
    await en._gather(None, None, None, q="q", facet=_V, group_id="g", settings=s)
    assert calls[0]["rerank_pool"] == 17 and calls[0]["settings"] is s


@pytest.mark.parametrize("kind,pool", [("comparison", 19), ("enumeration", 19),
                                       ("single", None)])
async def test_eval_evidence_passes_the_pool_for_facet_shapes(monkeypatch, kind, pool):
    calls = []

    async def _search(graphiti, driver, **kw):
        calls.append(kw)
        return {"results": [], "count": 0}

    async def _shortlist(*a, **kw):
        return []

    monkeypatch.setattr(er, "search_local", _search)
    monkeypatch.setattr(er, "shortlist_communities", _shortlist)
    facets = (_V, _A) if kind != "single" else (Facet("", Scope()),)
    await er._evidence(None, None, None, _settings(local_facet_rerank_pool=19), "q",
                       QuestionShape(kind, facets))
    assert calls and all(c.get("rerank_pool") == pool for c in calls)


# --- task 17: a scannable scope's own candidates always join the pool ---------------

class _CapDriver(_ScanDriver):
    """The scan query reporting `n` scope episodes (n > max_episodes = over the cap).

    The episode-count pre-check reports `n`; an over-cap `n` makes `_scoped_scan`
    return before any further query, so the full scan is never reached."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    async def execute_query(self, query, **kw):
        if query is search_mod._EPISODE_COUNT:
            return [{"n": self.n}], None, None
        return await super().execute_query(query, **kw)


async def test_a_pooled_scannable_scope_merges_scan_candidates_into_a_full_pool(
        monkeypatch):
    # Run 8: AWS's "Restore testing" facts never entered the corpus-wide hybrid
    # search, so the post-filter filled the pool without them; the scope's own
    # cosine candidates must join the pool before the single rerank.
    calls = []

    async def _rerank(query, documents, *, top_k, settings):
        calls.append((list(documents), top_k))
        return [(documents.index("fact s3"), 1.0)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    driver = _CapDriver(n=10)
    g = _ScanGraph(100)                    # the post-filter alone fills the pool
    res = await search_mod.search_local(
        g, driver, q="q", k=2, group_id="g", scope=_A.scope, rerank_pool=10,
        settings=_settings(**_RERANK, local_scoped_scan_candidates=5))
    assert driver.ks == [5]                # scanned although the pool was full
    assert len(calls) == 1                 # one rerank over the merged pool
    docs, top_k = calls[0]
    assert top_k == 2
    assert docs == [f"fact e{i}" for i in range(10)] + [f"fact s{i}" for i in range(5)]
    assert _uuids(res) == ["s3", "e0"]


async def test_pooled_scan_excludes_uuids_already_in_the_pool(monkeypatch):
    seen = {}

    async def _scan(graphiti, driver, *, q, scope, k, group_id, exclude,
                    include_invalid=False, max_episodes=6000, settings=None,
                    rerank=True, episode_count=None):
        seen.update(k=k, exclude=set(exclude), rerank=rerank)
        return [_edge("e1"), _edge("s0")]

    async def _rerank(query, documents, *, top_k, settings):
        seen["docs"] = list(documents)
        return []

    monkeypatch.setattr(search_mod, "_scoped_scan", _scan)
    monkeypatch.setattr(search_mod, "rerank", _rerank)
    await search_mod.search_local(_Graph(100), None, q="q", k=2, group_id="g",
                                  scope=_A.scope, rerank_pool=3,
                                  settings=_settings(**_RERANK))
    assert seen["k"] == 200 and seen["rerank"] is False
    assert seen["exclude"] == {"e0", "e1", "e2"}
    assert seen["docs"] == ["fact e0", "fact e1", "fact e2", "fact s0"]


async def test_a_pooled_over_cap_scope_does_not_scan(monkeypatch):
    calls = []

    async def _rerank(query, documents, *, top_k, settings):
        calls.append(len(documents))
        return [(0, 1.0)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    driver = _CapDriver(n=7000)
    res = await search_mod.search_local(
        _ScanGraph(100), driver, q="q", k=2, group_id="g", scope=_V.scope,
        rerank_pool=10, scan_max_episodes=6000, settings=_settings(**_RERANK))
    assert calls == [10]                   # the post-filter pool only
    assert _uuids(res) == ["e0", "e1"]


async def test_a_pooled_scan_failure_keeps_the_post_filter_pool(monkeypatch, caplog):
    async def _boom(*a, **kw):
        raise RuntimeError("neo4j down")

    calls = []

    async def _rerank(query, documents, *, top_k, settings):
        calls.append(len(documents))
        return [(4, 1.0), (1, 0.5)]

    monkeypatch.setattr(search_mod, "_scoped_scan", _boom)
    monkeypatch.setattr(search_mod, "rerank", _rerank)
    with caplog.at_level(logging.WARNING, logger="answer_api.search"):
        res = await search_mod.search_local(_Graph(100), None, q="q", k=2, group_id="g",
                                            scope=_A.scope, rerank_pool=10,
                                            settings=_settings(**_RERANK))
    assert "scoped scan failed" in caplog.text
    assert calls == [10]
    assert _uuids(res) == ["e4", "e1"]


async def test_a_non_pooled_full_scoped_search_still_does_not_scan(monkeypatch):
    async def _scan(*a, **kw):
        raise AssertionError("must not scan")

    monkeypatch.setattr(search_mod, "_scoped_scan", _scan)
    res = await search_mod.search_local(_Graph(100), None, q="q", k=3, group_id="g",
                                        scope=_A.scope, settings=_settings(**_RERANK))
    assert _uuids(res) == ["e0", "e1", "e2"]


class _CountingDriver(_ScanDriver):
    """`_ScanDriver` whose episode count is `n` (an Exception instance raises)."""

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.counts = 0

    async def execute_query(self, query, **kw):
        if query is search_mod._EPISODE_COUNT:
            self.counts += 1
            if isinstance(self.n, Exception):
                raise self.n
            return [{"n": self.n}], None, None
        return await super().execute_query(query, **kw)


async def _identity_rerank(query, documents, *, top_k, settings):
    return [(i, 1.0) for i in range(min(top_k, len(documents)))]


async def test_a_scannable_pooled_scope_fetches_once_then_scans(monkeypatch):
    # Run 9: 120 -> 400 -> 450 corpus-wide fetches cost ~20 s per small-scope facet
    # although the scan below them always supplies the scope's own facts.
    monkeypatch.setattr(search_mod, "rerank", _identity_rerank)
    g, driver = _ScanGraph(2), _CountingDriver(1040)
    await search_mod.search_local(g, driver, q="q", k=3, group_id="g", scope=_A.scope,
                                  rerank_pool=10, settings=_settings(**_RERANK))
    assert g.limits == [30]            # one fetch at the pool's first limit
    assert driver.counts == 1          # the scan reuses the count
    assert driver.ks == [200]


async def test_an_over_cap_pooled_scope_widens_the_fetch_and_skips_the_scan(monkeypatch):
    monkeypatch.setattr(search_mod, "rerank", _identity_rerank)
    g, driver = _ScanGraph(2), _CountingDriver(6001)
    await search_mod.search_local(g, driver, q="q", k=3, group_id="g", scope=_V.scope,
                                  rerank_pool=10, scan_max_episodes=6000,
                                  settings=_settings(**_RERANK))
    assert g.limits == [30, 100, 300]
    assert driver.ks == []             # no scan query for a scope over the cap
    assert driver.counts == 1


async def test_a_failed_episode_count_widens_the_fetch_as_before(monkeypatch, caplog):
    monkeypatch.setattr(search_mod, "rerank", _identity_rerank)
    g, driver = _ScanGraph(2), _CountingDriver(RuntimeError("neo4j down"))
    with caplog.at_level(logging.WARNING, logger="answer_api.search"):
        res = await search_mod.search_local(g, driver, q="q", k=3, group_id="g",
                                            scope=_A.scope, rerank_pool=10,
                                            settings=_settings(**_RERANK))
    assert g.limits == [30, 100, 300]
    assert "scope episode count failed" in caplog.text
    assert _uuids(res) == ["e0", "e1"]  # the post-filter's results survive
