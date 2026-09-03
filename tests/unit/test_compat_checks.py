import asyncio

from compat import checks
from compat.model import CallableCheck, CypherCheck


def _all():
    return (checks.server_checks() + checks.bootstrap_checks()
            + checks.vector_checks() + checks.fulltext_checks())


def test_every_check_has_a_nonempty_name_and_group():
    for c in _all():
        assert c.name and c.group


def test_group_labels_are_the_spec_names():
    groups = {c.group for c in _all()}
    assert groups == {"server", "bootstrap", "vector", "fulltext"}


def test_apoc_check_is_informational():
    apoc = [c for c in checks.server_checks() if "apoc" in c.name.lower()]
    assert len(apoc) == 1
    assert apoc[0].informational is True


def test_gds_check_is_callable_so_it_can_skip():
    gds = [c for c in checks.server_checks() if "gds" in c.name.lower()]
    assert len(gds) == 1
    assert isinstance(gds[0], CallableCheck)


def test_vector_index_ddl_check_is_informational_and_prefixed():
    ddl = [c for c in checks.vector_checks()
           if isinstance(c, CypherCheck) and "CREATE VECTOR INDEX" in c.cypher]
    assert ddl, "expected an informational CREATE VECTOR INDEX probe"
    for c in ddl:
        assert c.informational is True
        assert "compat_" in c.cypher


def test_no_check_calls_apoc_procedures():
    # APOC is not installed on the target and is referenced nowhere in the stack.
    for c in _all():
        if isinstance(c, CypherCheck):
            assert "apoc." not in c.cypher.lower() or c.informational


def test_every_harness_index_is_compat_prefixed():
    for c in _all():
        if isinstance(c, CypherCheck) and "CREATE " in c.cypher and "INDEX" in c.cypher:
            assert "compat_" in c.cypher


def test_cypher_checks_never_issue_an_unscoped_delete():
    for c in _all():
        if isinstance(c, CypherCheck) and "DELETE" in c.cypher.upper():
            assert "compat-check" in c.cypher or "$g" in c.cypher


def test_lucene_escaping_check_covers_the_metacharacters():
    ft = [c for c in checks.fulltext_checks() if "lucene" in c.name.lower()]
    assert len(ft) == 1
    assert isinstance(ft[0], CallableCheck)


def _later():
    return (checks.graphiti_write_checks() + checks.graphiti_search_checks()
            + checks.our_cypher_checks())


def test_later_group_labels_are_the_spec_names():
    assert {c.group for c in _later()} == {
        "graphiti-write", "graphiti-search", "our-cypher"}


def test_search_checks_cover_exactly_the_two_recipes_used_in_src():
    names = " ".join(c.name.lower() for c in checks.graphiti_search_checks())
    assert "rrf" in names
    assert "node_distance" in names or "node distance" in names


def test_our_cypher_group_covers_the_two_bare_call_subquery_sites():
    names = " ".join(c.name.lower() for c in checks.our_cypher_checks())
    assert "corpus cursor" in names       # theme_builder/cli.py:69
    assert "staleness sweep" in names     # graph_extract/staleness_sweep.py:52


def test_our_cypher_group_covers_every_module_the_spec_lists():
    names = " ".join(c.name.lower() for c in checks.our_cypher_checks())
    for needle in ["resolve_citations", "vendor", "freshness", "timeline",
                   "touched_entities", "load_persisted", "leiden"]:
        assert needle in names, f"missing a check for {needle}"


def test_our_cypher_checks_are_all_callable_so_failures_report_procedural():
    from compat.model import CallableCheck as CC
    for c in checks.our_cypher_checks():
        assert isinstance(c, CC)


def test_graphiti_write_group_exercises_the_bulk_dynamic_label_construct():
    """`.save()` interpolates labels as literal text; only the BULK save uses
    Cypher's native `SET n:$(node.labels)`. That construct is the 5.22 fault line,
    so it must be asserted explicitly."""
    ddl = [c for c in checks.graphiti_write_checks()
           if isinstance(c, CypherCheck) and "SET n:$(" in c.cypher]
    assert len(ddl) == 1


def test_all_synthetic_uuids_are_compat_prefixed():
    for value in [checks.EP_UUID, checks.ENT_A, checks.ENT_B, checks.ENT_C,
                  checks.FACT_AB, checks.FACT_BC]:
        assert value.startswith("compat-")


def test_all_checks_returns_every_group_in_registry_order():
    groups = [c.group for c in checks.all_checks([0.5] * 768)]
    first_seen = list(dict.fromkeys(groups))
    assert first_seen == ["server", "bootstrap", "vector", "fulltext",
                          "graphiti-write", "graphiti-search", "our-cypher", "e2e"]


def test_all_checks_substitutes_the_fabricated_vector():
    from compat.model import CypherCheck as CypherC
    vector = [0.25] * 768
    for c in checks.all_checks(vector):
        if isinstance(c, CypherC) and "v" in c.params:
            assert c.params["v"] == vector, f"{c.name} kept a None placeholder"


def test_e2e_check_never_references_synthesis():
    """Scoped to the e2e function's own source (not the whole module) so that
    explanatory comments elsewhere naming global_search/drift don't trip it."""
    import inspect

    from compat.checks import _e2e_ingest_and_retrieve
    source = inspect.getsource(_e2e_ingest_and_retrieve)
    for banned in ["answer_local", "synthesize", "_synthesis_client_and_model",
                   "judge", "global_search", "drift_search"]:
        assert banned not in source, f"e2e must not depend on the synthesis tier ({banned})"


def test_every_structural_id_is_compat_prefixed():
    """SAFETY: MERGE against a real vendor/article id would stamp group_id onto a
    production node, which teardown would then DELETE. Ids must be literal and
    unmistakably ours."""
    for value in [checks.VENDOR_ID, checks.PRODUCT_ID, checks.SOURCE_ID,
                  checks.ARTICLE_ID, checks.E2E_ARTICLE_ID]:
        assert value.startswith("compat-check-"), value


def test_orphan_ids_are_compat_prefixed():
    for value in [checks.EP_ORPHAN, checks.FACT_ORPHAN]:
        assert value.startswith("compat-")


def test_article_urls_use_the_reserved_invalid_tld():
    """A fabricated URL must never look like real documentation in a report."""
    for url in [checks.ARTICLE_URL, checks.E2E_ARTICLE_URL]:
        assert ".invalid/" in url


def test_bootstrap_group_writes_the_structural_chain():
    names = " ".join(c.name.lower() for c in checks.bootstrap_checks())
    assert "structural fixture" in names


def test_graphiti_write_group_links_the_article_to_the_episode():
    names = [c.name.lower() for c in checks.graphiti_write_checks()]
    joined = " ".join(names)
    assert "provenance link" in joined
    # The link must come AFTER the episode write: registry order is execution order,
    # and Provenance.link MATCHes an existing (:Episodic).
    write_idx = next(i for i, n in enumerate(names) if "synthetic graph" in n)
    link_idx = next(i for i, n in enumerate(names) if "provenance link" in n)
    assert write_idx < link_idx


def test_structural_write_precedes_the_episode_write_across_groups():
    groups = [c.group for c in checks.all_checks([0.5] * 768)]
    assert groups.index("bootstrap") < groups.index("graphiti-write")


def test_our_cypher_group_writes_then_reads_the_community():
    names = [c.name.lower() for c in checks.our_cypher_checks()]
    write_idx = next(i for i, n in enumerate(names) if "write community" in n)
    read_idx = next(i for i, n in enumerate(names) if "load_persisted" in n)
    shortlist_idx = next(i for i, n in enumerate(names) if "shortlist" in n)
    assert write_idx < read_idx
    assert write_idx < shortlist_idx


def test_sweep_runs_before_the_timeline_flag_check():
    """_timeline_sweep_flags asserts the orphan carries expired_by_sweep, so the
    sweep must have already run."""
    names = [c.name.lower() for c in checks.our_cypher_checks()]
    sweep_idx = next(i for i, n in enumerate(names) if "staleness sweep" in n)
    flags_idx = next(i for i, n in enumerate(names) if "timeline sweep flags" in n)
    assert sweep_idx < flags_idx


def test_fake_embedder_returns_the_vector_shortlist_expects():
    # `import asyncio` goes at the TOP of the test file, not here: CI runs
    # `ruff check src tests` and E402 applies to test files too.
    emb = checks._FakeEmbedder([0.25] * 768)
    got = asyncio.run(emb.create_batch(["anything"]))
    assert got == [[0.25] * 768]


def test_community_entry_satisfies_every_shortlist_filter():
    """shortlist_communities drops rows with a falsy embedding, and _rank_hits SKIPS
    rows whose cited_fact_uuids is empty. A community failing either would make the
    shortlist check pass vacuously."""
    entry = checks._community_entry([0.5] * 768)
    assert entry["level"] == checks.COMMUNITY_LEVEL
    assert entry["cited_fact_uuids"]
    assert entry["embedding"]
    assert entry["member_uuids"]
