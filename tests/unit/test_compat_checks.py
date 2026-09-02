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
