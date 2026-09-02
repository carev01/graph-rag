from compat.model import CheckResult
from compat.report import render, verdict


def _r(name, status, **kw):
    return CheckResult(name=name, group=kw.pop("group", "g"), status=status, **kw)


def test_verdict_go_when_all_pass():
    assert verdict([_r("a", "pass"), _r("b", "pass")]) == "GO"


def test_verdict_go_with_config_when_every_failure_passes_under_cypher5():
    results = [_r("a", "pass"), _r("b", "fail", cypher5_retry="pass")]
    assert verdict(results) == "GO_WITH_CONFIG"


def test_verdict_no_go_when_a_failure_also_fails_under_cypher5():
    results = [_r("a", "fail", cypher5_retry="pass"), _r("b", "fail", cypher5_retry="fail")]
    assert verdict(results) == "NO_GO"


def test_verdict_no_go_when_a_procedural_check_fails():
    # cypher5_retry is None for CallableChecks -> cannot be excused by config
    assert verdict([_r("a", "fail", cypher5_retry=None)]) == "NO_GO"


def test_verdict_ignores_skips_and_informational():
    results = [
        _r("a", "pass"),
        _r("b", "skip", detail="llm unreachable"),
        _r("c", "fail", cypher5_retry="fail", informational=True),
    ]
    assert verdict(results) == "GO"


def test_verdict_go_on_empty_results():
    assert verdict([]) == "GO"


def test_render_contains_verdict_matrix_and_sections():
    results = [
        _r("kernel version", "pass", group="server", detail="2026.07.1"),
        _r("apoc present", "fail", group="server", detail="not installed", informational=True),
        _r("e2e ingest", "skip", group="e2e", detail="embedder unreachable"),
        _r("sweep", "fail", group="our-cypher", cypher5_retry="pass", detail="boom"),
    ]
    out = render(results, target={"uri": "bolt://h:7687", "kernel": "2026.07.1"})
    assert "GO_WITH_CONFIG" in out
    assert "| kernel version |" in out
    assert "(info)" in out                    # informational marker
    assert "## Not verified" in out
    assert "embedder unreachable" in out
    assert "db.query.default_language=CYPHER_5" in out   # recommended action
    assert "bolt://h:7687" in out


def test_render_reports_teardown_error_prominently():
    out = render([_r("a", "pass")], target={"uri": "u"}, teardown_error="delete failed")
    assert "manual cleanup required" in out.lower()
    assert "compat-check" in out
    assert "delete failed" in out


def test_render_marks_procedural_retry_as_not_applicable():
    out = render([_r("a", "fail", group="e2e", cypher5_retry=None, detail="x")],
                 target={"uri": "u"})
    assert "n/a (procedural)" in out
