import importlib.util
import json as _json
import random
from pathlib import Path

from graph_extract.config import ExtractSettings
from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS

_spec = importlib.util.spec_from_file_location(
    "chunk_ab", Path(__file__).parents[2] / "scripts" / "chunk_ab.py")
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

S = ExtractSettings(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                    docext_base_url="https://x", docext_read_key="k")


def _art(i, vendor, n_chunks, tok=300):
    return {"id": f"a{i}", "vendor": vendor, "title": f"T{i}",
            "chunks": [{"t": f"chunk {j} " + "w " * 50, "tok": tok} for j in range(n_chunks)]}


def test_today_episode_count_uses_the_production_builder():
    assert ab.today_episode_count(_art(0, "V", 5), S) == 5


def test_select_prefers_multi_episode_articles_across_vendors():
    sample = [_art(i, v, n) for i, (v, n) in enumerate(
        [("A", 5), ("A", 5), ("A", 1), ("B", 4), ("B", 2), ("C", 6)])]
    picked = ab.select_articles(sample, 3, random.Random(0), S)
    assert {a["vendor"] for a in picked} == {"A", "B", "C"}
    assert all(ab.today_episode_count(a, S) >= 3 for a in picked)


def test_article_cost_uses_the_tier_price():
    assert ab.article_cost(1_000_000, 0, "cheap") == 0.09
    assert ab.article_cost(0, 1_000_000, "strong") == 2.00


def test_summarise_totals_and_ratios():
    rows = [{"episodes": 3, "facts": 10, "entities": 6, "seconds": 30.0, "cost": 0.02},
            {"episodes": 1, "facts": 5, "entities": 4, "seconds": 10.0, "cost": 0.01}]
    s = ab.summarise(rows)
    assert s["episodes"] == 4 and s["facts"] == 15 and s["entities"] == 10
    assert abs(s["cost"] - 0.03) < 1e-9 and s["seconds"] == 40.0


def test_smoke_refusal_tally_not_moved():
    a_rows = [{"episodes": 1, "prompt_tokens": 100}]
    b_rows = [{"episodes": 2, "prompt_tokens": 200}]
    refusal = ab._smoke_refusal(a_rows, b_rows, tally_moved=False)
    assert "usage tally did not move" in refusal


def test_smoke_refusal_metering_missing():
    a_rows = [{"episodes": 1, "prompt_tokens": 100}]
    b_rows = [{"episodes": 0, "prompt_tokens": 0}, {"episodes": 1, "prompt_tokens": 0}]
    refusal = ab._smoke_refusal(a_rows, b_rows, tally_moved=True)
    assert "prompt_tokens == 0" in refusal and "episodes > 0" in refusal


def test_smoke_refusal_variable_not_engaged():
    a_rows = [{"episodes": 2, "prompt_tokens": 100}]
    b_rows = [{"episodes": 3, "prompt_tokens": 200}]
    refusal = ab._smoke_refusal(a_rows, b_rows, tally_moved=True)
    assert "variable is not engaged" in refusal


def test_smoke_refusal_passes_all_gates():
    a_rows = [{"episodes": 5, "prompt_tokens": 100}]
    b_rows = [{"episodes": 3, "prompt_tokens": 200}]
    refusal = ab._smoke_refusal(a_rows, b_rows, tally_moved=True)
    assert refusal is None


def test_uncapped_salience_drops_only_the_fixed_count():
    assert "10-25 facts" in CHEAP_TIER_SALIENCE
    assert "10-25 facts" not in ab.CHEAP_TIER_SALIENCE_UNCAPPED
    assert "NOT EXHAUSTIVE" in ab.CHEAP_TIER_SALIENCE_UNCAPPED  # anti-per-cell rules stay


def test_coverage_arm_instructions_carry_the_marker_on_both_tiers():
    strong, cheap = ab.ARM_INSTRUCTIONS["pack1200_coverage"]
    assert strong.startswith(EXTRACTION_INSTRUCTIONS) and ab.COVERAGE_MARKER in strong
    assert cheap.startswith(EXTRACTION_INSTRUCTIONS) and ab.COVERAGE_MARKER in cheap
    assert "10-25 facts" not in cheap
    assert ab.ARMS["pack1200_coverage"] == ab.ARMS["pack1200"]


def test_directive_seen_requires_an_extraction_prompt_with_the_marker(tmp_path):
    p = tmp_path / "cap.jsonl"

    def rec(name, text):
        return _json.dumps(
            {"prompt_name": name, "messages": [{"role": "user", "content": text}]})

    p.write_text(rec("dedupe_nodes.nodes", ab.COVERAGE_MARKER) + "\n"
                 + rec("extract_nodes.extract_text", "no marker") + "\n")
    assert ab.directive_seen(p, ab.COVERAGE_MARKER) is False
    p.write_text(p.read_text() + rec("extract_edges.edge", "x " + ab.COVERAGE_MARKER) + "\n")
    assert ab.directive_seen(p, ab.COVERAGE_MARKER) is True


def test_directive_seen_is_false_for_a_missing_capture(tmp_path):
    assert ab.directive_seen(tmp_path / "absent.jsonl", ab.COVERAGE_MARKER) is False


def test_directive_seen_recognises_responses_parse_input_shape(tmp_path):
    """Strong tier (gpt-5-mini, responses.parse) stores the prompt under "input",
    not "messages" -- the gate must not silently pass it by only reading messages."""
    p = tmp_path / "cap.jsonl"

    def rec(tier, name, text):
        return _json.dumps({"tier": tier, "prompt_name": name,
                            "input": [{"role": "user", "content": text}]})

    p.write_text(rec("strong", "extract_nodes.extract_text", "x " + ab.COVERAGE_MARKER) + "\n")
    assert ab.directive_seen(p, ab.COVERAGE_MARKER) is True


def test_directive_seen_requires_the_marker_on_every_tier_present(tmp_path):
    """A capture where the cheap tier's extraction prompt carries the marker but the
    strong tier's does not must fail the gate -- proof is required per tier, not once
    across the whole capture."""
    p = tmp_path / "cap.jsonl"

    def msg_rec(tier, name, text):
        return _json.dumps({"tier": tier, "prompt_name": name,
                            "messages": [{"role": "user", "content": text}]})

    def input_rec(tier, name, text):
        return _json.dumps({"tier": tier, "prompt_name": name,
                            "input": [{"role": "user", "content": text}]})

    p.write_text(msg_rec("cheap", "extract_nodes.extract_text", "x " + ab.COVERAGE_MARKER) + "\n"
                 + input_rec("strong", "extract_nodes.extract_text", "no marker here") + "\n")
    assert ab.directive_seen(p, ab.COVERAGE_MARKER) is False

    p.write_text(p.read_text()
                 + input_rec("strong", "extract_edges.edge", "y " + ab.COVERAGE_MARKER) + "\n")
    assert ab.directive_seen(p, ab.COVERAGE_MARKER) is True


def test_reused_rows_must_match_the_article_order():
    prev = {"today": {"rows": [{"article_id": "a"}, {"article_id": "b"}]}}
    assert [r["article_id"] for r in ab.reused_rows(prev, "today", ["a", "b"])] == ["a", "b"]
    import pytest
    with pytest.raises(SystemExit):
        ab.reused_rows(prev, "today", ["b", "a"])


def test_reused_rows_raises_systemexit_on_missing_arm():
    import pytest
    prev = {"other_arm": {"rows": []}}
    with pytest.raises(SystemExit, match="missing key 'today'"):
        ab.reused_rows(prev, "today", [])


def test_reused_rows_raises_systemexit_on_missing_rows_key():
    import pytest
    prev = {"today": {"summary": {}}}
    with pytest.raises(SystemExit, match="missing key 'rows'"):
        ab.reused_rows(prev, "today", [])


def test_prepare_capture_clears_stale_file(tmp_path):
    p = tmp_path / "cap.jsonl"
    p.write_text("stale marker\n")
    assert p.exists()
    result = ab.prepare_capture(p)
    assert result == p
    assert not p.exists()


def test_prepare_capture_handles_missing_file(tmp_path):
    p = tmp_path / "cap.jsonl"
    assert not p.exists()
    result = ab.prepare_capture(p)
    assert result == p
    assert not p.exists()
