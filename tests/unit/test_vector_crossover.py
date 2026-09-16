"""Pure helpers of the vector-index crossover harness.

The harness reports numbers that a build/no-build decision rests on, so its
failure mode is "reports something that is not what it claims". These tests
pin the claims: that a fallback to synthetic vectors is visible rather than
silent, that recall is scored against the right ground truth, and that the
disk guard refuses a step before it fills the root filesystem rather than
after.
"""
from __future__ import annotations

import random

from scripts.vector_crossover import (
    build_pool,
    estimated_bytes,
    fits,
    rank1_agrees,
    recall_at_10,
)


def test_a_pool_built_without_seeds_announces_that_it_is_a_fallback():
    """i.i.d. vectors are near-equidistant in 768 dimensions, which is the worst
    case for an index and produces recall numbers that do NOT transfer to real
    embeddings. A run that silently fell back would publish those numbers as if
    they were comparable."""
    pool = build_pool(n=5, dim=8, rng=random.Random(0), seeds=None)
    assert pool.is_fallback is True
    assert "FALLBACK" in pool.provenance
    assert len(pool.vectors) == 5
    assert all(len(v) == 8 for v in pool.vectors)


def test_a_pool_built_from_seeds_is_not_a_fallback_and_says_what_it_used():
    seeds = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    pool = build_pool(n=4, dim=4, rng=random.Random(0), seeds=seeds)
    assert pool.is_fallback is False
    assert "2" in pool.provenance, "the provenance must name how many seeds it resampled"
    assert len(pool.vectors) == 4


def test_pool_vectors_are_unit_length():
    """Cosine similarity is scale-free, but unit vectors keep the planted-distance
    fixtures in the integration tests meaningful."""
    pool = build_pool(n=3, dim=16, rng=random.Random(1), seeds=None)
    for v in pool.vectors:
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-9


def test_a_pool_is_reproducible_for_a_given_seed():
    """A measurement nobody can re-run is not evidence."""
    a = build_pool(n=4, dim=8, rng=random.Random(7), seeds=None)
    b = build_pool(n=4, dim=8, rng=random.Random(7), seeds=None)
    assert a.vectors == b.vectors


def test_resampling_stays_near_its_seed():
    """Resampling exists to preserve cluster structure. A vector that wandered off
    its seed would be i.i.d. by another name, and the fallback flag would then be
    lying about which distribution was measured."""
    seeds = [[1.0, 0.0, 0.0, 0.0]]
    pool = build_pool(n=20, dim=4, rng=random.Random(3), seeds=seeds, noise=0.05)
    for v in pool.vectors:
        cosine = sum(a * b for a, b in zip(v, seeds[0]))
        assert cosine > 0.9, f"resampled vector drifted off its seed: cosine {cosine}"


def test_recall_is_scored_against_brute_force_as_ground_truth():
    brute = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert recall_at_10(brute, brute) == 1.0
    assert recall_at_10(brute, ["a", "b", "c", "d", "e", "x", "y", "z", "p", "q"]) == 0.5
    assert recall_at_10(brute, []) == 0.0


def test_recall_ignores_order_but_rank1_does_not():
    """recall@10 is a set measure; whether the single best neighbour was found is a
    different question and dedup cares about it, so both are reported."""
    brute = ["a", "b", "c"]
    assert recall_at_10(brute, ["c", "b", "a"]) == 1.0
    assert rank1_agrees(brute, ["c", "b", "a"]) is False
    assert rank1_agrees(brute, ["a", "c", "b"]) is True
    assert rank1_agrees([], []) is False


def test_recall_over_fetch_scores_only_the_top_ten():
    """At k=50 the index returns 50 candidates; the harness keeps the best 10 and
    scores THOSE. Scoring all 50 against a 10-item truth would report inflated
    recall that no caller would ever see."""
    brute = [f"t{i}" for i in range(10)]
    index_50 = brute[:10] + [f"x{i}" for i in range(40)]
    assert recall_at_10(brute, index_50) == 1.0
    index_50_bad = [f"x{i}" for i in range(10)] + brute
    assert recall_at_10(brute, index_50_bad) == 0.0, \
        "the true neighbours sat below rank 10, so a caller taking 10 would miss them"


def test_the_disk_guard_refuses_a_step_that_would_not_fit():
    """1M edges x 768 float64 is ~6 GB of property store before the index and the
    transaction logs. Filling the root filesystem mid-sweep would take the Docker
    daemon with it."""
    one_million = estimated_bytes(1_000_000)
    assert one_million > 6 * 1024**3, "estimate must include store overhead, not just raw floats"
    assert fits(1_000_000, free=one_million + 6 * 1024**3) is True
    assert fits(1_000_000, free=one_million) is False, "no headroom left for the index"
    assert fits(10_000, free=20 * 1024**3) is True


def test_the_disk_guard_headroom_grows_with_the_step():
    """A flat reserve would shrink to nothing in relative terms exactly as the
    steps got big enough for the index to matter."""
    small = estimated_bytes(10_000)
    large = estimated_bytes(1_000_000)
    # small steps sit on the 5 GiB floor
    assert fits(10_000, free=small + 5 * 1024**3) is True
    assert fits(10_000, free=small + 4 * 1024**3) is False
    # large steps reserve half their own estimate, which exceeds the floor
    assert large // 2 > 5 * 1024**3, "the 1M step must be past the floor for this to test anything"
    assert fits(1_000_000, free=large + large // 2) is True
    assert fits(1_000_000, free=large + 5 * 1024**3) is False, \
        "the flat floor must not be what decides a large step"


def test_the_report_marks_measured_and_projected_rows_apart():
    """Section 1.2 of the production review marked its own extrapolated rows
    `[inference]` three orders of magnitude out. A report that presented a
    projection as a measurement would repeat the habit this project keeps paying
    for."""
    from scripts.vector_crossover import StepResult, format_report

    steps = [
        StepResult(n=10_000, brute_ms=690.0, index_ms=12.0, control_ms=688.0,
                   recall={10: (1.0, 1.0), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                   insert_bare_ms=900.0, insert_indexed_ms=1400.0),
        StepResult(n=50_000, brute_ms=2810.0, index_ms=14.0, control_ms=2805.0,
                   recall={10: (0.98, 0.95), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                   insert_bare_ms=4400.0, insert_indexed_ms=7000.0),
    ]
    # measured_to is 10,000, so the 50,000 row is past where measurement stopped
    # and must be labelled as a projection.
    out = format_report(steps, provenance="resampled from 3469 real values",
                        measured_to=10_000)
    assert "[measured]" in out
    assert "[inference]" in out, "a row beyond the last measured step must be labelled"
    assert "resampled from 3469 real values" in out, "provenance must travel into the report"
    assert "10,000" in out or "10000" in out


def test_the_report_leads_with_the_fallback_warning_when_vectors_are_synthetic():
    """A reader skimming for the headline number must not miss that recall came
    from i.i.d. vectors."""
    from scripts.vector_crossover import StepResult, format_report

    steps = [StepResult(n=10_000, brute_ms=690.0, index_ms=12.0, control_ms=688.0,
                        recall={10: (1.0, 1.0), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                        insert_bare_ms=900.0, insert_indexed_ms=1400.0)]
    out = format_report(steps, provenance="i.i.d. Gaussian -- FALLBACK", measured_to=10_000)
    assert "FALLBACK" in out.split("\n")[0] or "FALLBACK" in out.split("\n")[1], \
        "the fallback warning must be at the top, not buried below the table"


def test_the_control_column_is_flagged_when_it_diverges():
    """The control is brute force WITH the index present; it must match brute
    force without it. A silent divergence would invalidate every latency row, so
    the report has to shout rather than print two similar numbers."""
    from scripts.vector_crossover import StepResult, format_report

    steps = [StepResult(n=10_000, brute_ms=690.0, index_ms=12.0, control_ms=120.0,
                        recall={10: (1.0, 1.0), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                        insert_bare_ms=900.0, insert_indexed_ms=1400.0)]
    out = format_report(steps, provenance="resampled", measured_to=10_000)
    assert "CONTROL DIVERGED" in out
