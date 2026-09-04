"""The one definition of episode liveness. These are string-shape tests; Task 5
proves the composed Cypher actually behaves correctly against a real Neo4j."""
from __future__ import annotations

from graph_extract.episode_liveness import ALIVE_EPISODE, ALIVE_LINK


def test_link_predicate_covers_article_removal_and_edge_supersession():
    assert "a.removed" in ALIVE_LINK
    assert "he.superseded" in ALIVE_LINK


def test_episode_predicate_covers_episode_removal_and_live_links():
    assert "e.removed" in ALIVE_EPISODE
    assert "live_links" in ALIVE_EPISODE


def test_missing_properties_default_to_alive():
    """A missing property must mean ALIVE -- this is what makes the 161 existing
    unflagged edges correct without a backfill migration."""
    for pred in (ALIVE_LINK, ALIVE_EPISODE):
        for prop in ("a.removed", "he.superseded", "e.removed"):
            if prop in pred:
                assert f"coalesce({prop}, false)" in pred, prop


def test_liveness_never_consults_the_episode_node_superseded_flag():
    """Liveness is per-EDGE: one episode can be referenced by more than one article,
    so a node-level flag would kill it for every article at once.

    Substring check must exclude the edge alias `he.superseded` -- "he." ends in
    "e", so a naive `"e.superseded" in pred` false-positives on it.
    """
    for pred in (ALIVE_LINK, ALIVE_EPISODE):
        assert "e.superseded" not in pred.replace("he.superseded", "")
