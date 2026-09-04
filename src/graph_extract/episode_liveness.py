"""The single definition of episode liveness (design decision #3).

`HAS_EPISODE` carries two independent meanings and they must not be conflated:

* the edge's **existence** means the episode is CITABLE -- `resolve_citations`
  traverses it regardless of any flag, so a fact that was ever true stays
  attributable to the document version that asserted it;
* the edge's **`superseded` flag** means the episode is DEAD -- it no longer
  reflects current content.

Before this module the system used edge *existence* for both, so `Provenance.link`
deleted the edge on re-key (destroying the citation) while
`_supersede_trailing_episodes` kept it (so a shrunk article's facts never expired).

An episode is ALIVE iff its edge is not superseded, the episode is not removed, and
the article is not removed. A fact is LIVE iff at least one supporting episode is
alive. **The staleness sweep is the only evaluator** -- it materialises the result
into `invalid_at`, which search and timeline consume. Evaluating it a second time in
retrieval would reintroduce exactly the divergence this module exists to remove.

Every clause uses `coalesce(..., false)` so a MISSING property means alive: that is
what lets pre-existing edges keep working with no migration.
"""
from __future__ import annotations

#: Is this Article->Episodic link live? Requires `a` (:Article) and `he`
#: (the HAS_EPISODE relationship) in scope. Deliberately reads the EDGE's
#: `superseded`, never the Episodic node's -- one episode may be referenced by
#: several articles, and it dies only for the one that superseded it.
ALIVE_LINK = (
    "coalesce(a.removed, false) = false "
    "AND coalesce(he.superseded, false) = false"
)

#: Is this episode alive? Requires `e` (:Episodic, possibly NULL from an
#: OPTIONAL MATCH) and `live_links` (the count of links passing ALIVE_LINK).
ALIVE_EPISODE = (
    "e IS NOT NULL "
    "AND coalesce(e.removed, false) = false "
    "AND live_links > 0"
)
