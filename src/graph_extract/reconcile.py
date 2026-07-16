"""Structural <-> semantic `SAME_AS` reconciliation.

Design decision #5 (see CLAUDE.md / graphrag-docextractor-plan.md SS4): the
deterministic structural layer (`:Vendor`/`:Product`, written by graph-sync
from the DocExtractor catalog) and Graphiti's semantic layer
(`:Entity:Vendor`/`:Entity:Product`, extracted from article content) denote
the same real-world vendors/products but are never merged -- they're linked
via a `SAME_AS` edge instead, so structural provenance (design decision #1)
and Graphiti's own schema (design decision #5) both stay untouched.

Matching is conservative and alias-driven (`vendor_aliases.accepted_forms`),
never fuzzy: a structural node links to a same-kind semantic `:Entity` only
if the entity's lowercased name is exactly one of the structural name's
accepted surface forms. This is what keeps noise entities (e.g. an
`:Entity:Vendor {name:'vaults'}` that survived extraction as junk) from ever
picking up a spurious link.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from graph_extract.vendor_aliases import accepted_forms

# The two structural node kinds that get reconciled -- see the ontology's
# structural layer (Vendor/Product), never Source/Chapter/Article (those
# have no semantic counterpart).
_KINDS = ("Vendor", "Product")

_SCAN = (
    "MATCH (v:{kind}) WHERE NOT v:Entity "
    "RETURN elementId(v) AS id, v.name AS name"
)

# elementId(v) pins the exact structural node -- names are not guaranteed
# unique, so re-matching by name here could link (or unlink) the wrong node.
# MERGE makes this idempotent: rerunning never creates a duplicate SAME_AS.
_LINK = (
    "MATCH (v:{kind}) WHERE elementId(v) = $id "
    "MATCH (e:Entity:{kind} {{group_id: $g}}) WHERE toLower(e.name) IN $forms "
    "MERGE (v)-[:SAME_AS]->(e) "
    "RETURN count(e) AS c"
)


async def reconcile_same_as(driver: AsyncDriver, group_id: str) -> dict:
    """Link structural Vendor/Product nodes to their semantic Entity twins.

    For every structural node (label `:Vendor` or `:Product`, NOT also
    `:Entity`), compute its accepted surface forms and `MERGE` a `SAME_AS`
    edge to every same-kind `:Entity` in `group_id` whose lowercased name is
    one of those forms.

    Returns `{"structural_scanned": int, "linked": int,
    "unmatched_structural": list[str]}`. `structural_scanned` accumulates
    across BOTH kinds (Vendor + Product) -- a running total, not the size of
    whichever kind's scan happened to run last. `linked` counts SAME_AS
    edges merged (matched-and-created-or-already-present) this run.
    `unmatched_structural` lists structural node names that matched no
    semantic entity at all.
    """
    structural_scanned = 0
    linked = 0
    unmatched: list[str] = []

    async with driver.session() as s:
        for kind in _KINDS:
            r = await s.run(_SCAN.format(kind=kind))
            structs = [dict(rec) async for rec in r]
            structural_scanned += len(structs)

            for st in structs:
                forms = sorted(accepted_forms(st["name"] or ""))
                link_result = await s.run(
                    _LINK.format(kind=kind),
                    id=st["id"],
                    g=group_id,
                    forms=forms,
                )
                link_record = await link_result.single()
                assert link_record is not None  # count() always returns exactly one row
                c = link_record["c"]
                linked += c
                if c == 0:
                    unmatched.append(st["name"])

    return {
        "structural_scanned": structural_scanned,
        "linked": linked,
        "unmatched_structural": sorted(set(unmatched)),
    }
