"""Pure per-community report-context assembly. No I/O — the caller fetches the
member/fact rows from Neo4j; this ranks, orders, budgets, and labels them so the
report LLM can cite fact UUIDs. Kept pure so it is unit-testable without a DB."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntityRow:
    uuid: str
    name: str
    type: str
    summary: str
    degree: int


@dataclass
class FactRow:
    uuid: str
    fact: str
    valid_at: str | None
    invalid_at: str | None
    name: str
    # The fact edge's endpoint entities. Context selection gives the budget to facts
    # about the community's top entities first; a row without endpoints is still
    # included, in the recency-ordered fill after them.
    source_uuid: str = ""
    target_uuid: str = ""


@dataclass
class ContextResult:
    text: str
    fact_uuids: set[str]
    # uuid -> fact text, for exactly the facts included above. The report verifier
    # judges each finding against the text of the facts it cites; `text` renders
    # them into one blob, which is not machine-addressable. Defaulted so existing
    # constructions (several test fakes) keep working.
    fact_texts: dict[str, str] = field(default_factory=dict)


def _recency_order(facts: list[FactRow]) -> list[FactRow]:
    """Current facts (invalid_at is None) first, then superseded; each group by
    valid_at descending. reverse=True puts empty/None valid_at last."""
    current = sorted((f for f in facts if f.invalid_at is None),
                     key=lambda f: f.valid_at or "", reverse=True)
    superseded = sorted((f for f in facts if f.invalid_at is not None),
                        key=lambda f: f.valid_at or "", reverse=True)
    return current + superseded


def _select_order(top: list[EntityRow], facts: list[FactRow]) -> list[FactRow]:
    """Facts about the top entities first, the entities taking turns in degree order
    (each one's facts recency-ordered), then every other fact by recency.

    Recency alone was the order until 2026-10-11, and on a hub community it is
    effectively random: a 7,300-fact community kept its 42 newest facts and none of
    the 82 about `AWS Backup Vault Lock`, one of its top-30 entities, so the report --
    and every global answer built on it -- never mentioned Vault Lock. Taking turns
    stops one hub entity from spending the whole budget on itself."""
    ordered = _recency_order(facts)
    per_entity: dict[str, list[FactRow]] = {e.uuid: [] for e in top}
    for f in ordered:
        for end in (f.source_uuid, f.target_uuid):
            if end in per_entity:
                per_entity[end].append(f)
    out: list[FactRow] = []
    seen: set[str] = set()
    queues = [per_entity[e.uuid] for e in top]
    depth = 0
    while any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q) and q[depth].uuid not in seen:
                seen.add(q[depth].uuid)
                out.append(q[depth])
        depth += 1
    out.extend(f for f in ordered if f.uuid not in seen)
    return out


TOKENS_PER_FACT = 80


def context_budget(n_facts: int, *, base: int, cap: int) -> int:
    """Token budget for a community's context: `base`, grown at ~TOKENS_PER_FACT per
    fact up to `cap`, so a fact-rich community is not squeezed into the budget sized
    for a small one."""
    return max(base, min(cap, n_facts * TOKENS_PER_FACT))


def assemble_context(members: list[EntityRow], facts: list[FactRow], *,
                     top_entities: int, token_budget: int) -> ContextResult:
    """Render a community's context: top-degree members, then intra-community facts
    (`_select_order`: the top members' facts first, taking turns, then the rest by
    recency), each labelled `[uuid]`, truncated to a ~token_budget (approximated as
    budget*4 chars). Returns the text plus the set of fact UUIDs actually included
    (the citable universe)."""
    lines: list[str] = ["ENTITIES:"]
    top = sorted(members, key=lambda e: e.degree, reverse=True)[:top_entities]
    for e in top:
        lines.append(f"- {e.name} ({e.type}): {e.summary}")
    lines.append("\nFACTS:")
    char_budget = token_budget * 4
    used = len("\n".join(lines))
    included: set[str] = set()
    texts: dict[str, str] = {}
    for f in _select_order(top, facts):
        suffix = (f" (valid {f.valid_at}" + (f", invalid {f.invalid_at})" if f.invalid_at else ")")) \
            if f.valid_at else ""
        line = f"[{f.uuid}] {f.fact}{suffix}"
        if used + len(line) + 1 > char_budget:
            if included:
                break
            # must include at least one fact, but never blow the budget: clip it,
            # preserving the leading [uuid] label so it stays citable.
            room = max(len(f"[{f.uuid}] "), char_budget - used - 1)
            lines.append(line[:room])
            included.add(f.uuid)
            texts[f.uuid] = f.fact
            break
        lines.append(line)
        included.add(f.uuid)
        texts[f.uuid] = f.fact
        used += len(line) + 1
    return ContextResult(text="\n".join(lines), fact_uuids=included, fact_texts=texts)
