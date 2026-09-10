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


@dataclass
class ContextResult:
    text: str
    fact_uuids: set[str]
    # uuid -> fact text, for exactly the facts included above. The report verifier
    # judges each finding against the text of the facts it cites; `text` renders
    # them into one blob, which is not machine-addressable. Defaulted so existing
    # constructions (several test fakes) keep working.
    fact_texts: dict[str, str] = field(default_factory=dict)


def assemble_context(members: list[EntityRow], facts: list[FactRow], *,
                     top_entities: int, token_budget: int) -> ContextResult:
    """Render a community's context: top-degree members, then intra-community
    facts current-first / recency-desc, each labelled `[uuid]`, truncated to a
    ~token_budget (approximated as budget*4 chars). Returns the text plus the set
    of fact UUIDs actually included (the citable universe)."""
    lines: list[str] = ["ENTITIES:"]
    for e in sorted(members, key=lambda e: e.degree, reverse=True)[:top_entities]:
        lines.append(f"- {e.name} ({e.type}): {e.summary}")
    lines.append("\nFACTS:")
    char_budget = token_budget * 4
    used = len("\n".join(lines))
    included: set[str] = set()
    texts: dict[str, str] = {}
    # current facts (invalid_at is None) first, then superseded; each group by
    # valid_at descending (recency). reverse=True puts empty/None valid_at last.
    current = [f for f in facts if f.invalid_at is None]
    superseded = [f for f in facts if f.invalid_at is not None]
    current.sort(key=lambda f: f.valid_at or "", reverse=True)
    superseded.sort(key=lambda f: f.valid_at or "", reverse=True)
    for f in current + superseded:
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
