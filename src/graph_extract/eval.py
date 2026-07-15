"""Evaluation reports for the viability verdict.

Reports, each independently runnable against a live (or seeded test) graph:

- `dedup_report`   -- did canonical concepts merge to one `:Entity` node, and
                       did distinct concepts stay separate?
- `noise_report`   -- deterministic % of entities/facts flagged by
                       `noise_filter.is_noise` (no LLM).
- `dedup_report_v2` -- acceptance labels from `quality_labels`: merge counts +
                       cross-vendor support, distinct-pair collapse, and
                       suspect false merges (no LLM).
- `provenance_report` -- does fact -> episode -> article -> url resolve?
- `cost_report`    -- token usage from the module-level usage tally, plus a
                       simple, clearly-labeled full-corpus extrapolation.
- `fact_quality`   -- LLM-judge faithfulness sample (fact vs. its supporting
                       episode content) for a human spot-check.
- `type_precision` -- LLM-judge check of assigned entity types against the
                       ontology's type definitions.
"""
from __future__ import annotations

import re

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.noise_filter import is_noise
from graph_extract.ontology import ENTITY_TYPES
from graph_extract.provenance import Provenance
from graph_extract.usage import get_tally, instrument

# Rough, clearly-labeled assumption used only for the cost extrapolation --
# the actual AWS Backup docs corpus this project targets. Not derived from
# any live count; the CLI/report consumer should treat it as illustrative.
ASSUMED_FULL_CORPUS_EPISODES = 105_000


async def dedup_report(driver, group_id, canon_merge: list[str],
                       distinct_pairs: list[tuple[str, str]]) -> dict:
    out: dict = {"merge": {}, "distinct": [], "totals": {}}
    async with driver.session() as s:
        for name in canon_merge:
            r = await s.run("MATCH (e:Entity {group_id:$g}) WHERE toLower(e.name)=toLower($n) "
                            "RETURN count(e) AS c", g=group_id, n=name)
            out["merge"][name] = {"node_count": (await r.single())["c"]}
        for a, b in distinct_pairs:
            r = await s.run(
                "MATCH (e:Entity {group_id:$g}) WHERE toLower(e.name) IN [toLower($a),toLower($b)] "
                "RETURN count(e) AS c", g=group_id, a=a, b=b)
            c = (await r.single())["c"]
            out["distinct"].append({"pair": [a, b], "node_count": c, "collapsed": c < 2})
        r = await s.run("MATCH (e:Entity {group_id:$g}) RETURN count(e) AS c", g=group_id)
        out["totals"]["entities"] = (await r.single())["c"]
        # Secondary type label per node (e.g. :Entity:Capability) -> count.
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) UNWIND labels(e) AS l WITH l WHERE l <> 'Entity' "
            "RETURN l AS label, count(*) AS c", g=group_id)
        out["totals"]["by_label"] = {rec["label"]: rec["c"] async for rec in r}
    return out


_NOISE_SAMPLE_CAP = 20


async def noise_report(driver, group_id) -> dict:
    """Deterministic noise rate over entities and facts. NO LLM.

    An entity is noise per `noise_filter.is_noise` on its name; a fact
    (`RELATES_TO` edge) is noise when either endpoint entity is noise.
    """
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) RETURN e.name AS name", g=group_id)
        entity_names = [rec["name"] async for rec in r]
        r = await s.run(
            "MATCH (a:Entity)-[f:RELATES_TO {group_id:$g}]->(b:Entity) "
            "RETURN a.name AS a, b.name AS b", g=group_id)
        fact_endpoints = [(rec["a"], rec["b"]) async for rec in r]

    noisy_entities = [n for n in entity_names if is_noise(n or "")]
    noisy_facts = sum(
        1 for a, b in fact_endpoints if is_noise(a or "") or is_noise(b or ""))

    e_total, f_total = len(entity_names), len(fact_endpoints)
    return {
        "entities": {
            "total": e_total,
            "noise": len(noisy_entities),
            "rate": (len(noisy_entities) / e_total) if e_total else 0.0,
            "by_flag_sample": noisy_entities[:_NOISE_SAMPLE_CAP],
        },
        "facts": {
            "total": f_total,
            "noise": noisy_facts,
            "rate": (noisy_facts / f_total) if f_total else 0.0,
        },
    }


async def _cross_vendor_names(session, group_id) -> set[str]:
    r = await session.run(
        "MATCH (v:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
        "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(:Episodic)-[:MENTIONS]->"
        "(e:Entity {group_id:$g}) "
        "WITH e, count(DISTINCT v.name) AS nv WHERE nv > 1 RETURN e.name AS name",
        g=group_id)
    return {rec["name"] async for rec in r}


async def dedup_report_v2(driver, group_id, labels) -> dict:
    """Acceptance-label dedup report. Deterministic, NO LLM.

    `labels` is the `graph_extract.quality_labels` module (SHOULD_MERGE,
    SHOULD_DISTINCT, VENDOR_TOKENS as data).
    """
    out: dict = {"should_merge": {}, "should_distinct": [], "suspect_false_merge": {}}
    async with driver.session() as s:
        cross_vendor = await _cross_vendor_names(s, group_id)
        cross_vendor_lower = {n.lower() for n in cross_vendor}

        for name in labels.SHOULD_MERGE:
            r = await s.run(
                "MATCH (e:Entity {group_id:$g}) WHERE toLower(e.name)=toLower($n) "
                "RETURN count(e) AS c", g=group_id, n=name)
            out["should_merge"][name] = {
                "node_count": (await r.single())["c"],
                "cross_vendor": name.lower() in cross_vendor_lower,
            }

        def _forms(member):
            return [member] if isinstance(member, str) else list(member)

        async def _node_ids(forms):
            r = await s.run(
                "MATCH (e:Entity {group_id:$g}) "
                "WHERE toLower(e.name) IN $forms "
                "RETURN collect(DISTINCT elementId(e)) AS ids",
                g=group_id, forms=[f.lower() for f in forms])
            return set((await r.single())["ids"])

        for a, b in labels.SHOULD_DISTINCT:
            a_ids = await _node_ids(_forms(a))
            b_ids = await _node_ids(_forms(b))
            if not a_ids or not b_ids:
                state = "absent"
            elif a_ids & b_ids:
                state = "merged"
            else:
                state = "distinct"
            out["should_distinct"].append({
                "pair": [a, b],
                "state": state,
                "collapsed": state == "merged",
                "a_nodes": len(a_ids), "b_nodes": len(b_ids),
            })

    # Vendor-branded names (contain a VENDOR_TOKENS token) that nonetheless
    # have cross-vendor episode support -> likely false merges.
    suspects = sorted(
        n for n in cross_vendor
        if any(tok in n.lower() for tok in labels.VENDOR_TOKENS))
    out["suspect_false_merge"] = {"names": suspects, "count": len(suspects)}
    return out


async def provenance_report(driver, group_id, sample: int) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() RETURN f.uuid AS uuid "
            "ORDER BY rand() LIMIT $n", g=group_id, n=sample)
        uuids = [rec["uuid"] async for rec in r]

    prov = Provenance(driver)
    resolved = 0
    dangling = 0
    examples: list[dict] = []
    for u in uuids:
        chain = await prov.resolve_chain(u)
        ok = any(c.get("url") and c.get("article_id") for c in chain)
        if ok:
            resolved += 1
        else:
            dangling += 1
        examples.append({"uuid": u, "resolved": ok, "chain": chain})

    total = len(uuids)
    return {
        "sampled": total,
        "resolved": resolved,
        "dangling": dangling,
        "resolution_rate": (resolved / total) if total else 0.0,
        "examples": examples,
    }


async def cost_report(episodes_processed: int = 0,
                      assumed_full_corpus_episodes: int = ASSUMED_FULL_CORPUS_EPISODES) -> dict:
    """Read the module-level usage tally and extrapolate to a full corpus.

    `episodes_processed` is the number of episodes actually ingested during
    the run that produced the current tally (the caller -- Task 11's pilot
    run -- knows this from its `IngestResult.episodes_added`). When it's not
    supplied (or is 0) the extrapolation field is left as 0.0 rather than
    dividing by zero; the raw tally is still returned either way.
    """
    t = get_tally()
    total_tokens = t.prompt_tokens + t.completion_tokens
    tokens_per_episode = (total_tokens / episodes_processed) if episodes_processed else 0.0
    return {
        "prompt_tokens": t.prompt_tokens,
        "completion_tokens": t.completion_tokens,
        "total_tokens": total_tokens,
        "calls": t.calls,
        "by_call": dict(t.by_call),
        "episodes_processed": episodes_processed,
        "extrapolation": {
            "tokens_per_episode": tokens_per_episode,
            "assumed_full_corpus_episodes": assumed_full_corpus_episodes,
            "estimated_full_corpus_tokens": tokens_per_episode * assumed_full_corpus_episodes,
            "note": "tokens_per_episode * assumed_full_corpus_episodes; "
                    "assumed_full_corpus_episodes is a rough, labeled guess, not measured.",
        },
    }


_JUDGE_PROMPT = (
    "Is the following FACT supported by the TEXT below it? "
    "You may reason about it first, but you MUST end your response with your "
    "verdict as a final single word on its own: yes or no.\n\n"
    "FACT: {fact}\n\nTEXT: {content}"
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_YES_NO_RE = re.compile(r"\b(yes|no)\b")


def _parse_yes_no(content: str | None) -> bool | None:
    """Parse a judge verdict out of possibly reasoning-model output.

    `glm-5.2` (the judge) is a reasoning model that commonly emits
    chain-of-thought text before its final yes/no, so a naive
    `.startswith("yes")` misreads "reasoning... yes" as "no". Instead: strip
    any `<think>...</think>` preamble, then take the LAST word-boundary
    yes/no token in the remaining text -- for a reasoning-then-answer
    response, the final verdict is last. Returns `None` if no yes/no token
    is found (unparseable), rather than silently guessing "no".
    """
    if content is None:
        return None
    text = _THINK_TAG_RE.sub("", content.strip().lower())
    matches = _YES_NO_RE.findall(text)
    if not matches:
        return None
    return matches[-1] == "yes"


def _judge_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    if settings.judge_base_url:
        base_url, model = settings.judge_base_url, settings.judge_model
        api_key = settings.judge_api_key or "not-needed"
    else:
        # Fall back to the LLM endpoint + its key (may be a cloud key, e.g. OpenRouter).
        base_url, model = settings.llm_base_url, settings.llm_model
        api_key = settings.judge_api_key or settings.llm_api_key
    if base_url == settings.llm_base_url and model == settings.llm_model:
        # The resolved judge is the extraction model itself -- never let the
        # judge silently grade its own output. A shared/empty judge_api_key
        # (reusing llm_api_key) is fine as long as base_url/model differ.
        raise ValueError(
            "Judge model resolves to the extraction model (self-judging). "
            "Set JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY in .env to an "
            "independent judge (e.g. glm-5.2:cloud)."
        )
    client = instrument(AsyncOpenAI(api_key=api_key, base_url=base_url))
    return client, model


async def fact_quality(driver, settings: ExtractSettings, sample: int) -> dict:
    async with driver.session() as s:
        # Judge each fact against ALL its supporting episodes (not just the
        # first), scoped to this group_id.
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
            "UNWIND f.episodes AS epu "
            "MATCH (e:Episodic {uuid: epu}) WHERE e.content IS NOT NULL "
            "WITH f, collect(e.content) AS contents WHERE size(contents) > 0 "
            "RETURN f.uuid AS uuid, f.fact AS fact, contents AS contents "
            "ORDER BY rand() LIMIT $n", g=settings.group_id, n=sample)
        rows = [dict(rec) async for rec in r]

    client, model = _judge_client_and_model(settings)
    results: list[dict] = []
    supported = 0
    unsupported = 0
    unparseable = 0
    try:
        for row in rows:
            content = "\n---\n".join(row["contents"])
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                          "content": _JUDGE_PROMPT.format(fact=row["fact"], content=content)}],
                temperature=0.0,
                max_tokens=600,
            )
            raw_answer = resp.choices[0].message.content or ""
            verdict = _parse_yes_no(raw_answer)
            if verdict is True:
                supported += 1
            elif verdict is False:
                unsupported += 1
            else:
                unparseable += 1
            results.append({"uuid": row["uuid"], "fact": row["fact"], "verdict": verdict,
                            "raw_answer": raw_answer})
    finally:
        await client.close()

    total = len(results)
    parsed = supported + unsupported
    return {
        "sampled": total,
        "supported": supported,
        "unsupported": unsupported,
        "unparseable": unparseable,
        "precision": (supported / parsed) if parsed else 0.0,
        "results": results,
    }


def _type_definitions() -> str:
    return "\n".join(
        f"- {name}: {(model.__doc__ or '').strip()}"
        for name, model in ENTITY_TYPES.items())


_TYPE_JUDGE_PROMPT = (
    "You are auditing entity typing in a knowledge graph about backup products.\n"
    "Given the ENTITY name below, pick the single best type from these "
    "definitions:\n{definitions}\n\n"
    "You may reason first, but you MUST end your response with the chosen "
    "type name as a final single word on its own line.\n\n"
    "ENTITY: {name}"
)

_TYPE_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in ENTITY_TYPES) + r")\b", re.IGNORECASE)

_CANON_TYPE = {t.lower(): t for t in ENTITY_TYPES}


def _parse_type(content: str | None) -> str | None:
    """Extract the judge's final type verdict from reasoning-model output.

    GLM (the judge) reasons before answering, so type names may appear
    mid-reasoning; the FINAL occurrence is the verdict. Strips any
    `<think>...</think>` preamble, matches known type names
    case-insensitively, and returns the last match canonicalized -- or
    `None` when no type name appears (unparseable).
    """
    if content is None:
        return None
    text = _THINK_TAG_RE.sub("", content.strip())
    matches = _TYPE_NAME_RE.findall(text)
    if not matches:
        return None
    return _CANON_TYPE[matches[-1].lower()]


async def type_precision(driver, settings: ExtractSettings, sample: int) -> dict:
    """LLM-judge check that entities carry the right type label.

    Samples `(name, type)` pairs -- type is the secondary node label next to
    `:Entity` -- and asks the JUDGE model (GLM, via `_judge_client_and_model`;
    never the extraction model) to pick the best type from the ontology's
    type definitions. `max_tokens=600` because the judge is a reasoning
    model and a small cap truncates the final answer to empty.
    """
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) "
            "UNWIND labels(e) AS l WITH e, l WHERE l <> 'Entity' "
            "WITH e, collect(l)[0] AS type "
            "RETURN e.name AS name, type AS type ORDER BY rand() LIMIT $n",
            g=settings.group_id, n=sample)
        rows = [dict(rec) async for rec in r]

    client, model = _judge_client_and_model(settings)
    definitions = _type_definitions()
    per_type: dict[str, dict] = {}
    misclassifications: list[dict] = []
    correct = 0
    unparseable = 0
    try:
        for row in rows:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": _TYPE_JUDGE_PROMPT.format(
                               definitions=definitions, name=row["name"])}],
                temperature=0.0,
                max_tokens=600,
            )
            raw_answer = resp.choices[0].message.content or ""
            judged = _parse_type(raw_answer)
            assigned = row["type"]
            bucket = per_type.setdefault(assigned, {"sampled": 0, "correct": 0})
            bucket["sampled"] += 1
            if judged is None:
                unparseable += 1
            elif judged == assigned:
                correct += 1
                bucket["correct"] += 1
            else:
                misclassifications.append(
                    {"name": row["name"], "assigned": assigned, "judged": judged,
                     "raw_answer": raw_answer})
    finally:
        await client.close()

    for bucket in per_type.values():
        bucket["precision"] = (
            bucket["correct"] / bucket["sampled"]) if bucket["sampled"] else 0.0
    total = len(rows)
    parsed = total - unparseable
    return {
        "sampled": total,
        "correct": correct,
        "unparseable": unparseable,
        "precision": (correct / parsed) if parsed else 0.0,
        "per_type": per_type,
        "misclassifications": misclassifications,
    }
