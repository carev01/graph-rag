"""Evaluation reports for the viability verdict.

Four reports, each independently runnable against a live (or seeded test)
graph:

- `dedup_report`   -- did canonical concepts merge to one `:Entity` node, and
                       did distinct concepts stay separate?
- `provenance_report` -- does fact -> episode -> article -> url resolve?
- `cost_report`    -- token usage from the module-level usage tally, plus a
                       simple, clearly-labeled full-corpus extrapolation.
- `fact_quality`   -- LLM-judge faithfulness sample (fact vs. its supporting
                       episode content) for a human spot-check.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.provenance import Provenance
from graph_extract.usage import get_tally, instrument

# Rough, clearly-labeled assumption used only for the cost extrapolation --
# the actual AWS Backup docs corpus this project targets. Not derived from
# any live count; the CLI/report consumer should treat it as illustrative.
ASSUMED_FULL_CORPUS_EPISODES = 105_000


async def dedup_report(driver, group_id, canon_merge: list[str],
                       distinct_pairs: list[tuple[str, str]]) -> dict:
    out = {"merge": {}, "distinct": [], "totals": {}}
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


async def provenance_report(driver, sample: int) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO]->() RETURN f.uuid AS uuid "
            "ORDER BY rand() LIMIT $n", n=sample)
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
    "Answer with a single word: yes or no.\n\n"
    "FACT: {fact}\n\nTEXT: {content}"
)


def _judge_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    if settings.judge_base_url:
        base_url, model = settings.judge_base_url, settings.judge_model
    else:
        base_url, model = settings.llm_base_url, settings.llm_model
    client = instrument(AsyncOpenAI(api_key="not-needed", base_url=base_url))
    return client, model


async def fact_quality(driver, settings: ExtractSettings, sample: int) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO]->() "
            "WITH f, f.episodes[0] AS epu "
            "MATCH (e:Episodic {uuid: epu}) WHERE e.content IS NOT NULL "
            "RETURN DISTINCT f.uuid AS uuid, f.fact AS fact, e.content AS content "
            "ORDER BY rand() LIMIT $n", n=sample)
        rows = [dict(rec) async for rec in r]

    client, model = _judge_client_and_model(settings)
    results: list[dict] = []
    yes = 0
    for row in rows:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                      "content": _JUDGE_PROMPT.format(fact=row["fact"], content=row["content"])}],
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        verdict = answer.startswith("yes")
        if verdict:
            yes += 1
        results.append({"uuid": row["uuid"], "fact": row["fact"], "verdict": verdict,
                        "raw_answer": answer})

    total = len(results)
    return {
        "sampled": total,
        "precision": (yes / total) if total else 0.0,
        "results": results,
    }
