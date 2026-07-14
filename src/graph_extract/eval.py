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

import re

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

    `gpt-oss-20b` (the judge) is a reasoning model that commonly emits
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
    for row in rows:
        content = "\n---\n".join(row["contents"])
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                      "content": _JUDGE_PROMPT.format(fact=row["fact"], content=content)}],
            temperature=0.0,
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
