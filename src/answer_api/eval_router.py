"""@live harness: run router_golden.json through the /answer router, score routing
accuracy + citation grounding + LLM-judge faithfulness, run the comparative broad
pass, and write docs/superpowers/router-eval-report.md.

    uv run --extra dev python -m answer_api.eval_router
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from answer_api import router as router_mod
from answer_api.drift import _REFUSAL as _DRIFT_REFUSAL
from answer_api.golden import precision_at_k
from answer_api.global_search import _REFUSAL as _GLOBAL_REFUSAL
from answer_api.global_search import _map_client_and_model
from answer_api.router import _cheap_classify_client
from answer_api.router_eval import _parse_judge_score, aggregate, routing_hit
from answer_api.synthesize import _REFUSAL as _SYNTH_REFUSAL
from answer_api.synthesize import (
    _range_markers, _synthesis_client_and_model, _usable_content,
)
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti

logger = logging.getLogger(__name__)
_QUESTIONS = Path(__file__).with_name("router_golden.json")

_JUDGE_PROMPT = (
    "You are scoring a backup-products ANSWER for faithfulness to its CITED FACTS. "
    "Score 0-5 how fully the answer's claims are supported by ONLY these facts "
    "(5 = every claim supported, 0 = unsupported/hallucinated). Reply with ONLY the "
    "integer.\n\nQUESTION: {q}\n\nANSWER:\n{answer}\n\nCITED FACTS:\n{facts}"
)


def _eval_judge_client_and_model(settings) -> tuple[AsyncOpenAI, str]:
    """Resolve the faithfulness judge, INDEPENDENT of synthesis.

    Uses `eval_judge_*` when set, else falls back to the judge/synthesis tier --
    and then refuses that fallback, because a judge scoring answers it wrote
    itself inflates faithfulness in a way the score cannot reveal. This mirrors
    the guard `graph_extract.eval._judge_client_and_model` already applies
    against judging with the extraction model.

    Failing loudly here is the point: a silently self-graded eval is worse than
    no eval, because it reads as evidence.
    """
    base = settings.eval_judge_base_url or settings.judge_base_url
    model = settings.eval_judge_model or settings.judge_model
    key = settings.eval_judge_api_key or settings.judge_api_key or "not-needed"
    if not base or not model:
        raise ValueError(
            "No eval judge configured. Set EVAL_JUDGE_BASE_URL / EVAL_JUDGE_MODEL "
            "(or JUDGE_BASE_URL / JUDGE_MODEL) in .env.")
    if base == settings.judge_base_url and model == settings.judge_model:
        raise ValueError(
            f"Eval judge resolves to the synthesis model ({model!r}) -- it would "
            "grade its own answers and inflate faithfulness. Set EVAL_JUDGE_MODEL "
            "to a different model, ideally a different family so the two do not "
            "share failure modes.")
    return AsyncOpenAI(api_key=key, base_url=base), model


async def _faithfulness_judge(client, model, q, answer, cited_facts) -> int | None:
    """Score faithfulness 0-5, or return None if the judge could not be measured.

    GLM-5.2 is a reasoning model: on heavy inputs it can burn the whole token
    budget on reasoning tokens and return NO content at all
    (finish_reason='length', content=None). Blindly parsing that with
    `_parse_judge_score` yields 0 -- "I could not measure this answer" recorded
    as "this answer is completely unfaithful", which corrupts the mean. So: give
    the first attempt a large reasoning budget, retry once with an even larger
    one if the reply is unusable, and if it is STILL unusable, return None
    rather than a fabricated 0. Whether a digit was actually found is checked
    directly against the raw content -- `_parse_judge_score` itself is not
    trusted to distinguish "no digit" from "judge said 0".
    """
    if not answer or not answer.strip():
        logger.warning(
            "faithfulness judge skipped for question %r: blank answer would score "
            "5 by vacuous truth; recording as unscored, not measured", q)
        return None
    stripped = answer.strip()
    # A refusal is non-blank, so it survives the check above and reaches the
    # judge with "CITED FACTS: (none)" -- which the judge scores 5 on "no
    # claims, so every claim is supported". That is the same vacuous-truth
    # inflation the blank-answer guard exists to prevent, just reached through
    # a different string. The three refusal strings are module-local and
    # worded differently per mode on purpose (design review) -- compare
    # against all three rather than merging them into one shared constant.
    if stripped in (_SYNTH_REFUSAL, _GLOBAL_REFUSAL, _DRIFT_REFUSAL):
        logger.warning(
            "faithfulness judge skipped for question %r: refusal answer would "
            "score 5 by vacuous truth; recording as unscored, not measured", q)
        return None

    facts_block = "\n".join(f"- {f}" for f in cited_facts) or "(none)"
    prompt = _JUDGE_PROMPT.format(q=q, answer=answer, facts=facts_block)

    async def _attempt(max_tokens: int):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        # `_usable_content` already handles the choices-less-HTTP-200 case (seen
        # in production, theme_builder/report.py commit a0603ec) and the
        # None/whitespace-only content case -- `resp.choices[0]` unguarded here
        # was the very IndexError this eval was written to catch downstream.
        content = _usable_content(resp)
        finish_reason = resp.choices[0].finish_reason if resp.choices else "no-choices"
        usable = (content is not None and finish_reason != "length"
                  and re.search(r"-?\d+", content) is not None)
        return finish_reason, content, usable

    finish_reason, content, usable = await _attempt(8000)
    if usable:
        return _parse_judge_score(content)

    finish_reason, content, usable = await _attempt(16000)
    if usable:
        return _parse_judge_score(content)

    logger.warning(
        "faithfulness judge unmeasurable for question %r after retry "
        "(finish_reason=%s); recording as unscored, not 0", q, finish_reason)
    return None


async def _cited_fact_texts(driver, group_id, fact_uuids) -> list[str]:
    if not fact_uuids:
        return []
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
            "RETURN f.fact AS fact", g=group_id, u=list(fact_uuids))
        return [rec["fact"] async for rec in r]


async def _score_one(clients, q, mode_override, settings):
    graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel, jc, jm = clients
    env = await router_mod.answer_router(
        graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel,
        q=q["question"], mode_override=mode_override, vendor=None, settings=settings)
    ghit = (precision_at_k(env["citations"], q["expected_article_ids"])
            if q["expected_article_ids"] else None)
    facts = await _cited_fact_texts(driver, settings.group_id,
                                    [c["fact_uuid"] for c in env["citations"]])
    faith = await _faithfulness_judge(jc, jm, q["question"], env["answer"], facts)
    return env, ghit, faith


async def run_eval(clients, questions, settings) -> dict:
    per_question: list[dict] = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        started = time.monotonic()
        try:
            env, ghit, faith = await _score_one(clients, q, None, settings)
            chosen = env["routing"]["chosen"]
            # `cited` and `ranges` are what the faithfulness score is silently
            # conditioned on (BACKLOG 0d): the judge sees only the cited facts,
            # and a surviving range "[1]-[26]" means it saw 2 of 26. The harness
            # does not persist answer text, so without these two numbers a
            # range-shorthand answer and a well-cited one are indistinguishable.
            rec: dict = {"question": q["question"], "intent": q["intent"], "chosen": chosen,
                         "routing_hit": routing_hit(chosen, q["expected_modes"]),
                         "grounding_hit": ghit, "faithfulness": faith,
                         "cited": len(env["citations"]),
                         "ranges": len(_range_markers(env["answer"])), "failed": False}
            if q["intent"] in ("global", "drift"):
                comp: dict = {}
                for m in ("local", "global", "drift"):
                    _e, _g, _f = await _score_one(clients, q, m, settings)
                    comp[m] = {"grounding_hit": _g, "faithfulness": _f}
                rec["comparative"] = comp
        except Exception:
            logger.warning("eval question failed: %s", q["question"], exc_info=True)
            rec = {"question": q["question"], "intent": q["intent"], "chosen": "error",
                   "routing_hit": False, "grounding_hit": None, "faithfulness": None,
                   "cited": None, "ranges": None, "failed": True}
        elapsed = time.monotonic() - started
        # A 2h09m run printed nothing until it finished, so a hung run and a working
        # one looked identical. One line per question makes progress visible.
        print(f"[{i}/{total}] {rec['intent']:8} -> {rec['chosen']:8} "
              f"routing={'-' if rec.get('failed') else rec['routing_hit']} "
              f"grounding={rec['grounding_hit']} "
              f"faith={rec['faithfulness'] if rec['faithfulness'] is not None else '-'} "
              f"cited={rec['cited']} ranges={rec['ranges']} "
              f"{elapsed:.0f}s", flush=True)
        per_question.append(rec)
    return aggregate(per_question)


def format_report(summary: dict) -> str:
    faith_mean = summary["faithfulness_mean"]
    faith_mean_str = f"{faith_mean:.2f}" if faith_mean is not None else "N/A"
    lines = ["# /answer Router Golden-Set Eval\n",
             f"Questions: {summary['n']} (failed: {summary['questions_failed']})\n",
             f"**Routing accuracy: {summary['routing_accuracy']:.2f}**",
             f"by intent: {summary['routing_by_intent']}",
             f"Grounding precision: {summary['grounding_precision']}",
             f"by mode: {summary['grounding_by_mode']}",
             f"Faithfulness mean: {faith_mean_str} "
             f"(unscored: {summary['faithfulness_unscored']}/{summary['n']})",
             f"by mode: {summary['faithfulness_by_mode']}\n",
             f"Comparative (broad): {summary['comparative']}",
             f"drift_wins: {summary['drift_wins']}\n",
             "## Per question\n",
             "| intent | chosen | routing | grounding | faithfulness | cited | ranges "
             "| question |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary["per_question"]:
        faith_cell = r["faithfulness"] if r["faithfulness"] is not None else "-"
        routing_cell = "-" if r.get("failed") else r["routing_hit"]
        # .get(): summaries written before the cited/ranges columns existed.
        cited_cell = r.get("cited") if r.get("cited") is not None else "-"
        ranges_cell = r.get("ranges") if r.get("ranges") is not None else "-"
        lines.append(f"| {r['intent']} | {r['chosen']} | {routing_cell} | "
                     f"{r['grounding_hit']} | {faith_cell} | {cited_cell} | "
                     f"{ranges_cell} | {r['question']} |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    settings = get_extract_settings()
    questions = json.loads(_QUESTIONS.read_text())
    graphiti = build_graphiti(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    embedder = build_embedder(settings)
    sc, sm = _synthesis_client_and_model(settings)
    jc, jm = _eval_judge_client_and_model(settings)
    mc, mm = _map_client_and_model(settings)
    cc, cmodel = _cheap_classify_client(settings)
    clients = (graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel, jc, jm)
    try:
        summary = await run_eval(clients, questions, settings)
        report = format_report(summary)
        out = (Path(__file__).resolve().parents[2]
               / "docs" / "superpowers" / "router-eval-report.md")
        out.write_text(report)
        print(report)
    finally:
        await graphiti.close()
        await driver.close()
        await sc.close()
        await jc.close()
        await mc.close()
        await embedder.client.close()   # standalone embedder owns a dedicated pool
        if cc is not None:
            await cc.close()


if __name__ == "__main__":
    asyncio.run(main())
