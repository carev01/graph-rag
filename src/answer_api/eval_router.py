"""@live harness: run router_golden.json through the /answer router, score routing
accuracy + citation grounding + LLM-judge faithfulness, run the comparative broad
pass, and write docs/superpowers/router-eval-report.md.

    uv run --extra dev python -m answer_api.eval_router
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from neo4j import AsyncGraphDatabase

from answer_api import router as router_mod
from answer_api.golden import precision_at_k
from answer_api.global_search import _map_client_and_model
from answer_api.router import _cheap_classify_client
from answer_api.router_eval import _parse_judge_score, aggregate, routing_hit
from answer_api.synthesize import _synthesis_client_and_model
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


async def _faithfulness_judge(client, model, q, answer, cited_facts) -> int:
    facts_block = "\n".join(f"- {f}" for f in cited_facts) or "(none)"
    resp = await client.chat.completions.create(
        # GLM-5.2 is a reasoning model: a tiny cap burns the whole budget on
        # reasoning tokens and returns EMPTY content (finish_reason='stop',
        # content='') -> _parse_judge_score -> 0 for every question. Give reasoning
        # headroom; the final content is then just the bare integer. (Same lesson as
        # theme-builder reports / the type_precision judge.)
        model=model, temperature=0, max_tokens=2000,
        messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
            q=q, answer=answer, facts=facts_block)}])
    return _parse_judge_score(resp.choices[0].message.content or "")


async def _cited_fact_texts(driver, group_id, fact_uuids) -> list[str]:
    if not fact_uuids:
        return []
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
            "RETURN f.fact AS fact", g=group_id, u=list(fact_uuids))
        return [rec["fact"] async for rec in r]


async def _score_one(clients, q, mode_override, settings):
    graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel = clients
    env = await router_mod.answer_router(
        graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel,
        q=q["question"], mode_override=mode_override, vendor=None, settings=settings)
    ghit = (precision_at_k(env["citations"], q["expected_article_ids"])
            if q["expected_article_ids"] else None)
    facts = await _cited_fact_texts(driver, settings.group_id,
                                    [c["fact_uuid"] for c in env["citations"]])
    faith = await _faithfulness_judge(sc, sm, q["question"], env["answer"], facts)
    return env, ghit, faith


async def run_eval(clients, questions, settings) -> dict:
    per_question: list[dict] = []
    for q in questions:
        try:
            env, ghit, faith = await _score_one(clients, q, None, settings)
            chosen = env["routing"]["chosen"]
            rec: dict = {"question": q["question"], "intent": q["intent"], "chosen": chosen,
                         "routing_hit": routing_hit(chosen, q["expected_modes"]),
                         "grounding_hit": ghit, "faithfulness": faith}
            if q["intent"] in ("global", "drift"):
                comp: dict = {}
                for m in ("local", "global", "drift"):
                    _e, _g, _f = await _score_one(clients, q, m, settings)
                    comp[m] = {"grounding_hit": _g, "faithfulness": _f}
                rec["comparative"] = comp
        except Exception:
            logger.warning("eval question failed: %s", q["question"], exc_info=True)
            rec = {"question": q["question"], "intent": q["intent"], "chosen": "error",
                   "routing_hit": False, "grounding_hit": None, "faithfulness": 0}
        per_question.append(rec)
    return aggregate(per_question)


def format_report(summary: dict) -> str:
    lines = ["# /answer Router Golden-Set Eval\n",
             f"Questions: {summary['n']}\n",
             f"**Routing accuracy: {summary['routing_accuracy']:.2f}**",
             f"by intent: {summary['routing_by_intent']}",
             f"Grounding precision: {summary['grounding_precision']}",
             f"by mode: {summary['grounding_by_mode']}",
             f"Faithfulness mean: {summary['faithfulness_mean']:.2f}",
             f"by mode: {summary['faithfulness_by_mode']}\n",
             f"Comparative (broad): {summary['comparative']}",
             f"drift_wins: {summary['drift_wins']}\n",
             "## Per question\n",
             "| intent | chosen | routing | grounding | faithfulness | question |",
             "|---|---|---|---|---|---|"]
    for r in summary["per_question"]:
        lines.append(f"| {r['intent']} | {r['chosen']} | {r['routing_hit']} | "
                     f"{r['grounding_hit']} | {r['faithfulness']} | {r['question']} |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    settings = get_extract_settings()
    questions = json.loads(_QUESTIONS.read_text())
    graphiti = build_graphiti(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    embedder = build_embedder(settings)
    sc, sm = _synthesis_client_and_model(settings)
    mc, mm = _map_client_and_model(settings)
    cc, cmodel = _cheap_classify_client(settings)
    clients = (graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel)
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
        await mc.close()
        if cc is not None:
            await cc.close()


if __name__ == "__main__":
    asyncio.run(main())
