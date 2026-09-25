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

from answer_api import attribution, router as router_mod
from answer_api.drift import _REFUSAL as _DRIFT_REFUSAL
from answer_api.golden import precision_at_k
from answer_api.global_search import _REFUSAL as _GLOBAL_REFUSAL
from answer_api.global_search import _map_client_and_model
from answer_api.router import _cheap_classify_client
from answer_api.router_eval import (
    BAG_MARKERS, _parse_attribution, _parse_judge_score, aggregate, bag_share,
    markers_per_sentence, routing_hit,
)
from answer_api.scope import ScopeResolver
from answer_api.synthesize import _REFUSAL as _SYNTH_REFUSAL
from answer_api.synthesize import (
    _range_markers, _synthesis_client_and_model, _usable_content,
)
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti
from graph_extract.usage import bounded_llm_client

logger = logging.getLogger(__name__)
_QUESTIONS = Path(__file__).with_name("router_golden.json")

_JUDGE_PROMPT = (
    "You are scoring a backup-products ANSWER for faithfulness to its CITED FACTS. "
    "Score 0-5 how fully the answer's claims are supported by ONLY these facts "
    "(5 = every claim supported, 0 = unsupported/hallucinated). Reply with ONLY the "
    "integer.\n\nQUESTION: {q}\n\nANSWER:\n{answer}\n\nCITED FACTS:\n{facts}"
)

# Calibrated on scripts/calibrate_attribution_judge.py (known-count probes). The
# first cut ("attributes something to a vendor or product not among the labels")
# counted every component, workload or tool a correctly labelled fact names --
# Backup center, Azure Files, PowerShell, MABS -- as a misattribution: 6 on a
# deterministic Azure Backup timeline render. A misattribution is a claim moved
# to a DIFFERENT vendor/product line than its fact's label.
_ATTRIBUTION_PROMPT = (
    "Each cited fact is labelled (Vendor · Product) with the vendor and product whose "
    "documentation states it. Count the claims in the ANSWER that credit a vendor or "
    "product with something its cited fact states for a DIFFERENT vendor or product -- "
    "e.g. a fact labelled (AWS · AWS Backup) presented as true of Azure Backup, or "
    "presented as true of both. Naming components, features, workloads, tools, services "
    "or platforms that the cited fact itself mentions (a console, a storage service, a "
    "database, a CLI) is NOT a misattribution. Check every sentence against the labels "
    "of the markers it cites and count each misattributed claim separately. Reply with "
    "ONLY the integer (0 if none)."
    "\n\nQUESTION: {q}\n\nANSWER:\n{answer}\n\nCITED FACTS:\n{facts}"
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
    return bounded_llm_client(base, key,
                              reasoning_effort=settings.eval_judge_reasoning_effort), model


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


async def judge_attribution(client, model, q, answer, labelled_facts: str) -> int | None:
    """Count claims the ANSWER attributes to a vendor/product not on its cited
    facts' (Vendor · Product) labels (spec §6). Uses the SAME independent
    eval-judge client as `_faithfulness_judge` and the same bounded max_tokens
    retry: an unusable reply (no content, or `finish_reason == 'length'`) must
    never be coerced to 0 -- that would record "could not be measured" as "zero
    misattributions", the same defect the faithfulness judge guards against
    (memory: "LLM empty reply coerced to a value")."""
    prompt = _ATTRIBUTION_PROMPT.format(q=q, answer=answer, facts=labelled_facts)

    async def _attempt(max_tokens: int):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        content = _usable_content(resp)
        finish_reason = resp.choices[0].finish_reason if resp.choices else "no-choices"
        usable = (content is not None and finish_reason != "length"
                  and re.search(r"\d+", content) is not None)
        return finish_reason, content, usable

    finish_reason, content, usable = await _attempt(8000)
    if usable:
        return _parse_attribution(content)

    finish_reason, content, usable = await _attempt(16000)
    if usable:
        return _parse_attribution(content)

    logger.warning(
        "attribution judge unmeasurable for question %r after retry "
        "(finish_reason=%s); recording as unscored, not 0", q, finish_reason)
    return None


def _labelled_facts(citations: list[dict], texts: dict[str, str]) -> str:
    """One `fact_line` per citation, in citation order, paired by `fact_uuid`
    (R7) -- never by position. `_cited_fact_texts`'s Cypher `IN`-scan row order
    is not guaranteed to match the requested uuid list, and a fact_uuid shared
    by two citations would shift a positional pairing for every citation after
    it. A citation whose uuid has no text (edge gone) is skipped, not
    mislabeled against a neighbour's fact."""
    lines = [attribution.fact_line(c["marker"], texts[c["fact_uuid"]], c.get("sources", []))
             for c in citations if c["fact_uuid"] in texts]
    return "\n".join(lines) or "(none)"


async def _cited_fact_texts(driver, group_id, fact_uuids) -> list[str]:
    if not fact_uuids:
        return []
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
            "RETURN f.fact AS fact", g=group_id, u=list(fact_uuids))
        return [rec["fact"] async for rec in r]


async def _cited_fact_texts_by_uuid(driver, group_id, fact_uuids) -> dict[str, str]:
    """Fact text keyed by uuid (R7): `_cited_fact_texts`'s list return is fine
    for the faithfulness judge, which folds every fact into one unordered bag
    and never needs a fact tied back to a specific citation. The attribution
    judge's labelled fact lines DO need that tie -- each citation's own
    marker/sources must sit beside its own fact text, and Cypher's `IN`-scan
    row order is not guaranteed to match the requested list (nor survive a
    fact_uuid shared by two citations). Kept as a sibling function rather than
    changing `_cited_fact_texts`'s shape: that list is also persisted verbatim
    into `raw["cited_facts"]` and consumed as a bag by the faithfulness prompt,
    so reshaping it would ripple beyond this fix."""
    if not fact_uuids:
        return {}
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
            "RETURN f.uuid AS uuid, f.fact AS fact", g=group_id, u=list(fact_uuids))
        return {rec["uuid"]: rec["fact"] async for rec in r}


async def _score_one(clients, q, mode_override, settings, resolver):
    graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel, jc, jm = clients
    scope = resolver.resolve(q["question"])
    env = await router_mod.answer_router(
        graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel,
        q=q["question"], mode_override=mode_override, scope=scope, settings=settings)
    ghit = (precision_at_k(env["citations"], q["expected_article_ids"])
            if q["expected_article_ids"] else None)
    facts = await _cited_fact_texts(driver, settings.group_id,
                                    [c["fact_uuid"] for c in env["citations"]])
    faith = await _faithfulness_judge(jc, jm, q["question"], env["answer"], facts)
    texts = await _cited_fact_texts_by_uuid(driver, settings.group_id,
                                            [c["fact_uuid"] for c in env["citations"]])
    labelled = _labelled_facts(env["citations"], texts)
    misattributed = await judge_attribution(jc, jm, q["question"], env["answer"], labelled)
    return env, ghit, faith, facts, misattributed


async def run_eval(clients, questions, settings, resolver) -> dict:
    per_question: list[dict] = []
    # Every scored column this harness has gained -- `cited`, `ranges`, `mps`,
    # `bag_share` -- cost a full paid eval run to observe, because the harness
    # scored each answer and then discarded it. `raw` keeps the material each of
    # those was derived from, so the next question asked of a run is a re-read
    # rather than a re-run.
    raw: list[dict] = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        started = time.monotonic()
        try:
            env, ghit, faith, facts, misattributed = await _score_one(
                clients, q, None, settings, resolver)
            chosen = env["routing"]["chosen"]
            # `cited` and `ranges` are what the faithfulness score is silently
            # conditioned on (BACKLOG 0d): the judge sees only the cited facts,
            # and a surviving range "[1]-[26]" means it saw 2 of 26. The harness
            # does not persist answer text, so without these two numbers a
            # range-shorthand answer and a well-cited one are indistinguishable.
            # `mps` (markers per sentence, BACKLOG 0b): the distribution that
            # separates one marker per claim from 19 markers pasted on one
            # sentence -- identical in `cited`, `ranges` and the judge score.
            # `bag_share` is what `mps` cannot say: how much of the citation
            # mass sits in those bags. Half the global-mode answers carried a
            # >=8-marker sentence and all of them scored 4-5.
            mps = markers_per_sentence(env["answer"])
            rec: dict = {"question": q["question"], "intent": q["intent"], "chosen": chosen,
                         "routing_hit": routing_hit(chosen, q["expected_modes"]),
                         "grounding_hit": ghit, "faithfulness": faith,
                         "cited": len(env["citations"]),
                         "ranges": len(_range_markers(env["answer"])),
                         "mps_mean": round(sum(mps) / len(mps), 1) if mps else None,
                         "mps_max": max(mps) if mps else None,
                         "bag_share": _round_or_none(bag_share(env["answer"])),
                         "misattributed": misattributed,
                         "failed": False}
            raw.append({"question": q["question"], "intent": q["intent"], "chosen": chosen,
                        "answer": env["answer"], "cited_facts": facts,
                        "citations": env["citations"]})
            if q["intent"] in ("global", "drift"):
                comp: dict = {}
                for m in ("local", "global", "drift"):
                    _e, _g, _f, _facts, _mis = await _score_one(
                        clients, q, m, settings, resolver)
                    comp[m] = {"grounding_hit": _g, "faithfulness": _f}
                    raw.append({"question": q["question"], "intent": q["intent"],
                                "chosen": f"comparative:{m}", "answer": _e["answer"],
                                "cited_facts": _facts, "citations": _e["citations"]})
                rec["comparative"] = comp
        except Exception:
            logger.warning("eval question failed: %s", q["question"], exc_info=True)
            rec = {"question": q["question"], "intent": q["intent"], "chosen": "error",
                   "routing_hit": False, "grounding_hit": None, "faithfulness": None,
                   "cited": None, "ranges": None, "mps_mean": None, "mps_max": None,
                   "bag_share": None, "misattributed": None, "failed": True}
        elapsed = time.monotonic() - started
        # A 2h09m run printed nothing until it finished, so a hung run and a working
        # one looked identical. One line per question makes progress visible.
        print(f"[{i}/{total}] {rec['intent']:8} -> {rec['chosen']:8} "
              f"routing={'-' if rec.get('failed') else rec['routing_hit']} "
              f"grounding={rec['grounding_hit']} "
              f"faith={rec['faithfulness'] if rec['faithfulness'] is not None else '-'} "
              f"cited={rec['cited']} ranges={rec['ranges']} mps={_mps_cell(rec)} "
              f"bag={_cell(rec.get('bag_share'))} "
              f"misattributed={_cell(rec.get('misattributed'))} {elapsed:.0f}s", flush=True)
        per_question.append(rec)
    summary = aggregate(per_question)
    summary["raw"] = raw
    return summary


def _round_or_none(x: float | None) -> float | None:
    return None if x is None else round(x, 2)


def _cell(x) -> str:
    """A measure that could not be taken renders as `-`, never as a legitimate
    value -- an uncited answer scoring 0.0 on bag_share would read as the best
    possible citation hygiene."""
    return "-" if x is None else str(x)


def _mps_cell(rec: dict) -> str:
    """`mean/max` markers per sentence, or `-` (no cited sentence, or a record
    written before the metric existed)."""
    if rec.get("mps_mean") is None:
        return "-"
    return f"{rec['mps_mean']}/{rec['mps_max']}"


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
             f"by mode: {summary['faithfulness_by_mode']}",
             f"Markers per sentence by mode (mean/max): {summary.get('mps_by_mode')}",
             f"Share of citations in >={BAG_MARKERS}-marker sentences, by mode: "
             f"{summary.get('bag_share_by_mode')}\n",
             f"Misattributed claims: {summary['misattributed_total']} across "
             f"{summary['misattributed_answers']} answers "
             f"(unscored: {summary['misattributed_unscored']})",
             f"Comparative (broad): {summary['comparative']}",
             f"drift_wins: {summary['drift_wins']}\n",
             "## Per question\n",
             "| intent | chosen | routing | grounding | faithfulness | cited | ranges "
             "| mps | bag | misattributed | question |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["per_question"]:
        faith_cell = r["faithfulness"] if r["faithfulness"] is not None else "-"
        routing_cell = "-" if r.get("failed") else r["routing_hit"]
        # .get(): summaries written before the cited/ranges/bag/misattributed
        # columns existed.
        lines.append(f"| {r['intent']} | {r['chosen']} | {routing_cell} | "
                     f"{r['grounding_hit']} | {faith_cell} | {_cell(r.get('cited'))} | "
                     f"{_cell(r.get('ranges'))} | {_mps_cell(r)} | "
                     f"{_cell(r.get('bag_share'))} | {_cell(r.get('misattributed'))} | "
                     f"{r['question']} |")
    return "\n".join(lines) + "\n"


def _raw_json(raw: object) -> str:
    """The scored answers as JSON. Answers carry `datetime`s (a fact's `valid_at`),
    which `json.dumps` rejects -- and this is the LAST write of a paid run, after
    the report, so a crash here silently threw away the only copy of the answers
    (2026-09-25). ISO-8601, like everything else the API returns."""
    return json.dumps(raw, indent=2, ensure_ascii=False,
                      default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o))


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
    resolver = await ScopeResolver.load(driver)
    try:
        summary = await run_eval(clients, questions, settings, resolver)
        report = format_report(summary)
        docs = Path(__file__).resolve().parents[2] / "docs" / "superpowers"
        (docs / "router-eval-report.md").write_text(report)
        # The answers this run scored, so a new metric can be computed from a
        # past run instead of buying another one. Not committed -- it is run
        # output, and it is large.
        (docs / "router-eval-raw.json").write_text(_raw_json(summary["raw"]))
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
