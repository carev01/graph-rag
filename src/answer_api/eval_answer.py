"""Run the golden questions through /answer (answer_local) and report
answer-groundedness: does the synthesized answer cite >=1 fact whose source is
an expected article? Live: needs the semantic graph + GLM-5.2.

    uv run --extra dev python -m answer_api.eval_answer [--k 15]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from neo4j import AsyncGraphDatabase

from answer_api.golden import precision_at_k
from answer_api.synthesize import (
    _REFUSAL,
    _URL_RE,
    _synthesis_client_and_model,
    answer_local,
)
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti

_QUESTIONS = Path(__file__).with_name("golden_questions.json")


async def main(k: int) -> None:
    settings = get_extract_settings()
    questions = json.loads(_QUESTIONS.read_text())
    graphiti = build_graphiti(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    synth_client, synth_model = _synthesis_client_and_model(settings)
    grounded = 0
    refusals = 0
    try:
        print(f"# Answer-groundedness eval (k={k}, {len(questions)} questions, {synth_model})\n")
        for q in questions:
            out = await answer_local(
                graphiti, driver, synth_client, synth_model,
                q=q["question"], k=k, vendor=q.get("vendor"), group_id=settings.group_id)
            is_grounded = precision_at_k(out["citations"], q["expected_article_ids"])
            is_refusal = out["answer"].strip() == _REFUSAL
            # only an actual https?:// URL is a design-decision-#2 breach;
            # the bare word "HTTPS" (a protocol name) is fine.
            has_url = bool(_URL_RE.search(out["answer"]))
            grounded += 1 if is_grounded else 0
            refusals += 1 if is_refusal else 0
            mark = "GROUNDED" if is_grounded else ("REFUSED " if is_refusal else "UNGROUND")
            flag = "  !!URL-IN-ANSWER" if has_url else ""
            print(f"[{mark} cited={out['cited']}/{out['retrieved']}]{flag} {q['question']}")
    finally:
        await graphiti.close()
        await driver.close()
        await synth_client.close()
    n = len(questions)
    print(f"\nanswer_groundedness@{k} = {grounded}/{n} = {grounded / n:.3f}")
    print(f"refusals = {refusals}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    asyncio.run(main(ap.parse_args().k))
