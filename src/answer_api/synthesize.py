from __future__ import annotations

import re

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument

_URL_RE = re.compile(r"https?://\S+")
_MARKER_RE = re.compile(r"\[(\d+)\]")

_PROMPT = (
    "You are answering a question about backup products using ONLY the numbered "
    "facts below. Cite every claim inline with its [N] marker. Do NOT use outside "
    "knowledge. Do NOT write any URL or link. If the facts do not answer the "
    "question, reply exactly: "
    "\"I don't have enough information to answer that from the available sources.\"\n\n"
    "FACTS:\n{facts}\n\nQUESTION: {q}\n\nAnswer:"
)

_REFUSAL = "I don't have enough information to answer that from the available sources."


def _finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]:
    """Deterministic design-decision #2 enforcement: strip any URL the model
    emitted (it must never write one), then keep the ordered-unique [N] markers
    that map to a retrieved fact (drop invented ones)."""
    text = _URL_RE.sub("", raw).strip()
    cited: list[int] = []
    for m in _MARKER_RE.findall(text):
        n = int(m)
        if n in marker_map and n not in cited:
            cited.append(n)
    return text, cited


def _synthesis_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    """Synthesis LLM = GLM-5.2 via the judge_* config (shared endpoint for now;
    point at a dedicated synthesis model later without touching the judge)."""
    if not settings.judge_base_url:
        raise ValueError(
            "No synthesis model configured. Set JUDGE_BASE_URL / JUDGE_MODEL / "
            "JUDGE_API_KEY in .env (synthesis currently uses the GLM-5.2 judge endpoint)."
        )
    client = instrument(AsyncOpenAI(
        api_key=settings.judge_api_key or "not-needed", base_url=settings.judge_base_url))
    return client, settings.judge_model
