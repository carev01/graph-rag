"""Community report generation (strong/synthesis LLM) + deterministic citation
validation. The model references fact UUIDs per finding; we keep only the UUIDs
that are real (present in the community context) and strip any URL — design
decision #2 (the LLM never authors a citation URL)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument
from theme_builder.context import ContextResult

# A URL must never survive into a report (design decision #2). Case-insensitive,
# stops at brackets so it can't swallow an adjacent token.
_URL_RE = re.compile(r"https?://[^\s\[\]]+", re.IGNORECASE)

_PROMPT = (
    "You are analyzing a COMMUNITY of related entities from backup-product "
    "documentation. Using ONLY the numbered FACTS below (each tagged with a "
    "[uuid]), write an analytical community report. Cite the supporting fact "
    "[uuid]s for every finding. Do NOT use outside knowledge. Do NOT write any "
    "URL or link. Respond with ONLY a JSON object of this shape:\n"
    '{{"title": str, "summary": str (one paragraph), '
    '"full_report": [{{"finding": str, "fact_ids": [uuid, ...]}}], '
    '"rating": number 0-10 (thematic importance), "rating_explanation": str, '
    '"tags": [str, ...] (workloads/vendors covered)}}\n\n'
    "COMMUNITY CONTEXT:\n{context}"
)


@dataclass
class CommunityReport:
    title: str
    summary: str
    full_report: str          # the findings list, serialized to a JSON string
    rating: float
    rating_explanation: str
    tags: list[str]
    cited_fact_uuids: list[str]


def _report_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    """Resolve the report tier. Uses report_llm_* when set, else falls back to
    the synthesis/judge tier (GLM-5.2)."""
    base = settings.report_llm_base_url or settings.judge_base_url
    model = settings.report_llm_model or settings.judge_model
    key = settings.report_llm_api_key or settings.judge_api_key or "not-needed"
    if not base or not model:
        raise ValueError(
            "No report model configured. Set report_llm_* or the judge_* (GLM-5.2) "
            "config in .env.")
    # Timeout + throughput routing. Measured 2026-09-08 on z-ai/glm-5.3-flash with
    # an identical prompt: Z.AI served it at 29 tok/s (7.7s), Parasail at 9.4 tok/s
    # (24.2s), and one unpinned call took 380s. Reports are generated SEQUENTIALLY
    # (see cli.py), so per-call routing luck multiplies across every community --
    # unpinned, a 60-community build is anywhere from 30 minutes to 6 hours. A
    # bounded timeout also stops a bad route from hanging the whole batch.
    client = instrument(AsyncOpenAI(api_key=key, base_url=base,
                                    timeout=180.0, max_retries=3))
    if "openrouter" in base:
        client = _prefer_fast_provider(client, settings.report_reasoning_effort)
    return client, model


def _prefer_fast_provider(client: AsyncOpenAI, reasoning_effort: str = "") -> AsyncOpenAI:
    """Route by throughput, and bound reasoning so it cannot eat the output budget.

    Reasoning tokens count as completion tokens, so on a reasoning model an
    unbounded effort level competes with the report text for the same cap --
    and losing that race truncates the JSON, which drops the community.
    """
    orig = client.chat.completions.create

    async def create(*args, **kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        provider = dict(extra.get("provider") or {})
        provider.setdefault("sort", "throughput")
        provider.setdefault("allow_fallbacks", True)
        extra["provider"] = provider
        if reasoning_effort:
            extra.setdefault("reasoning", {"effort": reasoning_effort})
        kwargs["extra_body"] = extra
        return await orig(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def _extract_json(raw: str) -> dict | None:
    """Parse a JSON object from possibly fenced / chatty model output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _strip(s: str) -> str:
    return _URL_RE.sub("", s or "").strip()


async def generate_report(client: AsyncOpenAI, model: str,
                          context: ContextResult,
                          max_tokens: int = 16000) -> CommunityReport | None:
    """One LLM call (+ one retry on unparseable JSON). Validates fact_ids against
    the community's real fact UUIDs (drops hallucinations) and strips URLs.
    Returns None if the model never produced valid JSON."""
    prompt = _PROMPT.format(context=context.text)
    obj: dict | None = None
    for _ in range(2):
        resp = await client.chat.completions.create(
            # GLM-5.2 is a reasoning model: a small cap truncates the JSON to empty
            # (finish_reason='length', content='') on larger communities -> parse
            # fail -> skipped report. 8000 gives the reasoning + report headroom
            # (measured: 3000 skipped ~24% of communities, 8000 skipped ~0). Same
            # lesson as answer_api/synthesize + the type_precision judge.
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    if obj is None:
        return None
    raw_findings = obj.get("full_report")
    findings: list[dict] = []
    cited: list[str] = []
    if isinstance(raw_findings, list):
        for f in raw_findings:
            if not isinstance(f, dict):
                continue  # structurally-off finding -> skip, don't crash
            fids = [fid for fid in (f.get("fact_ids") or []) if fid in context.fact_uuids]
            for fid in fids:
                if fid not in cited:
                    cited.append(fid)
            # carry ONLY sanitized finding text + validated ids (drop unknown keys,
            # which could smuggle a model-authored URL past the strip)
            findings.append({"finding": _strip(str(f.get("finding", ""))), "fact_ids": fids})
    try:
        rating = float(obj.get("rating", 0) or 0)
    except (TypeError, ValueError):
        rating = 0.0
    return CommunityReport(
        title=_strip(str(obj.get("title", ""))),
        summary=_strip(str(obj.get("summary", ""))),
        full_report=json.dumps(findings),
        rating=rating,
        rating_explanation=_strip(str(obj.get("rating_explanation", ""))),
        tags=[_strip(str(t)) for t in (obj.get("tags") or [])],
        cited_fact_uuids=cited)
