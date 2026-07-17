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
    return instrument(AsyncOpenAI(api_key=key, base_url=base)), model


def _extract_json(raw: str) -> dict | None:
    """Parse a JSON object from possibly fenced / chatty model output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _strip(s: str) -> str:
    return _URL_RE.sub("", s or "").strip()


async def generate_report(client: AsyncOpenAI, model: str,
                          context: ContextResult) -> CommunityReport | None:
    """One LLM call (+ one retry on unparseable JSON). Validates fact_ids against
    the community's real fact UUIDs (drops hallucinations) and strips URLs.
    Returns None if the model never produced valid JSON."""
    prompt = _PROMPT.format(context=context.text)
    obj: dict | None = None
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    if obj is None:
        return None
    findings = obj.get("full_report") or []
    cited: list[str] = []
    for f in findings:
        for fid in (f.get("fact_ids") or []):
            if fid in context.fact_uuids and fid not in cited:
                cited.append(fid)
        f["finding"] = _strip(str(f.get("finding", "")))
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
        tags=[str(t) for t in (obj.get("tags") or [])],
        cited_fact_uuids=cited)
