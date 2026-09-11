"""Community report generation (strong/synthesis LLM) + deterministic citation
validation. The model references fact UUIDs per finding; we keep only the UUIDs
that are real (present in the community context) and strip any URL — design
decision #2 (the LLM never authors a citation URL)."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument, usable_content
from theme_builder.context import ContextResult

logger = logging.getLogger(__name__)

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

_VERIFY_PROMPT = (
    "You are checking a community report for claims its own evidence does not support.\n"
    "For each FINDING below, decide whether EVERY claim it makes is stated by the FACTS "
    "listed under it. A finding is UNSUPPORTED if it adds anything the facts do not "
    "state -- a number, a limit, a duration, a product name, a mechanism, or a "
    "requirement -- EVEN IF THAT CLAIM IS TRUE IN THE REAL WORLD. Judge only against "
    "the facts shown; outside knowledge is exactly what we are detecting.\n"
    "Also judge whether the SUMMARY is supported by the facts shown anywhere below.\n"
    "Respond with ONLY JSON: {{\"unsupported\": [finding numbers], "
    "\"summary_supported\": true or false}}\n\n"
    "SUMMARY: {summary}\n\n{blocks}"
)


_SUMMARY_PROMPT = (
    "Write ONE paragraph summarising the STATEMENTS below for a community titled "
    "\"{title}\". Summarise only what these statements say -- do NOT add any detail, "
    "number, product name or limit that is not in them, and do NOT write any URL. "
    "Reply with the paragraph only.\n\nSTATEMENTS:\n{statements}"
)


@dataclass
class VerifyResult:
    unsupported: set[int]      # 1-based finding indices
    summary_supported: bool


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


def _verify_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    """Resolve the report VERIFIER, independent of the report writer.

    Uses verify_llm_* when set, else eval_judge_*, and then refuses that result if
    it is the report model. A model checking its own findings for outside-knowledge
    leakage will not find any -- and the report gives no sign that the check was
    vacuous. This mirrors answer_api.eval_router._eval_judge_client_and_model.
    """
    base = settings.verify_llm_base_url or settings.eval_judge_base_url
    model = settings.verify_llm_model or settings.eval_judge_model
    key = (settings.verify_llm_api_key or settings.eval_judge_api_key
           or settings.judge_api_key or "not-needed")
    if not base or not model:
        raise ValueError(
            "No report verifier configured. Set VERIFY_LLM_BASE_URL / VERIFY_LLM_MODEL "
            "(or EVAL_JUDGE_*) in .env.")
    report_model = settings.report_llm_model or settings.judge_model
    # Compare the MODEL ALONE. Also requiring the base URL to match let the exact
    # same weights reached through a second gateway/provider slip past and grade
    # their own findings -- and the shared failure modes we are guarding against
    # travel with the model, not with the endpoint that serves it.
    if model == report_model:
        raise ValueError(
            f"Report verifier resolves to the report model ({model!r}) -- it would "
            "check its own findings and confirm them. Set VERIFY_LLM_MODEL to a "
            "different model, ideally a different family so the two do not share "
            "failure modes.")
    client = instrument(AsyncOpenAI(api_key=key, base_url=base,
                                    timeout=180.0, max_retries=3))
    if "openrouter" in base:
        client = _prefer_fast_provider(client, settings.verify_reasoning_effort)
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


async def _regenerate_summary(client: AsyncOpenAI, model: str, findings: list[dict],
                              title: str, *, max_tokens: int = 4000) -> str | None:
    """Rewrite the summary from the findings that survived verification.

    The summary is inherently synthetic -- one paragraph generalising a whole
    community -- so verifying it against raw facts rejected legitimate summarising
    and cost 15 of 41 communities across two rebuilds. Regenerating it from verified
    findings makes it inherit their support instead.

    `max_tokens` matches the rest of this file's headroom (verify_report's 4000, the
    report writer's 8000+) rather than a tight cap: the report/judge tier is a
    REASONING model whose thinking tokens count against the same completion budget,
    so a small cap returns finish_reason='length' with EMPTY content -- which lands
    in the "keep the original summary" fallback and silently undoes the regeneration.
    """
    statements = "\n".join(f"- {f.get('finding', '')}" for f in findings)
    resp = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(
            title=title, statements=statements)}])
    text = usable_content(resp)
    return _strip(text) if text else None


async def verify_report(client: AsyncOpenAI, model: str, findings: list[dict],
                        summary: str, fact_texts: dict[str, str], *,
                        max_tokens: int = 4000) -> VerifyResult | None:
    """Check each finding against the text of the facts it cites.

    Returns None when verification could NOT be completed (unusable reply after a
    retry). None is not "supported": writing an unverified report because the
    verifier hiccuped is the defect this whole slice exists to prevent.
    """
    blocks = []
    for i, f in enumerate(findings, 1):
        texts = [fact_texts[u] for u in f.get("fact_ids", []) if u in fact_texts]
        listed = "\n".join(f"   - {t}" for t in texts) or "   (no facts cited)"
        blocks.append(f"FINDING {i}: {f.get('finding', '')}\nFACTS:\n{listed}")
    prompt = _VERIFY_PROMPT.format(summary=summary, blocks="\n\n".join(blocks))
    for budget in (max_tokens, max_tokens * 3):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=budget,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(usable_content(resp) or "")
        if obj is None:
            logger.warning("report verifier returned no usable JSON (budget=%d); retrying",
                           budget)
            continue
        raw = obj.get("unsupported")
        if not isinstance(raw, list):
            # A parseable reply whose schema drifted -- a renamed key
            # ({"verdicts": [...]}), a nested one ({"result": {...}}), or a null --
            # must NOT collapse to "nothing is unsupported". That is an unusable
            # reply reading as "supported", the exact defect this verifier exists to
            # prevent. No response_format json_schema is used here, and cheap tiers
            # drift like this. Treat it as unusable and retry.
            logger.warning(
                "report verifier reply has no list-typed 'unsupported' key "
                "(keys=%s, budget=%d); retrying", sorted(obj)[:8], budget)
            continue
        idx = {int(x) for x in raw
               if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit())}
        return VerifyResult(unsupported=idx,
                            summary_supported=bool(obj.get("summary_supported", False)))
    logger.warning(
        "report verifier failed twice; the report will be STAGED unverified "
        "(kept in the graph without an embedding, so it is unreachable from every "
        "answering path) -- recover it with `theme-build --verify-pending`")
    return None


async def _generate_once(client: AsyncOpenAI, model: str, prompt: str,
                         max_tokens: int) -> dict | None:
    """One LLM call plus one retry on an unparseable/contentless reply."""
    for _ in range(2):
        resp = await client.chat.completions.create(
            # GLM-5.2 is a reasoning model: a small cap truncates the JSON to empty
            # (finish_reason='length', content='') on larger communities -> parse
            # fail -> skipped report. 8000 gives the reasoning + report headroom
            # (measured: 3000 skipped ~24% of communities, 8000 skipped ~0). Same
            # lesson as answer_api/synthesize + the type_precision judge.
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        # A flaky provider can return HTTP 200 with an error payload and NO
        # choices; indexing [0] then raises and the caller drops the community
        # entirely. Observed 3/29 on the first real theme-build. usable_content
        # folds that in with contentless replies so both take the retry.
        obj = _extract_json(usable_content(resp) or "")
        if obj is not None:
            return obj
    return None


def _build_report(obj: dict, context: ContextResult) -> tuple[CommunityReport, list[dict]]:
    """Parse one report payload. Returns the report AND its findings list, so the
    caller can verify the findings and rebuild the report with a subset."""
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
    report = CommunityReport(
        title=_strip(str(obj.get("title", ""))),
        summary=_strip(str(obj.get("summary", ""))),
        full_report=json.dumps(findings),
        rating=rating,
        rating_explanation=_strip(str(obj.get("rating_explanation", ""))),
        tags=[_strip(str(t)) for t in (obj.get("tags") or [])],
        cited_fact_uuids=cited)
    return report, findings


def _with_findings(report: CommunityReport, findings: list[dict]) -> CommunityReport:
    """Same report carrying only `findings`, with cited_fact_uuids recomputed so a
    dropped finding's facts do not stay listed as cited."""
    cited: list[str] = []
    for f in findings:
        for fid in f["fact_ids"]:
            if fid not in cited:
                cited.append(fid)
    return replace(report, full_report=json.dumps(findings), cited_fact_uuids=cited)


@dataclass
class ReportStats:
    """Out-parameter for counters the caller surfaces. generate_report's return type
    is load-bearing for ten call sites, so the counts travel separately."""
    findings_dropped: int = 0
    reverified: bool = False
    unverified: bool = False
    # The generated report when verification could NOT be completed. Returned
    # separately from generate_report's return value, which stays None so nothing
    # unverified is ever mistaken for a usable report -- but the content is kept so
    # a transient verifier outage does not cost a full regeneration (spec 4.7).
    staged_report: "CommunityReport | None" = None


_RETRY_NOTE = (
    "\n\nYour previous answer was REJECTED. These findings state things the facts "
    "you cited do not state, even if they are true in the real world:\n{offenders}\n"
    "Rewrite the report using ONLY what the facts state. Do not restore the "
    "rejected claims."
)


async def generate_report(client: AsyncOpenAI, model: str,
                          context: ContextResult,
                          max_tokens: int = 16000, *,
                          verifier=None,
                          stats: ReportStats | None = None) -> CommunityReport | None:
    """Generate one community report, optionally verified against its own facts.

    With `verifier` set: every finding is checked against the text of the facts it
    cites. On violation the report is regenerated ONCE with the offenders named;
    findings still unsupported are dropped. The summary verdict is never a reason
    to skip the report -- see `_regenerate_summary`. Returns None -- report skipped
    -- if every finding is dropped, or if verification could not be completed.
    Without `verifier` the behaviour is exactly as before.
    """
    prompt = _PROMPT.format(context=context.text)
    obj = await _generate_once(client, model, prompt, max_tokens)
    if obj is None:
        return None
    report, findings = _build_report(obj, context)
    if verifier is None or not findings:
        # Nothing to verify: an empty report is already handled downstream by
        # writeback, and calling the verifier with no findings wastes a request.
        return report

    result = await verifier(findings, report.summary, context.fact_texts)
    if result is None:
        if stats is not None:
            stats.unverified = True
            stats.staged_report = report
        return None

    if result.unsupported:
        if stats is not None:
            stats.reverified = True
        offenders = "\n".join(
            f"- {findings[i - 1]['finding']}" for i in sorted(result.unsupported)
            if 1 <= i <= len(findings))
        retry_obj = await _generate_once(
            client, model, prompt + _RETRY_NOTE.format(offenders=offenders), max_tokens)
        if retry_obj is not None:
            report, findings = _build_report(retry_obj, context)
            if not findings:
                # The retry came back with no findings at all. Re-verifying an empty
                # list returns unsupported=set(), which walks straight past the
                # "every finding dropped" guard below and writes a citation-free
                # report -- empty full_report, no cited facts, and a summary
                # generated from an empty STATEMENTS block. Skip it here instead.
                logger.warning("report retry returned no findings; skipping the report")
                return None
            result = await verifier(findings, report.summary, context.fact_texts)
            if result is None:
                if stats is not None:
                    stats.unverified = True
                    stats.staged_report = report
                return None

    if result.unsupported:
        kept = [f for i, f in enumerate(findings, 1) if i not in result.unsupported]
        if stats is not None:
            stats.findings_dropped = len(findings) - len(kept)
        if not kept:
            return None
        report = _with_findings(report, kept)
        findings = kept

    # The summary verdict (result.summary_supported) is deliberately IGNORED -- see
    # the docstring on _regenerate_summary. Rewrite it from the surviving findings so
    # it inherits their support; keep the original if the summariser returns nothing,
    # since a summariser hiccup must not cost a verified report.
    new_summary = await _regenerate_summary(client, model, findings, report.title)
    if new_summary:
        report = replace(report, summary=new_summary)
    else:
        # Visible, because the kept summary may still describe a dropped finding.
        logger.warning(
            "summary regeneration returned nothing usable for %r; keeping the "
            "original summary (it was written before verification)", report.title)
    return report
