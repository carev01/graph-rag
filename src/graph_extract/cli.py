from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx
import typer
from neo4j import AsyncDriver, AsyncGraphDatabase

from graph_extract import quality_labels
from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.eval import (
    cost_report,
    dedup_report,
    dedup_report_v2,
    fact_quality,
    noise_report,
    provenance_report,
    type_precision,
)
from graph_extract.graph_cleanup import (
    prune_noise_entities,
    retype_region_entities,
    tombstone_navigation_articles,
)
from graph_extract.graphiti_client import (
    build_graphiti, build_cheap_graphiti, init_indices, ExtractionTier,
)
from graph_extract.dedup_guard import DedupIndexStats, install_dedup_guard
from graph_extract.llm_timing import PromptTimings
from graph_extract.ingest_driver import IngestDriver
from graph_extract.merge_duplicates import apply_merges, plan_merges
from graphiti_core import Graphiti
from graph_extract.ontology import EXTRACTION_INSTRUCTIONS, CHEAP_TIER_SALIENCE
from graph_extract.probe import DEFAULT_MODES, run_probe
from graph_extract.provenance import Provenance
from graph_extract.reconcile import reconcile_same_as
from graph_extract.staleness_sweep import sweep_stale_facts
from graph_extract.warmup import WarmupGate
from docext.client import make_docext_client

app = typer.Typer()
eval_app = typer.Typer()
app.add_typer(eval_app, name="eval")
logger = logging.getLogger(__name__)

# Task 11 spec: canonical dedup targets for the pilot eval -- concepts that
# should merge to a single :Entity node, and pairs that must stay distinct.
DEFAULT_CANON_MERGE = ["immutability", "cross-region copy", "Kubernetes", "RPO"]
DEFAULT_DISTINCT_PAIRS = [
    ("Amazon S3", "Azure Blob Storage"),
    ("AWS Backup", "Azure Backup"),
]

# Task 4 spec: quality-baseline/quality-report artifacts committed under
# docs/superpowers so a baseline captured once can be diffed against later
# reruns (e.g. after the ontology-v2 / noise-prune slice-2b changes land).
QUALITY_BASELINE_PATH = Path("docs/superpowers/slice-2b-quality-baseline.json")
QUALITY_REPORT_PATH = Path("docs/superpowers/slice-2b-quality-report.md")
DEFAULT_TYPE_PRECISION_SAMPLE = 100


def _dump(obj: object) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


async def _build_ingest_driver(
    settings: ExtractSettings,
) -> tuple[IngestDriver, Graphiti, httpx.AsyncClient, AsyncDriver]:
    """Construct the real dependency graph the `ingest` command needs.

    If any step after opening a resource fails, every resource already
    opened here is closed before the exception propagates -- without this,
    a failure partway through construction would return nothing to the
    caller, so the caller's own `try/finally` (which only covers the code
    *after* this function returns) would never run and the already-open
    graphiti/client/driver would leak.
    """
    graphiti = build_graphiti(settings)
    cheap_graphiti: Graphiti | None = None
    docext: httpx.AsyncClient | None = None
    driver: AsyncDriver | None = None
    try:
        docext = make_docext_client(
            base_url=settings.docext_base_url, read_key=settings.docext_read_key,
            admin_key=settings.docext_admin_key, verify_tls=settings.docext_verify_tls,
            admin=False,
        )
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        await init_indices(graphiti)
        provenance = Provenance(driver)
        strong_tier = ExtractionTier("strong", graphiti, EXTRACTION_INSTRUCTIONS,
                                     settings.max_chunk_tokens)
        # Detection on the strong tier too (does gpt-5-mini ever do this? -- the
        # counters answer that); retry only makes sense for the cheap tier.
        # One timings object across BOTH tiers: the question is how the provider
        # behaves under our concurrency, and the tiers interleave on one host.
        timings = PromptTimings()
        strong_guard = install_dedup_guard(graphiti, fallback=None,
                                           unscoped=DedupIndexStats(), timings=timings)
        cheap_tier: ExtractionTier | None = None
        if settings.extraction_routing and settings.cheap_llm_api_key:
            cheap_graphiti = build_cheap_graphiti(settings)
            await init_indices(cheap_graphiti)
            install_dedup_guard(
                cheap_graphiti,
                # `.raw` = the strong client's UNGUARDED method, so a retry is not
                # counted a second time by the strong tier's own guard.
                fallback=strong_guard.raw if settings.dedup_retry_on_strong else None,
                unscoped=DedupIndexStats(), timings=timings)
            cheap_tier = ExtractionTier(
                "cheap", cheap_graphiti,
                EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE,
                settings.cheap_max_chunk_tokens)
        # ingest_warmup_articles=0 is OFF: no gate at all, so `ingest_source`
        # takes the byte-for-byte pre-warm-up fan-out (a gate at threshold 0
        # would answer False every time, which is equivalent but not identical).
        warmup_gate = (
            WarmupGate(driver, settings.group_id, settings.ingest_warmup_articles)
            if settings.ingest_warmup_articles > 0 else None)
        ingest = IngestDriver(settings, strong_tier, cheap_tier, docext, provenance, driver,
                              warmup_gate=warmup_gate)
        ingest.timings = timings          # BACKLOG 31: reported by the ingest command
    except Exception:
        for closer in (
            cheap_graphiti.close if cheap_graphiti is not None else None,
            driver.close if driver is not None else None,
            docext.aclose if docext is not None else None,
            graphiti.close,
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:
                logger.exception(
                    "error closing a resource while cleaning up after a "
                    "_build_ingest_driver failure"
                )
        raise
    return ingest, graphiti, docext, driver


async def _build_driver(settings: ExtractSettings) -> AsyncDriver:
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


async def _run_quality_reports(
    driver: AsyncDriver, settings: ExtractSettings, sample: int
) -> dict:
    """Run the three slice-2b quality reports and combine them into one dict.

    Shared by `quality-baseline` (writes this as the JSON baseline) and
    `quality-report` (re-runs it and diffs against that baseline). `noise`
    and `dedup` are deterministic Cypher; `type_precision` is the only one
    that calls the JUDGE model (GLM, via `_judge_client_and_model` inside
    `type_precision` itself -- never the extraction model).
    """
    return {
        "noise": await noise_report(driver, settings.group_id),
        "dedup": await dedup_report_v2(driver, settings.group_id, quality_labels),
        "type_precision": await type_precision(driver, settings, sample),
    }


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_delta(before: object, after: object) -> str:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return f"{after - before:+.3f}" if isinstance(after, float) else f"{after - before:+d}"
    return "n/a"


def _render_quality_report_md(baseline: dict, current: dict) -> str:
    lines: list[str] = ["# Slice 2b quality report", ""]

    lines.append("## Noise")
    b_noise, c_noise = baseline.get("noise", {}), current.get("noise", {})
    for key in ("entities", "facts"):
        b, c = b_noise.get(key, {}), c_noise.get(key, {})
        b_rate, c_rate = b.get("rate"), c.get("rate")
        lines.append(
            f"- {key} noise rate: baseline={_fmt(b_rate)} -> "
            f"current={_fmt(c_rate)} (delta {_fmt_delta(b_rate, c_rate)}), "
            f"total baseline={_fmt(b.get('total'))} current={_fmt(c.get('total'))}"
        )
    lines.append("")

    lines.append("## Dedup (v2)")
    b_dedup, c_dedup = baseline.get("dedup", {}), current.get("dedup", {})
    b_merge, c_merge = b_dedup.get("should_merge", {}), c_dedup.get("should_merge", {})
    lines.append("### should_merge")
    for name in sorted(set(b_merge) | set(c_merge)):
        b_count = b_merge.get(name, {}).get("node_count")
        c_count = c_merge.get(name, {}).get("node_count")
        lines.append(
            f"- {name}: baseline={_fmt(b_count)} -> current={_fmt(c_count)} "
            f"(delta {_fmt_delta(b_count, c_count)})"
        )
    lines.append("")
    lines.append("### should_distinct (collapsed = bad)")
    b_distinct = {tuple(p["pair"]): p for p in b_dedup.get("should_distinct", [])}
    c_distinct = {tuple(p["pair"]): p for p in c_dedup.get("should_distinct", [])}
    for pair in sorted(set(b_distinct) | set(c_distinct)):
        b_collapsed = b_distinct.get(pair, {}).get("collapsed")
        c_collapsed = c_distinct.get(pair, {}).get("collapsed")
        lines.append(f"- {list(pair)}: baseline collapsed={b_collapsed} -> current collapsed={c_collapsed}")
    lines.append("")
    b_suspect = b_dedup.get("suspect_false_merge", {}).get("count")
    c_suspect = c_dedup.get("suspect_false_merge", {}).get("count")
    lines.append(
        f"### suspect_false_merge count: baseline={_fmt(b_suspect)} -> "
        f"current={_fmt(c_suspect)} (delta {_fmt_delta(b_suspect, c_suspect)})"
    )
    lines.append("")

    lines.append("## Type precision")
    b_tp, c_tp = baseline.get("type_precision", {}), current.get("type_precision", {})
    b_prec, c_prec = b_tp.get("precision"), c_tp.get("precision")
    lines.append(
        f"- overall precision: baseline={_fmt(b_prec)} -> current={_fmt(c_prec)} "
        f"(delta {_fmt_delta(b_prec, c_prec)}); "
        f"sampled baseline={_fmt(b_tp.get('sampled'))} current={_fmt(c_tp.get('sampled'))}"
    )
    lines.append("")
    lines.append("### per_type precision")
    b_per_type, c_per_type = b_tp.get("per_type", {}), c_tp.get("per_type", {})
    for t in sorted(set(b_per_type) | set(c_per_type)):
        b_p = b_per_type.get(t, {}).get("precision")
        c_p = c_per_type.get(t, {}).get("precision")
        lines.append(
            f"- {t}: baseline={_fmt(b_p)} -> current={_fmt(c_p)} "
            f"(delta {_fmt_delta(b_p, c_p)})"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


@app.command("ingest")
def ingest(
    source_id: str = typer.Option(..., "--source-id"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        ingest_driver, graphiti, docext, driver = await _build_ingest_driver(settings)
        try:
            res = await ingest_driver.ingest_source(source_id, limit)
            typer.echo(
                f"ingest complete: articles={res.articles} "
                f"episodes_added={res.episodes_added} "
                f"episodes_skipped={res.episodes_skipped} "
                f"routing={'hybrid' if ingest_driver._cheap is not None else 'strong-only'}"
            )
            # Out-of-range dedup indices (BACKLOG 4). `dup_in_invalidation_range` vs
            # `dup_beyond_range` is the decisive check for the index-space-confusion
            # hypothesis; `retry_clean` vs `retry_dirty` says whether the strong tier
            # reads the same prompt correctly.
            typer.echo(f"dedup indices: {res.dedup.summary()}")
            # BACKLOG 31: where an episode's wall time actually goes, and whether
            # the provider runs our 20-wide concurrent dedup calls in parallel.
            typer.echo(ingest_driver.timings.report())
            # Emit the cost report HERE, in-process: the usage tally is
            # process-local, so a separate `eval cost` invocation would see an
            # empty tally. This is the only path that reports real extraction
            # cost for the run. Covers extraction-LLM chat.completions tokens;
            # the cross-encoder is not called during add_episode (search-time
            # only), and TEI embeddings are not token-metered.
            typer.echo("--- cost (this run) ---")
            _dump(await cost_report(res.episodes_added))
        finally:
            await driver.close()
            await docext.aclose()
            await graphiti.close()
            if ingest_driver._cheap is not None:
                await ingest_driver._cheap.graphiti.close()

    asyncio.run(_run())


@eval_app.command("dedup")
def eval_dedup() -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            report = await dedup_report(
                driver, settings.group_id, DEFAULT_CANON_MERGE, DEFAULT_DISTINCT_PAIRS
            )
            _dump(report)
        finally:
            await driver.close()

    asyncio.run(_run())


@eval_app.command("provenance")
def eval_provenance(sample: int = typer.Option(20, "--sample")) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            report = await provenance_report(driver, settings.group_id, sample)
            _dump(report)
        finally:
            await driver.close()

    asyncio.run(_run())


@eval_app.command("cost")
def eval_cost(
    episodes_processed: int = typer.Option(0, "--episodes-processed"),
) -> None:
    async def _run() -> None:
        report = await cost_report(episodes_processed)
        _dump(report)

    asyncio.run(_run())


@eval_app.command("quality")
def eval_quality(sample: int = typer.Option(20, "--sample")) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            report = await fact_quality(driver, settings, sample)
            _dump(report)
        finally:
            await driver.close()

    asyncio.run(_run())


@app.command("quality-baseline")
def quality_baseline(
    sample: int = typer.Option(DEFAULT_TYPE_PRECISION_SAMPLE, "--sample"),
) -> None:
    """Run noise/dedup/type_precision and save the result as the quality baseline.

    Writes `docs/superpowers/slice-2b-quality-baseline.json`. `quality-report`
    diffs later runs against this file.
    """

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            report = await _run_quality_reports(driver, settings, sample)
        finally:
            await driver.close()

        QUALITY_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUALITY_BASELINE_PATH.write_text(json.dumps(report, indent=2, default=str))
        typer.echo(f"wrote baseline: {QUALITY_BASELINE_PATH}")
        _dump(report)

    asyncio.run(_run())


@app.command("quality-report")
def quality_report(
    sample: int = typer.Option(DEFAULT_TYPE_PRECISION_SAMPLE, "--sample"),
) -> None:
    """Re-run noise/dedup/type_precision and diff against the saved baseline.

    Writes `docs/superpowers/slice-2b-quality-report.md`. Fails with a clear
    message if `quality-baseline` hasn't been run yet.
    """
    if not QUALITY_BASELINE_PATH.exists():
        typer.echo(
            f"error: no baseline found at {QUALITY_BASELINE_PATH}. "
            "Run `quality-baseline` first.",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        baseline = json.loads(QUALITY_BASELINE_PATH.read_text())
    except json.JSONDecodeError:
        typer.echo(
            f"error: baseline at {QUALITY_BASELINE_PATH} is malformed JSON. "
            "Re-run `quality-baseline` to regenerate it.",
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            current = await _run_quality_reports(driver, settings, sample)
        finally:
            await driver.close()

        markdown = _render_quality_report_md(baseline, current)
        QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUALITY_REPORT_PATH.write_text(markdown)
        typer.echo(f"wrote report: {QUALITY_REPORT_PATH}")
        typer.echo("--- summary ---")
        typer.echo(
            f"noise entities rate: {current['noise']['entities']['rate']:.3f} "
            f"(baseline {baseline.get('noise', {}).get('entities', {}).get('rate', 'n/a')})"
        )
        typer.echo(
            f"type precision: {current['type_precision']['precision']:.3f} "
            f"(baseline {baseline.get('type_precision', {}).get('precision', 'n/a')})"
        )

    asyncio.run(_run())


@app.command("cleanup")
def cleanup(
    force_demote: bool = typer.Option(
        False, "--force-demote",
        help="Bypass the region demote-guard (use after confirming a large "
             "batch of :Region demotions is legitimate, e.g. a junk backlog).",
    ),
) -> None:
    """Run the deterministic post-ingest correctors: prune noise entities,
    then retype (promote+demote) region entities against the gazetteer."""

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            prune = await prune_noise_entities(driver, settings.group_id)
            retype = await retype_region_entities(
                driver, settings.group_id, force=force_demote)
            _dump({"prune": prune, "retype": retype})
        finally:
            await driver.close()

    asyncio.run(_run())


@app.command("merge-duplicates")
def merge_duplicates(
    apply: bool = typer.Option(
        False, "--apply",
        help="Actually merge. Without this the command only reports. Never run it "
             "while an ingest or semantic worker is running: an in-flight episode "
             "that resolved onto a node this deletes loses its facts silently."),
) -> None:
    """Merge :Entity nodes sharing a byte-identical name within one group_id.

    Report-only by default. Concurrent ingest creates these: the A/B measured 30
    at N=4, every one an exact-name collision on a hub entity. They are not
    static -- graphiti escalates to an LLM dedup call on EVERY later mention of
    an ambiguous name, so a duplicate is a permanent tax on the hottest names in
    the corpus.
    """

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            if apply:
                typer.echo(
                    "WARNING: --apply must not run while any ingest or semantic "
                    "worker is running. graphiti saves facts with MATCH "
                    "(source:Entity {uuid}) ... MERGE; if an in-flight episode "
                    "resolved an entity to a node this deletes, that edge is "
                    "SILENTLY not written -- no error, no log, no retry. This "
                    "command cannot check that condition for you.", err=True)
                # `apply_merges` stops and raises on the first failing group
                # without returning, and the committed groups' audit (edges
                # moved, self-loops, dropped summary texts) exists nowhere but
                # in this list. Print it, then let the error out unchanged --
                # including on Ctrl-C, when the record is worth more, not less.
                committed: list[dict] = []
                try:
                    _dump(await apply_merges(driver, settings.group_id, committed=committed))
                except BaseException:
                    _dump({"stopped": True, "committed": committed,
                           "totals": {"groups": len(committed),
                                      "merged": sum(len(g["losers"]) for g in committed)}})
                    typer.echo(
                        f"merge-duplicates: stopped after {len(committed)} committed "
                        "group(s). The JSON above is their audit; the group that failed "
                        "was rolled back whole and nothing after it was attempted. "
                        "The error follows.", err=True)
                    raise
            else:
                _dump(await plan_merges(driver, settings.group_id))
        finally:
            await driver.close()

    asyncio.run(_run())


@app.command("maintenance")
def maintenance() -> None:
    """Full housekeeping pass: prune noise, retype regions, tombstone
    navigation-article episodes, sweep stale facts, reconcile structural<->
    semantic SAME_AS. Composes the deterministic jobs; run weekly (scheduling
    is a deployment concern -- see maintenance-runbook). Navigation
    tombstoning runs before the sweep so the same run expires the now-
    unsupported nav facts."""

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            prune = await prune_noise_entities(driver, settings.group_id)
            retype = await retype_region_entities(driver, settings.group_id)
            navigation = await tombstone_navigation_articles(driver, settings.group_id)
            sweep = await sweep_stale_facts(driver, settings.group_id)
            reconcile = await reconcile_same_as(driver, settings.group_id)
            _dump({"prune": prune, "retype": retype, "navigation": navigation,
                   "sweep": sweep, "reconcile": reconcile})
        finally:
            await driver.close()

    asyncio.run(_run())


@app.command("sweep")
def sweep() -> None:
    """Expire RELATES_TO facts whose supporting episodes are all dead (an
    episode is dead when its HAS_EPISODE edge is flagged superseded, or the
    episode or its article is removed) -- the deterministic backstop for
    staleness Graphiti's own contradiction-detection can't see.
    """

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            result = await sweep_stale_facts(driver, settings.group_id)
            _dump(result)
        finally:
            await driver.close()

    asyncio.run(_run())


@app.command("reconcile")
def reconcile() -> None:
    """Link structural :Vendor/:Product nodes to their semantic :Entity
    twins via alias-matched SAME_AS edges (design decision #5: link, never
    merge). Idempotent -- safe to rerun after every bootstrap/incremental
    batch or whenever vendor_aliases gains new entries.
    """

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            result = await reconcile_same_as(driver, settings.group_id)
            _dump(result)
        finally:
            await driver.close()

    asyncio.run(_run())


@app.command("probe")
def probe(
    article_ids: str = typer.Option(..., "--article-ids", help="Comma-separated article IDs"),
    n: int = typer.Option(3, "--n", help="Chunks per article"),
) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        ids = [a.strip() for a in article_ids.split(",") if a.strip()]
        report = await run_probe(settings, ids, n, DEFAULT_MODES)
        _dump(report)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
