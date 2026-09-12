"""Per-`prompt_name` LLM timing, including latency vs in-flight concurrency.

Why this exists (BACKLOG 31). Two throughput questions could not be answered from
the code, and both were blocking decisions:

  1. **Where does an episode's ~45 s go?** Only `dedupe_edges.resolve_edge` was
     ever counted. `_extract_edge_timestamps` issues a SECOND LLM call per new
     fact (`edge_operations.py:813`) that nothing has measured, and node dedup and
     summaries are likewise invisible.
  2. **Does the provider serialise our concurrent calls?** `resolve_extracted_edges`
     dispatches its dedup calls 20-wide (`semaphore_gather` with no
     `max_coroutines`), so IF the provider runs them in parallel the phase costs
     the slowest call and reducing call COUNT saves nothing. If it queues them,
     count is back on the critical path and the whole optimisation strategy
     inverts.

Question 2 is the one that decides strategy, and it is answerable only by
correlating each call's latency with how many were in flight when it was
dispatched. That is what `by_inflight` reports: if median latency is flat across
the buckets the calls really are parallel; if it climbs roughly linearly, they are
being queued.

Cost: an integer increment and a `perf_counter` pair per call. No LLM spend --
this rides on runs that were happening anyway.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# In-flight buckets. Chosen to straddle graphiti's SEMAPHORE_LIMIT of 20 and the
# mean 6.9 facts per episode, so the common case lands mid-range rather than in an
# edge bucket.
_BUCKETS = ((1, 1), (2, 4), (5, 9), (10, 19), (20, 10**9))


def _bucket(n: int) -> str:
    for lo, hi in _BUCKETS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    return "?"


@dataclass
class PromptStat:
    calls: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    # latency samples keyed by the in-flight bucket at dispatch
    by_inflight: dict[str, list[float]] = field(default_factory=dict)

    def add(self, ms: float, inflight: int) -> None:
        self.calls += 1
        self.latencies_ms.append(ms)
        self.by_inflight.setdefault(_bucket(inflight), []).append(ms)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[k]


@dataclass
class PromptTimings:
    """Aggregate across every LLM call the guard sees, keyed by prompt_name."""

    by_prompt: dict[str, PromptStat] = field(default_factory=dict)
    inflight: int = 0
    max_inflight: int = 0

    def start(self) -> int:
        """Register a dispatch; returns the in-flight count INCLUDING this call."""
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        return self.inflight

    def finish(self, prompt_name: str, ms: float, inflight_at_dispatch: int) -> None:
        self.inflight = max(0, self.inflight - 1)
        self.by_prompt.setdefault(prompt_name, PromptStat()).add(ms, inflight_at_dispatch)

    def merge(self, other: PromptTimings) -> None:
        for name, st in other.by_prompt.items():
            mine = self.by_prompt.setdefault(name, PromptStat())
            mine.calls += st.calls
            mine.latencies_ms.extend(st.latencies_ms)
            for b, xs in st.by_inflight.items():
                mine.by_inflight.setdefault(b, []).extend(xs)
        self.max_inflight = max(self.max_inflight, other.max_inflight)

    def report(self) -> str:
        """Human-readable, printed with the ingest cost block."""
        if not self.by_prompt:
            return "llm timing: no calls observed"
        rows = sorted(self.by_prompt.items(),
                      key=lambda kv: sum(kv[1].latencies_ms), reverse=True)
        total_ms = sum(sum(st.latencies_ms) for _, st in rows)
        out = [f"llm timing by prompt (max in-flight {self.max_inflight}, "
               f"total in-call {total_ms / 1000:.0f}s):"]
        for name, st in rows:
            share = 100 * sum(st.latencies_ms) / total_ms if total_ms else 0
            out.append(
                f"  {name:38} n={st.calls:5d} "
                f"p50={_pct(st.latencies_ms, 50):7.0f}ms "
                f"p95={_pct(st.latencies_ms, 95):7.0f}ms "
                f"sum={sum(st.latencies_ms) / 1000:6.0f}s ({share:4.1f}%)")
        out.append("  latency by in-flight at dispatch "
                   "(flat => genuinely parallel; rising => provider queues us):")
        for name, st in rows:
            cells = " ".join(
                f"{b}:{statistics.median(xs):.0f}ms/n={len(xs)}"
                for b, xs in sorted(st.by_inflight.items(),
                                    key=lambda kv: int(kv[0].split("-")[0].rstrip("+"))))
            out.append(f"    {name:36} {cells}")
        return "\n".join(out)
