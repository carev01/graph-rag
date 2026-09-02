"""Pure data model for the Neo4j compatibility harness. No I/O, no heavy imports —
report.py and its unit tests depend only on this."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["pass", "fail", "skip"]
Verdict = Literal["GO", "GO_WITH_CONFIG", "NO_GO"]

#: Every node/edge the harness writes carries this group_id, and teardown deletes
#: exactly this namespace. Not configurable: it is the isolation contract.
COMPAT_GROUP_ID = "compat-check"


@dataclass(frozen=True)
class CheckResult:
    name: str
    group: str
    status: Status
    detail: str = ""
    #: Only set when status == "fail" AND the check was a CypherCheck. None means
    #: a language-version retry was not applicable (a procedural CallableCheck).
    cypher5_retry: Status | None = None
    #: Informational results are rendered but never move the verdict.
    informational: bool = False


@dataclass(frozen=True)
class CypherCheck:
    """A declarative statement. Because the runner holds the text, it can re-run a
    failure as "CYPHER 5 " + cypher and record cypher5_retry."""
    name: str
    group: str
    cypher: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Optional predicate over the returned rows. Absent => "did not raise" passes.
    expect: Callable[[list[dict]], bool] | None = None
    informational: bool = False


@dataclass(frozen=True)
class CallableCheck:
    """An arbitrary async call into graphiti or one of our modules. Returns a short
    detail string on success; raises to fail; raises SkipCheck to skip."""
    name: str
    group: str
    fn: Callable[["CheckContext"], Awaitable[str]]
    informational: bool = False


Check = CypherCheck | CallableCheck


class SkipCheck(Exception):
    """Raised by a CallableCheck to report `skip` rather than `fail` — used when a
    dependency outside Neo4j (LLM, embedder, GDS plugin) is unavailable, which is
    not a Neo4j incompatibility."""


@dataclass
class CheckContext:
    """Shared clients handed to every CallableCheck so checks never build their own."""
    driver: Any                 # neo4j.AsyncDriver
    graphiti: Any               # graphiti_core.Graphiti | None
    settings: Any               # graph_extract.config.ExtractSettings
    embedding: list[float]      # a fabricated 768-d vector, reused everywhere
