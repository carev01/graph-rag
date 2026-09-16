"""Measure what a Neo4j vector index buys over graphiti's brute-force cosine scan.

Phase B step 1 (`docs/superpowers/specs/2026-09-16-vector-index-crossover-design.md`).
Measurement only: this implements no `SearchInterface` and changes nothing under
`src/`.

Why it exists: the live graph holds 3,469 facts and the corpus projects to ~5.3M,
so every claim about the index at scale is currently a linear extrapolation three
orders of magnitude past the measured range.

There is no crossover to find -- at 3,096 facts the index already won (255 ms
brute vs 144 ms indexed). The open question is whether its curve stays FLAT while
brute force climbs at the measured 0.053 ms/fact. That needs range, not size.

`SEARCH` does not exist on Neo4j 2026.07.1. The server deprecates
`db.index.vector.queryRelationships` "in favour of SEARCH", but SEARCH is not
valid Cypher under either CYPHER 5 or CYPHER 25, and SHOW PROCEDURES offers only
the two procedures. Probed 2026-09-16. The deprecated procedures are what works.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

DIM = 768
BYTES_PER_FLOAT = 8
# Neo4j's property store costs more than the raw floats: block headers, the
# dynamic array record chain, and the copy the page cache holds. 2x is a
# deliberately conservative multiplier -- the guard's job is to refuse early.
STORE_OVERHEAD = 2.0


@dataclass(frozen=True)
class VectorPool:
    """Vectors plus WHERE THEY CAME FROM.

    Provenance travels with the data because the two sources are not
    interchangeable: resampled vectors keep the clustering of real embeddings,
    i.i.d. ones are near-equidistant in 768 dimensions, which is the worst case
    for HNSW traversal and would make the index look worse than reality.
    """

    vectors: list[list[float]]
    provenance: str
    is_fallback: bool


def _normalise(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0.0:
        return v
    return [x / norm for x in v]


def build_pool(
    n: int,
    dim: int,
    rng: random.Random,
    seeds: list[list[float]] | None = None,
    noise: float = 0.15,
) -> VectorPool:
    """`n` unit vectors, resampled from `seeds` with Gaussian noise when given.

    Without seeds this falls back to i.i.d. Gaussian and SAYS SO -- a silent
    fallback would publish recall numbers that do not transfer to real
    embeddings as though they did.
    """
    if seeds:
        vectors = [
            _normalise([x + rng.gauss(0.0, noise) for x in seeds[rng.randrange(len(seeds))]])
            for _ in range(n)
        ]
        return VectorPool(
            vectors,
            f"resampled from {len(seeds)} real fact_embedding values (noise sigma={noise})",
            False,
        )
    vectors = [_normalise([rng.gauss(0.0, 1.0) for _ in range(dim)]) for _ in range(n)]
    return VectorPool(
        vectors,
        "i.i.d. Gaussian -- FALLBACK, the live graph was unreachable. Latency "
        "conclusions hold; RECALL NUMBERS ARE NOT COMPARABLE to real embeddings.",
        True,
    )


def recall_at_10(brute_uuids: list[str], index_uuids: list[str]) -> float:
    """Share of brute force's true top-10 that the index's top-10 also contains.

    Both sides are cut to 10 deliberately. When the index is asked for 50
    candidates the caller still keeps 10, so scoring all 50 against a 10-item
    truth would report a recall no caller ever experiences.
    """
    truth = brute_uuids[:10]
    if not truth:
        return 0.0
    return len(set(truth) & set(index_uuids[:10])) / len(truth)


def rank1_agrees(brute_uuids: list[str], index_uuids: list[str]) -> bool:
    """Did the index find the single nearest neighbour? Dedup cares about this
    separately from set recall."""
    return bool(brute_uuids) and bool(index_uuids) and brute_uuids[0] == index_uuids[0]


def estimated_bytes(n_edges: int, dim: int = DIM) -> int:
    return int(n_edges * dim * BYTES_PER_FLOAT * STORE_OVERHEAD)


def free_bytes(path: str = "/var/lib/docker") -> int:
    """Free space where the container's store will land. Falls back to `/` when
    the Docker root is not readable from here."""
    target = path if os.path.exists(path) else "/"
    st = os.statvfs(target)
    return st.f_bavail * st.f_frsize


def fits(
    n_edges: int,
    free: int,
    headroom_bytes: int = 5 * 1024**3,
    dim: int = DIM,
) -> bool:
    """Whether a step fits with room for the HNSW index and transaction logs.

    The headroom is not politeness: filling the root filesystem mid-sweep takes
    the Docker daemon down with it.
    """
    return estimated_bytes(n_edges, dim) + headroom_bytes <= free
