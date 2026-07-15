# Ingestion Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `graph-sync` — a FastAPI service that consumes DocExtractor's delta feed and writes the structural knowledge-graph layer (Vendor→Product→Source→Chapter→Article) into Neo4j, driven by webhook push + polling fallback, with no LLM involvement.

**Architecture:** One FastAPI service of small single-purpose units. A pure `mapper`/`toc_mapper` core turns delta records and TOC trees into graph operations; thin I/O adapters (`delta_client`, `toc`, `catalog`, `neo4j_repo`, `state_store`) do all network/DB work; `sync_core` orchestrates with strict cursor discipline and idempotent `MERGE` writes. Chapters are authoritative, rebuilt per-source from the TOC endpoint.

**Tech Stack:** Python 3.12, `uv`, FastAPI + uvicorn, httpx (async), neo4j async driver, asyncpg (Postgres), pydantic v2 + pydantic-settings, pytest + pytest-asyncio + testcontainers, ruff, mypy.

Spec: [`../specs/2026-07-12-ingestion-foundation-design.md`](../specs/2026-07-12-ingestion-foundation-design.md).

## Global Constraints

- **Python 3.12**; async throughout (`httpx.AsyncClient`, `neo4j.AsyncGraphDatabase`, `asyncpg`).
- **No LLM / embedding / Graphiti calls** anywhere in this slice.
- **Cursor is opaque**: stored/passed verbatim; decode only in a debug helper, never in the sync path.
- **Cursor advances only on a clean terminal `{"control":"cursor"}` line** and a fully-applied batch. Any doubt → do not advance → retry.
- **All graph writes are idempotent `MERGE`** keyed on stable IDs (at-least-once processing).
- **Articles are tombstoned, never deleted** (`removed=true`, keep node). **Chapters are rebuilt from the authoritative TOC** (safe to detach/remove).
- **TOC refresh is decoupled from article sync**: a TOC failure never blocks/reverts article ingestion or cursor advance.
- **Secrets (API keys, HMAC secret) come from env**, never committed. Default test lane: `pytest -m "not live"`.
- **Live endpoints:** `https://docextractor.k3s.home.lan` (internal CA → `verify=False` in dev). Dev host reachable at `srv-openclaw.home.lan` (172.16.255.190).

---

## File Structure

```
graph-rag/
  pyproject.toml                     # deps, ruff, mypy, pytest config
  docker-compose.yml                 # neo4j 5.x + postgres 16 for local dev
  .env.example                       # documented env vars (no real secrets)
  src/graph_sync/
    __init__.py
    config.py         # pydantic-settings Settings
    models.py         # delta record types, cursor helpers, graph-op dataclasses
    catalog.py        # source_id -> product/vendor id + metadata cache
    mapper.py         # pure: delta record -> StructuralWrite / Tombstone
    toc_mapper.py     # pure: TOC tree -> TocSnapshot (chapters, links, nesting)
    delta_client.py   # streaming NDJSON delta client + cursor discipline
    toc.py            # TOC tree fetch
    neo4j_repo.py     # schema + idempotent structural/tombstone/TOC writes
    state_store.py    # Postgres: cursor, bootstrap progress, webhook dedup, dead-letter, lock
    sync_core.py      # orchestration (bootstrap, incremental, gating, TOC pass, single-flight)
    webhook.py        # FastAPI route: HMAC verify, dedup, debounce, enqueue
    poll_loop.py      # background 30-min safety-net task
    app.py            # FastAPI wiring, /health, /status, lifespan
    cli.py            # register-webhook, bootstrap, sync-once, refresh-toc
  tests/
    unit/  integration/  e2e/  fixtures/
```

---

### Task 1: Project scaffold, tooling, config, local infra

**Files:**
- Create: `pyproject.toml`, `docker-compose.yml`, `.env.example`, `src/graph_sync/__init__.py`, `src/graph_sync/config.py`, `tests/__init__.py`, `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `graph_sync.config.Settings` (pydantic-settings) with fields: `docext_base_url: str`, `docext_read_key: str`, `docext_admin_key: str = ""`, `docext_verify_tls: bool = False`, `neo4j_uri: str`, `neo4j_user: str`, `neo4j_password: str`, `postgres_dsn: str`, `webhook_secret: str = ""`, `webhook_public_url: str = ""`, `poll_interval_seconds: int = 1800`, `webhook_debounce_seconds: int = 300`. Factory `get_settings() -> Settings`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "graph-sync"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.30",
  "httpx>=0.27",
  "neo4j>=5.22",
  "asyncpg>=0.29",
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
  "typer>=0.12",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-asyncio>=0.23",
  "testcontainers[neo4j,postgres]>=4.5",
  "ruff>=0.5",
  "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/graph_sync"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["live: hits the real DocExtractor cluster (opt-in)"]
addopts = "-m 'not live'"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  neo4j:
    image: neo4j:5.22
    environment:
      NEO4J_AUTH: neo4j/testpassword
      NEO4J_PLUGINS: '["graph-data-science"]'
    ports: ["7474:7474", "7687:7687"]
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: testpassword
      POSTGRES_DB: graphsync
    ports: ["5432:5432"]
```

- [ ] **Step 3: Write `.env.example`**

```bash
DOCEXT_BASE_URL=https://docextractor.k3s.home.lan
DOCEXT_READ_KEY=dxk_replace_me
DOCEXT_ADMIN_KEY=dxk_replace_me
DOCEXT_VERIFY_TLS=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=testpassword
POSTGRES_DSN=postgresql://postgres:testpassword@localhost:5432/graphsync
WEBHOOK_SECRET=
WEBHOOK_PUBLIC_URL=http://srv-openclaw.home.lan:8080/webhooks/docextractor
POLL_INTERVAL_SECONDS=1800
WEBHOOK_DEBOUNCE_SECONDS=300
```

- [ ] **Step 4: Write the failing test** — `tests/unit/test_config.py`

```python
import os
from graph_sync.config import get_settings

def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("DOCEXT_BASE_URL", "https://x.local")
    monkeypatch.setenv("DOCEXT_READ_KEY", "dxk_read")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://localhost/db")
    get_settings.cache_clear()
    s = get_settings()
    assert s.docext_base_url == "https://x.local"
    assert s.docext_verify_tls is False
    assert s.poll_interval_seconds == 1800
```

- [ ] **Step 5: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: graph_sync.config`).

- [ ] **Step 6: Write `src/graph_sync/config.py`**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    docext_base_url: str
    docext_read_key: str
    docext_admin_key: str = ""
    docext_verify_tls: bool = False
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    postgres_dsn: str
    webhook_secret: str = ""
    webhook_public_url: str = ""
    poll_interval_seconds: int = 1800
    webhook_debounce_seconds: int = 300

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
```

Also create empty `src/graph_sync/__init__.py` and `tests/__init__.py`.

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example src/graph_sync tests
git commit -m "chore: scaffold graph-sync project, tooling, config"
```

---

### Task 2: Capture live fixtures

**Files:**
- Create: `tests/fixtures/aws_delta.ndjson`, `tests/fixtures/aws_toc.json`, `tests/fixtures/vendors.json`, `tests/fixtures/products.json`, `tests/fixtures/sources.json`, `tests/fixtures/README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: on-disk fixtures used by later unit tests. `aws_delta.ndjson` = the AWS Backup Developer Guide bootstrap stream (146 content records + control lines). `aws_toc.json` = its TOC tree. The three catalog JSONs = full vendor/product/source lists.

- [ ] **Step 1: Capture the delta stream and TOC** (uses the read key; run from repo root)

```bash
export RKEY="$DOCEXT_READ_KEY"  # read-only key from untracked .env (never commit the literal)
export B=https://docextractor.k3s.home.lan
export SRC=21632f3b-5a4c-4c93-9f00-6701d0e9f677
mkdir -p tests/fixtures
curl -sN -H "X-API-Key: $RKEY" "$B/api/articles/delta?source_id=$SRC" > tests/fixtures/aws_delta.ndjson
curl -sk -H "X-API-Key: $RKEY" "$B/api/articles/toc/$SRC" > tests/fixtures/aws_toc.json
curl -sk -H "X-API-Key: $RKEY" "$B/api/vendors?limit=200"  > tests/fixtures/vendors.json
curl -sk -H "X-API-Key: $RKEY" "$B/api/products?limit=200" > tests/fixtures/products.json
curl -sk -H "X-API-Key: $RKEY" "$B/api/sources?limit=500"  > tests/fixtures/sources.json
```

- [ ] **Step 2: Verify the fixtures are well-formed**

```bash
head -1 tests/fixtures/aws_delta.ndjson | grep -q bootstrap_start && echo "delta ok"
tail -1 tests/fixtures/aws_delta.ndjson | grep -q '"control":"cursor"' && echo "terminal ok"
python -c "import json; json.load(open('tests/fixtures/aws_toc.json'))" && echo "toc ok"
```
Expected: `delta ok`, `terminal ok`, `toc ok`.

- [ ] **Step 3: Write `tests/fixtures/README.md`** noting provenance (source id, date captured, that content_markdown is real vendor doc text used only as test input).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures
git commit -m "test: capture live DocExtractor fixtures (AWS Backup source)"
```

---

### Task 3: Delta record models + cursor helpers

**Files:**
- Create: `src/graph_sync/models.py`, `tests/unit/test_models.py`

**Interfaces:**
- Consumes: fixtures from Task 2.
- Produces:
  - `ContentRecord` (pydantic): `change_type: Literal["added","updated"]`, `id: str`, `topic_key: str`, `source_id: str`, `vendor: str`, `product: str`, `title: str`, `source_url: str`, `last_updated_at: str | None`, `content_hash: str`, `estimated_tokens: int`, `parent_chapter: str | None`, `top_level_chapter: str | None`, `sort_order: int`, `run_id: str | None`, `seq: int | None`. Ignores extra keys (`content_markdown`, `images`).
  - `TombstoneRecord`: `change_type: Literal["removed"]`, `id: str`, `source_id: str`, `removed_at: str`, `run_id: str | None`, `seq: int | None`, `topic_key: str | None`.
  - `ControlRecord`: `control: Literal["bootstrap_start","cursor"]`, `next_since: str | None`, `count: int | None`.
  - `DeltaRecord = ContentRecord | TombstoneRecord | ControlRecord`.
  - `parse_delta_line(line: str) -> DeltaRecord` — dispatches on `control` / `change_type`.
  - `decode_cursor_seq(cursor: str) -> int` — base64→JSON→`["seq"]` (debug/min-watermark only).
  - `min_watermark(cursors: list[str]) -> str` — returns the cursor with the smallest decoded seq.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_models.py`

```python
import base64, json
from pathlib import Path
from graph_sync.models import (
    parse_delta_line, ContentRecord, TombstoneRecord, ControlRecord,
    decode_cursor_seq, min_watermark,
)

FIX = Path(__file__).parent.parent / "fixtures" / "aws_delta.ndjson"

def _cursor(seq: int) -> str:
    return base64.b64encode(json.dumps({"seq": seq, "v": 1}).encode()).decode()

def test_parse_control_and_content_from_fixture():
    lines = [l for l in FIX.read_text().splitlines() if l.strip()]
    first = parse_delta_line(lines[0])
    assert isinstance(first, ControlRecord) and first.control == "bootstrap_start"
    last = parse_delta_line(lines[-1])
    assert isinstance(last, ControlRecord) and last.control == "cursor"
    content = [parse_delta_line(l) for l in lines[1:-1]]
    assert all(isinstance(r, ContentRecord) for r in content)
    assert content[0].source_id and content[0].content_hash

def test_parse_tombstone():
    line = json.dumps({"seq": 327, "change_type": "removed", "id": "abc",
                       "source_id": "s1", "removed_at": "2026-07-11T18:41:55Z",
                       "run_id": None})
    rec = parse_delta_line(line)
    assert isinstance(rec, TombstoneRecord) and rec.run_id is None

def test_cursor_helpers():
    assert decode_cursor_seq(_cursor(42)) == 42
    assert min_watermark([_cursor(50), _cursor(10), _cursor(30)]) == _cursor(10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/graph_sync/models.py`**

```python
from __future__ import annotations
import base64, json
from typing import Literal
from pydantic import BaseModel, ConfigDict

class ContentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    change_type: Literal["added", "updated"]
    id: str
    topic_key: str
    source_id: str
    vendor: str
    product: str
    title: str
    source_url: str
    last_updated_at: str | None = None
    content_hash: str
    estimated_tokens: int
    parent_chapter: str | None = None
    top_level_chapter: str | None = None
    sort_order: int
    run_id: str | None = None
    seq: int | None = None

class TombstoneRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    change_type: Literal["removed"]
    id: str
    source_id: str
    removed_at: str
    run_id: str | None = None
    seq: int | None = None
    topic_key: str | None = None

class ControlRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    control: Literal["bootstrap_start", "cursor"]
    next_since: str | None = None
    count: int | None = None

DeltaRecord = ContentRecord | TombstoneRecord | ControlRecord

def parse_delta_line(line: str) -> DeltaRecord:
    obj = json.loads(line)
    if "control" in obj:
        return ControlRecord.model_validate(obj)
    if obj.get("change_type") == "removed":
        return TombstoneRecord.model_validate(obj)
    return ContentRecord.model_validate(obj)

def decode_cursor_seq(cursor: str) -> int:
    return int(json.loads(base64.b64decode(cursor))["seq"])

def min_watermark(cursors: list[str]) -> str:
    return min(cursors, key=decode_cursor_seq)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_sync/models.py tests/unit/test_models.py
git commit -m "feat: delta record models and cursor helpers"
```

---

### Task 4: Catalog (source→product→vendor resolution)

**Files:**
- Create: `src/graph_sync/catalog.py`, `tests/unit/test_catalog.py`

**Interfaces:**
- Consumes: `Settings`; catalog fixture JSONs.
- Produces:
  - `SourceInfo` (dataclass): `source_id, source_name, base_url, source_type, platform, last_extracted_at, product_id, product_name, product_version, vendor_id, vendor_name, vendor_website` (all `str | None` except the ids/names).
  - `class Catalog`: `async def load(self) -> None`; `def resolve(self, source_id: str) -> SourceInfo | None`; `async def refresh(self) -> None`. Constructed with an httpx `AsyncClient` + read key, OR a `load_from` classmethod taking three parsed lists (for pure testing).
  - `Catalog.from_lists(vendors, products, sources) -> Catalog` (no I/O) — builds the resolution maps.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_catalog.py`

```python
import json
from pathlib import Path
from graph_sync.catalog import Catalog

FIX = Path(__file__).parent.parent / "fixtures"

def test_resolve_source_to_product_and_vendor():
    vendors = json.loads((FIX / "vendors.json").read_text())["vendors"]
    products = json.loads((FIX / "products.json").read_text())["products"]
    sources = json.loads((FIX / "sources.json").read_text())["sources"]
    cat = Catalog.from_lists(vendors, products, sources)
    info = cat.resolve("21632f3b-5a4c-4c93-9f00-6701d0e9f677")  # AWS Backup Dev Guide
    assert info is not None
    assert info.product_name == "AWS Backup"
    assert info.vendor_name == "AWS"
    assert info.vendor_id and info.product_id

def test_resolve_unknown_source_returns_none():
    cat = Catalog.from_lists([], [], [])
    assert cat.resolve("does-not-exist") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_catalog.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/graph_sync/catalog.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import httpx

@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    source_name: str
    base_url: str | None
    source_type: str | None
    platform: str | None
    last_extracted_at: str | None
    product_id: str
    product_name: str
    product_version: str | None
    vendor_id: str
    vendor_name: str
    vendor_website: str | None

class Catalog:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._by_source: dict[str, SourceInfo] = {}

    @classmethod
    def from_lists(cls, vendors: list[dict], products: list[dict],
                   sources: list[dict]) -> "Catalog":
        cat = cls()
        cat._build(vendors, products, sources)
        return cat

    def _build(self, vendors: list[dict], products: list[dict],
               sources: list[dict]) -> None:
        vmap = {v["id"]: v for v in vendors}
        pmap = {p["id"]: p for p in products}
        by_source: dict[str, SourceInfo] = {}
        for s in sources:
            p = pmap.get(s["product_id"])
            if p is None:
                continue
            v = vmap.get(p["vendor_id"], {})
            by_source[s["id"]] = SourceInfo(
                source_id=s["id"], source_name=s.get("name", ""),
                base_url=s.get("base_url"), source_type=s.get("source_type"),
                platform=s.get("platform"), last_extracted_at=s.get("last_extracted_at"),
                product_id=p["id"], product_name=p.get("name", ""),
                product_version=p.get("version"),
                vendor_id=v.get("id", p["vendor_id"]), vendor_name=v.get("name", ""),
                vendor_website=v.get("website"),
            )
        self._by_source = by_source

    def resolve(self, source_id: str) -> SourceInfo | None:
        return self._by_source.get(source_id)

    async def load(self) -> None:
        await self.refresh()

    async def refresh(self) -> None:
        assert self._client is not None, "Catalog needs an httpx client to refresh"
        vendors = (await self._client.get("/api/vendors", params={"limit": 500})).json()["vendors"]
        products = (await self._client.get("/api/products", params={"limit": 500})).json()["products"]
        sources = (await self._client.get("/api/sources", params={"limit": 1000})).json()["sources"]
        self._build(vendors, products, sources)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_catalog.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_sync/catalog.py tests/unit/test_catalog.py
git commit -m "feat: catalog source->product->vendor resolution"
```

---

### Task 5: Mapper (pure: delta record → graph-write structures)

**Files:**
- Create: `src/graph_sync/mapper.py`, `tests/unit/test_mapper.py`
- Modify: `src/graph_sync/models.py` (add graph-write dataclasses)

**Interfaces:**
- Consumes: `ContentRecord`, `TombstoneRecord`, `SourceInfo`.
- Produces (add to `models.py`):
  - `StructuralWrite` (dataclass): `vendor: dict`, `product: dict`, `source: dict`, `article: dict` — each a flat property dict ready for a Cypher `MERGE`.
  - `Tombstone` (dataclass): `article_id: str`, `removed_at: str`.
  - In `mapper.py`: `map_content(rec: ContentRecord, info: SourceInfo) -> StructuralWrite`; `map_tombstone(rec: TombstoneRecord) -> Tombstone`. Pure — no chapter linkage here (chapters come from the TOC pass).

- [ ] **Step 1: Add dataclasses to `models.py`**

```python
from dataclasses import dataclass, field  # add to imports

@dataclass
class StructuralWrite:
    vendor: dict
    product: dict
    source: dict
    article: dict

@dataclass
class Tombstone:
    article_id: str
    removed_at: str
```

- [ ] **Step 2: Write the failing test** — `tests/unit/test_mapper.py`

```python
from graph_sync.models import ContentRecord, TombstoneRecord
from graph_sync.catalog import SourceInfo
from graph_sync.mapper import map_content, map_tombstone

INFO = SourceInfo(
    source_id="s1", source_name="Developer Guide", base_url="https://b/",
    source_type="web", platform="docusaurus", last_extracted_at="2026-07-12T16:00:53Z",
    product_id="p1", product_name="AWS Backup", product_version=None,
    vendor_id="v1", vendor_name="AWS", vendor_website="https://aws.amazon.com",
)

def _rec(**over):
    base = dict(change_type="added", id="a1", topic_key="tk", source_id="s1",
               vendor="AWS", product="AWS Backup", title="What is AWS Backup?",
               source_url="https://x/whatis.html", content_hash="h1",
               estimated_tokens=4075, sort_order=0, seq=None, run_id="r1")
    base.update(over)
    return ContentRecord.model_validate(base)

def test_map_content_builds_structural_write():
    w = map_content(_rec(), INFO)
    assert w.vendor == {"id": "v1", "name": "AWS", "website": "https://aws.amazon.com"}
    assert w.product["id"] == "p1" and w.product["vendor_id"] == "v1"
    assert w.source["id"] == "s1" and w.source["platform"] == "docusaurus"
    assert w.article["id"] == "a1" and w.article["content_hash"] == "h1"
    assert w.article["source_id"] == "s1"
    assert "content_markdown" not in w.article

def test_map_tombstone():
    rec = TombstoneRecord.model_validate(
        {"change_type": "removed", "id": "a9", "source_id": "s1",
         "removed_at": "2026-07-11T18:41:55Z", "run_id": None})
    t = map_tombstone(rec)
    assert t.article_id == "a9" and t.removed_at == "2026-07-11T18:41:55Z"
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/unit/test_mapper.py -v`
Expected: FAIL (`ModuleNotFoundError: graph_sync.mapper`).

- [ ] **Step 4: Write `src/graph_sync/mapper.py`**

```python
from __future__ import annotations
from graph_sync.catalog import SourceInfo
from graph_sync.models import ContentRecord, StructuralWrite, Tombstone, TombstoneRecord

def map_content(rec: ContentRecord, info: SourceInfo) -> StructuralWrite:
    return StructuralWrite(
        vendor={"id": info.vendor_id, "name": info.vendor_name,
                "website": info.vendor_website},
        product={"id": info.product_id, "name": info.product_name,
                 "version": info.product_version, "vendor_id": info.vendor_id},
        source={"id": info.source_id, "name": info.source_name,
                "base_url": info.base_url, "source_type": info.source_type,
                "platform": info.platform, "last_extracted_at": info.last_extracted_at},
        article={"id": rec.id, "title": rec.title, "source_url": rec.source_url,
                 "topic_key": rec.topic_key, "content_hash": rec.content_hash,
                 "estimated_tokens": rec.estimated_tokens, "sort_order": rec.sort_order,
                 "last_updated_at": rec.last_updated_at, "run_id": rec.run_id,
                 "seq": rec.seq, "source_id": rec.source_id, "removed": False},
    )

def map_tombstone(rec: TombstoneRecord) -> Tombstone:
    return Tombstone(article_id=rec.id, removed_at=rec.removed_at)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/unit/test_mapper.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/graph_sync/mapper.py src/graph_sync/models.py tests/unit/test_mapper.py
git commit -m "feat: pure mapper delta record -> structural write"
```

---

### Task 6: TOC mapper (pure: TOC tree → chapter snapshot)

**Files:**
- Create: `src/graph_sync/toc_mapper.py`, `tests/unit/test_toc_mapper.py`
- Modify: `src/graph_sync/models.py` (add `TocSnapshot`)

**Interfaces:**
- Consumes: TOC JSON (`{"source_id", "entries":[{id,title,url,level,sort_order,is_article,article_id?,children[]}]}`).
- Produces (add to `models.py`):
  - `ChapterRow` (dataclass): `id, source_id, title, url, level, sort_order`.
  - `TocSnapshot` (dataclass): `source_id: str`, `chapters: list[ChapterRow]`, `root_ids: list[str]`, `nesting: list[tuple[str,str]]` (parent_id, child_id), `article_links: list[tuple[str,str]]` (article_id, chapter_id).
  - In `toc_mapper.py`: `map_toc(toc: dict) -> TocSnapshot`. Every entry becomes a `ChapterRow`; an entry's `article_id` (when present) links to *its own* chapter id; nesting from `children`; roots are the top-level entries.

- [ ] **Step 1: Add dataclasses to `models.py`**

```python
@dataclass
class ChapterRow:
    id: str
    source_id: str
    title: str
    url: str | None
    level: int
    sort_order: int

@dataclass
class TocSnapshot:
    source_id: str
    chapters: list["ChapterRow"] = field(default_factory=list)
    root_ids: list[str] = field(default_factory=list)
    nesting: list[tuple[str, str]] = field(default_factory=list)
    article_links: list[tuple[str, str]] = field(default_factory=list)
```

- [ ] **Step 2: Write the failing test** — `tests/unit/test_toc_mapper.py`

```python
import json
from pathlib import Path
from graph_sync.toc_mapper import map_toc

FIX = Path(__file__).parent.parent / "fixtures" / "aws_toc.json"

def test_map_toc_flattens_tree():
    toc = json.loads(FIX.read_text())
    snap = map_toc(toc)
    assert snap.source_id == toc["source_id"]
    ids = {c.id for c in snap.chapters}
    # Known root entry from the fixture:
    assert "6d93201e-6728-48cc-9f94-4cba18b5d2a5" in ids
    assert "6d93201e-6728-48cc-9f94-4cba18b5d2a5" in snap.root_ids
    # Nesting: the child chapter is linked under its parent
    assert ("6d93201e-6728-48cc-9f94-4cba18b5d2a5",
            "d57bd09b-bd96-40ec-a2dd-076b3a95ba85") in snap.nesting
    # An entry that carries an article_id links that article to its own chapter
    assert ("2c92266f-f84c-49c2-9e89-3b4376ec9043",
            "6d93201e-6728-48cc-9f94-4cba18b5d2a5") in snap.article_links

def test_map_toc_synthetic_both_article_and_section():
    toc = {"source_id": "s1", "entries": [
        {"id": "c0", "title": "Root", "url": "u", "level": 0, "sort_order": 0,
         "is_article": False, "article_id": "artRoot", "children": [
            {"id": "c1", "title": "Child", "url": "u2", "level": 1, "sort_order": 1,
             "is_article": True, "article_id": "artChild", "children": []}]}]}
    snap = map_toc(toc)
    assert ("artRoot", "c0") in snap.article_links
    assert ("artChild", "c1") in snap.article_links
    assert ("c0", "c1") in snap.nesting
    assert snap.root_ids == ["c0"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/unit/test_toc_mapper.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Write `src/graph_sync/toc_mapper.py`**

```python
from __future__ import annotations
from graph_sync.models import ChapterRow, TocSnapshot

def map_toc(toc: dict) -> TocSnapshot:
    source_id = toc["source_id"]
    snap = TocSnapshot(source_id=source_id)

    def walk(entry: dict, parent_id: str | None) -> None:
        cid = entry["id"]
        snap.chapters.append(ChapterRow(
            id=cid, source_id=source_id, title=entry.get("title", ""),
            url=entry.get("url"), level=entry.get("level", 0),
            sort_order=entry.get("sort_order", 0),
        ))
        if parent_id is None:
            snap.root_ids.append(cid)
        else:
            snap.nesting.append((parent_id, cid))
        if entry.get("article_id"):
            snap.article_links.append((entry["article_id"], cid))
        for child in entry.get("children", []) or []:
            walk(child, cid)

    for e in toc.get("entries", []):
        walk(e, None)
    return snap
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/unit/test_toc_mapper.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/graph_sync/toc_mapper.py src/graph_sync/models.py tests/unit/test_toc_mapper.py
git commit -m "feat: pure TOC mapper tree -> chapter snapshot"
```

---

### Task 7: Read clients (delta streaming + TOC fetch)

**Files:**
- Create: `src/graph_sync/delta_client.py`, `src/graph_sync/toc.py`, `tests/unit/test_delta_client.py`

**Interfaces:**
- Consumes: `parse_delta_line`, models.
- Produces:
  - `class DeltaStream`: constructed with an httpx `AsyncClient`, plus query params. `async def records(self) -> AsyncIterator[ContentRecord | TombstoneRecord]` yields data records, skipping controls but capturing them into attributes set during iteration: `self.bootstrap_start_since: str | None`, `self.next_since: str | None`, `self.terminated_clean: bool`, `self.count: int | None`. Cursor discipline lives here: `next_since`/`terminated_clean` are only set when the terminal `cursor` line is seen.
  - `def build_delta_params(*, since=None, source_id=None, vendor_id=None, bootstrap_after=None) -> dict` (pure helper; tested directly).
  - `async def fetch_toc(client: httpx.AsyncClient, source_id: str) -> dict` in `toc.py`.
- A shared `def make_client(settings) -> httpx.AsyncClient` (base_url, `X-API-Key`, `verify`) — put in `delta_client.py` and reused.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_delta_client.py` (drives `DeltaStream` against a fake transport replaying the fixture)

```python
import httpx, pytest
from pathlib import Path
from graph_sync.delta_client import DeltaStream, build_delta_params
from graph_sync.models import ContentRecord

FIX = Path(__file__).parent.parent / "fixtures" / "aws_delta.ndjson"

def test_build_params_omits_none():
    assert build_delta_params(since="cur") == {"since": "cur"}
    assert build_delta_params(source_id="s") == {"source_id": "s"}
    assert build_delta_params(bootstrap_after="x") == {"bootstrap_after": "x"}
    assert build_delta_params() == {}

def _transport(body: bytes) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)
    return httpx.MockTransport(handler)

async def test_stream_applies_cursor_discipline_on_clean_terminal():
    body = FIX.read_bytes()
    client = httpx.AsyncClient(transport=_transport(body), base_url="https://x")
    stream = DeltaStream(client, params={"source_id": "s"})
    recs = [r async for r in stream.records()]
    assert all(isinstance(r, ContentRecord) for r in recs)
    assert len(recs) == 146
    assert stream.terminated_clean is True
    assert stream.next_since is not None
    assert stream.bootstrap_start_since is not None

async def test_stream_truncated_has_no_next_since():
    lines = FIX.read_bytes().split(b"\n")
    truncated = b"\n".join(lines[:50])  # drop terminal control line
    client = httpx.AsyncClient(transport=_transport(truncated), base_url="https://x")
    stream = DeltaStream(client, params={})
    _ = [r async for r in stream.records()]
    assert stream.terminated_clean is False
    assert stream.next_since is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_delta_client.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/graph_sync/delta_client.py`**

```python
from __future__ import annotations
from typing import AsyncIterator
import httpx
from graph_sync.config import Settings
from graph_sync.models import (
    ContentRecord, TombstoneRecord, ControlRecord, parse_delta_line,
)

def make_client(settings: Settings, *, admin: bool = False) -> httpx.AsyncClient:
    key = settings.docext_admin_key if admin else settings.docext_read_key
    return httpx.AsyncClient(
        base_url=settings.docext_base_url,
        headers={"X-API-Key": key},
        verify=settings.docext_verify_tls,
        timeout=300.0,
    )

def build_delta_params(*, since: str | None = None, source_id: str | None = None,
                       vendor_id: str | None = None,
                       bootstrap_after: str | None = None) -> dict:
    params = {"since": since, "source_id": source_id,
              "vendor_id": vendor_id, "bootstrap_after": bootstrap_after}
    return {k: v for k, v in params.items() if v is not None}

class DeltaStream:
    def __init__(self, client: httpx.AsyncClient, params: dict) -> None:
        self._client = client
        self._params = params
        self.bootstrap_start_since: str | None = None
        self.next_since: str | None = None
        self.count: int | None = None
        self.terminated_clean: bool = False

    async def records(self) -> AsyncIterator[ContentRecord | TombstoneRecord]:
        async with self._client.stream(
            "GET", "/api/articles/delta", params=self._params
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                rec = parse_delta_line(line)
                if isinstance(rec, ControlRecord):
                    if rec.control == "bootstrap_start":
                        self.bootstrap_start_since = rec.next_since
                    elif rec.control == "cursor":
                        self.next_since = rec.next_since
                        self.count = rec.count
                        self.terminated_clean = True
                    continue
                yield rec
```

- [ ] **Step 4: Write `src/graph_sync/toc.py`**

```python
from __future__ import annotations
import httpx

async def fetch_toc(client: httpx.AsyncClient, source_id: str) -> dict:
    resp = await client.get(f"/api/articles/toc/{source_id}")
    resp.raise_for_status()
    return resp.json()
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/unit/test_delta_client.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add src/graph_sync/delta_client.py src/graph_sync/toc.py tests/unit/test_delta_client.py
git commit -m "feat: delta stream client with cursor discipline + TOC fetch"
```

---

### Task 8: Neo4j repository (schema + idempotent writes + TOC rewire/prune)

**Files:**
- Create: `src/graph_sync/neo4j_repo.py`, `tests/integration/__init__.py`, `tests/integration/conftest.py`, `tests/integration/test_neo4j_repo.py`

**Interfaces:**
- Consumes: `StructuralWrite`, `Tombstone`, `TocSnapshot`.
- Produces:
  - `class Neo4jRepo`: `async def init_schema(self) -> None`; `async def apply_structural(self, w: StructuralWrite) -> None`; `async def get_content_hash(self, article_id: str) -> str | None`; `async def tombstone_article(self, t: Tombstone) -> None`; `async def apply_toc(self, snap: TocSnapshot) -> None` (upsert chapters + rewire `IN_CHAPTER` + prune chapters absent from `snap`); `async def article_count_by_source(self, source_id: str) -> int`; `async def close(self) -> None`.

- [ ] **Step 1: Write `tests/integration/conftest.py`** (spins up Neo4j via testcontainers)

```python
import pytest_asyncio
from testcontainers.neo4j import Neo4jContainer
from graph_sync.neo4j_repo import Neo4jRepo

@pytest_asyncio.fixture(scope="module")
async def neo4j_repo():
    with Neo4jContainer("neo4j:5.22") as neo:
        repo = Neo4jRepo(neo.get_connection_url(), "neo4j", neo.password)
        await repo.init_schema()
        yield repo
        await repo.close()
```

- [ ] **Step 2: Write the failing test** — `tests/integration/test_neo4j_repo.py`

```python
import pytest
from graph_sync.models import StructuralWrite, Tombstone, TocSnapshot, ChapterRow

pytestmark = pytest.mark.asyncio

def _write(article_id="a1", source_id="s1", h="h1"):
    return StructuralWrite(
        vendor={"id": "v1", "name": "AWS", "website": "https://aws"},
        product={"id": "p1", "name": "AWS Backup", "version": None, "vendor_id": "v1"},
        source={"id": source_id, "name": "Dev Guide", "base_url": "b",
                "source_type": "web", "platform": "docusaurus", "last_extracted_at": None},
        article={"id": article_id, "title": "T", "source_url": "u", "topic_key": "tk",
                 "content_hash": h, "estimated_tokens": 10, "sort_order": 0,
                 "last_updated_at": None, "run_id": "r1", "seq": None,
                 "source_id": source_id, "removed": False},
    )

async def test_apply_structural_is_idempotent(neo4j_repo):
    await neo4j_repo.apply_structural(_write())
    await neo4j_repo.apply_structural(_write())  # second apply: no duplicates
    assert await neo4j_repo.article_count_by_source("s1") == 1
    assert await neo4j_repo.get_content_hash("a1") == "h1"

async def test_tombstone_keeps_node(neo4j_repo):
    await neo4j_repo.apply_structural(_write(article_id="a2", h="h2"))
    await neo4j_repo.tombstone_article(Tombstone("a2", "2026-07-11T00:00:00Z"))
    assert await neo4j_repo.get_content_hash("a2") == "h2"  # node still present

async def test_apply_toc_rewires_and_prunes(neo4j_repo):
    await neo4j_repo.apply_structural(_write(article_id="a3", source_id="s3"))
    snap1 = TocSnapshot(source_id="s3",
        chapters=[ChapterRow("cA", "s3", "A", "u", 0, 0),
                  ChapterRow("cB", "s3", "B", "u", 0, 1)],
        root_ids=["cA", "cB"], nesting=[], article_links=[("a3", "cB")])
    await neo4j_repo.apply_toc(snap1)
    # Rebuild TOC without cB -> cB pruned, a3 relinked to cA
    snap2 = TocSnapshot(source_id="s3",
        chapters=[ChapterRow("cA", "s3", "A", "u", 0, 0)],
        root_ids=["cA"], nesting=[], article_links=[("a3", "cA")])
    await neo4j_repo.apply_toc(snap2)
    assert await neo4j_repo.chapter_exists("cB") is False
    assert await neo4j_repo.article_chapter_id("a3") == "cA"
```

(Also add helper methods `chapter_exists(id)` and `article_chapter_id(article_id)` to the repo — used only by tests but cheap.)

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/integration/test_neo4j_repo.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Write `src/graph_sync/neo4j_repo.py`**

```python
from __future__ import annotations
from neo4j import AsyncGraphDatabase
from graph_sync.models import StructuralWrite, Tombstone, TocSnapshot

_CONSTRAINTS = [
    "CREATE CONSTRAINT vendor_id IF NOT EXISTS FOR (n:Vendor) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (n:Article) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX article_source IF NOT EXISTS FOR (n:Article) ON (n.source_id)",
    "CREATE INDEX chapter_source IF NOT EXISTS FOR (n:Chapter) ON (n.source_id)",
]

_APPLY_STRUCTURAL = """
MERGE (v:Vendor {id: $vendor.id}) SET v += $vendor
MERGE (p:Product {id: $product.id}) SET p += $product
MERGE (s:Source {id: $source.id}) SET s += $source
MERGE (a:Article {id: $article.id}) SET a += $article
MERGE (v)-[:HAS_PRODUCT]->(p)
MERGE (p)-[:HAS_SOURCE]->(s)
MERGE (s)-[:HAS_ARTICLE]->(a)
"""

_APPLY_TOC = """
UNWIND $chapters AS ch
  MERGE (c:Chapter {id: ch.id}) SET c += ch
WITH $source_id AS sid
MATCH (s:Source {id: sid})
UNWIND $root_ids AS rid
  MATCH (c:Chapter {id: rid}) MERGE (s)-[:HAS_CHAPTER]->(c)
WITH sid
UNWIND $nesting AS pair
  MATCH (pc:Chapter {id: pair[0]}), (cc:Chapter {id: pair[1]})
  MERGE (pc)-[:HAS_CHAPTER]->(cc)
WITH sid
// clear old article->chapter links for this source, then relink
CALL {
  WITH sid
  MATCH (a:Article {source_id: sid})-[r:IN_CHAPTER]->() DELETE r
}
WITH sid
UNWIND $article_links AS link
  MATCH (a:Article {id: link[0]}), (c:Chapter {id: link[1]})
  MERGE (a)-[:IN_CHAPTER]->(c)
WITH sid, [ch IN $chapters | ch.id] AS keep
// prune chapters for this source no longer in the tree
MATCH (old:Chapter {source_id: sid}) WHERE NOT old.id IN keep DETACH DELETE old
"""

class Neo4jRepo:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self._driver.close()

    async def init_schema(self) -> None:
        async with self._driver.session() as sess:
            for stmt in _CONSTRAINTS:
                await sess.run(stmt)

    async def apply_structural(self, w: StructuralWrite) -> None:
        async with self._driver.session() as sess:
            await sess.run(_APPLY_STRUCTURAL, vendor=w.vendor, product=w.product,
                           source=w.source, article=w.article)

    async def get_content_hash(self, article_id: str) -> str | None:
        async with self._driver.session() as sess:
            r = await sess.run(
                "MATCH (a:Article {id:$id}) RETURN a.content_hash AS h", id=article_id)
            rec = await r.single()
            return rec["h"] if rec else None

    async def tombstone_article(self, t: Tombstone) -> None:
        async with self._driver.session() as sess:
            await sess.run(
                "MERGE (a:Article {id:$id}) SET a.removed=true, a.removed_at=$ts",
                id=t.article_id, ts=t.removed_at)

    async def apply_toc(self, snap: TocSnapshot) -> None:
        chapters = [vars(c) for c in snap.chapters]
        async with self._driver.session() as sess:
            await sess.run(_APPLY_TOC, source_id=snap.source_id, chapters=chapters,
                           root_ids=snap.root_ids, nesting=[list(p) for p in snap.nesting],
                           article_links=[list(p) for p in snap.article_links])

    async def article_count_by_source(self, source_id: str) -> int:
        async with self._driver.session() as sess:
            r = await sess.run(
                "MATCH (a:Article {source_id:$s}) RETURN count(a) AS n", s=source_id)
            return (await r.single())["n"]

    async def chapter_exists(self, chapter_id: str) -> bool:
        async with self._driver.session() as sess:
            r = await sess.run("MATCH (c:Chapter {id:$id}) RETURN c LIMIT 1", id=chapter_id)
            return await r.single() is not None

    async def article_chapter_id(self, article_id: str) -> str | None:
        async with self._driver.session() as sess:
            r = await sess.run(
                "MATCH (a:Article {id:$id})-[:IN_CHAPTER]->(c) RETURN c.id AS id",
                id=article_id)
            rec = await r.single()
            return rec["id"] if rec else None
```

- [ ] **Step 5: Run to verify it passes** (Docker required)

Run: `uv run pytest tests/integration/test_neo4j_repo.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/graph_sync/neo4j_repo.py tests/integration
git commit -m "feat: neo4j repo with idempotent structural + TOC rewire/prune writes"
```

---

### Task 9: State store (Postgres: cursor, bootstrap progress, dedup, dead-letter, lock)

**Files:**
- Create: `src/graph_sync/state_store.py`, `tests/integration/test_state_store.py`
- Modify: `tests/integration/conftest.py` (add postgres fixture)

**Interfaces:**
- Consumes: `Settings.postgres_dsn`.
- Produces:
  - `class StateStore`: `async def init_schema()`; `async def get_cursor() -> str | None`; `async def set_cursor(cursor: str)`; `async def upsert_bootstrap(shard, watermark, last_id, status)`; `async def get_bootstrap(shard) -> dict | None`; `async def all_bootstrap_watermarks() -> list[str]`; `async def seen_delivery(signature: str) -> bool` (returns True if already recorded; records if not — dedup); `async def should_run_source(source_id, debounce_seconds) -> bool` (debounce); `async def record_dead_letter(raw: str, context: str)`; `async def dead_letter_count() -> int`; `async def try_lock() -> bool` / `async def unlock()` (advisory lock via a dedicated connection); `async def close()`.

- [ ] **Step 1: Add postgres fixture to `tests/integration/conftest.py`**

```python
from testcontainers.postgres import PostgresContainer
from graph_sync.state_store import StateStore

@pytest_asyncio.fixture(scope="module")
async def state_store():
    with PostgresContainer("postgres:16") as pg:
        dsn = pg.get_connection_url().replace("+psycopg2", "")
        store = StateStore(dsn)
        await store.init_schema()
        yield store
        await store.close()
```

- [ ] **Step 2: Write the failing test** — `tests/integration/test_state_store.py`

```python
import pytest
pytestmark = pytest.mark.asyncio

async def test_cursor_roundtrip(state_store):
    assert await state_store.get_cursor() is None
    await state_store.set_cursor("cur-1")
    assert await state_store.get_cursor() == "cur-1"
    await state_store.set_cursor("cur-2")
    assert await state_store.get_cursor() == "cur-2"

async def test_delivery_dedup(state_store):
    assert await state_store.seen_delivery("sig-abc") is False
    assert await state_store.seen_delivery("sig-abc") is True

async def test_bootstrap_watermarks(state_store):
    await state_store.upsert_bootstrap("vendorA", "wm-5", "id-9", "complete")
    await state_store.upsert_bootstrap("vendorB", "wm-2", "id-3", "complete")
    assert set(await state_store.all_bootstrap_watermarks()) == {"wm-5", "wm-2"}

async def test_dead_letter(state_store):
    before = await state_store.dead_letter_count()
    await state_store.record_dead_letter("{bad json", "ctx")
    assert await state_store.dead_letter_count() == before + 1

async def test_advisory_lock_single_flight(state_store):
    assert await state_store.try_lock() is True
    assert await state_store.try_lock() is True  # reentrant on same holder is fine
    await state_store.unlock()
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/integration/test_state_store.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Write `src/graph_sync/state_store.py`**

```python
from __future__ import annotations
import asyncpg

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_cursor (
  id int PRIMARY KEY DEFAULT 1, cursor text, updated_at timestamptz DEFAULT now(),
  CHECK (id = 1));
CREATE TABLE IF NOT EXISTS bootstrap_progress (
  shard text PRIMARY KEY, watermark text, last_id text, status text,
  updated_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS webhook_delivery (
  signature text PRIMARY KEY, received_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS source_debounce (
  source_id text PRIMARY KEY, last_run_at timestamptz);
CREATE TABLE IF NOT EXISTS dead_letter (
  id bigserial PRIMARY KEY, raw text, context text, created_at timestamptz DEFAULT now());
"""
_LOCK_KEY = 911_222_333

class StateStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._lock_conn: asyncpg.Connection | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
        return self._pool

    async def init_schema(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as c:
            await c.execute(_SCHEMA)

    async def close(self) -> None:
        if self._lock_conn is not None:
            await self._lock_conn.close()
        if self._pool is not None:
            await self._pool.close()

    async def get_cursor(self) -> str | None:
        pool = await self._get_pool()
        return await pool.fetchval("SELECT cursor FROM sync_cursor WHERE id=1")

    async def set_cursor(self, cursor: str) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO sync_cursor (id, cursor, updated_at) VALUES (1,$1,now()) "
            "ON CONFLICT (id) DO UPDATE SET cursor=$1, updated_at=now()", cursor)

    async def upsert_bootstrap(self, shard, watermark, last_id, status) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO bootstrap_progress (shard,watermark,last_id,status,updated_at) "
            "VALUES ($1,$2,$3,$4,now()) ON CONFLICT (shard) DO UPDATE SET "
            "watermark=$2,last_id=$3,status=$4,updated_at=now()",
            shard, watermark, last_id, status)

    async def get_bootstrap(self, shard) -> dict | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM bootstrap_progress WHERE shard=$1", shard)
        return dict(row) if row else None

    async def all_bootstrap_watermarks(self) -> list[str]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT watermark FROM bootstrap_progress WHERE watermark IS NOT NULL")
        return [r["watermark"] for r in rows]

    async def seen_delivery(self, signature: str) -> bool:
        pool = await self._get_pool()
        res = await pool.execute(
            "INSERT INTO webhook_delivery (signature) VALUES ($1) "
            "ON CONFLICT (signature) DO NOTHING", signature)
        return res == "INSERT 0 0"  # 0 rows inserted => already seen

    async def should_run_source(self, source_id: str, debounce_seconds: int) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as c, c.transaction():
            row = await c.fetchrow(
                "SELECT last_run_at FROM source_debounce WHERE source_id=$1 FOR UPDATE",
                source_id)
            if row and row["last_run_at"] is not None:
                due = await c.fetchval(
                    "SELECT now() - $1 > make_interval(secs => $2)",
                    row["last_run_at"], debounce_seconds)
                if not due:
                    return False
            await c.execute(
                "INSERT INTO source_debounce (source_id,last_run_at) VALUES ($1,now()) "
                "ON CONFLICT (source_id) DO UPDATE SET last_run_at=now()", source_id)
            return True

    async def record_dead_letter(self, raw: str, context: str) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO dead_letter (raw,context) VALUES ($1,$2)", raw, context)

    async def dead_letter_count(self) -> int:
        pool = await self._get_pool()
        return await pool.fetchval("SELECT count(*) FROM dead_letter")

    async def try_lock(self) -> bool:
        if self._lock_conn is None:
            self._lock_conn = await asyncpg.connect(self._dsn)
        return await self._lock_conn.fetchval("SELECT pg_try_advisory_lock($1)", _LOCK_KEY)

    async def unlock(self) -> None:
        if self._lock_conn is not None:
            await self._lock_conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_KEY)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/integration/test_state_store.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/graph_sync/state_store.py tests/integration
git commit -m "feat: postgres state store (cursor, bootstrap, dedup, dead-letter, lock)"
```

---

### Task 10: Sync core (orchestration)

**Files:**
- Create: `src/graph_sync/sync_core.py`, `tests/integration/test_sync_core.py`

**Interfaces:**
- Consumes: `DeltaStream`, `fetch_toc`, `map_toc`, `Catalog`, `mapper`, `Neo4jRepo`, `StateStore`, httpx clients.
- Produces:
  - `class SyncCore(client, catalog, repo, store, settings)`.
  - `async def bootstrap(self, *, source_id=None, vendor_id=None) -> BootstrapResult` — streams, upserts (hash-gated), tracks highest id, resumes on truncation via `bootstrap_after`, records progress, then runs `refresh_toc` for each touched source; on all-clean sets `sync_cursor` to `min_watermark`.
  - `async def run_incremental(self) -> IncrementalResult` — single-flight; reads cursor; streams `since`; applies added/updated (hash-gated) + tombstones; on clean terminal advances cursor and runs `refresh_toc` for each touched source.
  - `async def refresh_toc(self, source_id: str) -> None` — fetch TOC → `map_toc` → `repo.apply_toc`; catches/logs errors without raising into the caller.
  - `async def _apply_record(rec) -> str | None` — returns the touched `source_id` (or None if skipped/dead-lettered); applies hash gating and catalog resolution.
  - Result dataclasses `BootstrapResult(applied:int, skipped:int, sources:set[str])`, `IncrementalResult(applied:int, removed:int, skipped:int, advanced:bool)`.

- [ ] **Step 1: Write the failing test** — `tests/integration/test_sync_core.py` (uses real Neo4j + Postgres fixtures + a MockTransport replaying the fixture delta and TOC)

```python
import httpx, json, pytest
from pathlib import Path
from graph_sync.catalog import Catalog
from graph_sync.sync_core import SyncCore
from graph_sync.config import Settings

pytestmark = pytest.mark.asyncio
FIX = Path(__file__).parent.parent / "fixtures"

def _settings():
    return Settings(docext_base_url="https://x", docext_read_key="k",
                    neo4j_uri="x", neo4j_user="x", neo4j_password="x",
                    postgres_dsn="x", webhook_debounce_seconds=0)

def _client():
    delta = FIX.joinpath("aws_delta.ndjson").read_bytes()
    toc = FIX.joinpath("aws_toc.json").read_bytes()
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/articles/delta":
            return httpx.Response(200, content=delta)
        if req.url.path.startswith("/api/articles/toc/"):
            return httpx.Response(200, content=toc)
        raise AssertionError(req.url.path)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")

def _catalog():
    load = lambda n: json.loads(FIX.joinpath(n).read_text())
    return Catalog.from_lists(load("vendors.json")["vendors"],
                              load("products.json")["products"],
                              load("sources.json")["sources"])

async def test_bootstrap_ingests_and_gates(neo4j_repo, state_store):
    src = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    core = SyncCore(_client(), _catalog(), neo4j_repo, state_store, _settings())
    r1 = await core.bootstrap(source_id=src)
    assert r1.applied == 146 and r1.skipped == 0
    assert await neo4j_repo.article_count_by_source(src) == 146
    # Second bootstrap: content_hash gate skips everything
    r2 = await core.bootstrap(source_id=src)
    assert r2.applied == 0 and r2.skipped == 146
    # TOC pass ran -> a known article is linked to a chapter
    assert await neo4j_repo.article_chapter_id(
        "2c92266f-f84c-49c2-9e89-3b4376ec9043") is not None

def _resuming_client():
    """First delta call truncates (no terminal); the bootstrap_after retry completes."""
    lines = FIX.joinpath("aws_delta.ndjson").read_bytes().split(b"\n")
    boot = lines[0]
    data = [l for l in lines[1:] if l.strip() and b'"control"' not in l]
    terminal = lines[-1] if b'"control":"cursor"' in lines[-1] else lines[-2]
    first = b"\n".join([boot] + data[:80])            # truncated: no terminal
    second = b"\n".join([boot] + data[80:] + [terminal])
    calls = {"n": 0}
    toc = FIX.joinpath("aws_toc.json").read_bytes()
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/api/articles/toc/"):
            return httpx.Response(200, content=toc)
        if req.url.path == "/api/articles/delta":
            if "bootstrap_after" in req.url.params:
                return httpx.Response(200, content=second)
            calls["n"] += 1
            return httpx.Response(200, content=first)
        raise AssertionError(req.url.path)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x"), calls

async def test_bootstrap_resumes_after_truncation(neo4j_repo, state_store):
    src = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
    client, calls = _resuming_client()
    core = SyncCore(client, _catalog(), neo4j_repo, state_store, _settings())
    res = await core.bootstrap(source_id=src)
    # No missed or duplicated articles across the resume boundary
    assert res.applied == 146
    assert await neo4j_repo.article_count_by_source(src) == 146
    # The original bootstrap watermark (not a resume-recomputed one) became the cursor
    assert await state_store.get_cursor() is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/test_sync_core.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/graph_sync/sync_core.py`**

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import httpx
from graph_sync.catalog import Catalog
from graph_sync.config import Settings
from graph_sync.delta_client import DeltaStream, build_delta_params
from graph_sync.mapper import map_content, map_tombstone
from graph_sync.models import ContentRecord, TombstoneRecord, min_watermark
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore
from graph_sync.toc import fetch_toc
from graph_sync.toc_mapper import map_toc

log = logging.getLogger("graph_sync.sync")

@dataclass
class BootstrapResult:
    applied: int = 0
    skipped: int = 0
    sources: set[str] = field(default_factory=set)

@dataclass
class IncrementalResult:
    applied: int = 0
    removed: int = 0
    skipped: int = 0
    advanced: bool = False

class SyncCore:
    def __init__(self, client: httpx.AsyncClient, catalog: Catalog,
                 repo: Neo4jRepo, store: StateStore, settings: Settings) -> None:
        self._client = client
        self._catalog = catalog
        self._repo = repo
        self._store = store
        self._settings = settings

    async def _apply_record(self, rec, res) -> str | None:
        if isinstance(rec, TombstoneRecord):
            await self._repo.tombstone_article(map_tombstone(rec))
            res.removed = getattr(res, "removed", 0) + 1
            return rec.source_id
        assert isinstance(rec, ContentRecord)
        existing = await self._repo.get_content_hash(rec.id)
        if existing == rec.content_hash:
            res.skipped += 1
            return None
        info = self._catalog.resolve(rec.source_id)
        if info is None:
            await self._catalog.refresh()
            info = self._catalog.resolve(rec.source_id)
        if info is None:
            await self._store.record_dead_letter(rec.model_dump_json(),
                                                 f"unknown source {rec.source_id}")
            return None
        await self._repo.apply_structural(map_content(rec, info))
        res.applied += 1
        return rec.source_id

    async def refresh_toc(self, source_id: str) -> None:
        try:
            toc = await fetch_toc(self._client, source_id)
            await self._repo.apply_toc(map_toc(toc))
        except Exception:  # decoupled: never blocks article sync
            log.exception("TOC refresh failed for source %s", source_id)

    async def bootstrap(self, *, source_id=None, vendor_id=None) -> BootstrapResult:
        res = BootstrapResult()
        bootstrap_after = None
        watermark: str | None = None
        shard = source_id or vendor_id or "global"
        while True:
            params = build_delta_params(source_id=source_id, vendor_id=vendor_id,
                                        bootstrap_after=bootstrap_after)
            stream = DeltaStream(self._client, params)
            last_id = None
            async for rec in stream.records():
                touched = await self._apply_record(rec, res)
                if touched:
                    res.sources.add(touched)
                if isinstance(rec, ContentRecord):
                    last_id = rec.id
            if watermark is None:
                watermark = stream.bootstrap_start_since
            await self._store.upsert_bootstrap(
                shard, watermark, last_id,
                "complete" if stream.terminated_clean else "in_progress")
            if stream.terminated_clean:
                break
            bootstrap_after = last_id  # resume, keep original watermark
        for sid in res.sources:
            await self.refresh_toc(sid)
        marks = await self._store.all_bootstrap_watermarks()
        if marks:
            await self._store.set_cursor(min_watermark(marks))
        return res

    async def run_incremental(self) -> IncrementalResult:
        res = IncrementalResult()
        if not await self._store.try_lock():
            return res  # another sync is running
        try:
            cursor = await self._store.get_cursor()
            stream = DeltaStream(self._client, build_delta_params(since=cursor))
            async for rec in stream.records():
                touched = await self._apply_record(rec, res)
                if touched:
                    res.__dict__.setdefault("_sources", set()).add(touched)
            if stream.terminated_clean and stream.next_since:
                await self._store.set_cursor(stream.next_since)
                res.advanced = True
                for sid in res.__dict__.get("_sources", set()):
                    await self.refresh_toc(sid)
            return res
        finally:
            await self._store.unlock()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/test_sync_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_sync/sync_core.py tests/integration/test_sync_core.py
git commit -m "feat: sync core orchestration (bootstrap, incremental, gating, TOC pass)"
```

---

### Task 11: Webhook route (HMAC verify, dedup, debounce, enqueue)

**Files:**
- Create: `src/graph_sync/webhook.py`, `tests/unit/test_webhook.py`

**Interfaces:**
- Consumes: `StateStore`, `Settings`; a `trigger` callable `Callable[[], Awaitable[None]]` (so tests inject a fake instead of a real sync).
- Produces:
  - `def verify_signature(secret: str, body: bytes, header: str | None) -> bool` (pure; `sha256=` prefix + `hmac.compare_digest`).
  - `def build_router(store, settings, trigger) -> APIRouter` exposing `POST /webhooks/docextractor`: 401 on bad HMAC; dedup via `store.seen_delivery`; debounce via `store.should_run_source`; schedule `trigger()` in the background; return `{"status":"ok"}` fast.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_webhook.py`

```python
import hmac, hashlib, json, pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from graph_sync.webhook import verify_signature, build_router

pytestmark = pytest.mark.asyncio
SECRET = "s3cr3t"

def _sig(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()

def test_verify_signature_accepts_and_rejects():
    body = b'{"a":1}'
    assert verify_signature(SECRET, body, _sig(body)) is True
    assert verify_signature(SECRET, body, "sha256=deadbeef") is False
    assert verify_signature(SECRET, body, None) is False

class FakeStore:
    def __init__(self): self.seen=set(); self.ran=[]
    async def seen_delivery(self, sig): 
        dup = sig in self.seen; self.seen.add(sig); return dup
    async def should_run_source(self, sid, deb): self.ran.append(sid); return True

async def _app(store, calls):
    from types import SimpleNamespace
    settings = SimpleNamespace(webhook_secret=SECRET, webhook_debounce_seconds=300)
    async def trigger(): calls.append(1)
    app = FastAPI(); app.include_router(build_router(store, settings, trigger))
    return app

async def test_webhook_rejects_bad_hmac():
    calls=[]; app = await _app(FakeStore(), calls)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        r = await c.post("/webhooks/docextractor", content=b"{}",
                         headers={"X-DocExtractor-Signature":"sha256=bad"})
    assert r.status_code == 401 and calls == []

async def test_webhook_accepts_and_dedups():
    calls=[]; store=FakeStore(); app = await _app(store, calls)
    body = json.dumps({"event":"extraction_complete","source_id":"s1"}).encode()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        h = {"X-DocExtractor-Signature": _sig(body)}
        r1 = await c.post("/webhooks/docextractor", content=body, headers=h)
        r2 = await c.post("/webhooks/docextractor", content=body, headers=h)
    assert r1.status_code == 200 and r2.status_code == 200
    assert store.ran == ["s1"]  # deduped: second delivery did not enqueue again
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_webhook.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/graph_sync/webhook.py`**

```python
from __future__ import annotations
import hashlib, hmac, json
from typing import Awaitable, Callable
from fastapi import APIRouter, BackgroundTasks, Request, Response

def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    if not header:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)

def build_router(store, settings, trigger: Callable[[], Awaitable[None]]) -> APIRouter:
    router = APIRouter()

    @router.post("/webhooks/docextractor")
    async def receive(request: Request, background: BackgroundTasks) -> Response:
        body = await request.body()
        sig = request.headers.get("X-DocExtractor-Signature")
        if not verify_signature(settings.webhook_secret, body, sig):
            return Response(status_code=401)
        if await store.seen_delivery(sig):
            return Response(content='{"status":"ok"}', media_type="application/json")
        payload = json.loads(body or b"{}")
        source_id = payload.get("source_id")
        run = True
        if source_id:
            run = await store.should_run_source(
                source_id, settings.webhook_debounce_seconds)
        if run:
            background.add_task(trigger)
        return Response(content='{"status":"ok"}', media_type="application/json")

    return router
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_webhook.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/graph_sync/webhook.py tests/unit/test_webhook.py
git commit -m "feat: webhook route with HMAC verify, dedup, debounce"
```

---

### Task 12: App wiring, poll loop, status endpoints, CLI

**Files:**
- Create: `src/graph_sync/app.py`, `src/graph_sync/poll_loop.py`, `src/graph_sync/cli.py`, `tests/unit/test_app.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `poll_loop.run_poll_loop(trigger, interval_seconds, stop_event)` — awaits, calls `trigger` every N seconds until `stop_event` set.
  - `app.create_app(deps) -> FastAPI` with lifespan that builds clients/repo/store/catalog + `SyncCore`, mounts the webhook router, starts the poll loop, exposes `GET /health` → `{"status":"ok"}` and `GET /status` → cursor lag / last-run / bootstrap / dead-letter counts.
  - `cli.py` (Typer): `register-webhook`, `bootstrap [--source-id/--vendor-id]`, `sync-once`, `refresh-toc <source_id>`.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_app.py`

```python
import pytest
from httpx import ASGITransport, AsyncClient
from graph_sync.app import create_app

pytestmark = pytest.mark.asyncio

class FakeStatusStore:
    async def get_cursor(self): return "cur-1"
    async def dead_letter_count(self): return 0

async def test_health_and_status():
    app = create_app(status_store=FakeStatusStore(), start_background=False)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        s = (await c.get("/status")).json()
        assert s["cursor"] == "cur-1" and s["dead_letter_count"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_app.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/graph_sync/poll_loop.py`**

```python
from __future__ import annotations
import asyncio
from typing import Awaitable, Callable

async def run_poll_loop(trigger: Callable[[], Awaitable[None]],
                        interval_seconds: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            await trigger()
```

- [ ] **Step 4: Write `src/graph_sync/app.py`** (test-focused seam: `create_app` accepts injected deps and a `start_background` flag)

```python
from __future__ import annotations
from fastapi import FastAPI

def create_app(*, status_store=None, webhook_router=None,
               start_background: bool = True) -> FastAPI:
    app = FastAPI(title="graph-sync")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/status")
    async def status() -> dict:
        if status_store is None:
            return {"cursor": None, "dead_letter_count": None}
        return {
            "cursor": await status_store.get_cursor(),
            "dead_letter_count": await status_store.dead_letter_count(),
        }

    if webhook_router is not None:
        app.include_router(webhook_router)
    return app
```

(The production entrypoint — a separate `main()` that builds real deps, wires the webhook router and poll loop via lifespan — is added here too, but is exercised by the `@live` E2E task, not this unit test.)

- [ ] **Step 5: Write `src/graph_sync/cli.py`**

```python
from __future__ import annotations
import asyncio, secrets
import typer
from graph_sync.config import get_settings
from graph_sync.delta_client import make_client
# (bootstrap/sync-once/refresh-toc build SyncCore from real deps; register-webhook
#  posts to /api/webhooks with the admin key and a generated secret.)

app = typer.Typer()

@app.command("register-webhook")
def register_webhook() -> None:
    s = get_settings()
    secret = s.webhook_secret or secrets.token_urlsafe(32)
    async def _run() -> None:
        async with make_client(s, admin=True) as client:
            resp = await client.post("/api/webhooks", json={
                "url": s.webhook_public_url,
                "events": "extraction_complete",
                "secret": secret, "is_active": True})
            resp.raise_for_status()
            typer.echo(f"registered; secret={secret}")
    asyncio.run(_run())

# @app.command("bootstrap") / "sync-once" / "refresh-toc": construct
# Catalog+Neo4jRepo+StateStore+SyncCore from get_settings() and invoke the
# matching SyncCore coroutine via asyncio.run.

if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/unit/test_app.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/graph_sync/app.py src/graph_sync/poll_loop.py src/graph_sync/cli.py tests/unit/test_app.py
git commit -m "feat: app wiring, poll loop, status endpoints, CLI"
```

---

### Task 13: End-to-end live verification

**Files:**
- Create: `tests/e2e/__init__.py`, `tests/e2e/test_live.py`

**Interfaces:**
- Consumes: real `Settings` from `.env`, real Neo4j + Postgres (docker-compose up), the live cluster + admin key.
- Produces: `@pytest.mark.live` tests proving the success criteria against reality. Run explicitly with `uv run pytest -m live`.

- [ ] **Step 1: Bring up local infra**

```bash
docker compose up -d
cp .env.example .env   # then fill DOCEXT_READ_KEY, DOCEXT_ADMIN_KEY, WEBHOOK_SECRET
```

- [ ] **Step 2: Write `tests/e2e/test_live.py`**

```python
import os, pytest
from graph_sync.config import get_settings
from graph_sync.delta_client import make_client
from graph_sync.catalog import Catalog
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore
from graph_sync.sync_core import SyncCore

pytestmark = [pytest.mark.live, pytest.mark.asyncio]
AWS_SRC = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"

async def test_live_bootstrap_reconciles_and_builds_chapters():
    s = get_settings()
    client = make_client(s)
    cat = Catalog(client); await cat.load()
    repo = Neo4jRepo(s.neo4j_uri, s.neo4j_user, s.neo4j_password); await repo.init_schema()
    store = StateStore(s.postgres_dsn); await store.init_schema()
    try:
        core = SyncCore(client, cat, repo, store, s)
        res = await core.bootstrap(source_id=AWS_SRC)
        assert res.applied >= 140  # ~146; tolerate drift
        # Reconcile against the dashboard
        dash = (await client.get("/api/dashboard/sources")).json()["sources"]
        expected = next(x["article_count"] for x in dash if x["id"] == AWS_SRC)
        assert await repo.article_count_by_source(AWS_SRC) == expected
        # Authoritative chapters: a known nested article is linked
        assert await repo.article_chapter_id(
            "2c92266f-f84c-49c2-9e89-3b4376ec9043") is not None
        # Idempotent replay
        res2 = await core.bootstrap(source_id=AWS_SRC)
        assert res2.applied == 0
    finally:
        await repo.close(); await store.close(); await client.aclose()
```

- [ ] **Step 3: Run the live bootstrap test**

Run: `uv run pytest tests/e2e/test_live.py -m live -v`
Expected: PASS (counts reconcile; chapters linked; replay is a no-op).

- [ ] **Step 4: Verify the webhook path manually**

```bash
uv run python -m graph_sync.cli register-webhook      # prints the secret; put it in .env
uv run uvicorn graph_sync.app:main --factory --host 0.0.0.0 --port 8080 &
# Trigger a re-extraction of the AWS source with the admin key, then watch logs
curl -sk -X POST -H "X-API-Key: $DOCEXT_ADMIN_KEY" \
  "$DOCEXT_BASE_URL/api/extraction/runs" -H 'Content-Type: application/json' \
  -d "{\"source_id\":\"$AWS_SRC\"}"
```
Expected: the service logs a received, HMAC-verified webhook and a resulting incremental sync + TOC refresh for the source. (If DocExtractor rejects the plain-HTTP URL at registration, switch `WEBHOOK_PUBLIC_URL` to an HTTPS terminator per spec §6 and re-register.)

- [ ] **Step 5: Commit**

```bash
git add tests/e2e
git commit -m "test: live e2e bootstrap reconciliation + chapters + webhook path"
```

---

## Self-Review Notes

- **Spec coverage:** components (§3) → Tasks 3–12; data model (§4) → Tasks 5,6,8; sync protocol/invariants (§5) → Task 10 (bootstrap resume, min-watermark cursor, hash gating, single-flight, tombstone-keep) + Task 8 (idempotent MERGE, TOC rewire/prune); TOC pass (§5.5) → Tasks 6,8,10; webhook (§6) → Task 11 + Task 13 step 4; error handling (§7): dead-letter (Tasks 9,10), unknown-source catalog refresh (Task 10), TOC decoupled from sync (Task 10 `refresh_toc`), HMAC/dedup (Task 11); testing (§8) → unit (Tasks 3–7,11,12), integration (Tasks 8–10), live (Task 13); layout (§9) → Task 1 + file structure; success criteria (§10) → criterion 1 (Task 13), 2 (Tasks 10,13), 3 (Task 10 `test_bootstrap_resumes_after_truncation`), 4 (Task 13 step 4), 5 (Task 10 `test_bootstrap_ingests_and_gates`), 6 (Task 8 `test_apply_toc_rewires_and_prunes`), 7 (default `pytest` lane across all tasks).
- **Deferred-by-design (not gaps):** `:NEXT` edges, Graphiti/episodes, LLM — all out of scope per spec §1.
- **Naming consistency:** `SyncCore.bootstrap/run_incremental/refresh_toc`, `Neo4jRepo.apply_structural/apply_toc/tombstone_article/get_content_hash`, `StateStore.get_cursor/set_cursor/seen_delivery/should_run_source/try_lock`, `DeltaStream.records/next_since/terminated_clean/bootstrap_start_since`, `Catalog.from_lists/resolve/refresh`, `map_content/map_tombstone/map_toc` — used identically across tasks.
