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
