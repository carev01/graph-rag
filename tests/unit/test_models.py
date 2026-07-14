import base64
import json
from pathlib import Path
from graph_sync.models import (
    parse_delta_line, ContentRecord, TombstoneRecord, ControlRecord,
    decode_cursor_seq, min_watermark,
)

FIX = Path(__file__).parent.parent / "fixtures" / "aws_delta.ndjson"

def _cursor(seq: int) -> str:
    return base64.b64encode(json.dumps({"seq": seq, "v": 1}).encode()).decode()

def test_parse_control_and_content_from_fixture():
    lines = [ln for ln in FIX.read_text().splitlines() if ln.strip()]
    first = parse_delta_line(lines[0])
    assert isinstance(first, ControlRecord) and first.control == "bootstrap_start"
    last = parse_delta_line(lines[-1])
    assert isinstance(last, ControlRecord) and last.control == "cursor"
    content = [parse_delta_line(ln) for ln in lines[1:-1]]
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
