"""The eval's raw-answer dump must survive the datetimes answers carry."""
import json
from datetime import datetime, timezone

from answer_api.eval_router import _raw_json


def test_raw_json_serialises_datetimes_as_iso8601():
    raw = [{"q": "x", "results": [{"valid_at": datetime(2026, 9, 25, 1, 2, 3,
                                                         tzinfo=timezone.utc)}]}]
    out = json.loads(_raw_json(raw))
    assert out[0]["results"][0]["valid_at"] == "2026-09-25T01:02:03+00:00"
