"""build_graphiti must install the gate, and the default must be suspended --
the behaviour it would otherwise enable is measured to be wrong."""
from __future__ import annotations

from graph_extract.config import ExtractSettings


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


def test_the_default_is_suspended():
    assert _settings().ingest_detect_contradictions is False


def test_the_flag_can_be_enabled_explicitly():
    assert _settings(ingest_detect_contradictions=True).ingest_detect_contradictions is True
