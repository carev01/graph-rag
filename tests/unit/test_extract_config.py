from graph_extract.config import get_extract_settings


def test_extract_settings_defaults(monkeypatch):
    monkeypatch.setenv("DOCEXT_BASE_URL", "https://x")
    monkeypatch.setenv("DOCEXT_READ_KEY", "k")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    get_extract_settings.cache_clear()
    s = get_extract_settings()
    assert s.embed_dim == 768
    assert s.group_id == "backup-docs"
    assert s.llm_client_mode == "structured"
    assert s.max_chunk_tokens == 1800
