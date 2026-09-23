from graph_extract.config import ExtractSettings, get_extract_settings


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
    assert s.llm_client_mode == "generic_json_schema"
    assert s.max_chunk_tokens == 1800


def test_vector_search_defaults():
    s = ExtractSettings(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u",
                        neo4j_password="p", docext_base_url="https://x",
                        docext_read_key="k")
    assert s.vector_search_enabled is True
    assert s.vector_search_fetch_k == 200
