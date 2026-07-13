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
