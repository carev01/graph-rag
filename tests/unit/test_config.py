from graph_sync.config import get_settings
from graph_extract.config import ExtractSettings

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


_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def test_routing_defaults_on_with_cheap_ling():
    s = ExtractSettings(_env_file=None, **_MIN)
    assert s.extraction_routing is True
    assert s.cheap_llm_model == "inclusionai/ling-2.6-flash"
    assert s.cheap_llm_base_url == "https://openrouter.ai/api/v1"
    assert s.cheap_llm_client_mode == "generic_json_schema"
    # the FIELD default is empty (a key must be supplied via .env); assert the
    # declared default directly so a real key in the deployment .env can't leak
    # into this hermetic check.
    assert ExtractSettings.model_fields["cheap_llm_api_key"].default == ""
    assert s.cheap_max_chunk_tokens == 900


def test_dense_thresholds_default():
    s = ExtractSettings(_env_file=None, **_MIN)
    assert s.dense_table_line_ratio == 0.25
    assert s.dense_pipe_count == 200


def test_global_search_defaults():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert ExtractSettings.model_fields["map_llm_base_url"].default == ""
    assert s.global_shortlist_k == 10
    assert s.global_default_level == 1
    assert s.global_map_relevance_min == 2


def test_drift_defaults():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.drift_primer_level == 1
    assert s.drift_primer_k == 5
    assert s.drift_max_followups == 4
    assert s.drift_followup_k == 8
    assert s.drift_iterations == 1


def test_router_default_mode():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.router_default_mode == "drift"


def test_theme_refresh_tau_default():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.theme_refresh_jaccard_tau == 0.5
