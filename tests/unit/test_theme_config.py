from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def test_theme_defaults():
    s = ExtractSettings(_env_file=None, **_MIN)
    assert s.report_llm_base_url == "" and s.report_llm_model == "" and s.report_llm_api_key == ""
    assert s.leiden_min_community_size == 3
    assert s.leiden_max_levels == 3
    assert s.report_token_budget == 12000
    assert s.report_top_entities == 30
