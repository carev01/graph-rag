def test_build_cheap_graphiti_uses_cheap_model(monkeypatch):
    import graph_extract.graphiti_client as gc
    from graph_extract.config import ExtractSettings

    seen = {}
    monkeypatch.setattr(gc, "build_graphiti", lambda cfg: seen.update(
        model=cfg.llm_model, base=cfg.llm_base_url, key=cfg.llm_api_key,
        mode=cfg.llm_client_mode) or "G")

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                        cheap_llm_api_key="or-key")
    out = gc.build_cheap_graphiti(s)
    assert out == "G"
    assert seen == {"model": "inclusionai/ling-2.6-flash",
                    "base": "https://openrouter.ai/api/v1",
                    "key": "or-key", "mode": "generic_json_schema"}


def test_extraction_tier_holds_fields():
    from graph_extract.graphiti_client import ExtractionTier
    t = ExtractionTier(name="cheap", graphiti="G", instructions="I", max_chunk_tokens=900)
    assert (t.name, t.graphiti, t.instructions, t.max_chunk_tokens) == ("cheap", "G", "I", 900)
