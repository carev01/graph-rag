def test_build_cheap_graphiti_uses_cheap_model(monkeypatch):
    import graph_extract.graphiti_client as gc
    from graph_extract.config import ExtractSettings

    seen = {}
    monkeypatch.setattr(gc, "build_graphiti", lambda cfg, *, tier="strong": seen.update(
        model=cfg.llm_model, base=cfg.llm_base_url, key=cfg.llm_api_key,
        mode=cfg.llm_client_mode, layout=cfg.llm_cache_layout, tier=tier) or "G")

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                        cheap_llm_api_key="or-key")
    out = gc.build_cheap_graphiti(s)
    assert out == "G"
    assert seen == {"model": "inclusionai/ling-2.6-flash",
                    "base": "https://openrouter.ai/api/v1",
                    "key": "or-key", "mode": "generic_json_schema", "layout": False,
                    "tier": "cheap"}


def test_extraction_tier_holds_fields():
    from graph_extract.graphiti_client import ExtractionTier
    t = ExtractionTier(name="cheap", graphiti="G", instructions="I", max_chunk_tokens=900)
    assert (t.name, t.graphiti, t.instructions, t.max_chunk_tokens) == ("cheap", "G", "I", 900)


def _s(**kw):
    from graph_extract.config import ExtractSettings
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p", **kw)


def test_the_cheap_tier_carries_its_own_cache_layout_flag(monkeypatch):
    import graph_extract.graphiti_client as gc
    seen = {}
    monkeypatch.setattr(gc, "build_graphiti",
                        lambda cfg, *, tier="strong": seen.update(layout=cfg.llm_cache_layout))
    gc.build_cheap_graphiti(_s(cheap_llm_cache_layout=True))
    assert seen["layout"] is True


def test_the_cache_layout_wrapper_is_installed_only_when_configured(monkeypatch):
    """Inert-knob guard: the flag must reach the client, not merely exist."""
    import graph_extract.graphiti_client as gc
    calls: list = []
    monkeypatch.setattr(gc, "wrap_client", lambda client, learner: calls.append(learner) or client)
    base = dict(llm_base_url="https://openrouter.ai/api/v1", llm_api_key="k",
                llm_client_mode="generic_json_schema")
    gc._llm_client(_s(**base, llm_cache_layout=True), tier="cheap")
    assert len(calls) == 1
    gc._llm_client(_s(**base), tier="cheap")
    assert len(calls) == 1, "off by default"
