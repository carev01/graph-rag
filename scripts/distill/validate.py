"""Deterministic validation of a teacher answer (spec section 4).

Three checks, run in order, any one short-circuiting the rest -- an empty list
means the answer is valid, training-target quality, and ready for `store.put`:

1. **Schema.** `answer` parses against the prompt's own response model
   (`scripts.build_extraction_dataset.RESPONSE_MODELS[prompt_name].model_validate`).
   A pydantic `ValidationError` is a hard reject: there is no derivable fix for
   a malformed JSON object.
2. **Index ranges.** `scripts.build_extraction_dataset.validate_capture`, reused
   verbatim rather than re-implemented: every defect it names is a real
   rejection site in graphiti 0.30.1 or this project's own `dedup_guard`, so a
   training target with one of these defects would teach the student an answer
   production already has to repair or discard.
3. **Grounding** (`extract_nodes.extract_text` only). The gate
   (`graph_extract.grounding.filter_entities`) silently REPAIRS a reply at
   ingest time -- drops an ungrounded entity, injects the documented product,
   renames an alias into it. A training target must need none of that repair,
   so this validator REJECTS instead of fixing: an answer the gate would change
   fails here, with every changed entity named in the reason.

`build_grounding_context` rebuilds the gate's `GroundingContext` from the
prompt's own `<TEXT>` block (never the live graph) and the article's
structural product/vendor, exactly as `sample.json` records them -- the
harness's `intake` command is what resolves "which article" (via the key's
askers) before calling it.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import ValidationError

from graph_extract.grounding import (
    EXTRACT_TEXT_PROMPT_NAME,
    GroundingContext,
    GroundingStats,
    filter_entities,
)
from scripts.build_extraction_dataset import RESPONSE_MODELS, capture_episode_content, validate_capture


def build_grounding_context(prompt_name: str, messages: list[dict[str, str]], *,
                            product: str, vendor: str,
                            stats: GroundingStats | None = None,
                            ) -> GroundingContext | None:
    """The gate's context for this prompt, or None when grounding does not apply
    (any prompt other than `extract_text`) or the prompt has no parseable
    `<TEXT>` block (passed through -- an unparseable prompt is not this
    validator's business, `validate_capture`'s own `prompt_entity_types_unparseable`
    etc. already covers it)."""
    if prompt_name != EXTRACT_TEXT_PROMPT_NAME or len(messages) < 2:
        return None
    text = capture_episode_content(prompt_name, messages[1]["content"])
    if text is None:
        return None
    return GroundingContext(text=text, allowed=tuple(n for n in (product, vendor) if n),
                            stats=stats if stats is not None else GroundingStats(),
                            product=product)


def _entity_names(payload: Any) -> Counter[str]:
    entities = payload.get("extracted_entities") if isinstance(payload, dict) else None
    if not isinstance(entities, list):
        return Counter()
    return Counter(e.get("name") for e in entities
                  if isinstance(e, dict) and isinstance(e.get("name"), str))


def _grounding_reasons(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """One reason per entity the gate's filter/inject/rename pass would change --
    the multiset difference between the answer as given and what `filter_entities`
    returns. A rename shows up as one dropped name (the alias) and one added name
    (the canonical product), which is exactly the pair a reviewer needs to see."""
    before_names, after_names = _entity_names(before), _entity_names(after)
    dropped = sorted((before_names - after_names).elements())
    added = sorted((after_names - before_names).elements())
    reasons = [f"grounding: {name!r} is not grounded in the chunk text "
              f"(the gate would drop it)" for name in dropped]
    reasons += [f"grounding: {name!r} is missing or should be the documented product "
               f"(the gate would add or rename to it)" for name in added]
    return reasons


def validate_answer(prompt_name: str, messages: list[dict[str, str]], answer: dict[str, Any],
                    *, grounding_ctx: GroundingContext | None = None) -> list[str]:
    """The checks above, in order. Returns the human-readable rejection reasons,
    or an empty list when `answer` is valid."""
    model = RESPONSE_MODELS.get(prompt_name)
    if model is None:
        return [f"unknown_prompt_type: {prompt_name}"]
    try:
        obj = model.model_validate(answer)
    except ValidationError as exc:
        return [f"schema_invalid: {exc}"]
    defects = validate_capture(prompt_name, messages, obj, allow_overlong_summaries=False)
    if defects:
        return defects
    if prompt_name == EXTRACT_TEXT_PROMPT_NAME and grounding_ctx is not None:
        filtered = filter_entities(answer, grounding_ctx)
        if filtered != answer:
            return _grounding_reasons(answer, filtered)
    return []
