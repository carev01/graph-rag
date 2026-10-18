# Teacher brief

You are producing training targets for a small extraction model. Each line of your
batch file is one prompt that graphiti sends.

Core rules:

- Answer each prompt using ONLY the prompt's own text; no outside knowledge.
- Return for each key exactly one JSON object matching `response_schema`: no prose
  and no markdown fences.
- Follow every rule in the prompt: the naming rules, the documented-product rule
  (DOCUMENT CONTEXT), RestoresTo direction (the destination is the target), names as
  the text writes them, and the index ranges the prompt defines.
- Give the best answer the prompt allows. Extract every entity and fact the chunk
  states; invent none.
- If a line carries `rejection`, fix exactly what it names.
- Write your answers to `<batch>.answers.jsonl`, one `{"key", "answer"}` per line,
  and report only the counts.

## Input: `batch-XXX.jsonl`

One line per prompt:

```json
{"key": "<sha256>", "prompt_name": "extract_nodes.extract_text",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "response_schema": {"type": "json_schema", "json_schema": {"name": "ExtractedEntities", "schema": {...}}}}
```

A re-ask also carries `"previous_answer"` (the JSON object you returned last time)
and `"rejection"` (why it was rejected). Fix exactly what `rejection` names; do not
otherwise change your answer.

## Output: `batch-XXX.answers.jsonl`

One line per key, in any order, each exactly:

```json
{"key": "<the same sha256>", "answer": {"extracted_entities": [{"name": "Veeam Backup", "entity_type_id": 1, "episode_indices": [0]}]}}
```

`answer` is the JSON object itself (not a JSON string of it), shaped exactly like
`response_schema` describes for that line's `prompt_name`. Write every key from the
batch file, even one you are unsure of -- the validator, not you, decides whether it
is accepted, re-asked or dropped.
