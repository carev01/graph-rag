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

Judgement calls, settled once so every teacher answers them the same way:

- Fact text states what is configured, supported or required -- not the UI click-path
  used to do it.
- Edge dedup: a new fact that restates an existing fact in other words ("supports X"
  vs "facilitates backups of X") IS a duplicate; a new fact that contains an existing
  fact plus added detail (a number, a qualifier, a condition) is NOT.
- Summaries: when updating an existing summary, drop its sentences that are about a
  different entity.
- An entity named only in a breadcrumb or section heading counts as mentioned only if
  the body text also refers to it.
- Echo each line's `ordinal` and `check` unchanged next to its `key`: they prove the
  answer is filed under the prompt it answers. An answer whose `ordinal`/`check` do
  not match its key's line is rejected as "answer filed under the wrong prompt".
- Write your answers to `<batch>.answers.jsonl`, one
  `{"key", "ordinal", "check", "answer"}` per line, and report only the counts.

## Input: `batch-XXX.jsonl`

One line per prompt:

```json
{"key": "<sha256>", "ordinal": <n>, "check": "<12 hex chars>",
 "prompt_name": "extract_nodes.extract_text",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "response_schema": {"type": "json_schema", "json_schema": {"name": "ExtractedEntities", "schema": {...}}}}
```

`ordinal` is the line's 1-based position in the file; `check` is a short fingerprint
of the line's user message. Copy both, never compute them.

A re-ask also carries `"previous_answer"` (the JSON object you returned last time)
and `"rejection"` (why it was rejected). Fix exactly what `rejection` names; do not
otherwise change your answer.

## Output: `batch-XXX.answers.jsonl`

One line per key, in any order, each exactly:

```json
{"key": "<the same sha256>", "ordinal": <the same n>, "check": "<the same 12 hex chars>", "answer": {"extracted_entities": [{"name": "<product>", "entity_type_id": 1, "episode_indices": [0]}]}}
```

`answer` is the JSON object itself (not a JSON string of it), shaped exactly like
`response_schema` describes for that line's `prompt_name`. Write every key from the
batch file, even one you are unsure of -- the validator, not you, decides whether it
is accepted, re-asked or dropped.
