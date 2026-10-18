# Reviewer brief

For each sampled answer, judge it against its prompt for:

- missed facts;
- invented facts or entities;
- wrong subject;
- wrong RestoresTo direction;
- a wrong duplicate or index choice.

## Input: `review-sample.jsonl`

One line per sampled answer, already accepted by the deterministic validator:

```json
{"key": "<sha256>", "prompt_name": "extract_nodes.extract_text",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "answer": {"extracted_entities": [{"name": "Veeam Backup", "entity_type_id": 1, "episode_indices": [0]}]}}
```

Read `messages` as the prompt graphiti sent (the chunk text is inside it, tagged
`<TEXT>`/`<CURRENT_MESSAGE>`/`<CURRENT MESSAGE>` depending on `prompt_name`); `answer`
is what the teacher returned for that prompt. Judge `answer` against `messages`
alone -- the same "only the prompt's own text" rule the teacher follows.

## Output

Write one line per answer you reviewed, to the path you were given, each exactly:

```json
{"key": "<the same sha256>", "verdict": "ok", "issues": []}
```

or, when something is wrong:

```json
{"key": "<the same sha256>", "verdict": "issue", "issues": ["invented entity \"Hyper-V\": not in the chunk text"]}
```

`verdict` is `"ok"` or `"issue"`; `issues` is a list of short, specific findings (one
of the five categories above, naming the entity/fact/index at fault) -- empty when
`verdict` is `"ok"`. Review every line of the input file; report only the counts of
`"ok"` vs `"issue"` once you are done.
