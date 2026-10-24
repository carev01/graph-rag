# DPO labelling brief (dpo4)

You label data for a small extraction model. Each line of your batch file is one task.
Every task's `prompt` is self-contained: it holds its own instructions, the documentation
chunk and the facts or reply to judge. Use ONLY the prompt's own text; no outside
knowledge.

- Answer every line with exactly one JSON object, as the prompt's last paragraph
  specifies -- no prose, no markdown fences.
- Quotes are checked by code, character for character (runs of whitespace count as one
  space) against the chunk's CURRENT MESSAGE. Copy them; never paraphrase. A quote that
  is not in the CURRENT MESSAGE costs the fact.
- Be strict about support. A fact the text does not state is unsupported, however
  plausible.
- Echo each line's `key`, `ordinal` and `check` unchanged next to your answer.
- If a line carries `rejection`, fix exactly what it names (your earlier answer is in
  `previous_answer`). This is the last chance: a second rejection drops the task.

Write your answers to `<batch>.answers.jsonl`, one line per task:

    {"key": "<sha256>", "ordinal": <n>, "check": "<12 hex>", "answer": {...}}

then report only the counts (answered, skipped).
