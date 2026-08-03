# Cassettes

Content-addressed recordings of every outbound call, so the repo runs offline.

- `llm/` — Anthropic Messages API requests and responses
- `http/` — OpenAlex and arXiv responses

The filename is the SHA-256 of the request. Each file stores the full request
alongside the response, so a cassette reads as a transcript and a changed
prompt produces a new file rather than a silently-wrong cache hit.

## ⚠️ These recordings are synthetic

The committed cassettes were **generated**, not recorded from real API calls.
`scripts/seed_cassettes.py` runs the real graph against a small fabricated
corpus of seven papers and a rule-based stand-in for the model.

**What that does prove:** the graph, routing, screening, extraction, citation
verification, cost accounting, budget enforcement, cassette keying and the
AutoGen review board all work end to end.

**What it does not prove:** anything about how well Claude actually performs
this task. The synthetic extractor copies quotes verbatim by construction, so
groundedness of 1.0 is guaranteed rather than earned.

To measure the real thing:

```bash
export SCHOLAR_GRAPH_ANTHROPIC_API_KEY=sk-ant-...
SCHOLAR_GRAPH_LLM_MODE=record uv run scholar-graph research "your question"
```

Then re-baseline `evals/thresholds.json` from what live runs actually achieve.
See `docs/evaluation.md`.

## Regenerating

```bash
make seed
```

The seeder refuses to write a recording that produced no sources or failed
verification — a demo that only *looks* healthy is worse than an obviously
broken one.

**Seed with default settings.** Document caps and round limits change request
bodies, and therefore cassette keys. Recording under non-default settings
produces cassettes a normal run will never match.
