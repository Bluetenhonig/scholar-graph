# Operations

What you would need to know to run this and be woken up by it.

## Deploy

```bash
docker build -t scholar-graph .
docker run -p 8000:8000 \
  -e SCHOLAR_GRAPH_LLM_MODE=live \
  -e SCHOLAR_GRAPH_ANTHROPIC_API_KEY=sk-ant-... \
  -e SCHOLAR_GRAPH_LOG_FORMAT=json \
  -e SCHOLAR_GRAPH_MAX_USD_PER_RUN=0.50 \
  -v scholar-graph-data:/data \
  -e SCHOLAR_GRAPH_CHECKPOINT_DB=/data/checkpoints.sqlite \
  scholar-graph
```

`GET /healthz` reports liveness and the active LLM mode. Check the mode: a
service accidentally running in `replay` will answer every question with a
cassette miss.

## Cost control

Three independent layers, because any one of them can be misconfigured:

1. **`MAX_USD_PER_RUN`** — checked *before* every call, so it is a ceiling
   rather than a post-mortem. Nodes degrade rather than dying when it bites.
2. **Structural bounds** — `MAX_SEARCH_ROUNDS`, `MAX_DOCUMENTS`,
   `MAX_REVISIONS`. These bound the run even if pricing changes underneath you.
3. **`REQUIRE_APPROVAL_OVER_USD`** — a human sees anything projected to be
   unusually expensive before it runs.

Per-call cost lands in the structured logs (`llm.call` with `usd` and
`run_usd`), and per-run totals are on every report. To alert on cost, alert on
`run_usd` rather than on a monthly invoice.

**When prices change:** update `src/scholar_graph/llm/pricing.py`. Because
replayed cost is recomputed from token counts rather than read from the
cassette, historical cassettes re-price themselves and the eval cost gate stays
honest.

## Observability

Structured logs (`LOG_FORMAT=json` in production), every line carrying `run_id`
via a `contextvar`. One `grep run_id=...` reconstructs a single run out of
interleaved concurrent traffic.

Events worth building dashboards on:

| Event | Watch for |
| --- | --- |
| `llm.call` | `usd`, `run_usd` — cost per stage and per run |
| `verify.done` | `citation_precision`, `groundedness`, `issues` — quality, per run |
| `route.revise` / `route.revision_capped` | Rising revisions means drafting quality is degrading |
| `route.search_capped` | Runs hitting the retrieval bound; the corpus or queries may be wrong |
| `search.provider_failed` | Provider health |
| `research.budget_exhausted` | Budget too tight, or a run looping unexpectedly |
| `panel.skipped` / `panel.failed` | Review coverage silently dropping |
| `synthesize.degraded` | Runs completing without a written synthesis |

`timed()` wraps the run with `duration_ms` and an `outcome` field on both the
success and failure paths.

## Failure modes and what to do

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `CassetteMiss` in production | Service is in `replay` mode | Set `LLM_MODE=live` |
| Every run warns "search failed" | Provider outage or egress blocked | Check OpenAlex/arXiv reachability; runs degrade but return thin reports |
| Reports arrive with citation issues | Model drifting, or retrieved abstracts too thin | Inspect `verify.done` issues; consider raising `MAX_DOCUMENTS` |
| Costs climbing | More search rounds firing | Check `route.search_again` rate; tighten `MAX_SEARCH_ROUNDS` |
| Runs stuck `awaiting_approval` | Nobody is approving | Lower `REQUIRE_APPROVAL_OVER_USD`, or wire approvals to a real queue |
| `LLMRefusal` | Safety classifiers declined | Expected for some topics; surfaces as a failed run with the category logged |

## Rate limits and retries

The Anthropic SDK retries 429/5xx with backoff (`max_retries=3`). HTTP search
retries only genuinely transient statuses (408/425/429/5xx and transport
errors) with exponential backoff and jitter — a 404 is an answer, not a hiccup.

Nothing here implements a global rate limiter. Under real concurrency you would
want one; today, concurrency is bounded only by how many runs you start.

## Known limits before this is production-ready

Stated plainly, because pretending otherwise is how you get paged:

1. **The API's run registry is in-process.** `RunStore` is a dict. Multiple
   replicas will not see each other's runs. Fix: back it with the same store as
   the checkpointer (Postgres) — LangGraph ships a Postgres checkpointer.
2. **SQLite checkpointing is single-node.** Fine for one instance and a mounted
   volume; use `langgraph-checkpoint-postgres` for HA.
3. **Background tasks die with the process.** A restart mid-run orphans it. The
   checkpoint survives, so a resume endpoint could recover it, but nothing
   currently reaps orphans automatically.
4. **No authentication.** The API is unauthenticated. Put it behind a gateway,
   or add auth before exposing it.
5. **No global concurrency limit.** Nothing stops N simultaneous runs from
   exhausting rate limits together.
6. **Retrieval is abstracts-only.** No full-text fetching, so findings are
   limited to what an abstract states. Full-text would need PDF handling,
   licensing care, and a much larger context budget.
