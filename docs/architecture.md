# Architecture

## Shape of the system

```
CLI ─┐
API ─┼─→ ResearchService ─→ LangGraph state machine ─→ nodes ─→ LLMProvider ─→ Anthropic
Eval ┘         │                      │                  │           │
               │                      │                  └─→ SearchService ─→ OpenAlex / arXiv
               │                      │
               │                      └─→ checkpointer (SQLite / memory)
               └─→ BudgetTracker (shared across every call in the run)
```

Three entry points, one service. The CLI, HTTP API and eval harness all go
through `ResearchService`, so they cannot drift apart in how they build
dependencies, handle interrupts, or assemble a report. A bug fixed in one is
fixed in all three.

## Why a state graph rather than a while-loop

The obvious implementation of a research agent is a `while` loop around a
tool-calling model: let it search, read, and write until it says it is done.
That is simpler, and it is the wrong shape here, for three reasons.

1. **Termination.** A loop that ends when the model says so is a loop with no
   upper bound on cost or time. Here, every cycle is an explicit edge with an
   explicit counter — `max_search_rounds`, `MAX_REVISIONS` — and the routing
   functions are pure, synchronous, and unit-tested without a model.

2. **Suspension.** Pausing for human approval mid-run means serialising
   everything and resuming later, possibly in another process. A graph with a
   checkpointer gets that from `interrupt()`. A while-loop would need its own
   state-machine — which is to say, this.

3. **Different work needs different models.** Screening abstracts and writing
   the final synthesis are not the same job and should not use the same model.
   Discrete nodes make per-stage model and effort choices explicit, and make
   the cost of each stage visible in the logs.

## State channels

State is a flat, JSON-serialisable `TypedDict` — that is what the checkpointer
persists, and a run that cannot be persisted cannot be resumed.

| Channel | Reducer | Why |
| --- | --- | --- |
| `documents` | `merge_documents` (append, dedupe) | **Append-only on purpose.** Citation markers are positions in this list; reordering or removing entries mid-run would silently renumber every citation in the draft. |
| `screened_ids` / `rejected_ids` | append | Screening filters by *marking*, never by deleting from `documents`. Discard decisions stay auditable, and the append-only invariant above survives. |
| `cited_sources` | overwrite | The exact ordered list the writer saw, frozen at synthesis time. Verification and rendering index into **this**, never into `documents`, so a later retrieval round cannot invalidate a draft's citations. |
| `notes` | append | Evidence accumulates across rounds. |
| `warnings` | append | Degradation is recorded, not swallowed. |
| `max_search_rounds` | overwrite | Seeded from settings so routing reads only from state — routers stay pure functions of state, with no dependency injection. |

### The bug this design prevents

An earlier version filtered rejected sources out of `documents` directly. With
an additive reducer that silently did nothing, so rejects reappeared. Worse, if
it *had* worked, it would have shifted every index — turning `[S3]` in a draft
into a citation of a different paper. Splitting "what we retrieved" from "what
survived screening" from "what the writer saw" removes the whole class of
error.

## Dependency injection

Nodes receive collaborators through LangGraph's `configurable`, not through
closures or module globals:

```python
config = {"configurable": {"thread_id": run_id, "deps": Deps(...)}}
```

`Deps` bundles settings, the LLM provider, the search service and the budget
tracker. This keeps one compiled graph reusable across runs, and lets tests
substitute collaborators without patching imports.

## The provider chokepoint

Every model call in the system — graph nodes *and* the AutoGen review board —
goes through `LLMProvider`. That single chokepoint is what makes record/replay,
cost accounting and budget enforcement **properties of the system** rather than
things each call site must remember.

This is why the AutoGen adapter exists (see `docs/adr/0003-autogen-adapter.md`):
using AutoGen's own Anthropic client would have been two lines and would have
punched a hole straight through all three properties.

## Failure policy

The rule: **degrade on external failure, propagate on internal error.**

| Failure | Response |
| --- | --- |
| One search provider errors | Continue with the other; warn |
| All search providers error | Warn, continue with no new documents |
| Extraction batch fails | Drop that batch, keep the rest |
| Budget exhausted at synthesis | Return extracted findings, correctly cited |
| Budget exhausted elsewhere | Service-level salvage from last checkpoint |
| Review panel fails / not installed | Ship without critiques |
| Verification fails twice | Ship with defects named in the output |
| **Cassette miss** | **Propagate.** A stale recording is a developer error; degrading would produce a confident empty report |
| Malformed model JSON | Propagate with the model name and schema in the message |

## Concurrency

Search fans out across providers and queries with `asyncio.gather`. Extraction
batches run concurrently. Both use `return_exceptions=True` so one failure
degrades rather than cancelling its siblings.

Extraction batches are deliberately small (4 sources). Large batches make the
model blur quote spans across documents, which surfaces later as an
`unsupported_quote` verification failure — expensive to discover and cheap to
prevent.
