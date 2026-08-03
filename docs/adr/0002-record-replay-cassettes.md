# ADR 0002 — Content-addressed record/replay for every external call

**Status:** accepted

## Context

LLM applications are hard to test for three compounding reasons: calls cost
money, responses are non-deterministic, and CI needs a secret to run at all.
The usual responses are to mock the SDK client, or to skip integration tests.

Both are unsatisfying. Mocking the SDK means the request-construction code —
schema normalisation, prompt assembly, parameter choices — is never exercised,
which is exactly where the bugs live.

## Decision

Hash every outbound request (model, system, messages, output config) into a
content-addressed cassette on disk. Three modes: `live`, `record`, `replay`.
Apply the same treatment to search HTTP calls.

Fake **only** the network boundary — in tests, `_call_api` is the single
overridden method.

## Rationale

- **The interesting code still runs.** Cassette keying, cost accounting, budget
  enforcement, structured-output schema generation and pydantic validation all
  execute for real in every test. Only the HTTP call is substituted.
- **Prompt changes are visible.** The system prompt is part of the hash, so an
  edit produces a new key and a loud miss with instructions to re-record —
  rather than silently reusing a response to a different question.
- **A cassette is a readable transcript.** Storing the full request alongside
  the response costs disk and buys debuggability.
- **CI needs no secret and no budget.**

## Design details

- **Cost is recomputed on replay** from stored token counts and the price
  table, not read from the cassette. So updating prices re-prices history, and
  eval cost assertions remain meaningful without spending anything.
- **A cassette miss propagates.** It is a developer error, not a transient
  provider failure, so it is deliberately re-raised past the handlers that
  degrade on provider outages.

  This is not hypothetical. During development, a miss *was* swallowed by the
  search node's degradation handler; the run produced a confident report with
  zero sources and a vacuous 100% citation precision. The explicit re-raise
  exists so that cannot recur.

## Consequences

**Good:** a fresh clone runs the full suite and a complete end-to-end research
run offline, deterministically, for $0.

**Bad:**

- Cassettes are a committed artifact that must be regenerated when prompts
  change. Mitigated by the actionable error message.
- Replay proves the *plumbing* works, not that the *model* performs well. Only
  live runs measure that, which is why `docs/evaluation.md` states the caveat
  explicitly rather than letting a green badge imply more than it means.
- The shipped cassettes are synthetic, and are labelled as such everywhere they
  are mentioned.
