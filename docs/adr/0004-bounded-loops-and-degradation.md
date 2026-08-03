# ADR 0004 — Bound every loop; degrade rather than fail

**Status:** accepted

## Context

This agent has three cycles: search → screen → extract → coverage → search;
verify → revise → verify; and panel → revise. Each could in principle run
forever, and each costs money per iteration.

The common pattern is to let the model decide when it is done ("respond DONE
when you have enough information"). That places the termination condition
inside the component least able to guarantee it.

## Decision

1. Every loop has an explicit numeric bound, checked in a **pure routing
   function** that takes state and returns the next node.
2. A hard USD budget is checked **before** each call, not tallied after it.
3. When a bound or budget bites, the system **degrades to a useful partial
   result** rather than raising.

## Rationale

**On bounds.** A model deciding it is done is a heuristic; a counter is a
guarantee. Pure routing functions are also directly unit-testable — the routing
tests in `test_graph.py` need no model, no network and no fixtures, and they
cover the cases that matter (cap reached with gaps still open, panel requesting
a revision past the cap).

**On degradation.** This is meant to run unattended. Failing a whole run
because arXiv is having a bad afternoon throws away everything already paid
for. Concretely:

- Budget exhausted at synthesis → return the extracted findings, correctly
  cited. They were already paid for, and they pass the same verification a
  written report does.
- Verification defects surviving two repairs → ship *with the defects named*.
  A report that says where it is weak is more useful than no report; a third
  rewrite of a claim the evidence does not support will not fix the evidence.
- Review panel unavailable → ship without critiques.

**The line.** Degradation applies to *external* failures. Internal errors —
cassette misses, malformed model JSON — propagate, because they indicate a bug
and hiding them produces confidently-wrong output. The one time this line was
blurred during development, a swallowed cassette miss produced an empty report
claiming 100% citation precision. See ADR 0002.

## Consequences

**Good:** no unbounded spend, no infinite loops, partial results instead of
lost work. Every degradation is recorded in `warnings` and surfaced in the
report and the logs, so it is visible rather than silent.

**Bad:** more code paths, and each needs a test — `TestDegradation` and
`TestSearchLoop` exist for exactly this. There is also a real risk of
degrading so gracefully that a chronically broken system looks healthy;
mitigated by putting warnings in the report body and by alerting on
`synthesize.degraded` and `search.provider_failed` (see `docs/operations.md`).
