# ADR 0001 — Verify citations mechanically, not with a model

**Status:** accepted

## Context

The defining failure of a research agent is not a wrong answer. It is a
*plausible* answer with citations that do not support it. A wrong answer gets
challenged; a confidently-cited wrong answer gets forwarded.

Two options for catching it:

1. **LLM-as-judge** — ask a model whether each claim is supported by its cited
   source.
2. **Mechanical verification** — resolve markers against retrieved sources and
   check quoted spans against source text with string algorithms.

## Decision

Mechanical verification, as a blocking gate. The LLM review board exists too,
but it is advisory and cannot substitute for the mechanical check.

## Rationale

- **A judge is itself unverified.** Using a model to check a model's citations
  means the guarantee is only as good as the judge — and nothing checks the
  judge. The failure mode we care about most (fabricated support) is precisely
  the one a language model is prone to accepting.
- **Determinism.** The same draft yields the same verdict, forever. That is
  what allows citation precision to be a CI gate rather than a noisy metric.
- **Cost and latency.** Verification is free and instant. A judge would add a
  call per claim.
- **It cannot be argued with.** The checker has no prompt to be injected into
  and no reasoning to be led astray. Whatever the draft says about itself, the
  quote either appears in the source or it does not.

## Design consequence

The extraction step is forced to produce a `quote` field copied verbatim. That
constraint exists *because* of this decision — it is what makes claims
checkable at all. Prose alone could not be verified this way.

Quote matching tolerates transcription drift (dropped articles, normalised
whitespace, case) but is **order-preserving**, so reordering — which can invert
meaning — fails. The threshold is 90% of tokens recoverable in order.

## Consequences

**Good:** the strongest guarantee in the system, and the cheapest. Catches a
real quote attributed to the wrong paper, which human review essentially never
catches.

**Bad:** it verifies *attribution*, not *correctness*. A correctly-quoted
source can still be misinterpreted, and a claim can be true but poorly
supported. That gap is what the AutoGen panel is for, and it is stated in
`docs/evaluation.md` rather than glossed.
