# Evaluation

```bash
make eval                      # replay: free, deterministic, CI-safe
uv run python evals/run_eval.py --mode live --out report.md
```

Non-zero exit when any metric crosses `evals/thresholds.json`. That is what
turns this from a dashboard into a regression gate.

## What is measured, and why

| Metric | Definition | Why this one |
| --- | --- | --- |
| **citation precision** | share of `[Sn]` markers resolving to a retrieved source | Catches the model inventing source numbers — the failure a reader is least able to detect. |
| **groundedness** | share of extracted quotes that exist verbatim in the source they name | Catches attribution errors, including a *real* quote pinned to the *wrong* paper. |
| **verification pass rate** | share of runs shipping with zero unresolved defects | Precision and groundedness are averages; this is the all-or-nothing view. |
| **coverage** | share of expected terms present in the answer | A perfectly-cited answer to the wrong question still fails the user. |
| **source rate** | share of runs meeting the case's `min_sources` | Guards against a run that "passes" by citing almost nothing. |
| **cost / latency** | max and mean USD and seconds per run | Quality regressions are loud; cost regressions are silent until the invoice. |
| **completion rate** | share of runs that finished at all | An unattended eval that pauses for approval is a failed eval. |

### Why thresholds are absolute, not relative

`min_citation_precision` is `1.0`. Not "no worse than last week" — a citation
that doesn't resolve is a defect, and a ratchet that drifts downward one
acceptable-looking step at a time is how a quality bar disappears.

Cost is the exception, and is a ceiling rather than a target: it should be
tightened when the system gets cheaper, never loosened to make a red build
green.

## The honest caveat

**The committed thresholds are calibrated against synthetic cassettes.** With
those, the synthetic extractor copies quotes verbatim by construction, so
groundedness of 1.0 is close to guaranteed. The suite is therefore currently
testing *the harness and the verification logic*, not the model's ability to
research.

Against live models the numbers will be lower and the interesting work begins:

1. Record real cassettes (`--mode record`) over a broader dataset.
2. Re-baseline `thresholds.json` from observed live performance.
3. Commit the cassettes so CI keeps gating deterministically against real
   model behaviour.

This is called out rather than papered over because a green eval badge that
only measures a fixture is worse than no badge.

## Extending the dataset

`evals/dataset.jsonl`, one case per line:

```json
{"id": "short-slug", "question": "...", "must_mention": ["term"], "min_sources": 4}
```

`must_mention` is deliberately substring-based (`"quantis"` matches both
"quantisation" and "quantization"). It is a coarse coverage proxy — cheap,
deterministic, and no LLM judge in the scoring loop.

**Why no LLM-as-judge.** A model scoring another model's output is itself
unverified, non-deterministic, and costs money per eval run. The metrics here
are mechanical on purpose: they can be recomputed identically forever, and they
cannot be argued with. An LLM judge would be the right addition for qualities
these metrics genuinely cannot reach — argument quality, whether the synthesis
is *insightful* — as a separate, clearly-labelled signal, not folded into the
gate.

## What is not measured yet

Named honestly, because the gaps matter:

- **Answer correctness.** Coverage checks that a term appears, not that the
  claim about it is right. Needs expert-labelled reference answers.
- **Recall.** Nothing checks whether the search *missed* the key paper. Needs a
  dataset with known-relevant sources per question.
- **Robustness.** No adversarial questions, no ambiguous or unanswerable ones,
  no prompt-injection cases in retrieved abstracts.
- **Live-model variance.** Single runs; no repeat-and-report-spread.
