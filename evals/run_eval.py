"""Evaluation harness.

Runs the dataset through the real graph and scores the output on metrics that
would matter to someone relying on the result:

  citation precision   do the [Sn] markers resolve to retrieved sources?
  groundedness         do the quotes exist in the sources they name?
  verification pass    did the run ship with zero unresolved citation defects?
  coverage             does the answer address the facets the question asks about?
  cost and latency     what does one answer cost, and how long does it take?

Defaults to replay, so `make eval` is free and deterministic and CI needs no
secret. Point it at `--mode live` to measure the real thing.

Exit code is non-zero when any metric crosses `thresholds.json`, which is what
turns this from a dashboard into a regression gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scholar_graph.config import LLMMode, Settings, set_settings  # noqa: E402
from scholar_graph.domain import ResearchReport  # noqa: E402
from scholar_graph.observability import configure_logging  # noqa: E402
from scholar_graph.service import ResearchService, build_deps  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


@dataclass
class CaseResult:
    id: str
    question: str
    ok: bool
    citation_precision: float = 0.0
    groundedness: float = 0.0
    verification_passed: bool = False
    covered_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    sources: int = 0
    notes: int = 0
    search_rounds: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def coverage(self) -> float:
        total = len(self.covered_terms) + len(self.missing_terms)
        return 1.0 if total == 0 else len(self.covered_terms) / total


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def score(case: dict[str, Any], report: ResearchReport, seconds: float) -> CaseResult:
    haystack = f"{report.summary}\n{report.body}".lower()
    required = [t.lower() for t in case.get("must_mention", [])]
    covered = [t for t in required if t in haystack]
    missing = [t for t in required if t not in haystack]

    return CaseResult(
        id=case["id"],
        question=case["question"],
        ok=True,
        citation_precision=report.verification.citation_precision,
        groundedness=report.verification.groundedness,
        verification_passed=report.verification.passed,
        covered_terms=covered,
        missing_terms=missing,
        sources=len(report.sources),
        notes=report.verification.total_notes,
        search_rounds=report.search_rounds,
        usd=report.cost.usd,
        seconds=round(seconds, 2),
        warnings=report.warnings,
    )


async def run_case(case: dict[str, Any], settings: Settings) -> CaseResult:
    started = time.perf_counter()
    try:
        async with build_deps(settings) as deps:
            service = ResearchService(settings)
            result = await service.run(case["question"], deps=deps)
            await service.aclose()
    except Exception as exc:  # noqa: BLE001 - one bad case must not abort the sweep
        return CaseResult(
            id=case["id"],
            question=case["question"],
            ok=False,
            seconds=round(time.perf_counter() - started, 2),
            error=f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(result, ResearchReport):
        return CaseResult(
            id=case["id"],
            question=case["question"],
            ok=False,
            error="Run suspended for human approval; evals must be unattended.",
        )

    return score(case, result, time.perf_counter() - started)


def aggregate(results: list[CaseResult], cases: list[dict[str, Any]]) -> dict[str, float]:
    completed = [r for r in results if r.ok]
    if not completed:
        return {
            "citation_precision": 0.0,
            "groundedness": 0.0,
            "verification_pass_rate": 0.0,
            "coverage_rate": 0.0,
            "source_rate": 0.0,
            "mean_usd": 0.0,
            "max_usd": 0.0,
            "mean_seconds": 0.0,
            "max_seconds": 0.0,
            "completion_rate": 0.0,
        }

    minimums = {c["id"]: c.get("min_sources", 0) for c in cases}
    return {
        "citation_precision": statistics.mean(r.citation_precision for r in completed),
        "groundedness": statistics.mean(r.groundedness for r in completed),
        "verification_pass_rate": sum(r.verification_passed for r in completed) / len(completed),
        "coverage_rate": statistics.mean(r.coverage for r in completed),
        "source_rate": sum(r.sources >= minimums.get(r.id, 0) for r in completed) / len(completed),
        "mean_usd": statistics.mean(r.usd for r in completed),
        "max_usd": max(r.usd for r in completed),
        "mean_seconds": statistics.mean(r.seconds for r in completed),
        "max_seconds": max(r.seconds for r in completed),
        "completion_rate": len(completed) / len(results),
    }


def check(metrics: dict[str, float], thresholds: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    def at_least(metric: str, key: str) -> None:
        limit = thresholds.get(key)
        if limit is not None and metrics[metric] < limit:
            failures.append(f"{metric} {metrics[metric]:.3f} < required {limit}")

    def at_most(metric: str, key: str) -> None:
        limit = thresholds.get(key)
        if limit is not None and metrics[metric] > limit:
            failures.append(f"{metric} {metrics[metric]:.3f} > allowed {limit}")

    at_least("citation_precision", "min_citation_precision")
    at_least("groundedness", "min_groundedness")
    at_least("verification_pass_rate", "min_verification_pass_rate")
    at_least("coverage_rate", "min_coverage_rate")
    at_least("source_rate", "min_source_rate")
    at_most("max_usd", "max_usd_per_run")
    at_most("max_seconds", "max_seconds_per_run")

    if metrics["completion_rate"] < 1.0:
        failures.append(f"completion_rate {metrics['completion_rate']:.2f} < 1.00")

    return failures


def render(results: list[CaseResult], metrics: dict[str, float], failures: list[str]) -> str:
    lines = [
        "# Evaluation report",
        "",
        "| case | precision | grounded | verified | coverage | sources | $ | s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if not r.ok:
            lines.append(f"| {r.id} | — | — | ERROR | — | — | — | {r.seconds} |")
            continue
        lines.append(
            f"| {r.id} | {r.citation_precision:.0%} | {r.groundedness:.0%} | "
            f"{'yes' if r.verification_passed else 'NO'} | {r.coverage:.0%} | "
            f"{r.sources} | {r.usd:.4f} | {r.seconds} |"
        )

    lines += ["", "## Aggregate", ""]
    lines += [f"- **{k}**: {v:.4f}" for k, v in metrics.items()]

    if failures:
        lines += ["", "## Threshold failures", ""]
        lines += [f"- {f}" for f in failures]
    else:
        lines += ["", "All thresholds met."]

    for r in results:
        if r.error:
            lines += ["", f"### {r.id} error", "", f"```\n{r.error}\n```"]
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the scholar-graph evaluation suite.")
    parser.add_argument("--mode", default="replay", choices=[m.value for m in LLMMode])
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--thresholds", type=Path, default=EVAL_DIR / "thresholds.json")
    parser.add_argument("--out", type=Path, default=None, help="Write the markdown report here.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings(
        llm_mode=LLMMode(args.mode),
        checkpoint_db=None,
        require_approval_over_usd=1_000.0,  # unattended: never pause for a human
        log_level="WARNING",
    )
    set_settings(settings)
    configure_logging(settings.log_level, settings.log_format)

    cases = load_cases(args.dataset)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))

    results = [await run_case(case, settings) for case in cases]
    metrics = aggregate(results, cases)
    failures = check(metrics, thresholds)
    report = render(results, metrics, failures)

    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {"metrics": metrics, "failures": failures, "cases": [asdict(r) for r in results]},
                indent=2,
            ),
            encoding="utf-8",
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
