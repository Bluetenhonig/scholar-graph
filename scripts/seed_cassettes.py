"""Generate the synthetic cassettes that let this repo run offline.

WHAT THIS IS: a deterministic stand-in for the network. It runs the real
graph, the real provider, the real cassette keying and the real cost
accounting — and fakes only the two HTTP boundaries (Anthropic, and the
search APIs). The cassettes it writes are genuine cache entries for genuine
requests; only their *content* is synthesised.

WHAT THIS IS NOT: a recording of real model output. Nothing here tells you
how good Claude is at this task. For that, record real cassettes:

    SCHOLAR_GRAPH_LLM_MODE=record \
    SCHOLAR_GRAPH_ANTHROPIC_API_KEY=sk-... \
    scholar-graph research "your question"

Why bother: a reviewer can clone this repo and get a full, verified,
end-to-end run in seconds with no API key and no spend — and CI can assert on
citation precision, groundedness and cost without a secret or a bill.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scholar_graph.config import LLMMode, Settings, set_settings  # noqa: E402
from scholar_graph.domain import Document, ResearchReport  # noqa: E402
from scholar_graph.llm.budget import BudgetTracker  # noqa: E402
from scholar_graph.llm.pricing import estimate_cost  # noqa: E402
from scholar_graph.llm.provider import LLMProvider, LLMResponse  # noqa: E402
from scholar_graph.observability import configure_logging  # noqa: E402
from scholar_graph.service import ResearchService  # noqa: E402
from scholar_graph.state import Deps  # noqa: E402
from scholar_graph.tools import SearchService  # noqa: E402
from scholar_graph.tools.http import HttpFetcher  # noqa: E402

# --------------------------------------------------------------------------
# A small synthetic literature on LLM inference efficiency
# --------------------------------------------------------------------------

CORPUS: list[dict[str, Any]] = [
    {
        "id": "W2001",
        "provider": "openalex",
        "title": "Sparse attention patterns for efficient long-context inference",
        "abstract": (
            "We evaluate block-sparse and sliding-window attention across sequence lengths "
            "from 4k to 128k tokens. Block-sparse attention reduces inference latency by 38 "
            "percent at 32k context while accuracy on long-context benchmarks falls by less "
            "than one point. The saving grows with sequence length because attention cost is "
            "quadratic in context whereas the sparse variants are close to linear."
        ),
        "authors": ["Ada Lovelace", "Alan Turing"],
        "published": "2024-03-11",
        "venue": "Transactions on Efficient Machine Learning",
        "citations": 214,
    },
    {
        "id": "W2002",
        "provider": "openalex",
        "title": "A systematic study of post-training quantisation for language model serving",
        "abstract": (
            "Across seven open-weight models we find that eight-bit weight quantisation "
            "halves memory footprint with a mean quality loss of 0.4 points, whereas four-bit "
            "quantisation degrades multi-step reasoning benchmarks by 6.1 points on average. "
            "Quantising activations is consistently more damaging than quantising weights."
        ),
        "authors": ["Grace Hopper", "Katherine Johnson"],
        "published": "2024-06-02",
        "venue": "Journal of Machine Learning Systems",
        "citations": 187,
    },
    {
        "id": "2401.10101",
        "provider": "arxiv",
        "title": "Paged key-value caching for high-throughput transformer serving",
        "abstract": (
            "Fragmentation in the key-value cache wastes between 20 and 40 percent of GPU "
            "memory in production serving. Allocating the cache in fixed-size pages raises "
            "achievable batch size by 2.2 times and improves throughput by 1.9 times with no "
            "change to model outputs, since the technique is numerically exact."
        ),
        "authors": ["Barbara Liskov"],
        "published": "2024-01-18",
        "venue": "arXiv",
        "citations": 302,
    },
    {
        "id": "2402.20202",
        "provider": "arxiv",
        "title": "Speculative decoding with lightweight draft models",
        "abstract": (
            "A small draft model proposes tokens that the target model verifies in parallel. "
            "Because verification is exact, the output distribution is unchanged. We measure "
            "wall-clock speedups of 2.1 to 2.8 times on greedy decoding, with the acceptance "
            "rate of drafted tokens being the dominant factor in the realised gain."
        ),
        "authors": ["Edsger Dijkstra", "Frances Allen"],
        "published": "2024-02-29",
        "venue": "arXiv",
        "citations": 265,
    },
    {
        "id": "W2003",
        "provider": "openalex",
        "title": "Mixture-of-experts routing reduces active parameter count",
        "abstract": (
            "Sparse mixture-of-experts layers activate a fraction of parameters per token. "
            "We report a 4.3 times reduction in active parameters at matched quality, though "
            "total memory required to hold all experts increases, which shifts the bottleneck "
            "from compute to memory capacity in single-GPU deployments."
        ),
        "authors": ["John McCarthy"],
        "published": "2023-11-20",
        "venue": "Conference on Neural Systems",
        "citations": 158,
    },
    {
        "id": "W2004",
        "provider": "openalex",
        "title": "Continuous batching for large language model inference servers",
        "abstract": (
            "Static batching leaves accelerators idle while short requests wait for long ones. "
            "Continuous batching admits new requests at each decoding step and raises measured "
            "throughput by up to 23 times at the same latency target under bursty traffic."
        ),
        "authors": ["Leslie Lamport"],
        "published": "2023-09-05",
        "venue": "Symposium on Operating Systems",
        "citations": 401,
    },
    {
        "id": "W2005",
        "provider": "openalex",
        "title": "Nitrogen mineralisation in temperate grassland soils",
        "abstract": (
            "Nitrogen mineralisation rates were measured monthly across three grassland sites "
            "over two growing seasons. Rates correlated with soil temperature and moisture."
        ),
        "authors": ["Carl Linnaeus"],
        "published": "2021-04-04",
        "venue": "Ecology Letters",
        "citations": 22,
    },
]

DOCUMENTS: dict[str, Document] = {}
for entry in CORPUS:
    doc = Document(
        source_id=Document.make_source_id(entry["provider"], entry["id"]),
        provider=entry["provider"],
        external_id=entry["id"],
        title=entry["title"],
        abstract=entry["abstract"],
        authors=tuple(entry["authors"]),
        published=date.fromisoformat(entry["published"]),
        venue=entry["venue"],
        url=f"https://example.org/{entry['id']}",
        citation_count=entry["citations"],
    )
    DOCUMENTS[doc.source_id] = doc


DEMO_QUESTIONS = [
    "What techniques reduce large language model inference cost, and what do they trade away?",
    "How much can quantisation and sparse attention reduce serving cost without hurting quality?",
]


# --------------------------------------------------------------------------
# Synthetic model
# --------------------------------------------------------------------------


def _first_sentence(text: str) -> str:
    return text.split(". ")[0].strip().rstrip(".")


def _quote_from(abstract: str) -> str:
    """A verbatim span, so the citation verifier genuinely passes."""
    words = abstract.split()
    return " ".join(words[:18])


def _synthesise(purpose: str, user: str) -> str:
    if purpose == "plan":
        return json.dumps(
            {
                "interpretation": (
                    "Which engineering techniques lower the cost of serving large language "
                    "models, and what quality or resource cost does each carry?"
                ),
                "sub_questions": [
                    "Attention and architecture-level techniques",
                    "Numerical precision techniques",
                    "Serving and scheduling techniques",
                ],
                "search_queries": [
                    "sparse attention inference latency long context",
                    "post-training quantisation language model quality",
                    "key value cache paging continuous batching throughput",
                ],
                "success_criteria": [
                    "Names at least three distinct cost-reduction techniques",
                    "States the quality or resource trade-off for each technique",
                    "Distinguishes techniques that change model outputs from those that do not",
                ],
            }
        )

    source_ids = re.findall(r"source_id: (\S+)", user)

    if purpose == "screen":
        return json.dumps(
            {
                "verdicts": [
                    {
                        "source_id": sid,
                        "relevant": "grassland" not in DOCUMENTS[sid].title.lower(),
                        "reason": (
                            "Soil science, unrelated to model serving."
                            if "grassland" in DOCUMENTS[sid].title.lower()
                            else "Reports a concrete cost-reduction result."
                        ),
                    }
                    for sid in source_ids
                    if sid in DOCUMENTS
                ]
            }
        )

    if purpose == "extract":
        notes = []
        for sid in source_ids:
            doc = DOCUMENTS.get(sid)
            if doc is None:
                continue
            notes.append(
                {
                    "source_id": sid,
                    "claim": _first_sentence(doc.abstract) + ".",
                    "quote": _quote_from(doc.abstract),
                }
            )
        return json.dumps({"notes": notes})

    if purpose == "coverage":
        return json.dumps(
            {
                "satisfied_criteria": [
                    "Names at least three distinct cost-reduction techniques",
                    "States the quality or resource trade-off for each technique",
                    "Distinguishes techniques that change model outputs from those that do not",
                ],
                "unmet_criteria": [],
                "follow_up_queries": [],
            }
        )

    if purpose in {"synthesize", "revise"}:
        findings = re.findall(r"\[S(\d+)\] (.+)", user)
        seen: dict[str, str] = {}
        for marker, claim in findings:
            seen.setdefault(marker, claim.strip())
        markers = sorted(seen, key=int)[:6]

        paragraphs = []
        for marker in markers:
            paragraphs.append(
                f"{seen[marker]} This is one of the levers available to a serving team, "
                f"and its cost profile differs from the others listed here [S{marker}]."
            )
        citation_run = "".join(f"[S{m}]" for m in markers[:3])
        return json.dumps(
            {
                "summary": (
                    "Serving cost can be reduced along three independent axes: attention "
                    "sparsity, numerical precision, and scheduling of the key-value cache "
                    f"and request batches {citation_run}."
                ),
                "body": "\n\n".join(paragraphs),
                "limitations": (
                    "These results come from published benchmarks rather than one common "
                    "harness, so the reported speedups are not directly comparable to each "
                    "other. None of the sources reports cost under sustained production load."
                ),
            }
        )

    if purpose.startswith("panel:"):
        if purpose.endswith("adjudicator"):
            return (
                "accept - every claim resolves to a retrieved source and the trade-offs are "
                "stated explicitly."
            )
        if purpose.endswith("methodologist"):
            return (
                "The draft correctly separates exact techniques from lossy ones. The speedup "
                "figures are quoted from single-paper benchmarks; the limitations section "
                "already says so. No blocking concerns."
            )
        return (
            "Structure is organised by technique rather than by paper, which is the right "
            "shape for the question. The summary answers it directly. No blocking concerns."
        )

    return "(no synthetic response defined for this purpose)"


class SyntheticProvider(LLMProvider):
    """Fakes only the HTTP call; keying, costing and budgeting run for real."""

    async def _call_api(self, request: dict[str, Any], purpose: str) -> LLMResponse:
        user = request["messages"][0]["content"]
        text = _synthesise(purpose, user)
        # Plausible token counts so replayed cost numbers stay meaningful.
        return LLMResponse(
            text=text,
            cost=estimate_cost(
                request["model"],
                input_tokens=max(200, len(request["system"] + user) // 4),
                output_tokens=max(80, len(text) // 4),
            ),
            stop_reason="end_turn",
        )


class SyntheticFetcher(HttpFetcher):
    """Serves synthetic OpenAlex JSON and arXiv Atom from the corpus above."""

    async def _fetch(self, url: str, params: dict[str, Any]) -> str:
        if "openalex" in url:
            return json.dumps(
                {
                    "results": [
                        {
                            "id": f"https://openalex.org/{e['id']}",
                            "display_name": e["title"],
                            "abstract_inverted_index": _invert(e["abstract"]),
                            "authorships": [{"author": {"display_name": a}} for a in e["authors"]],
                            "publication_date": e["published"],
                            "primary_location": {"source": {"display_name": e["venue"]}},
                            "cited_by_count": e["citations"],
                            "doi": f"https://example.org/{e['id']}",
                        }
                        for e in CORPUS
                        if e["provider"] == "openalex"
                    ]
                }
            )

        entries = "".join(
            f"""<entry>
    <id>https://arxiv.org/abs/{e["id"]}</id>
    <title>{e["title"]}</title>
    <summary>{e["abstract"]}</summary>
    <published>{e["published"]}T00:00:00Z</published>
    {"".join(f"<author><name>{a}</name></author>" for a in e["authors"])}
  </entry>"""
            for e in CORPUS
            if e["provider"] == "arxiv"
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'
        )


def _invert(abstract: str) -> dict[str, list[int]]:
    inverted: dict[str, list[int]] = {}
    for position, word in enumerate(abstract.split()):
        inverted.setdefault(word, []).append(position)
    return inverted


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


async def main() -> int:
    # Seed with the *default* settings a user's first run will use. Anything
    # that differs (document caps, round limits, model choice) changes the
    # request bodies and therefore the cassette keys, and the recording would
    # silently fail to match on replay.
    settings = Settings(
        llm_mode=LLMMode.record,
        anthropic_api_key="synthetic-not-a-real-key",
        checkpoint_db=None,
        log_level="WARNING",
    )
    set_settings(settings)
    configure_logging(settings.log_level, settings.log_format)

    for question in DEMO_QUESTIONS:
        budget = BudgetTracker(settings.max_usd_per_run)
        search = SearchService(settings, fetcher=SyntheticFetcher(settings))
        deps = Deps(
            settings=settings,
            provider=SyntheticProvider(settings, budget),
            search=search,
            budget=budget,
        )
        service = ResearchService(settings)
        report = await service.run(question, deps=deps)
        await service.aclose()

        # Guard against seeding a demo that only looks healthy. An empty
        # source list means retrieval never happened, and a report with no
        # citations passes verification vacuously.
        if not isinstance(report, ResearchReport):
            raise SystemExit(f"Seeding paused unexpectedly for: {question}")
        if not report.sources:
            raise SystemExit(f"Seeding produced no sources for: {question}")
        if not report.verification.passed:
            raise SystemExit(f"Seeded run failed verification: {report.verification.issues}")

        print(  # noqa: T201
            f"seeded: {question[:58]}… "
            f"sources={len(report.sources)} "
            f"notes={report.verification.total_notes} "
            f"precision={report.verification.citation_precision:.0%} "
            f"critiques={len(report.critiques)} "
            f"cost=${report.cost.usd:.4f}"
        )

    print(f"\nCassettes written to {settings.cassette_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
