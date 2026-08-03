"""Graph nodes, grouped by the stage of the pipeline they belong to."""

from scholar_graph.nodes.panel import convene_panel
from scholar_graph.nodes.planning import approval_gate, assess_coverage, plan
from scholar_graph.nodes.retrieval import extract, screen, search
from scholar_graph.nodes.writing import check_citations, revise, synthesize

__all__ = [
    "approval_gate",
    "assess_coverage",
    "check_citations",
    "convene_panel",
    "extract",
    "plan",
    "revise",
    "screen",
    "search",
    "synthesize",
]
