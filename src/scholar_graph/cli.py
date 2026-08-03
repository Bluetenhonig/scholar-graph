"""Command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from scholar_graph.config import LLMMode, Settings, get_settings, set_settings
from scholar_graph.domain import ResearchReport
from scholar_graph.observability import configure_logging
from scholar_graph.service import PendingApproval, ResearchService

app = typer.Typer(
    add_completion=False,
    help="scholar-graph — an evidence-grounded research agent.",
    no_args_is_help=True,
)
console = Console()


def _apply_overrides(mode: str | None, budget: float | None, panel: bool | None) -> Settings:
    settings = get_settings()
    updates: dict[str, object] = {}
    if mode is not None:
        updates["llm_mode"] = LLMMode(mode)
    if budget is not None:
        updates["max_usd_per_run"] = budget
    if panel is not None:
        updates["enable_review_panel"] = panel
    if updates:
        settings = settings.model_copy(update=updates)
        set_settings(settings)
    configure_logging(settings.log_level, settings.log_format)
    return settings


def _render(report: ResearchReport, as_json: bool, output: Path | None) -> None:
    if as_json:
        payload = report.model_dump(mode="json")
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        text = report.to_markdown()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote {output}[/green]")
        return

    if as_json:
        console.print_json(text)
    else:
        console.print(Markdown(text))


def _render_pending(pending: PendingApproval) -> None:
    payload = pending.payload
    console.print("[yellow]This run needs approval before it spends money.[/yellow]")
    table = Table(show_header=False, box=None)
    table.add_row("run id", pending.run_id)
    table.add_row("projected", f"${payload.get('projected_usd', 0):.4f}")
    table.add_row("threshold", f"${payload.get('threshold_usd', 0):.4f}")
    table.add_row("budget cap", f"${payload.get('budget_usd', 0):.4f}")
    console.print(table)
    console.print(
        f"\nApprove with:  [bold]scholar-graph resume {pending.run_id} --decision approve[/bold]"
    )


@app.command()
def research(
    question: Annotated[str, typer.Argument(help="The research question.")],
    mode: Annotated[str | None, typer.Option("--mode", help="replay | record | live.")] = None,
    budget: Annotated[float | None, typer.Option("--budget", help="Max USD for this run.")] = None,
    panel: Annotated[
        bool | None, typer.Option("--panel/--no-panel", help="Run the AutoGen review board.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of markdown.")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the report to a file.")
    ] = None,
) -> None:
    """Research a question and print a cited report."""
    settings = _apply_overrides(mode, budget, panel)
    service = ResearchService(settings)

    result = asyncio.run(service.run(question))
    if isinstance(result, PendingApproval):
        _render_pending(result)
        raise typer.Exit(code=2)

    _render(result, as_json, output)
    if not result.verification.passed:
        raise typer.Exit(code=1)


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Run id from the approval prompt.")],
    decision: Annotated[str, typer.Option("--decision", help="approve | reject.")] = "approve",
    mode: Annotated[str | None, typer.Option("--mode")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Resume a run that paused for cost approval."""
    settings = _apply_overrides(mode, None, None)
    service = ResearchService(settings)

    result = asyncio.run(service.resume(run_id, decision))
    if isinstance(result, PendingApproval):
        _render_pending(result)
        raise typer.Exit(code=2)

    _render(result, as_json, output)


@app.command()
def cassettes() -> None:
    """Show what has been recorded, and therefore what can be replayed."""
    from scholar_graph.llm.cassette import CassetteStore

    settings = get_settings()
    table = Table(title="Recorded cassettes")
    table.add_column("namespace")
    table.add_column("count", justify="right")
    for namespace in ("llm", "http"):
        store = CassetteStore(settings.cassette_dir, namespace)
        table.add_row(namespace, str(store.count()))
    console.print(table)
    console.print(f"[dim]{settings.cassette_dir}[/dim]")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("scholar_graph.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()
