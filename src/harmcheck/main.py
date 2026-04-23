import asyncio
from pathlib import Path
from typing import Annotated

import typer

from . import analyze as analyze_mod
from .config import settings
from .datasets import load_prompts
from .judge import default_dst_path
from .judge import judge as judge_async
from .runner import default_output_path
from .runner import run as run_async

app = typer.Typer(
    name="harmcheck",
    help="Compare PT-BR harm-prompt responses across local LLMs.",
    no_args_is_help=True,
)


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="Path to a DADA dataset JSON")],
    language: Annotated[str, typer.Option(help="Language key: 'pt-br', 'en', ...")] = settings.default_language,
    limit: Annotated[int | None, typer.Option(help="Take first N prompts (after sampling)")] = None,
    sample_per_category: Annotated[
        int | None,
        typer.Option("--sample-per-category", "-s", help="Stratified sample N per category"),
    ] = None,
    min_quality_score: Annotated[
        float | None,
        typer.Option(
            "--min-quality-score",
            help="Drop rows whose MetricX translation score (DADA 02-evaluated sidecar) is below this",
        ),
    ] = None,
    quality_file: Annotated[
        Path | None,
        typer.Option("--quality-file", help="Explicit path to the 02-evaluated sidecar (auto-discovered otherwise)"),
    ] = None,
    models: Annotated[list[str] | None, typer.Option("--model", "-m", help="Models (repeat)")] = None,
    output: Annotated[Path | None, typer.Option(help="Output JSONL path")] = None,
    num_parallel: Annotated[int, typer.Option(help="Concurrent requests per model")] = settings.num_parallel,
    resume: Annotated[bool, typer.Option(help="Skip rows already in the output file")] = True,
    seed: Annotated[int, typer.Option(help="Seed for stratified sampling")] = 42,
) -> None:
    """Stream responses from each model into a JSONL file, resumable on restart."""
    selected = models or settings.models
    prompts = load_prompts(
        dataset,
        language=language,
        limit=limit,
        sample_per_category=sample_per_category,
        seed=seed,
        min_quality_score=min_quality_score,
        quality_file=quality_file,
    )
    if not prompts:
        typer.echo("No prompts after filtering. Check --language / dataset shape.", err=True)
        raise typer.Exit(1)

    out_path = output or default_output_path(dataset, settings.output_dir)

    typer.echo(f"dataset : {dataset}")
    typer.echo(f"prompts : {len(prompts)}  (language={language})")
    typer.echo(f"models  : {', '.join(selected)}")
    typer.echo(f"output  : {out_path}  (resume={resume}, parallel={num_parallel})")
    typer.echo("")

    asyncio.run(run_async(prompts, selected, out_path, num_parallel, resume))
    typer.echo("")
    typer.echo(f"Done. Run: harmcheckanalyze {out_path}")


@app.command()
def analyze(
    path: Annotated[Path, typer.Argument(help="A JSONL file produced by `harmcheckrun` or `harmcheckjudge`")],
) -> None:
    """Print refusal / error / length stats and cross-model divergence.
    Uses the LLM-as-judge label (`judge_label`) if the file has one,
    otherwise falls back to the `refused` heuristic."""
    analyze_mod.summarize(path)


@app.command()
def judge(
    src: Annotated[Path, typer.Argument(help="JSONL produced by `harmcheckrun`")],
    judge_model: Annotated[
        str, typer.Option("--judge-model", help="Model used as classifier")
    ] = settings.judge_model,
    output: Annotated[Path | None, typer.Option(help="Output JSONL (default: <src>.judged.jsonl)")] = None,
    num_parallel: Annotated[int, typer.Option(help="Concurrent judge requests")] = settings.num_parallel,
    resume: Annotated[bool, typer.Option(help="Skip rows already judged in output file")] = True,
) -> None:
    """Re-classify responses with an LLM-as-judge (REFUSAL / UNSAFE / SAFE / OTHER).
    Operates on an existing JSONL — does not re-run the target models."""
    dst = output or default_dst_path(src)
    typer.echo(f"source  : {src}")
    typer.echo(f"judge   : {judge_model}")
    typer.echo(f"output  : {dst}  (resume={resume}, parallel={num_parallel})")
    typer.echo("")
    asyncio.run(judge_async(src, dst, judge_model, num_parallel, resume))
    typer.echo("")
    typer.echo(f"Done. Run: harmcheckanalyze {dst}")


@app.command()
def divergence(
    path: Annotated[Path, typer.Argument(help="A JSONL file produced by `harmcheckrun`")],
    limit: Annotated[int, typer.Option(help="Max examples to show")] = 20,
) -> None:
    """Show prompts where models disagree on whether to refuse."""
    analyze_mod.disagreements(path, limit=limit)


@app.command()
def datasets() -> None:
    """List available DADA dataset files."""
    found = False
    for d in (settings.datasets_dir, settings.translated_dir):
        if d.exists():
            typer.echo(f"{d}:")
            for f in sorted(d.glob("*.json")):
                typer.echo(f"  {f.name}")
            found = True
    if not found:
        typer.echo("No dataset directories found. Check config.")
        raise typer.Exit(1)


@app.command()
def models() -> None:
    """List configured models."""
    for m in settings.models:
        typer.echo(f"  {m}")


if __name__ == "__main__":
    app()
