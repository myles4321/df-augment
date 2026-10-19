"""Command-line interface for df-augment."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="df-augment",
    help=(
        "Turn one parquet dataset into a larger, augmented parquet dataset.\n\n"
        "Every synthetic row records which original sample(s) it came from via\n"
        "the aug_source_group column, preventing augmented data from leaking\n"
        "into test sets when used with df-analyze or DeepTune."
    ),
    add_completion=False,
)


@app.command()
def augment(
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to the input dataset (.parquet, .csv, or .xlsx).",
        exists=True,
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Path for the augmented output parquet file.",
    ),
    techniques: list[str] = typer.Option(
        ...,
        "--techniques",
        "-t",
        help="Augmentation technique(s) to apply, e.g. --techniques gauss jitter.",
    ),
    multiplier: Optional[float] = typer.Option(
        None,
        "--multiplier",
        "-m",
        help="Each class gets round((m-1) * n) new rows. Cannot be used with --balance.",
        min=1.0,
    ),
    balance: bool = typer.Option(
        False,
        "--balance",
        help="Balance classes to the size of the largest class.",
    ),
    target: str = typer.Option(
        "target",
        "--target",
        "-y",
        help="Name of the target column.",
    ),
    group_col: Optional[str] = typer.Option(
        None,
        "--group-col",
        help="Column to use as the source group (e.g. patient_id). "
        "If not supplied, one group per row is assigned (S001, S002, ...).",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility.",
    ),
) -> None:
    """Augment a dataset and write a lineage-tracked parquet file."""
    typer.echo(f"df-augment — input: {data}")
    typer.echo(f"  techniques : {', '.join(techniques)}")
    typer.echo(f"  multiplier : {multiplier}")
    typer.echo(f"  balance    : {balance}")
    typer.echo(f"  target     : {target}")
    typer.echo(f"  group_col  : {group_col}")
    typer.echo(f"  seed       : {seed}")
    typer.echo(f"  output     : {out}")
    typer.echo("(Not yet implemented — Phase 1a)")


def main() -> None:
    app()
