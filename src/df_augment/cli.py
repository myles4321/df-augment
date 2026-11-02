"""Command-line interface for df-augment."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
    multiplier: float | None = typer.Option(
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
    group_col: str | None = typer.Option(
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
    from pydantic import ValidationError

    import df_augment
    from df_augment.config import AugmentConfig
    from df_augment.io import loader, writer

    # Build and validate config.
    try:
        config = AugmentConfig(
            data=data,
            out=out,
            techniques=techniques,
            multiplier=multiplier,
            balance=balance,
            target=target,
            group_col=group_col,
            seed=seed,
        )
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(1)

    # Load input.
    try:
        df = loader.load(config.data, config.group_col)
    except ValueError as exc:
        typer.echo(f"Load error: {exc}", err=True)
        raise typer.Exit(1)

    n_orig = len(df)

    # SHA-256 of the input file.
    sha256 = hashlib.sha256(config.data.read_bytes()).hexdigest()

    # Augment.
    result = df_augment.augment(df, config)

    # Write output.
    writer.write(result, config.out, sha256, config.model_dump())

    n_total = len(result)
    typer.echo(f"+ {n_orig} -> {n_total} rows written to {config.out}")


def main() -> None:
    app()
