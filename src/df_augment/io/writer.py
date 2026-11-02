"""Write the augmented DataFrame plus manifest and groups report."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd


def write(
    df: pd.DataFrame,
    out: Path,
    input_sha256: str,
    config_dict: dict[str, Any],
) -> None:
    """Persist the augmented dataset and supporting metadata files.

    Writes three files next to *out*:

    * ``<out>`` -- the augmented parquet file.
    * ``<out>.manifest.json`` -- run metadata.
    * ``<out>.groups_report.csv`` -- row count per source group.

    Parameters
    ----------
    df:
        The fully assembled output DataFrame (originals + synthetic rows).
        Must contain ``aug_is_original``, ``aug_source_group``, and
        ``aug_method`` columns.
    out:
        Destination path for the parquet file.
    input_sha256:
        SHA-256 hex digest of the input file.
    config_dict:
        Serialisable config snapshot (e.g. from ``AugmentConfig.model_dump()``
        with paths converted to strings).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # -- Parquet ----------------------------------------------------------------
    df.to_parquet(out, engine="pyarrow", row_group_size=1000, index=False)

    # -- Manifest ---------------------------------------------------------------
    pkg_version = version("df-augment")
    rows_original = int(df["aug_is_original"].sum())
    rows_synthetic = int((~df["aug_is_original"]).sum())
    method_counts: dict[str, int] = (
        df.groupby("aug_method")["aug_method"].count().to_dict()
    )

    # Convert Path objects to strings so JSON serialisation works.
    serialisable_config: dict[str, Any] = {
        k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items()
    }

    manifest = {
        "df_augment_version": pkg_version,
        "input_sha256": input_sha256,
        "rows_original": rows_original,
        "rows_synthetic": rows_synthetic,
        "methods": method_counts,
        "config": serialisable_config,
    }
    manifest_path = out.with_suffix("").with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # -- Groups report ----------------------------------------------------------
    groups_report = df.groupby("aug_source_group").size().reset_index(name="count")
    groups_path = out.with_suffix("").with_suffix(".groups_report.csv")
    groups_report.to_csv(groups_path, index=False)
