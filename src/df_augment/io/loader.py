"""Load tabular datasets into a pandas DataFrame with group-ID resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SUPPORTED = {".parquet", ".csv", ".xlsx"}


def load(path: Path, group_col: str | None = None) -> pd.DataFrame:
    """Read *path* and return a DataFrame with an internal ``_group_id`` column.

    Parameters
    ----------
    path:
        Input file. Accepted extensions: .parquet, .csv, .xlsx.
    group_col:
        If given and the column exists in the file, its string values become
        the group IDs. Otherwise one group per row is assigned (``S0001``, …).

    Raises
    ------
    ValueError
        For unsupported extensions or files that already contain ``aug_*``
        columns.
    """
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(
            f"Unsupported file extension {suffix!r}. Accepted: {sorted(_SUPPORTED)}."
        )

    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    aug_cols = [c for c in df.columns if str(c).startswith("aug_")]
    if aug_cols:
        raise ValueError(
            f"Input already contains aug_* columns: {aug_cols}. "
            "Pass --reaugment to override."
        )

    if group_col is not None and group_col in df.columns:
        group_ids = df[group_col].astype(str).tolist()
    else:
        width = max(4, len(str(len(df))))
        group_ids = [f"S{i + 1:0{width}d}" for i in range(len(df))]

    df = df.copy()
    df["_group_id"] = group_ids
    return df
