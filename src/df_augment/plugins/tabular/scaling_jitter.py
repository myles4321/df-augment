"""Scaling-jitter plug-in: multiplies continuous features by draws from N(1, s²)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from df_augment.plugins.base import Row
from df_augment.registry import register


@register
class ScalingJitterTabular:
    """Multiply numeric columns by a random scale factor drawn from N(1, scale_sd²).

    * Float columns are multiplied by a draw from N(1, scale_sd²) directly.
    * Integer columns receive the same multiplicative jitter, then are rounded
      and cast back to the original dtype.
    * Non-numeric columns (object, category, bool, etc.) are copied verbatim.
    * The target column, ``_group_id``, and all ``aug_*`` columns are never
      modified.

    If a column has a zero mean, the scale factor still applies — the result
    remains zero (0 × anything = 0), so no special-casing is needed.

    Parameters
    ----------
    scale_sd:
        Standard deviation of the multiplicative scale factor drawn from N(1, s²).
        Default 0.05 (5 % scale variation).
    """

    name = "scaling_jitter"
    modality = "tabular"
    parents_per_row = 1
    sizing = "multiplier"
    needs_fit = False

    def __init__(self, scale_sd: float = 0.05) -> None:
        self.scale_sd = scale_sd

    def generate(
        self,
        df: pd.DataFrame,
        n_per_class: dict[int | str, int],
        rng: np.random.Generator,
        pairs: list[tuple[str, str]] | None = None,
        target_col: str = "target",
    ) -> tuple[list[Row], list[list[str]], list[dict[str, Any]]]:
        """Sample rows per class and apply multiplicative scaling jitter.

        Parameters
        ----------
        df:
            Source DataFrame. Must already have ``aug_row_id`` on every row.
        n_per_class:
            How many synthetic rows to produce per class label.
        rng:
            Seeded NumPy Generator for this technique.
        pairs:
            Unused (single-parent technique).
        target_col:
            Name of the target/label column — never scaled.
        """
        feature_cols = [
            c for c in df.columns if not str(c).startswith("aug_") and c != "_group_id"
        ]

        numeric_to_scale = [
            c
            for c in feature_cols
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        float_cols = [c for c in numeric_to_scale if pd.api.types.is_float_dtype(df[c])]
        int_cols = [c for c in numeric_to_scale if pd.api.types.is_integer_dtype(df[c])]

        new_rows: list[Row] = []
        parent_ids_per_row: list[list[str]] = []
        params_per_row: list[dict[str, Any]] = []

        for cls, n_new in n_per_class.items():
            if n_new <= 0:
                continue
            class_df = df[df[target_col] == cls]
            if class_df.empty:
                continue

            indices = rng.integers(0, len(class_df), size=n_new)
            sampled = class_df.iloc[indices]

            for _, src_row in sampled.iterrows():
                aug_row_id: str = str(src_row["aug_row_id"])
                features: Row = {col: src_row[col] for col in feature_cols}

                # Each column gets its own independent scale factor.
                for c in float_cols:
                    scale = rng.normal(1.0, self.scale_sd)
                    features[c] = float(src_row[c]) * scale

                for c in int_cols:
                    scale = rng.normal(1.0, self.scale_sd)
                    val = float(src_row[c]) * scale
                    features[c] = df[c].dtype.type(round(val))

                new_rows.append(features)
                parent_ids_per_row.append([aug_row_id])
                params_per_row.append({"scale_sd": self.scale_sd})

        return new_rows, parent_ids_per_row, params_per_row
