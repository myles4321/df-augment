"""Gaussian-noise plug-in: perturbs continuous features to create synthetic rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from df_augment.plugins.base import Row
from df_augment.registry import register


@register
class GaussianNoiseTabular:
    """Add Gaussian noise to numeric columns to create synthetic rows.

    * Float columns receive noise drawn from N(0, (sigma_frac * σ_col)²).
    * Integer columns receive the same noise, then are rounded and cast back
      to the original dtype.
    * Non-numeric columns (object, category, bool, etc.) are copied verbatim.
    * The target column, ``_group_id``, and all ``aug_*`` columns are never
      modified.

    If a column has zero (or NaN) standard deviation the noise term is
    skipped and the value is copied verbatim — avoids ``rng.normal(0, 0)``
    raising ``ValueError``.

    Parameters
    ----------
    sigma_frac:
        Fraction of each column's standard deviation used as the noise σ.
        Default 0.05 (5 %).
    """

    name = "gaussian_noise"
    modality = "tabular"
    parents_per_row = 1
    sizing = "multiplier"
    needs_fit = False

    def __init__(self, sigma_frac: float = 0.05) -> None:
        self.sigma_frac = sigma_frac

    def generate(
        self,
        df: pd.DataFrame,
        n_per_class: dict[int | str, int],
        rng: np.random.Generator,
        pairs: list[tuple[str, str]] | None = None,
        target_col: str = "target",
    ) -> tuple[list[Row], list[list[str]], list[dict[str, Any]]]:
        """Sample rows per class and perturb numeric features with Gaussian noise.

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
            Name of the target/label column — never perturbed.
        """
        # All output columns: exclude aug_* metadata and the _group_id helper.
        feature_cols = [
            c for c in df.columns if not str(c).startswith("aug_") and c != "_group_id"
        ]

        # Columns eligible for noise: numeric only, never the target.
        numeric_to_perturb = [
            c
            for c in feature_cols
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        float_cols = [
            c for c in numeric_to_perturb if pd.api.types.is_float_dtype(df[c])
        ]
        int_cols = [
            c for c in numeric_to_perturb if pd.api.types.is_integer_dtype(df[c])
        ]

        # Pre-compute per-column noise std (sigma_frac * column std).
        # Store as float; NaN/zero std means no noise for that column.
        noise_stds: dict[str, float] = {
            c: self.sigma_frac * float(df[c].std()) for c in float_cols + int_cols
        }

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

                # Perturb float columns.
                for c in float_cols:
                    noise_std = noise_stds[c]
                    if noise_std > 0:
                        features[c] = float(src_row[c]) + rng.normal(0.0, noise_std)

                # Perturb integer columns: add noise, round, cast back.
                for c in int_cols:
                    noise_std = noise_stds[c]
                    if noise_std > 0:
                        val = float(src_row[c]) + rng.normal(0.0, noise_std)
                        features[c] = df[c].dtype.type(round(val))

                new_rows.append(features)
                parent_ids_per_row.append([aug_row_id])
                params_per_row.append({"sigma_frac": self.sigma_frac})

        return new_rows, parent_ids_per_row, params_per_row
