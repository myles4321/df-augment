"""Feature-masking plug-in: replaces a random fraction of numeric cells with the column median."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from df_augment.plugins.base import Row
from df_augment.registry import register


@register
class FeatureMaskingTabular:
    """Randomly replace a fraction of numeric feature values with the column median.

    * For each numeric, non-target column, each cell is independently masked
      with probability ``mask_frac``.  Masked cells receive the column median
      computed over the full source DataFrame.
    * Non-numeric columns (object, category, bool, etc.) are copied verbatim.
    * The target column, ``_group_id``, and all ``aug_*`` columns are never
      modified.
    * If a column's median is NaN (all-NaN column) the cell is left unchanged
      rather than being set to NaN.

    Parameters
    ----------
    mask_frac:
        Probability that any individual numeric cell is replaced by the column
        median.  Default 0.15 (15 %).
    """

    name = "feature_masking"
    modality = "tabular"
    parents_per_row = 1
    sizing = "multiplier"
    needs_fit = False

    def __init__(self, mask_frac: float = 0.15) -> None:
        self.mask_frac = mask_frac

    def generate(
        self,
        df: pd.DataFrame,
        n_per_class: dict[int | str, int],
        rng: np.random.Generator,
        pairs: list[tuple[str, str]] | None = None,
        target_col: str = "target",
    ) -> tuple[list[Row], list[list[str]], list[dict[str, Any]]]:
        """Sample rows per class and mask a random fraction of numeric cells.

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
            Name of the target/label column — never masked.
        """
        feature_cols = [
            c for c in df.columns if not str(c).startswith("aug_") and c != "_group_id"
        ]

        numeric_to_mask = [
            c
            for c in feature_cols
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Pre-compute medians once over the full source DataFrame.
        col_medians: dict[str, float] = {
            c: float(df[c].median()) for c in numeric_to_mask
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

                for c in numeric_to_mask:
                    if rng.random() < self.mask_frac:
                        median_val = col_medians[c]
                        if not np.isnan(median_val):
                            # Preserve the original dtype for integer columns.
                            if pd.api.types.is_integer_dtype(df[c]):
                                features[c] = df[c].dtype.type(round(median_val))
                            else:
                                features[c] = median_val

                new_rows.append(features)
                parent_ids_per_row.append([aug_row_id])
                params_per_row.append({"mask_frac": self.mask_frac})

        return new_rows, parent_ids_per_row, params_per_row
