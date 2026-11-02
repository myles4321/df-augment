"""Random-oversampling plug-in: copies existing rows without modification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from df_augment.plugins.base import Row
from df_augment.registry import register


@register
class RandomOversampleTabular:
    """Oversample rows by random sampling with replacement.

    Each synthetic row is an exact copy of a randomly chosen original row
    within the same class. No feature values are modified.

    * All columns (numeric, categorical, target) are copied verbatim.
    * The target column, ``_group_id``, and all ``aug_*`` columns are never
      included in the output feature dict.

    Parameters
    ----------
    (none — this technique has no tunable parameters)
    """

    name = "random_oversample"
    modality = "tabular"
    parents_per_row = 1
    sizing = "multiplier"
    needs_fit = False

    def generate(
        self,
        df: pd.DataFrame,
        n_per_class: dict[int | str, int],
        rng: np.random.Generator,
        pairs: list[tuple[str, str]] | None = None,
        target_col: str = "target",
    ) -> tuple[list[Row], list[list[str]], list[dict[str, Any]]]:
        """Sample rows per class and copy them verbatim.

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
            Name of the target/label column — included verbatim in output.
        """
        feature_cols = [
            c for c in df.columns if not str(c).startswith("aug_") and c != "_group_id"
        ]

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

                new_rows.append(features)
                parent_ids_per_row.append([aug_row_id])
                params_per_row.append({})

        return new_rows, parent_ids_per_row, params_per_row
