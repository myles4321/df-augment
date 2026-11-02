"""Identity plug-in: copies rows verbatim to validate the lineage core."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from df_augment.plugins.base import Row
from df_augment.registry import register


@register
class IdentityTabular:
    """Copy rows from the original dataset without any transformation.

    This plug-in exists solely to exercise the lineage pipeline. If
    lineage tracking is correct for verbatim copies it will be correct for
    any other technique built on top.
    """

    name = "identity"
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
        """For each class, sample *n_per_class[cls]* rows with replacement.

        The target column and all ``aug_*`` columns are excluded from the
        returned feature dicts.  Only ``aug_row_id`` is read from *df* to
        report parent IDs -- it is never included in the output rows.
        """
        # Columns to copy: exclude aug_* and the internal _group_id helper.
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
                params_per_row.append({"copied_from": aug_row_id})

        return new_rows, parent_ids_per_row, params_per_row
