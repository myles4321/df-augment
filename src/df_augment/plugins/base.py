"""Plug-in Protocol definition for df-augment techniques."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import pandas as pd

# A single row's feature values (no aug_* keys allowed).
Row = dict


class Technique(Protocol):
    """Interface every augmentation plug-in must satisfy."""

    name: str
    modality: Literal["tabular", "image", "any"]
    parents_per_row: Literal[1, 2]
    sizing: Literal["multiplier", "balance"]
    needs_fit: bool  # True = fits on all rows -> refused in v1

    def generate(
        self,
        df: pd.DataFrame,
        n_per_class: dict[int | str, int],
        rng: np.random.Generator,
        pairs: list[tuple[str, str]] | None = None,
        target_col: str = "target",
    ) -> tuple[list[Row], list[list[str]], list[dict]]:
        """Generate synthetic rows.

        Parameters
        ----------
        df:
            The original DataFrame. Already contains ``aug_row_id`` on every
            row (set by the tracker). Plug-ins read ``aug_row_id`` to report
            parent IDs only; they must never write ``aug_*`` output columns.
        n_per_class:
            How many new rows to produce per class label.
        rng:
            Seeded NumPy Generator for this technique.
        pairs:
            Pre-computed group pairs for two-parent techniques (SMOTE,
            MixUp). ``None`` for single-parent techniques.
        target_col:
            Name of the target column. Plug-ins use this to filter rows by
            class instead of hardcoding ``"target"``.

        Returns
        -------
        new_rows:
            List of feature-value dicts (no ``aug_*`` keys).
        parent_ids_per_row:
            For each new row, the list of ``aug_row_id`` strings of its
            parent(s).
        params_per_row:
            For each new row, a dict of technique-specific parameters that
            will be stored in ``aug_params`` as JSON.
        """
        ...
