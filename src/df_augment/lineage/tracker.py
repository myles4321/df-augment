"""LineageTracker: the only component allowed to create output rows."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd

from df_augment.lineage.groups import GroupResolver


class LineageTracker:
    """Records original and synthetic rows, filling all reserved aug_* columns.

    Rules enforced here
    -------------------
    1. Only this class creates rows -- plug-ins return feature dicts, never
       full rows.
    2. Every synthetic row must reference at least one known parent
       ``aug_row_id``.
    3. ``aug_source_group`` is resolved via the :class:`GroupResolver`, not by
       the plug-in.
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        seed: int,
        method: str,
    ) -> None:
        self._source_df = source_df
        self._seed = seed
        self._method = method
        self._rows: list[dict[str, Any]] = []
        # Maps aug_row_id -> group_id for all recorded originals.
        self._known_ids: dict[str, str] = {}
        # Counter per (method, sorted-parent-tuple) for deterministic IDs.
        self._counters: dict[tuple[str, ...], int] = defaultdict(int)

        # Feature column names (excludes _group_id and any aug_* cols).
        self._feature_cols = [
            c
            for c in source_df.columns
            if not str(c).startswith("aug_") and c != "_group_id"
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_original(self, row_idx: int) -> dict[str, Any]:
        """Record one original row and return the full row dict."""
        row = self._source_df.iloc[row_idx]
        gid: str = str(row["_group_id"])
        # Include row_idx to guarantee uniqueness when multiple rows share
        # the same group ID (e.g. multiple scans of one patient).
        aug_row_id = f"orig:{gid}:{row_idx}"

        record: dict[str, Any] = {col: row[col] for col in self._feature_cols}
        record["aug_row_id"] = aug_row_id
        record["aug_is_original"] = True
        record["aug_parent_ids"] = [aug_row_id]
        record["aug_source_group"] = gid
        record["aug_method"] = "original"
        record["aug_params"] = "{}"
        record["aug_seed"] = self._seed

        self._known_ids[aug_row_id] = gid
        self._rows.append(record)
        return record

    def record_synthetic(
        self,
        features: dict[str, Any],
        parent_ids: list[str],
        params: dict[str, Any],
        child_seed: int,
        resolver: GroupResolver,
    ) -> dict[str, Any]:
        """Record one synthetic row and return the full row dict.

        Parameters
        ----------
        features:
            Feature key→value pairs. Must not contain ``aug_*`` keys.
        parent_ids:
            ``aug_row_id`` values of the parent rows (1 or 2 entries).
        params:
            Plug-in-reported parameters for this row (stored as JSON).
        child_seed:
            Per-row seed from the technique's RNG child.
        resolver:
            The active :class:`GroupResolver` — used to derive
            ``aug_source_group``.

        Raises
        ------
        ValueError
            If any ``aug_*`` key appears in *features* or if any parent ID is
            not a known original.
        """
        aug_keys = [k for k in features if str(k).startswith("aug_")]
        if aug_keys:
            raise ValueError(
                f"Plug-in returned aug_* keys: {aug_keys}. "
                "Plug-ins must return feature values only."
            )

        for pid in parent_ids:
            if pid not in self._known_ids:
                raise ValueError(
                    f"Unknown parent aug_row_id {pid!r}. "
                    "All parents must be original rows recorded before synthesis."
                )

        parent_key = tuple(sorted(parent_ids))
        k = self._counters[parent_key]
        self._counters[parent_key] += 1
        aug_row_id = f"{self._method}:{':'.join(parent_ids)}:{k}"

        # Resolve source group via the resolver (never the plug-in).
        if len(parent_ids) == 1:
            source_group = resolver.group_for_single(self._known_ids[parent_ids[0]])
        else:
            gid_a = self._known_ids[parent_ids[0]]
            gid_b = self._known_ids[parent_ids[1]]
            source_group = resolver.group_for_pair(gid_a, gid_b)

        record: dict[str, Any] = dict(features)
        record["aug_row_id"] = aug_row_id
        record["aug_is_original"] = False
        record["aug_parent_ids"] = list(parent_ids)
        record["aug_source_group"] = source_group
        record["aug_method"] = self._method
        record["aug_params"] = json.dumps(params, sort_keys=True)
        record["aug_seed"] = child_seed

        self._rows.append(record)
        return record

    def build_output(self, resolver: GroupResolver) -> pd.DataFrame:
        """Assemble the final DataFrame: originals first, then synthetic rows.

        Applies :meth:`GroupResolver.relabel` to fix group IDs for rows whose
        group was merged after they were recorded (originals in a pair).
        """
        df = pd.DataFrame(self._rows)
        df = resolver.relabel(df)
        # Drop the internal helper column if it leaked through.
        if "_group_id" in df.columns:
            df = df.drop(columns=["_group_id"])
        return df.reset_index(drop=True)
