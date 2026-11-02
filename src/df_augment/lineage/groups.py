"""Union-find group resolver for lineage tracking."""

from __future__ import annotations

import pandas as pd


class GroupResolver:
    """Track how source groups merge as paired rows are created.

    Rules
    -----
    * Single-parent rows inherit their parent's group unchanged.
    * Two-parent rows merge the two groups into a pair ID of the form
      ``pair:<min>|<max>`` (lexicographically sorted so the ID is always
      deterministic regardless of argument order).
    * **Disjoint-pairing**: a group may appear in at most one pair.
      Attempting to pair a group a second time raises ``ValueError``.
    * After all rows have been generated, call :meth:`relabel` once to
      rewrite the ``aug_source_group`` column so every member of a merged
      pair carries the canonical pair ID.
    """

    def __init__(self, group_ids: list[str]) -> None:
        # Maps each original group ID to its current canonical label.
        self._canonical: dict[str, str] = {gid: gid for gid in group_ids}
        # Tracks which groups have already been paired (for the disjoint rule).
        self._paired: set[str] = set()

    def group_for_single(self, gid: str) -> str:
        """Return the canonical group for a single-parent row."""
        return self._canonical.get(gid, gid)

    def group_for_pair(self, gid_a: str, gid_b: str) -> str:
        """Merge *gid_a* and *gid_b* and return the new pair label.

        Idempotent when called multiple times for the same pair (as happens
        when many synthetic rows are generated from the same two groups).
        Raises ``ValueError`` only when a group is paired with a *different*
        group than the one it was first paired with.

        Raises
        ------
        ValueError
            If either group is already participating in a *different* pair.
        """
        # Idempotent: if the two groups are already established as a pair,
        # return the existing canonical label without re-triggering the check.
        canon_a = self._canonical.get(gid_a, gid_a)
        canon_b = self._canonical.get(gid_b, gid_b)
        if canon_a == canon_b and gid_a in self._paired and gid_b in self._paired:
            return canon_a

        for gid in (gid_a, gid_b):
            if gid in self._paired:
                raise ValueError(
                    f"Group {gid!r} is already part of a pair. "
                    "The disjoint-pairing rule allows each group in at most one pair."
                )

        pair_label = f"pair:{min(gid_a, gid_b)}|{max(gid_a, gid_b)}"
        self._canonical[gid_a] = pair_label
        self._canonical[gid_b] = pair_label
        self._paired.add(gid_a)
        self._paired.add(gid_b)
        return pair_label

    def relabel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with ``aug_source_group`` rewritten to
        canonical pair labels where applicable."""
        out = df.copy()
        out["aug_source_group"] = out["aug_source_group"].map(
            lambda g: self._canonical.get(g, g)
        )
        return out
