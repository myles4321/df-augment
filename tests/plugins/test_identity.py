"""Tests for the identity plug-in."""

from __future__ import annotations

import numpy as np
import pandas as pd

from df_augment.plugins.tabular.identity import IdentityTabular


def _make_df_with_ids() -> pd.DataFrame:
    """Small DataFrame that already has aug_row_id on every row."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=6),
            "f1": rng.normal(size=6),
            "target": [0, 0, 0, 1, 1, 1],
            "_group_id": [f"G{i}" for i in range(6)],
        }
    )
    df["aug_row_id"] = [f"orig:G{i}" for i in range(6)]
    return df


def test_generate_returns_correct_lengths() -> None:
    df = _make_df_with_ids()
    n_per_class = {0: 4, 1: 2}
    rng = np.random.default_rng(42)
    plugin = IdentityTabular()

    new_rows, parent_ids, params = plugin.generate(df, n_per_class, rng)

    assert len(new_rows) == 6
    assert len(parent_ids) == 6
    assert len(params) == 6


def test_parent_ids_are_valid_aug_row_ids() -> None:
    df = _make_df_with_ids()
    known_ids = set(df["aug_row_id"])
    rng = np.random.default_rng(0)
    plugin = IdentityTabular()

    _, parent_ids, _ = plugin.generate(df, {0: 3, 1: 3}, rng)

    for pids in parent_ids:
        assert len(pids) == 1
        assert pids[0] in known_ids


def test_feature_values_match_source_row() -> None:
    df = _make_df_with_ids()
    rng = np.random.default_rng(1)
    plugin = IdentityTabular()

    new_rows, parent_ids, _ = plugin.generate(df, {0: 2, 1: 2}, rng)

    # Build an index from aug_row_id to original row for quick lookup.
    id_to_row = {row["aug_row_id"]: row for _, row in df.iterrows()}
    feature_cols = ["f0", "f1", "target"]

    for row_dict, pids in zip(new_rows, parent_ids):
        src = id_to_row[pids[0]]
        for col in feature_cols:
            assert row_dict[col] == src[col], (
                f"Column {col!r}: expected {src[col]}, got {row_dict[col]}"
            )


def test_no_aug_columns_in_returned_rows() -> None:
    df = _make_df_with_ids()
    rng = np.random.default_rng(2)
    plugin = IdentityTabular()

    new_rows, _, _ = plugin.generate(df, {0: 3, 1: 3}, rng)

    for row_dict in new_rows:
        aug_keys = [k for k in row_dict if str(k).startswith("aug_")]
        assert not aug_keys, f"Found aug_* keys in output row: {aug_keys}"


def test_zero_n_per_class_returns_empty() -> None:
    df = _make_df_with_ids()
    rng = np.random.default_rng(3)
    plugin = IdentityTabular()

    new_rows, parent_ids, params = plugin.generate(df, {0: 0, 1: 0}, rng)

    assert new_rows == []
    assert parent_ids == []
    assert params == []
