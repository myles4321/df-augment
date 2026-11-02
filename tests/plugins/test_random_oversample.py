"""Tests for the RandomOversampleTabular plug-in."""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import df_augment
from df_augment.config import AugmentConfig
from df_augment.io.loader import load
from df_augment.plugins.tabular.random_oversample import RandomOversampleTabular

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_realistic_df_with_ids() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n_subjects, rows_per = 10, 2
    subject_ids = [
        f"S{i:02d}" for i in range(1, n_subjects + 1) for _ in range(rows_per)
    ]
    cities = ["Halifax", "Sydney", "Antigonish", "Truro", "Dartmouth"]
    n = n_subjects * rows_per
    df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "age": rng.integers(25, 65, size=n),
            "income": rng.uniform(30_000.0, 120_000.0, size=n),
            "city": [cities[i % len(cities)] for i in range(n)],
            "target": rng.integers(0, 2, size=n),
            "_group_id": subject_ids,
        }
    )
    df["aug_row_id"] = [f"orig:{sid}:{i}" for i, sid in enumerate(subject_ids)]
    return df


def _run_augment_oversample(source_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    pq.write_table(pa.Table.from_pandas(source_df, preserve_index=False), tmp_path)
    config = AugmentConfig(
        data=tmp_path,
        out="/tmp/aug_oversample_out.parquet",
        techniques=["random_oversample"],
        multiplier=2.0,
        seed=seed,
    )
    df_loaded = load(config.data, config.group_col)
    return df_augment.augment(df_loaded, config)


# ---------------------------------------------------------------------------
# Unit tests — exercise generate() directly
# ---------------------------------------------------------------------------


def test_all_columns_copied_verbatim() -> None:
    """Random oversampling must copy every column exactly — no modification."""
    df = _make_realistic_df_with_ids()
    n_per_class = df["target"].value_counts().to_dict()
    rng = np.random.default_rng(0)
    plugin = RandomOversampleTabular()

    new_rows, parent_ids, _ = plugin.generate(df, n_per_class, rng, target_col="target")

    id_to_row = {str(row["aug_row_id"]): row for _, row in df.iterrows()}
    for row_dict, pids in zip(new_rows, parent_ids):
        parent = id_to_row[pids[0]]
        assert row_dict["income"] == parent["income"]
        assert row_dict["age"] == parent["age"]
        assert row_dict["city"] == parent["city"]
        assert row_dict["target"] == parent["target"]


def test_target_column_never_modified() -> None:
    """The target label must match the parent's target exactly."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(1)
    plugin = RandomOversampleTabular()

    new_rows, parent_ids, _ = plugin.generate(
        df, {0: 5, 1: 5}, rng, target_col="target"
    )

    id_to_row = {str(row["aug_row_id"]): row for _, row in df.iterrows()}
    for row_dict, pids in zip(new_rows, parent_ids):
        assert row_dict["target"] == id_to_row[pids[0]]["target"]


def test_no_aug_columns_in_returned_rows() -> None:
    """Returned feature dicts must contain no aug_* keys."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(2)
    plugin = RandomOversampleTabular()

    new_rows, _, _ = plugin.generate(df, {0: 4, 1: 4}, rng, target_col="target")

    for row_dict in new_rows:
        aug_keys = [k for k in row_dict if str(k).startswith("aug_")]
        assert not aug_keys, f"Found aug_* keys in output row: {aug_keys}"


def test_params_are_empty_dict() -> None:
    """Random oversampling has no parameters — params dict must be empty."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(3)
    plugin = RandomOversampleTabular()

    _, _, params = plugin.generate(df, {0: 3, 1: 3}, rng, target_col="target")

    for p in params:
        assert p == {}


def test_zero_n_per_class_returns_empty() -> None:
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(4)
    plugin = RandomOversampleTabular()

    new_rows, parent_ids, params = plugin.generate(
        df, {0: 0, 1: 0}, rng, target_col="target"
    )

    assert new_rows == []
    assert parent_ids == []
    assert params == []


def test_correct_number_of_rows_produced() -> None:
    """generate() must return exactly as many rows as requested."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(5)
    plugin = RandomOversampleTabular()

    new_rows, parent_ids, params = plugin.generate(
        df, {0: 7, 1: 3}, rng, target_col="target"
    )

    assert len(new_rows) == 10
    assert len(parent_ids) == 10
    assert len(params) == 10


# ---------------------------------------------------------------------------
# Lineage invariants — all three fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name", ["small_tabular_df", "grouped_tabular_df", "realistic_tabular_df"]
)
def test_all_lineage_invariants_random_oversample(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    """All eight lineage invariants must hold when using random_oversample."""
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment_oversample(source_df)
    original_counts = source_df["target"].value_counts().to_dict()
    known_ids = set(result["aug_row_id"])
    id_to_group = dict(zip(result["aug_row_id"], result["aug_source_group"]))
    synthetic = result[~result["aug_is_original"]]

    # 1. Every row has a non-null, non-empty aug_source_group.
    assert result["aug_source_group"].notna().all()
    assert (result["aug_source_group"] != "").all()

    # 2. Every row has at least one parent.
    assert result["aug_parent_ids"].apply(lambda p: len(p) >= 1).all()

    # 3. Every parent ID exists in the output.
    for pids in result["aug_parent_ids"]:
        for pid in pids:
            assert pid in known_ids, f"Parent {pid!r} not found in output"

    # 4. Every synthetic row shares its parent's aug_source_group.
    for _, row in synthetic.iterrows():
        for pid in row["aug_parent_ids"]:
            assert row["aug_source_group"] == id_to_group[pid]

    # 5. No source group references more than two distinct original sources.
    for _group, sub in result.groupby("aug_source_group"):
        orig_parent_ids = {
            pid
            for pids in sub["aug_parent_ids"]
            for pid in pids
            if pid.startswith("orig:")
        }
        assert len(orig_parent_ids) <= 2

    # 6. multiplier=2 doubles each class (within rounding).
    result_counts = result.groupby("target").size().to_dict()
    for cls, n_orig in original_counts.items():
        assert result_counts[cls] == round(2.0 * n_orig)

    # 7. Original rows are unchanged.
    feature_cols = [
        c
        for c in source_df.columns
        if c not in ("subject_id",) and not str(c).startswith("aug_")
    ]
    originals = result[result["aug_is_original"]].reset_index(drop=True)
    for col in feature_cols:
        pd.testing.assert_series_equal(
            originals[col].reset_index(drop=True),
            source_df[col].reset_index(drop=True),
            check_names=False,
        )

    # 8. Determinism: same seed → identical output.
    result2 = _run_augment_oversample(source_df, seed=42)
    pd.testing.assert_frame_equal(result, result2)
