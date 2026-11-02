"""Tests for the ScalingJitterTabular plug-in."""

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
from df_augment.plugins.tabular.scaling_jitter import ScalingJitterTabular

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


def _run_augment_jitter(source_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    pq.write_table(pa.Table.from_pandas(source_df, preserve_index=False), tmp_path)
    config = AugmentConfig(
        data=tmp_path,
        out="/tmp/aug_jitter_out.parquet",
        techniques=["scaling_jitter"],
        multiplier=2.0,
        seed=seed,
    )
    df_loaded = load(config.data, config.group_col)
    return df_augment.augment(df_loaded, config)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_float_columns_are_scaled() -> None:
    """Synthetic income values must differ from their parent's income."""
    df = _make_realistic_df_with_ids()
    n_per_class = df["target"].value_counts().to_dict()
    rng = np.random.default_rng(0)
    plugin = ScalingJitterTabular(scale_sd=0.10)

    new_rows, parent_ids, _ = plugin.generate(df, n_per_class, rng, target_col="target")

    id_to_row = {str(row["aug_row_id"]): row for _, row in df.iterrows()}
    diffs = [
        abs(row["income"] - id_to_row[pids[0]]["income"])
        for row, pids in zip(new_rows, parent_ids)
    ]
    assert any(d > 0 for d in diffs), "No income values were scaled"


def test_integer_columns_stay_integer_dtype() -> None:
    """Age values in returned rows must be integer-typed."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(1)
    plugin = ScalingJitterTabular()

    new_rows, _, _ = plugin.generate(df, {0: 5, 1: 5}, rng, target_col="target")

    for row_dict in new_rows:
        age_val = row_dict["age"]
        assert isinstance(age_val, (int, np.integer)), (
            f"Expected integer age, got {type(age_val).__name__!r}: {age_val}"
        )


def test_non_numeric_columns_copied_verbatim() -> None:
    """City (object dtype) must be identical between parent and synthetic row."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(2)
    plugin = ScalingJitterTabular()

    new_rows, parent_ids, _ = plugin.generate(
        df, {0: 6, 1: 6}, rng, target_col="target"
    )

    id_to_row = {str(row["aug_row_id"]): row for _, row in df.iterrows()}
    for row_dict, pids in zip(new_rows, parent_ids):
        assert row_dict["city"] == id_to_row[pids[0]]["city"]


def test_target_column_never_modified() -> None:
    """The target label must be copied verbatim — never scaled."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(3)
    plugin = ScalingJitterTabular()

    new_rows, parent_ids, _ = plugin.generate(
        df, {0: 5, 1: 5}, rng, target_col="target"
    )

    id_to_row = {str(row["aug_row_id"]): row for _, row in df.iterrows()}
    for row_dict, pids in zip(new_rows, parent_ids):
        assert row_dict["target"] == id_to_row[pids[0]]["target"]


def test_no_aug_columns_in_returned_rows() -> None:
    """Returned feature dicts must contain no aug_* keys."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(4)
    plugin = ScalingJitterTabular()

    new_rows, _, _ = plugin.generate(df, {0: 4, 1: 4}, rng, target_col="target")

    for row_dict in new_rows:
        aug_keys = [k for k in row_dict if str(k).startswith("aug_")]
        assert not aug_keys, f"Found aug_* keys in output row: {aug_keys}"


def test_scale_sd_zero_copies_exactly() -> None:
    """With scale_sd=0, N(1, 0) always draws 1 — values must be unchanged."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(5)
    plugin = ScalingJitterTabular(scale_sd=0.0)

    new_rows, parent_ids, _ = plugin.generate(
        df, {0: 4, 1: 4}, rng, target_col="target"
    )

    id_to_row = {str(row["aug_row_id"]): row for _, row in df.iterrows()}
    for row_dict, pids in zip(new_rows, parent_ids):
        assert row_dict["income"] == pytest.approx(id_to_row[pids[0]]["income"])


def test_params_reports_scale_sd() -> None:
    """Each returned params dict must record the scale_sd used."""
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(6)
    sd = 0.08
    plugin = ScalingJitterTabular(scale_sd=sd)

    _, _, params = plugin.generate(df, {0: 3, 1: 3}, rng, target_col="target")

    for p in params:
        assert p == {"scale_sd": sd}


def test_zero_n_per_class_returns_empty() -> None:
    df = _make_realistic_df_with_ids()
    rng = np.random.default_rng(7)
    plugin = ScalingJitterTabular()

    new_rows, parent_ids, params = plugin.generate(
        df, {0: 0, 1: 0}, rng, target_col="target"
    )

    assert new_rows == []
    assert parent_ids == []
    assert params == []


# ---------------------------------------------------------------------------
# Lineage invariants — all three fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name", ["small_tabular_df", "grouped_tabular_df", "realistic_tabular_df"]
)
def test_all_lineage_invariants_scaling_jitter(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    """All eight lineage invariants must hold when using scaling_jitter."""
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment_jitter(source_df)
    original_counts = source_df["target"].value_counts().to_dict()
    known_ids = set(result["aug_row_id"])
    id_to_group = dict(zip(result["aug_row_id"], result["aug_source_group"]))
    synthetic = result[~result["aug_is_original"]]

    assert result["aug_source_group"].notna().all()
    assert (result["aug_source_group"] != "").all()
    assert result["aug_parent_ids"].apply(lambda p: len(p) >= 1).all()

    for pids in result["aug_parent_ids"]:
        for pid in pids:
            assert pid in known_ids

    for _, row in synthetic.iterrows():
        for pid in row["aug_parent_ids"]:
            assert row["aug_source_group"] == id_to_group[pid]

    for _group, sub in result.groupby("aug_source_group"):
        orig_parent_ids = {
            pid
            for pids in sub["aug_parent_ids"]
            for pid in pids
            if pid.startswith("orig:")
        }
        assert len(orig_parent_ids) <= 2

    result_counts = result.groupby("target").size().to_dict()
    for cls, n_orig in original_counts.items():
        assert result_counts[cls] == round(2.0 * n_orig)

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

    result2 = _run_augment_jitter(source_df, seed=42)
    pd.testing.assert_frame_equal(result, result2)
