"""Eight lineage invariants that must hold for every technique."""

from __future__ import annotations

import pandas as pd
import pytest

import df_augment
from df_augment.config import AugmentConfig
from df_augment.io.loader import load


def _run_augment(source_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Run the identity plug-in with multiplier=2 on *source_df*."""
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name

    pq.write_table(pa.Table.from_pandas(source_df, preserve_index=False), tmp_path)

    config = AugmentConfig(
        data=tmp_path,
        out="/tmp/aug_out.parquet",
        techniques=["identity"],
        multiplier=2.0,
        seed=seed,
    )
    df_loaded = load(config.data, config.group_col)
    return df_augment.augment(df_loaded, config)


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_every_row_has_nonnull_aug_source_group(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment(source_df)
    assert result["aug_source_group"].notna().all()
    assert (result["aug_source_group"] != "").all()


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_every_row_has_at_least_one_parent(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment(source_df)
    assert result["aug_parent_ids"].apply(lambda p: len(p) >= 1).all()


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_every_parent_id_exists_in_output(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment(source_df)
    known_ids = set(result["aug_row_id"])
    for pids in result["aug_parent_ids"]:
        for pid in pids:
            assert pid in known_ids, f"Parent {pid!r} not found in output"


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_derived_row_shares_parents_source_group(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment(source_df)
    id_to_group = dict(zip(result["aug_row_id"], result["aug_source_group"]))
    synthetic = result[~result["aug_is_original"]]
    for _, row in synthetic.iterrows():
        for pid in row["aug_parent_ids"]:
            assert row["aug_source_group"] == id_to_group[pid], (
                f"Synthetic row {row['aug_row_id']!r} has group "
                f"{row['aug_source_group']!r} but parent {pid!r} has "
                f"{id_to_group[pid]!r}"
            )


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_no_source_group_holds_more_than_two_sources(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment(source_df)
    # For the identity plug-in, each group has exactly one source.
    # This invariant ensures the disjoint-pairing rule holds in general.
    for group, sub in result.groupby("aug_source_group"):
        orig_parent_ids = {
            pid
            for pids in sub["aug_parent_ids"]
            for pid in pids
            if pid.startswith("orig:")
        }
        assert len(orig_parent_ids) <= 2, (
            f"Group {group!r} references {len(orig_parent_ids)} original sources"
        )


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_multiplier_two_doubles_each_class(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment(source_df)
    original_counts = source_df["target"].value_counts().to_dict()
    result_counts = result.groupby("target").size().to_dict()
    for cls, n_orig in original_counts.items():
        expected = round(2.0 * n_orig)
        assert result_counts[cls] == expected, (
            f"Class {cls}: expected {expected} rows, got {result_counts[cls]}"
        )


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_original_rows_unchanged(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result = _run_augment(source_df)
    originals = result[result["aug_is_original"]].reset_index(drop=True)
    feature_cols = [c for c in source_df.columns if c not in ("subject_id",)]
    for col in feature_cols:
        pd.testing.assert_series_equal(
            originals[col].reset_index(drop=True),
            source_df[col].reset_index(drop=True),
            check_names=False,
        )


@pytest.mark.parametrize("fixture_name", ["small_tabular_df", "grouped_tabular_df"])
def test_same_seed_gives_identical_output(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    source_df = request.getfixturevalue(fixture_name)
    result1 = _run_augment(source_df, seed=7)
    result2 = _run_augment(source_df, seed=7)
    pd.testing.assert_frame_equal(result1, result2)
