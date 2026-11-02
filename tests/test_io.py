"""Tests for io/loader.py."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from df_augment.io.loader import load


def test_load_parquet_auto_group_ids(tmp_path: pytest.TempPathFactory) -> None:
    df = pd.DataFrame({"f0": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    result = load(path)
    assert "_group_id" in result.columns
    assert list(result["_group_id"]) == ["S0001", "S0002", "S0003"]


def test_load_parquet_with_group_col(tmp_path: pytest.TempPathFactory) -> None:
    df = pd.DataFrame(
        {"subject_id": ["A", "B", "C"], "f0": [1.0, 2.0, 3.0], "target": [0, 1, 0]}
    )
    path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    result = load(path, group_col="subject_id")
    assert list(result["_group_id"]) == ["A", "B", "C"]


def test_load_refuses_aug_columns(tmp_path: pytest.TempPathFactory) -> None:
    df = pd.DataFrame({"f0": [1.0], "aug_source_group": ["G1"], "target": [0]})
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    with pytest.raises(ValueError, match="aug_\\*"):
        load(path)


def test_load_unsupported_extension(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "data.json"
    path.write_text("{}")

    with pytest.raises(ValueError, match="Unsupported"):
        load(path)


def test_load_preserves_dtypes(tmp_path: pytest.TempPathFactory) -> None:
    df = pd.DataFrame(
        {"age": pd.array([25, 30, 45], dtype="int64"), "target": [0, 1, 0]}
    )
    path = tmp_path / "typed.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    result = load(path)
    assert result["age"].dtype == pd.api.types.pandas_dtype("int64")
