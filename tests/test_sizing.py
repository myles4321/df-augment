"""Unit tests for sizing.rows_to_generate."""

from __future__ import annotations

import pytest

from df_augment.sizing import rows_to_generate


def test_multiplier_three_classes() -> None:
    counts = {0: 10, 1: 20, 2: 30}
    result = rows_to_generate(counts, multiplier=2.0)
    assert result == {0: 10, 1: 20, 2: 30}


def test_multiplier_one_gives_zero_new_rows() -> None:
    counts = {0: 10, 1: 15}
    result = rows_to_generate(counts, multiplier=1.0)
    assert result == {0: 0, 1: 0}


def test_balance_unequal_classes() -> None:
    counts = {0: 10, 1: 20, 2: 30}
    result = rows_to_generate(counts, balance=True)
    assert result == {0: 20, 1: 10, 2: 0}


def test_balance_already_balanced() -> None:
    counts = {0: 20, 1: 20, 2: 20}
    result = rows_to_generate(counts, balance=True)
    assert result == {0: 0, 1: 0, 2: 0}


def test_multiplier_below_one_raises() -> None:
    with pytest.raises(ValueError, match=">="):
        rows_to_generate({0: 10}, multiplier=0.5)


def test_both_multiplier_and_balance_raises() -> None:
    with pytest.raises(ValueError, match="both"):
        rows_to_generate({0: 10}, multiplier=2.0, balance=True)


def test_rounding_edge_case() -> None:
    # n=3, m=1.5 -> round(0.5 * 3) = round(1.5) = 2
    result = rows_to_generate({0: 3}, multiplier=1.5)
    assert result[0] == 2


def test_neither_set_returns_zeros() -> None:
    counts = {0: 10, 1: 5}
    result = rows_to_generate(counts)
    assert result == {0: 0, 1: 0}


def test_single_class_regression_multiplier() -> None:
    result = rows_to_generate({"regression": 100}, multiplier=3.0)
    assert result == {"regression": 200}


def test_single_class_balance_gives_zero() -> None:
    result = rows_to_generate({"regression": 50}, balance=True)
    assert result == {"regression": 0}
