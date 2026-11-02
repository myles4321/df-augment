"""Tiny synthetic datasets shared across tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def small_tabular_df() -> pd.DataFrame:
    """10 rows (1 per subject), 3 continuous features, binary target, no aug_* columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "subject_id": [f"S{i:02d}" for i in range(1, 11)],
            "f0": rng.normal(size=10),
            "f1": rng.normal(size=10),
            "f2": rng.normal(size=10),
            "target": rng.integers(0, 2, size=10),
        }
    )


@pytest.fixture()
def grouped_tabular_df() -> pd.DataFrame:
    """30 rows from 10 subjects (3 rows each), binary target, no aug_* columns."""
    rng = np.random.default_rng(1)
    subject_ids = [f"S{i:02d}" for i in range(1, 11) for _ in range(3)]
    return pd.DataFrame(
        {
            "subject_id": subject_ids,
            "f0": rng.normal(size=30),
            "f1": rng.normal(size=30),
            "f2": rng.normal(size=30),
            "target": rng.integers(0, 2, size=30),
        }
    )


@pytest.fixture()
def realistic_tabular_df() -> pd.DataFrame:
    """20 rows from 10 subjects (2 rows each) with real-looking mixed-type columns.

    Columns
    -------
    subject_id : str   -- group identifier
    age        : int64 -- integer feature (25–64)
    income     : float -- continuous feature (30 000–120 000)
    city       : str   -- categorical feature (non-numeric, copied verbatim)
    target     : int64 -- binary label
    """
    rng = np.random.default_rng(2)
    n_subjects, rows_per = 10, 2
    subject_ids = [
        f"S{i:02d}" for i in range(1, n_subjects + 1) for _ in range(rows_per)
    ]
    cities = ["Halifax", "Sydney", "Antigonish", "Truro", "Dartmouth"]
    return pd.DataFrame(
        {
            "subject_id": subject_ids,
            "age": rng.integers(25, 65, size=n_subjects * rows_per),
            "income": rng.uniform(30_000.0, 120_000.0, size=n_subjects * rows_per),
            "city": [cities[i % len(cities)] for i in range(n_subjects * rows_per)],
            "target": rng.integers(0, 2, size=n_subjects * rows_per),
        }
    )
