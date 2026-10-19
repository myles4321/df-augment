"""Tiny synthetic datasets shared across tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def small_tabular_df() -> pd.DataFrame:
    """10 rows, 3 continuous features, binary target, no aug_* columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f0": rng.normal(size=10),
            "f1": rng.normal(size=10),
            "f2": rng.normal(size=10),
            "target": rng.integers(0, 2, size=10),
        }
    )
