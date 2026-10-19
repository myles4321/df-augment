"""
Tabular data augmentation pipeline with parent tracking.

Steps:
  1. Read the raw tabular dataset (CSV / Excel)
  2. Convert it to Parquet
  3. Add a `source_group` column (S001, S002, ...) marking each row as a parent
  4. Augment, giving every new row an `aug_source_group` that points back
     to the `source_group` of the row it was created from
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1: read raw data
# ---------------------------------------------------------------------------
def read_raw(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


# ---------------------------------------------------------------------------
# Step 2: convert to Parquet
# ---------------------------------------------------------------------------
def to_parquet(df: pd.DataFrame, out_path: str) -> str:
    df.to_parquet(out_path, index=False)  # requires pyarrow
    return out_path


# ---------------------------------------------------------------------------
# Step 3: add source_group IDs
# ---------------------------------------------------------------------------
def add_source_group(parquet_path: str, out_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    width = max(3, len(str(len(df))))           # S001 ... S9999 etc.
    df.insert(0, "source_group",
              [f"S{i:0{width}d}" for i in range(1, len(df) + 1)])
    df.to_parquet(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Step 4: augment
# ---------------------------------------------------------------------------
def augment(
    parquet_path: str,
    out_path: str,
    n_per_row: int = 2,
    noise_scale: float = 0.05,
    seed: int = 42,
    exclude: tuple = (),
) -> pd.DataFrame:
    """
    For each original row, create `n_per_row` synthetic rows by adding
    Gaussian noise to numeric columns (std = noise_scale * column std).
    Categorical columns are copied from the parent.

    Output = originals + synthetic rows, with:
      - is_augmented     : False for originals, True for new rows
      - aug_source_group : the parent's source_group (blank for originals)
    """
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(parquet_path)

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in exclude]
    col_std = df[numeric_cols].std()

    new_rows = []
    for _, parent in df.iterrows():
        for _ in range(n_per_row):
            child = parent.copy()
            for c in numeric_cols:
                val = parent[c] + rng.normal(0, noise_scale * col_std[c])
                # keep integer columns as integers
                if pd.api.types.is_integer_dtype(df[c]):
                    val = int(round(val))
                child[c] = val
            child["aug_source_group"] = parent["source_group"]
            child["source_group"] = None   # synthetic rows are not parents
            new_rows.append(child)

    aug = pd.DataFrame(new_rows)
    aug["is_augmented"] = True

    orig = df.copy()
    orig["aug_source_group"] = None
    orig["is_augmented"] = False

    result = pd.concat([orig, aug], ignore_index=True)
    for c in numeric_cols:                     # restore dtypes after concat
        result[c] = result[c].astype(df[c].dtype)
    result.to_parquet(out_path, index=False)
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 140)

    # Sample raw dataset
    raw = pd.DataFrame({
        "age":    [34, 52, 27, 45, 61],
        "income": [52000.0, 87000.0, 39000.0, 64000.0, 71000.0],
        "city":   ["Halifax", "Sydney", "Antigonish", "Truro", "Halifax"],
        "bought": ["yes", "no", "yes", "no", "yes"],
    })
    raw.to_csv("raw_data.csv", index=False)

    print("STEP 1 - Raw dataset (raw_data.csv)")
    df = read_raw("raw_data.csv")
    print(df, "\n")

    print("STEP 2 - Converted to data.parquet")
    to_parquet(df, "data.parquet")
    print(pd.read_parquet("data.parquet").dtypes, "\n")

    print("STEP 3 - Added source_group (data_grouped.parquet)")
    grouped = add_source_group("data.parquet", "data_grouped.parquet")
    print(grouped, "\n")

    print("STEP 4 - Augmented (data_augmented.parquet)")
    result = augment("data_grouped.parquet", "data_augmented.parquet",
                     n_per_row=2)
    print(result.to_string(), "\n")

    # Traceability check: every new row maps to a real parent
    parents = set(grouped["source_group"])
    children = result[result["is_augmented"]]
    assert children["aug_source_group"].isin(parents).all()
    print(f"Check passed: all {len(children)} augmented rows map to a parent.")
    print("\nChildren of S002:")
    print(result[(result.source_group == "S002") |
                 (result.aug_source_group == "S002")].to_string())
