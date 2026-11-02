"""Compare augmentation techniques by running each through df-analyze.

Usage
-----
    uv run python compare_techniques.py

For each technique the script:
1. Balances the dataset (minority class → majority class size).
2. Multiplies the balanced dataset by 12× so df-analyze has enough rows.
3. Calls df-analyze and captures holdout-set performance.
4. Prints a comparison table sorted by balanced accuracy.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

import df_augment
from df_augment.config import AugmentConfig
from df_augment.io.loader import load

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
SOURCE_CSV = DATA_DIR / "debtors.csv"
DF_ANALYZE_SCRIPT = Path(
    "/Users/emjaay/Documents/STFX/ML and DM/df-analyze/df-analyze.py"
)
DF_ANALYZE_UV = Path("/Users/emjaay/Documents/STFX/ML and DM/df-analyze")

TECHNIQUES = [
    "gaussian_noise",
    "random_oversample",
    "scaling_jitter",
    "feature_masking",
]

TARGET_COL = "paid"
GROUP_COL = "debtor_id"
CONTINUOUS_COLS = ["income", "loan_amount"]


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def _augment_single_pass(technique: str, source_csv: Path) -> Path:
    """Augment with multiplier=12 in a single pass — returns path to parquet."""
    final_path = DATA_DIR / f"debtors_{technique}.parquet"

    config = AugmentConfig(
        data=str(source_csv),
        out=str(final_path),
        techniques=[technique],
        multiplier=12.0,
        seed=42,
        target=TARGET_COL,
        group_col=GROUP_COL,
    )
    df_loaded = load(config.data, config.group_col)
    df_final = df_augment.augment(df_loaded, config)
    df_final.to_parquet(str(final_path), index=False)

    counts = df_final[TARGET_COL].value_counts().to_dict()
    print(f"  {technique}: {df_final.shape[0]} rows — class counts: {counts}")
    return final_path


# ---------------------------------------------------------------------------
# df-analyze runner
# ---------------------------------------------------------------------------


def _run_df_analyze(parquet_path: Path, technique: str) -> dict[str, float | str]:
    """Run df-analyze and parse the best holdout-set row."""
    outdir = DATA_DIR / f"results_{technique}"
    outdir.mkdir(exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        str(DF_ANALYZE_SCRIPT),
        "--df",
        str(parquet_path),
        "--target",
        TARGET_COL,
        "--mode",
        "classify",
        "--grouper",
        GROUP_COL,
        "--continuous",
        *CONTINUOUS_COLS,
        "--outdir",
        str(outdir),
        "--verbosity",
        "0",
    ]

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(DF_ANALYZE_UV),
    )

    output = result.stdout + result.stderr
    return _parse_holdout_best(output, technique)


def _parse_holdout_best(output: str, technique: str) -> dict[str, float | str]:
    """Extract the best holdout row from df-analyze stdout."""
    lines = output.splitlines()

    # Find the holdout table header line.
    holdout_start = None
    for i, line in enumerate(lines):
        if "Holdout set performance" in line:
            holdout_start = i
            break

    if holdout_start is None:
        print(f"  WARNING: Could not find holdout table for {technique}")
        return {
            "technique": technique,
            "model": "N/A",
            "acc": float("nan"),
            "auroc": float("nan"),
            "bal_acc": float("nan"),
            "sens": float("nan"),
            "spec": float("nan"),
        }

    # Skip until the dashed separator line (data rows follow).
    data_rows = []
    past_header = False
    for line in lines[holdout_start:]:
        stripped = line.strip()
        if not stripped:
            if past_header and data_rows:
                break
            continue
        if set(stripped.replace(" ", "").replace("-", "")) == set():
            past_header = True
            continue
        if past_header and stripped and not stripped.startswith("model"):
            data_rows.append(stripped)

    if not data_rows:
        print(f"  WARNING: Empty holdout table for {technique}")
        return {
            "technique": technique,
            "model": "N/A",
            "acc": float("nan"),
            "auroc": float("nan"),
            "bal_acc": float("nan"),
            "sens": float("nan"),
            "spec": float("nan"),
        }

    # Parse the first data row (best model, sorted by acc descending by df-analyze).
    best = data_rows[0].split()
    try:
        return {
            "technique": technique,
            "model": best[0],
            "acc": float(best[3]),
            "auroc": float(best[4]),
            "bal_acc": float(best[5]),
            "sens": float(best[9]) if len(best) > 9 else float("nan"),
            "spec": float(best[10]) if len(best) > 10 else float("nan"),
        }
    except (IndexError, ValueError):
        return {
            "technique": technique,
            "model": " ".join(best[:2]),
            "acc": float("nan"),
            "auroc": float("nan"),
            "bal_acc": float("nan"),
            "sens": float("nan"),
            "spec": float("nan"),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not SOURCE_CSV.exists():
        print(f"ERROR: source dataset not found at {SOURCE_CSV}")
        sys.exit(1)

    if not DF_ANALYZE_SCRIPT.exists():
        print(f"ERROR: df-analyze not found at {DF_ANALYZE_SCRIPT}")
        sys.exit(1)

    print("=" * 60)
    print("STEP 1 — Augmenting with each technique")
    print("=" * 60)

    augmented_paths: dict[str, Path] = {}
    for technique in TECHNIQUES:
        print(f"\n[{technique}]")
        augmented_paths[technique] = _augment_single_pass(technique, SOURCE_CSV)

    print("\n" + "=" * 60)
    print("STEP 2 — Running df-analyze on each augmented file")
    print("=" * 60)

    rows: list[dict[str, float | str]] = []
    for technique, parquet_path in augmented_paths.items():
        print(f"\n[{technique}] running df-analyze...")
        row = _run_df_analyze(parquet_path, technique)
        rows.append(row)
        print(
            f"  best holdout — acc={row['acc']:.3f}  auroc={row['auroc']:.3f}  "
            f"bal_acc={row['bal_acc']:.3f}  sens={row['sens']:.3f}  spec={row['spec']:.3f}"
        )

    print("\n" + "=" * 60)
    print("RESULTS — Holdout Set Performance by Technique")
    print("(sorted by balanced accuracy, best first)")
    print("=" * 60)

    results_df = (
        pd.DataFrame(rows)
        .sort_values("bal_acc", ascending=False)
        .reset_index(drop=True)
    )

    results_df.index = results_df.index + 1  # 1-based rank
    results_df.index.name = "rank"

    print(
        results_df[
            ["technique", "model", "acc", "auroc", "bal_acc", "sens", "spec"]
        ].to_string(
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nColumns: acc=accuracy  auroc=area under ROC  bal_acc=balanced accuracy")
    print(
        "         sens=sensitivity (true positive rate)  spec=specificity (true negative rate)"
    )

    winner = results_df.iloc[0]
    print(
        f"\nBEST TECHNIQUE: {winner['technique']} "
        f"(balanced accuracy = {winner['bal_acc']:.3f})"
    )


if __name__ == "__main__":
    main()
