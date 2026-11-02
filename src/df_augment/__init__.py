"""df-augment: lineage-tracked tabular and image data augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Importing the tabular sub-package triggers @register decorators.
import df_augment.plugins.tabular  # noqa: F401
from df_augment import registry
from df_augment.cli import main
from df_augment.config import AugmentConfig
from df_augment.lineage.groups import GroupResolver
from df_augment.lineage.tracker import LineageTracker
from df_augment.sizing import rows_to_generate

__all__ = ["AugmentConfig", "augment", "main"]


def augment(df: pd.DataFrame, config: AugmentConfig) -> pd.DataFrame:
    """Augment *df* according to *config* and return the lineage-tracked result.

    The returned DataFrame contains all original rows (unchanged) followed by
    all synthetic rows, each with the full set of ``aug_*`` columns populated.

    Parameters
    ----------
    df:
        Input DataFrame. Must already have a ``_group_id`` column (added by
        :func:`df_augment.io.loader.load`).
    config:
        Validated augmentation configuration.

    Returns
    -------
    pd.DataFrame
        Originals + synthetic rows with all reserved ``aug_*`` columns.
    """
    target_col = config.target

    # --- Sizing -----------------------------------------------------------------
    class_counts: dict[int | str, int] = df[target_col].value_counts().to_dict()
    n_per_class = rows_to_generate(
        class_counts,
        multiplier=config.multiplier,
        balance=config.balance,
    )

    # --- One technique for now (Phase 1a) -------------------------------------
    technique_name = config.techniques[0]
    plugin_cls = registry.get(technique_name)
    plugin = plugin_cls()

    # --- Seed sequence ---------------------------------------------------------
    ss = np.random.SeedSequence(config.seed)
    (child_ss,) = ss.spawn(1)
    rng = np.random.default_rng(child_ss)
    child_seeds = iter(rng.integers(0, 2**31, size=sum(n_per_class.values()) + 1))

    # --- Group resolver --------------------------------------------------------
    group_ids: list[str] = df["_group_id"].astype(str).tolist()
    resolver = GroupResolver(group_ids)

    # --- Tracker ---------------------------------------------------------------
    tracker = LineageTracker(
        source_df=df,
        seed=config.seed,
        method=technique_name,
    )

    # Record all originals and attach aug_row_id so the plug-in can reference
    # them as parents.
    aug_row_ids: list[str] = []
    for i in range(len(df)):
        rec = tracker.record_original(i)
        aug_row_ids.append(rec["aug_row_id"])

    # Attach aug_row_id to a working copy for the plug-in.
    df_with_ids = df.copy()
    df_with_ids["aug_row_id"] = aug_row_ids

    # --- Generate rows --------------------------------------------------------
    new_rows, parent_ids_per_row, params_per_row = plugin.generate(
        df_with_ids, n_per_class, rng, target_col=config.target
    )

    # --- Record synthetic rows via tracker ------------------------------------
    for features, parent_ids, params in zip(
        new_rows, parent_ids_per_row, params_per_row
    ):
        child_seed = int(next(child_seeds))
        tracker.record_synthetic(
            features=features,
            parent_ids=parent_ids,
            params=params,
            child_seed=child_seed,
            resolver=resolver,
        )

    return tracker.build_output(resolver)
