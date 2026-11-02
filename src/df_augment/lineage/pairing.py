"""Pairing strategies: nearest-neighbour (for SMOTE) and random (for MixUp)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def nearest_neighbour_pairs(
    X: np.ndarray,
    labels: np.ndarray,
    group_ids: list[str],
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Greedy within-class nearest-neighbour pairing over source groups.

    One representative row per group (the first occurrence) is used for
    distance computation. Features are z-scored per call so that columns
    with large magnitudes do not dominate the distance metric.

    Parameters
    ----------
    X:
        Feature matrix (n_rows x n_features). Rows correspond 1-to-1 with
        *group_ids* and *labels*.
    labels:
        Class label for each row (one per group representative).
    group_ids:
        Group ID string for each row.
    rng:
        NumPy random Generator (used for tie-breaking in rare cases).

    Returns
    -------
    list of (group_id_a, group_id_b) tuples — only fully paired groups.
    """
    unique_labels = np.unique(labels)
    pairs: list[tuple[str, str]] = []
    free: set[str] = set(group_ids)

    # Z-score the full feature matrix once.
    std = X.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant columns
    X_scaled = (X - X.mean(axis=0)) / std

    for label in unique_labels:
        mask = labels == label
        class_indices = np.where(mask)[0]
        if len(class_indices) < 2:
            continue

        class_gids = [group_ids[i] for i in class_indices]
        class_X = X_scaled[class_indices]

        # Find the 2 nearest neighbours for every point within this class.
        k = min(2, len(class_indices))
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(class_X)
        distances, indices = nn.kneighbors(class_X)

        # Build candidate (dist, a_idx, b_idx) sorted by distance.
        candidates: list[tuple[float, int, int]] = []
        for row, (dists, nbrs) in enumerate(zip(distances, indices)):
            for dist, nbr in zip(dists[1:], nbrs[1:]):  # skip self (index 0)
                a_gid = class_gids[row]
                b_gid = class_gids[nbr]
                if a_gid < b_gid:  # deduplicate symmetric pairs
                    candidates.append((float(dist), row, nbr))

        candidates.sort(key=lambda t: t[0])

        for _, a_row, b_row in candidates:
            a_gid = class_gids[a_row]
            b_gid = class_gids[b_row]
            if a_gid in free and b_gid in free:
                pairs.append((a_gid, b_gid))
                free.discard(a_gid)
                free.discard(b_gid)

    return pairs


def random_pairs(
    group_ids: list[str],
    labels: np.ndarray,
    same_class: bool,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Random pairing of source groups.

    Parameters
    ----------
    group_ids:
        One entry per group (must align with *labels*).
    labels:
        Class label per group.
    same_class:
        If ``True``, pair only within the same class. If ``False``, pair
        across all groups regardless of class.
    rng:
        NumPy random Generator for shuffling.

    Returns
    -------
    list of (group_id_a, group_id_b) tuples.
    """
    pairs: list[tuple[str, str]] = []

    if not same_class:
        shuffled = list(group_ids)
        rng.shuffle(shuffled)  # type: ignore[arg-type]
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))
        return pairs

    unique_labels = np.unique(labels)
    for label in unique_labels:
        class_gids = [g for g, lbl in zip(group_ids, labels) if lbl == label]
        rng.shuffle(class_gids)  # type: ignore[arg-type]
        for i in range(0, len(class_gids) - 1, 2):
            pairs.append((class_gids[i], class_gids[i + 1]))

    return pairs
