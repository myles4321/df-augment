"""Compute how many new rows to generate per class."""

from __future__ import annotations


def rows_to_generate(
    class_counts: dict[int | str, int],
    multiplier: float | None = None,
    balance: bool = False,
) -> dict[int | str, int]:
    """Return the number of *new* (synthetic) rows needed per class.

    Parameters
    ----------
    class_counts:
        Mapping from class label to the number of existing rows in that class.
    multiplier:
        Target size = multiplier x current size. Each class gets
        ``round((multiplier - 1) * n_c)`` new rows. Must be >= 1.0.
    balance:
        Grow every class to the size of the largest class. Each class gets
        ``max_count - n_c`` new rows.

    Raises
    ------
    ValueError
        If both *multiplier* and *balance* are set, or if *multiplier* < 1.0.
    """
    if multiplier is not None and balance:
        raise ValueError("Cannot set both --multiplier and --balance. Choose one.")

    if multiplier is not None and multiplier < 1.0:
        raise ValueError(f"--multiplier must be >= 1.0, got {multiplier}.")

    if balance:
        max_count = max(class_counts.values())
        return {cls: max_count - n for cls, n in class_counts.items()}

    if multiplier is not None:
        return {cls: round((multiplier - 1) * n) for cls, n in class_counts.items()}

    # Neither set -- generate nothing.
    return {cls: 0 for cls in class_counts}
