"""Pydantic configuration model for df-augment."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator


class AugmentConfig(BaseModel):
    """All options that control one augmentation run."""

    data: Path
    out: Path
    techniques: list[str]
    multiplier: float | None = None
    balance: bool = False
    target: str = "target"
    group_col: str | None = None
    seed: int = 42

    @field_validator("techniques")
    @classmethod
    def techniques_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("--techniques must list at least one technique.")
        return v

    @field_validator("multiplier")
    @classmethod
    def multiplier_at_least_one(cls, v: float | None) -> float | None:
        if v is not None and v < 1.0:
            raise ValueError(f"--multiplier must be >= 1.0, got {v}.")
        return v

    @model_validator(mode="after")
    def multiplier_and_balance_exclusive(self) -> AugmentConfig:
        if self.multiplier is not None and self.balance:
            raise ValueError("Cannot set both --multiplier and --balance. Choose one.")
        return self
