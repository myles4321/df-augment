"""Tabular plug-ins -- importing this sub-package registers them all."""

from df_augment.plugins.tabular.feature_masking import FeatureMaskingTabular
from df_augment.plugins.tabular.gaussian_noise import GaussianNoiseTabular
from df_augment.plugins.tabular.identity import IdentityTabular
from df_augment.plugins.tabular.random_oversample import RandomOversampleTabular
from df_augment.plugins.tabular.scaling_jitter import ScalingJitterTabular

__all__ = [
    "FeatureMaskingTabular",
    "GaussianNoiseTabular",
    "IdentityTabular",
    "RandomOversampleTabular",
    "ScalingJitterTabular",
]
