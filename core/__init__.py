"""SelfFoil core."""

from .airfoil_embedding import (
    MAX_ABS_FREQ,
    N_COMPLEX_MODES,
    N_COORDS,
    N_FOURIER_REAL,
    AirfoilFourierEmbedding,
)
from .coord_fourier_regression import CoordFourierRegressionMLP, CoordToFourierMLP
from .geometry import remove_duplicate_closure, resample_closed_poly
from .polar_token_embedding import PolarTokenEmbedding
from .regime_transformer import (
    RegimeConditionedAeroTransformer,
    fourier_mse_loss,
)

__all__ = [
    "AirfoilFourierEmbedding",
    "CoordFourierRegressionMLP",
    "CoordToFourierMLP",
    "MAX_ABS_FREQ",
    "N_COMPLEX_MODES",
    "N_COORDS",
    "N_FOURIER_REAL",
    "PolarTokenEmbedding",
    "RegimeConditionedAeroTransformer",
    "fourier_mse_loss",
    "remove_duplicate_closure",
    "resample_closed_poly",
]
