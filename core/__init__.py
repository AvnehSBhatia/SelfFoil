"""SelfFoil core — polar tokens → low-frequency Fourier coefficients."""

from .airfoil_embedding import (
    MAX_ABS_FREQ,
    N_COMPLEX_MODES,
    N_COORDS,
    N_FOURIER_REAL,
    AirfoilFourierEmbedding,
)
from .constants import POLAR_ALPHA, POLAR_CD, POLAR_CL, POLAR_DIM, POLAR_MA, POLAR_RE
from .geometry import remove_duplicate_closure, resample_closed_poly
from .polar_token_embedding import PolarTokenEmbedding
from .polar_voting_moe import PolarVotingMoETransformer, fourier_mse_loss

__all__ = [
    "AirfoilFourierEmbedding",
    "MAX_ABS_FREQ",
    "N_COMPLEX_MODES",
    "N_COORDS",
    "N_FOURIER_REAL",
    "POLAR_ALPHA",
    "POLAR_CD",
    "POLAR_CL",
    "POLAR_DIM",
    "POLAR_MA",
    "POLAR_RE",
    "PolarTokenEmbedding",
    "PolarVotingMoETransformer",
    "fourier_mse_loss",
    "remove_duplicate_closure",
    "resample_closed_poly",
]
