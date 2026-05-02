"""SelfFoil core: geometry and Fourier airfoil embeddings."""

from .airfoil_embedding import (
    MAX_ABS_FREQ,
    N_COMPLEX_MODES,
    N_COORDS,
    N_FOURIER_REAL,
    AirfoilFourierEmbedding,
)
from .geometry import remove_duplicate_closure, resample_closed_poly

__all__ = [
    "AirfoilFourierEmbedding",
    "MAX_ABS_FREQ",
    "N_COMPLEX_MODES",
    "N_COORDS",
    "N_FOURIER_REAL",
    "remove_duplicate_closure",
    "resample_closed_poly",
]
