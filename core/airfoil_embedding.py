"""Analytic Fourier packing of Cartesian coordinates ↔ 50 real coefficients (NOT polar 5→16 `E(x)`)."""

from __future__ import annotations

import numpy as np

from .geometry import resample_closed_poly, resample_closed_poly_batched

# One contour sample count (FFT length); CSV convention.
N_COORDS = 250
# 50 reals = Re/Im pairs for 25 complex bins covering integer freqs with |k| ≤ 12.
N_FOURIER_REAL = 50
N_COMPLEX_MODES = 25
MAX_ABS_FREQ = 12


def _fft_bins_ordered(m: int, max_abs_k: int) -> list[int]:
    """Indices for np.fft.fft order: DC, +1..+max_abs_k, then -max_abs_k..-1."""
    if max_abs_k > m // 2:
        raise ValueError("max_abs_k exceeds Nyquist")
    pos = list(range(0, max_abs_k + 1))
    neg = [m + k for k in range(-max_abs_k, 0)]
    return pos + neg


_FFT_BINS = _fft_bins_ordered(N_COORDS, MAX_ABS_FREQ)
assert len(_FFT_BINS) == N_COMPLEX_MODES
_FFT_BINS_ARR = np.array(_FFT_BINS, dtype=np.intp)


class AirfoilFourierEmbedding:
    """
    Embedding engine: complex FFT on z = x + i·y along uniform arc-length samples.

    The 50-vector is **not** 50 complex exponentials; it is **50 real scalars**
    storing ``[Re(c₀), Im(c₀), …]`` for **25 complex** low-frequency FFT bins
    (integer frequencies with ``|k| ≤ 12``), ``M = 250``.
    """

    n_coords = N_COORDS
    n_fourier_real = N_FOURIER_REAL
    n_complex_modes = N_COMPLEX_MODES
    max_abs_freq = MAX_ABS_FREQ

    def encode(self, xy: np.ndarray, *, resample: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        xy
            Shape (250, 2). Closed polyline; duplicate closing vertex optional.
        resample
            If True (default), rebuild uniform arc-length samples from the polyline
            (matches CSV-style vertices). If False, treat ``xy`` as samples already
            on the FFT grid so :meth:`decode` → :meth:`encode` recovers coefficients
            exactly (up to float noise).
        Returns
        -------
        coeffs
            Shape (50,), ``float64``: interleaved Re/Im of packed FFT bins.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape != (N_COORDS, 2):
            raise ValueError(f"xy must have shape ({N_COORDS}, 2), got {xy.shape}")
        xy_u = resample_closed_poly(xy, N_COORDS) if resample else xy
        z = xy_u[:, 0] + 1j * xy_u[:, 1]
        zf = np.fft.fft(z)
        c = zf[_FFT_BINS_ARR]
        packed = np.empty(N_FOURIER_REAL, dtype=np.float64)
        packed[0::2] = c.real
        packed[1::2] = c.imag
        return packed

    def decode(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        coeffs
            Shape (50,) interleaved Re/Im in the same order as :meth:`encode`.

        Returns
        -------
        xy
            Shape (250, 2), Cartesian coordinates on the inverse-FFT grid.
        """
        c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if c.shape != (N_FOURIER_REAL,):
            raise ValueError(f"coeffs must have shape ({N_FOURIER_REAL},), got {c.shape}")
        zf = np.zeros(N_COORDS, dtype=np.complex128)
        cplx = c[0::2] + 1j * c[1::2]
        zf[_FFT_BINS_ARR] = cplx
        z = np.fft.ifft(zf)
        return np.stack([z.real, z.imag], axis=-1)

    def encode_batch(
        self, xy: np.ndarray, *, resample: bool = True
    ) -> np.ndarray:
        """
        Batched :meth:`encode` with a single vectorized FFT over the batch axis.

        Parameters
        ----------
        xy
            ``(B, 250, 2)`` float.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 3 or xy.shape[1:] != (N_COORDS, 2):
            raise ValueError(f"xy must have shape (B, {N_COORDS}, 2), got {xy.shape}")
        xy_u = resample_closed_poly_batched(xy, N_COORDS, resample=resample)
        z = xy_u[..., 0] + 1j * xy_u[..., 1]
        zf = np.fft.fft(z, axis=1)
        c = zf[:, _FFT_BINS_ARR]
        packed = np.empty((xy.shape[0], N_FOURIER_REAL), dtype=np.float64)
        packed[:, 0::2] = c.real
        packed[:, 1::2] = c.imag
        return packed
