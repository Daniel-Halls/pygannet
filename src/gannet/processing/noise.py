"""Noise estimation utilities for MRS spectra."""

import numpy as np
from numpy.typing import ArrayLike


def calc_noise(freq: ArrayLike, spec: ArrayLike) -> float:
    """Estimate noise level in downfield frequency-domain signal.

    Mirrors ``CalcNoise.m`` by detrending 8-9 ppm and 9-10 ppm regions with a
    second-order polynomial and returning twice the minimum standard deviation.

    Parameters
    ------------
    freq: ArrayLike
        Frequency axis in ppm.
    spec: ArrayLike
        Complex or real frequency-domain spectrum.

    Returns
    --------
        float: float object
            Estimated peak-to-peak noise metric (``2 * sigma``).
    """
    freq_arr = np.asarray(freq, dtype=np.float64).ravel()
    spec_arr = np.asarray(spec).ravel()
    degree = 2

    ind_a = (freq_arr >= 8.0) & (freq_arr <= 9.0)
    noise_a = np.real(spec_arr[ind_a])
    poly_a = np.polyfit(freq_arr[ind_a], noise_a, degree)
    noise_a = noise_a - np.polyval(poly_a, freq_arr[ind_a])

    ind_b = (freq_arr >= 9.0) & (freq_arr <= 10.0)
    noise_b = np.real(spec_arr[ind_b])
    poly_b = np.polyfit(freq_arr[ind_b], noise_b, degree)
    noise_b = noise_b - np.polyval(poly_b, freq_arr[ind_b])

    sigma = min(float(np.std(noise_a, ddof=1)), float(np.std(noise_b, ddof=1)))
    noise = 2.0 * sigma
    return noise
