"""Phantom-specific Lorentzian multiplet models from GannetFitPhantom.m."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def phantom_three_lorentz_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the phantom three-Lorentzian multiplet model (GABA, EtOH).

    Based on Marshall & Roe, 1978 (Analytical Chem).

    Parameters
    ------------
    x: ArrayLike
        11-element parameter vector ``[Ha, Hb, H0, T2, f0, J, phi_a, phi_b,
        phi_0, M, C]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    ha, hb, h0, t2, f0, j_cpl, phi_a, phi_b, phi_0, slope, offset = p[:11]

    dfa = f0 + j_cpl - f
    dfb = f0 - j_cpl - f
    df0 = f0 - f

    aa = np.cos(phi_a) * ((ha * t2) / (1.0 + dfa**2 * t2**2)) - np.sin(phi_a) * (
        (ha * dfa * t2**2) / (1.0 + dfa**2 * t2**2)
    )
    ab = np.cos(phi_b) * ((hb * t2) / (1.0 + dfb**2 * t2**2)) - np.sin(phi_b) * (
        (hb * dfb * t2**2) / (1.0 + dfb**2 * t2**2)
    )
    a0 = np.cos(phi_0) * ((h0 * t2) / (1.0 + df0**2 * t2**2)) - np.sin(phi_0) * (
        (h0 * df0 * t2**2) / (1.0 + df0**2 * t2**2)
    )

    result: NDArray[np.float64] = aa + a0 + ab + slope * df0 + offset
    return result


def phantom_two_lorentz_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the phantom two-Lorentzian multiplet model (Glx, Lac).

    Based on Marshall & Roe, 1978 (Analytical Chem).

    Parameters
    ------------
    x: ArrayLike
        9-element parameter vector
        ``[Ha, Hb, T2, f0, J, phi_a, phi_b, M, C]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    ha, hb, t2, f0, j_cpl, phi_a, phi_b, slope, offset = p[:9]

    dfa = f0 + j_cpl - f
    dfb = f0 - j_cpl - f

    aa = np.cos(phi_a) * ((ha * t2) / (1.0 + dfa**2 * t2**2)) - np.sin(phi_a) * (
        (ha * dfa * t2**2) / (1.0 + dfa**2 * t2**2)
    )
    ab = np.cos(phi_b) * ((hb * t2) / (1.0 + dfb**2 * t2**2)) - np.sin(phi_b) * (
        (hb * dfb * t2**2) / (1.0 + dfb**2 * t2**2)
    )

    result: NDArray[np.float64] = aa + ab + slope * (f0 - f) + offset
    return result


def phantom_six_lorentz_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the phantom six-Lorentzian multiplet model (GSH).

    Based on Marshall & Roe, 1978 (Analytical Chem).

    Parameters
    ------------
    x: ArrayLike
        17-element parameter vector ``[Ha..Hf, T2, f0, J, phi_a..phi_f, M, C]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    amps = p[0:6]
    t2, f0, j_cpl = p[6], p[7], p[8]
    phis = p[9:15]
    slope, offset = p[15], p[16]
    offsets = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]) * j_cpl

    total = np.zeros_like(f, dtype=np.float64)
    for amp, phi, shift in zip(amps, phis, offsets, strict=True):
        df = f0 + shift - f
        denom = 1.0 + df**2 * t2**2
        total = total + (
            np.cos(phi) * (amp * t2 / denom) - np.sin(phi) * (amp * df * t2**2 / denom)
        )

    result: NDArray[np.float64] = total + slope * (f0 - f) + offset
    return result
