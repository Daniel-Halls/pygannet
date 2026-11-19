"""Frequency-domain lineshape models matching Gannet MATLAB definitions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gauss_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a single Gaussian peak with a linear baseline.

    Parameters
    ------------
    x: ArrayLike
        Parameter vector ``[amp, width, f0, slope, offset]`` where
        ``width = -1 / (2 * sigma**2)``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    result: NDArray[np.float64] = (
        p[0] * np.exp(p[1] * (f - p[2]) ** 2) + p[3] * (f - p[2]) + p[4]
    )
    return result


def double_gauss_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a double-Gaussian peak model with a linear baseline.

    Parameters
    ------------
    x: ArrayLike
        Parameter vector ``[a1, w1, f1, a2, w2, f2, slope, offset]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    result: NDArray[np.float64] = (
        p[0] * np.exp(p[1] * (f - p[2]) ** 2)
        + p[3] * np.exp(p[4] * (f - p[5]) ** 2)
        + p[6] * (f - p[2])
        + p[7]
    )
    return result


def five_gauss_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a five-Gaussian model with a quadratic baseline.

    Parameters
    ------------
    x: ArrayLike
        18-element parameter vector for five Gaussians and quadratic baseline.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    result: NDArray[np.float64] = (
        p[0] * np.exp(p[1] * (f - p[2]) ** 2)
        + p[3] * np.exp(p[4] * (f - p[5]) ** 2)
        + p[6] * np.exp(p[7] * (f - p[8]) ** 2)
        + p[9] * np.exp(p[10] * (f - p[11]) ** 2)
        + p[12] * np.exp(p[13] * (f - p[14]) ** 2)
        + p[15]
        + p[16] * (f - p[2])
        + p[17] * (f - p[2]) ** 2
    )
    return result


def six_gauss_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a six-Gaussian model with a quadratic baseline.

    Parameters
    ------------
    x: ArrayLike
        21-element parameter vector for six Gaussians and quadratic baseline.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    result: NDArray[np.float64] = (
        p[0] * np.exp(p[1] * (f - p[2]) ** 2)
        + p[3] * np.exp(p[4] * (f - p[5]) ** 2)
        + p[6] * np.exp(p[7] * (f - p[8]) ** 2)
        + p[9] * np.exp(p[10] * (f - p[11]) ** 2)
        + p[12] * np.exp(p[13] * (f - p[14]) ** 2)
        + p[15] * np.exp(p[16] * (f - p[17]) ** 2)
        + p[18]
        + p[19] * (f - p[2])
        + p[20] * (f - p[2]) ** 2
    )
    return result


def gaba_glx_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the combined GABA+Glx three-Gaussian and sinusoidal baseline model.

    Parameters
    ------------
    x: ArrayLike
        12-element parameter vector for two Glx peaks, one GABA peak, linear
        baseline slope, and sine/cosine baseline terms.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    result: NDArray[np.float64] = (
        p[0] * np.exp(p[1] * (f - p[2]) ** 2)
        + p[3] * np.exp(p[4] * (f - p[5]) ** 2)
        + p[6] * np.exp(p[7] * (f - p[8]) ** 2)
        + p[9] * (f - p[2])
        + p[10] * np.sin(np.pi * f / 1.31 / 4.0)
        + p[11] * np.cos(np.pi * f / 1.31 / 4.0)
    )
    return result


def baseline_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a linear baseline model ``x[1] * (freq - x[0]) + x[2]``.

    Parameters
    ------------
    x: ArrayLike
        Parameter vector ``[f0, slope, offset]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated baseline values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    result: NDArray[np.float64] = p[1] * (f - p[0]) + p[2]
    return result


def lorentz_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a single phased Lorentzian peak with a linear baseline.

    Parameters
    ------------
    x: ArrayLike
        Parameter vector ``[area, hwhm, f0, phase, baseline0, baseline1]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    area, hwhm, f0, phase, baseline0, baseline1 = p[:6]
    denom = (f - f0) ** 2 + hwhm**2
    absorption = (1.0 / (2.0 * np.pi)) * area * hwhm / denom
    dispersion = (1.0 / (2.0 * np.pi)) * area * (f - f0) / denom
    result: NDArray[np.float64] = (
        np.cos(phase) * absorption
        + np.sin(phase) * dispersion
        + baseline0
        + baseline1 * (f - f0)
    )
    return result


def two_lorentz_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a double-Lorentzian model (e.g., Cr + Cho) with linear baseline.

    Parameters
    ------------
    x: ArrayLike
        7-element parameter vector ``[area, hwhm, f0, phase, b0, b1, ratio]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    area, hwhm, f0, phase, baseline0, baseline1, ratio = p[:7]
    d1 = (f - f0) ** 2 + hwhm**2
    d2 = (f - f0 - 0.18) ** 2 + hwhm**2
    absorption = (1.0 / (2.0 * np.pi)) * area * hwhm / d1 + (
        1.0 / (2.0 * np.pi)
    ) * area * ratio * hwhm / d2
    dispersion = (1.0 / (2.0 * np.pi)) * area * (f - f0) / d1 + (
        1.0 / (2.0 * np.pi)
    ) * area * ratio * (f - f0 - 0.18) / d2
    result: NDArray[np.float64] = (
        np.cos(phase) * absorption
        + np.sin(phase) * dispersion
        + baseline0
        + baseline1 * (f - f0)
    )
    return result


def three_lorentz_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a three-Lorentzian multiplet model with phase and quadratic baseline.

    Based on Marshall & Roe, Anal Chem, 1978;50(6):756-763.

    Parameters
    ------------
    x: ArrayLike
        10-element parameter vector ``[H, a, b, T2, f0, J, phi, M1, M2, B]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    amp, a, b, t2, f0, j_cpl, phi, m1, m2, offset = p[:10]

    df0 = f0 - f
    dfa = f0 + j_cpl - f
    dfb = f0 - j_cpl - f

    a0 = np.cos(phi) * ((amp * t2) / (1.0 + df0**2 * t2**2)) - np.sin(phi) * (
        (amp * df0 * t2**2) / (1.0 + df0**2 * t2**2)
    )
    aa = np.cos(phi) * ((amp * a * t2) / (1.0 + dfa**2 * t2**2)) - np.sin(phi) * (
        (amp * a * dfa * t2**2) / (1.0 + dfa**2 * t2**2)
    )
    ab = np.cos(phi) * ((amp * b * t2) / (1.0 + dfb**2 * t2**2)) - np.sin(phi) * (
        (amp * b * dfb * t2**2) / (1.0 + dfb**2 * t2**2)
    )

    result: NDArray[np.float64] = a0 + aa + ab + m1 * df0 + m2 * df0**2 + offset
    return result


def lorentz_gauss_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a Lorentzian-multiplied-by-Gaussian lineshape without phase.

    Parameters
    ------------
    x: ArrayLike
        6-element parameter vector ``[amp, inv_hwhm, f0, slope, offset, g_w]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    df = f - p[2]
    result: NDArray[np.float64] = (
        (p[0] / (p[1] ** 2 * df**2 + 1.0)) * np.exp(p[5] * df**2) + p[3] * df + p[4]
    )
    return result


def lorentz_gauss_model_p(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a phased Lorentzian-multiplied-by-Gaussian lineshape.

    Parameters
    ------------
    x: ArrayLike
        7-element parameter vector
        ``[amp, inv_hwhm, f0, slope, offset, g_w, phase]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    df = f - p[2]
    num = np.cos(p[6]) * p[0] + np.sin(p[6]) * p[0] * p[1] * df
    den = p[1] ** 2 * df**2 + 1.0
    result: NDArray[np.float64] = (num / den) * np.exp(p[5] * df**2) + p[3] * df + p[4]
    return result


def etoh_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the dual-Lorentzian ethanol model with linear baseline.

    Parameters
    ------------
    x: ArrayLike
        8-element parameter vector ``[a1, f1, w1, a2, f2, w2, slope, offset]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    l1 = p[0] / (1.0 + ((f - p[1]) / (p[2] / 2.0)) ** 2)
    l2 = p[3] / (1.0 + ((f - p[4]) / (p[5] / 2.0)) ** 2)
    baseline = p[6] * (f - p[2]) + p[7]
    result: NDArray[np.float64] = l1 + l2 + baseline
    return result


def lac_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the Lactate + BHB + MM3 multi-Gaussian model.

    Parameters
    ------------
    x: ArrayLike
        10-element parameter vector ``[a_lac, w_lac, f_lac, a_bhb, w_bhb,
        f_bhb, offset, slope, quad, a_mm3]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    result: NDArray[np.float64] = (
        p[0] * np.exp(p[1] * (f - p[2]) ** 2)
        + p[0] * np.exp(p[1] * (f - (p[2] - 0.055)) ** 2)
        + p[3] * np.exp(p[4] * (f - p[5]) ** 2)
        + p[9] * np.exp(p[4] * (f - (p[5] + 0.21)) ** 2)
        + p[6]
        + p[7] * (f - p[2])
        + p[8] * (f - p[2]) ** 2
    )
    return result


def voigt_model(x: ArrayLike, freq: ArrayLike) -> NDArray[np.float64]:
    """Evaluate a pseudo-Voigt lineshape model with linear baseline.

    Based on Marshall et al., Magn Reson Med. 1997;37(5):651-657.

    Parameters
    ------------
    x: ArrayLike
        6-element parameter vector ``[r, a, f0, width, slope, offset]``.
    freq: ArrayLike
        Frequency vector in ppm.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Evaluated model values at ``freq``.
    """
    p = np.asarray(x, dtype=np.float64)
    f = np.asarray(freq, dtype=np.float64)
    r, amp, f0, width, slope, offset = p[:6]
    lorentz = amp / (1.0 + ((f - f0) / (width / 2.0)) ** 2)
    gauss = amp * np.exp(-np.log(2.0) * ((f - f0) / (width / 2.0)) ** 2)
    mix = np.exp(-np.abs(r))
    result: NDArray[np.float64] = (
        mix * lorentz + (1.0 - mix) * gauss + slope * (f - f0) + offset
    )
    return result
