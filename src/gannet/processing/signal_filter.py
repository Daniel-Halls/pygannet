"""Spectral baseline modeling and transient lipid/water filtering."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.signal import hilbert
from scipy.sparse.linalg import spsolve

from gannet.config import MRSStruct


def _upcoef2(
    lo: NDArray[np.float64], hi: NDArray[np.float64], precis: int
) -> NDArray[np.float64]:
    """Iteratively upsample and convolve Haar filter coefficients."""
    out = hi.copy()
    for _ in range(1, precis):
        up = np.zeros(2 * out.size - 1, dtype=np.float64)
        up[0::2] = out
        out = np.convolve(up, lo, mode="full")
    return out


def _keep_vec(arr: NDArray[np.float64], target_len: int) -> NDArray[np.float64]:
    """Extract centered slice of length ``target_len`` from ``arr``."""
    if target_len < 0 or target_len >= arr.size:
        return arr
    delta = (arr.size - target_len) / 2.0
    first = int(np.floor(delta))
    last = arr.size - int(np.ceil(delta))
    result: NDArray[np.float64] = arr[first:last]
    return result


def cwt2(y: ArrayLike, scale: float) -> NDArray[np.float64]:
    """Compute continuous wavelet transform derivative used in baseline detection.

    Mirrors ``cwt2`` in ``SignalFilter.m``.

    Parameters
    ------------
    y: ArrayLike
        1D real input spectrum.
    scale: float
        Wavelet scale factor (typically 10).

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Wavelet transform coefficients matching ``y`` length.
    """
    precis = 10
    coef = np.sqrt(2.0) ** precis
    pas = 1.0 / (2**precis)
    lo = np.array([np.sqrt(2.0) * 0.5, np.sqrt(2.0) * 0.5], dtype=np.float64)
    hi = lo * np.array([1.0, -1.0], dtype=np.float64)
    nbpts = round((lo.size - 1) / pas + 2)

    psi = coef * _upcoef2(lo, hi, precis)
    psi = np.concatenate(([0.0], psi, [0.0]))
    x = np.linspace(0.0, (nbpts - 1) * pas, nbpts)
    step = x[1] - x[0]
    wav = np.cumsum(psi) * step
    x = x - x[0]

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    j = np.floor(np.arange(0.0, scale * x[-1] + 1e-12, 1.0) / (scale * step))
    j_int = j.astype(np.int64)
    if j_int.size <= 1:
        j_int = np.array([0, 0], dtype=np.int64)
    j_int = np.clip(j_int, 0, wav.size - 1)
    filt = wav[j_int][::-1]

    conv = np.convolve(y_arr, filt, mode="full")
    wy: NDArray[np.float64] = -np.sqrt(scale) * _keep_vec(np.diff(conv), y_arr.size)
    return wy


def whittaker(y: ArrayLike, diff_order: int, lam: float) -> NDArray[np.float64]:
    """Apply Whittaker penalized least-squares smoother (Eilers, 2003).

    Mirrors ``whittaker`` in ``SignalFilter.m``.

    Parameters
    ------------
    y: ArrayLike
        1D signal vector with zeros where peaks were masked out.
    diff_order: int
        Difference penalty order (typically 2).
    lam: float
        Smoothing parameter lambda.

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Smoothed baseline vector.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    m = y_arr.size
    if m <= diff_order:
        return y_arr.copy()

    eye = sparse.eye(m, format="csc")
    d_mat = eye
    for _ in range(diff_order):
        d_mat = d_mat[1:, :] - d_mat[:-1, :]
    w_vec = (y_arr != 0).astype(np.float64)
    w_mat = sparse.diags(w_vec, 0, shape=(m, m), format="csc")
    system_mat = w_mat + lam * (d_mat.T @ d_mat)
    z: NDArray[np.float64] = np.asarray(
        spsolve(system_mat, w_vec * y_arr), dtype=np.float64
    )
    return z


def baseline_modeling(
    spec_in: ArrayLike,
    lipid_flag: bool,
    water_flag: bool,
    mrs_struct: MRSStruct,
) -> dict[str, NDArray[np.float64]]:
    """Model spectral baseline including optional lipid and water regions.

    Parameters
    ------------
    spec_in: ArrayLike
        1D complex frequency-domain spectrum.
    lipid_flag: bool
        Whether unstable lipid contamination was detected.
    water_flag: bool
        Whether unstable residual water was detected.
    mrs_struct: MRSStruct
        MRS data structure containing ``p.sw``, ``p.LarmorFreq``, and ``ii``.

    Returns
    --------
        dict[str, NDArray[np.float64]]: dict object
            Dictionary with keys ``'real'`` and ``'imag'`` for complex baseline.
    """
    y = np.asarray(spec_in, dtype=np.complex128).ravel()
    wy = np.abs(cwt2(np.real(y), 10.0)) ** 2

    ii = int(mrs_struct["ii"])
    sw = float(mrs_struct["p"]["sw"][ii])
    larmor = float(mrs_struct["p"]["LarmorFreq"][ii])
    freq_range = sw / larmor
    n_wy = wy.size
    freq = (
        (n_wy + 1 - np.arange(1, n_wy + 1)) / n_wy * freq_range
        + 4.68
        - freq_range / 2.0
    )
    noise_lim = (freq <= 9.0) & (freq >= 8.0)
    sigma = float(np.std(wy[noise_lim], ddof=1)) if np.any(noise_lim) else 0.0

    baseline = np.zeros(n_wy, dtype=np.float64)
    k = 3.0
    for idx in range(max(n_wy - 4, 0)):
        win = slice(idx, idx + 5)
        h_val = float(np.max(wy[win]) - np.min(wy[win]))
        if h_val < k * sigma:
            baseline[win] = wy[win]

    lipid_lim = (freq <= 1.85) & (freq >= -2.0)
    water_lim = (freq <= 5.5) & (freq >= 3.5)
    if lipid_flag:
        baseline[lipid_lim] = wy[lipid_lim]
    if water_flag:
        baseline[water_lim] = wy[water_lim]

    z_real = np.real(y).copy()
    z_real[baseline == 0] = 0.0
    z_lipids = (
        whittaker(z_real[lipid_lim], 2, 10.0)
        if lipid_flag and np.any(lipid_lim)
        else None
    )
    z_water = (
        whittaker(z_real[water_lim], 2, 0.2)
        if water_flag and np.any(water_lim)
        else None
    )
    z_real = whittaker(z_real, 2, 1e3)
    if z_lipids is not None:
        z_real[lipid_lim] = z_lipids
    if z_water is not None:
        z_real[water_lim] = z_water

    z_imag = -np.imag(hilbert(z_real))
    result: dict[str, NDArray[np.float64]] = {"real": z_real, "imag": z_imag}
    return result


def signal_filter(
    spec_in: ArrayLike,
    lipid_flag: bool,
    water_flag: bool,
    transient_idx: int,
    mrs_struct: MRSStruct,
) -> NDArray[np.complex128]:
    """Filter unstable lipid and residual water signals from a single transient.

    Mirrors ``SignalFilter.m``.

    Parameters
    ------------
    spec_in: ArrayLike
        1D complex frequency-domain spectrum for one transient.
    lipid_flag: bool
        Whether unstable lipid contamination was detected.
    water_flag: bool
        Whether unstable residual water was detected.
    transient_idx: int
        Transient index used to seed deterministic Gaussian noise replacement.
    mrs_struct: MRSStruct
        MRS data structure containing acquisition parameters.

    Returns
    --------
        NDArray[np.complex128]: 1D complex numpy array
            Filtered time-domain FID for the transient.
    """
    spec_arr = np.asarray(spec_in, dtype=np.complex128).ravel()
    z = baseline_modeling(spec_arr, lipid_flag, water_flag, mrs_struct)
    y_real = np.real(spec_arr) - z["real"]
    y_imag = np.imag(spec_arr) - z["imag"]

    if water_flag:
        ii = int(mrs_struct["ii"])
        npoints = int(mrs_struct["p"]["npoints"][ii])
        sw = float(mrs_struct["p"]["sw"][ii])
        larmor = float(mrs_struct["p"]["LarmorFreq"][ii])
        freq_range = sw / larmor
        freq = (
            (npoints + 1 - np.arange(1, npoints + 1)) / npoints * freq_range
            + 4.68
            - freq_range / 2.0
        )
        water_lim = (freq <= 5.5) & (freq >= 3.5)
        noise_lim = (freq <= min(float(np.max(freq)), 12.0)) & (freq >= 8.0)

        rng = np.random.default_rng(int(transient_idx) + 1)
        n_w = int(np.sum(water_lim))
        if np.any(noise_lim) and n_w > 0:
            y_real[water_lim] = rng.standard_normal(n_w) * float(
                np.std(y_real[noise_lim], ddof=1)
            )
            y_imag[water_lim] = rng.standard_normal(n_w) * float(
                np.std(y_imag[noise_lim], ddof=1)
            )

    out: NDArray[np.complex128] = np.fft.ifft(np.fft.fftshift(y_real + 1j * y_imag))
    return out
