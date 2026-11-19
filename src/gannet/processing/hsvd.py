"""Hankel Singular Value Decomposition (HSVD) residual water removal."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def water_removal_hsvd(
    fid: ArrayLike,
    sw_khz: float,
    n_hsvd: int = 8,
    hsvd_low: float = -0.08,
    hsvd_high: float = 0.08,
    plotting_on: int = 0,
    use_points: int = 2048,
) -> NDArray[np.complex128]:
    """Remove residual water resonance from a time-domain FID using HSVD.

    Mirrors ``WaterRemovalHSVD.m`` (DataHandling = 3).

    Parameters
    ------------
    fid: ArrayLike
        Complex 1D time-domain FID array.
    sw_khz: float
        Spectral width in kHz (e.g., ``sw_hz / 1000.0``).
    n_hsvd: int
        Number of singular values / decaying exponential components.
        Defaults to 8.
    hsvd_low: float
        Lower bound of water removal band in kHz. Defaults to -0.08.
    hsvd_high: float
        Upper bound of water removal band in kHz. Defaults to 0.08.
    plotting_on: int
        Unused compatibility flag (no interactive plotting). Defaults to 0.
    use_points: int
        Number of initial FID points used to construct the Hankel matrix.
        Defaults to 2048.

    Returns
    --------
        NDArray[np.complex128]: 1D complex numpy array
            Water-removed time-domain FID.
    """
    _ = plotting_on
    fid_orig = np.asarray(fid, dtype=np.complex128).ravel()
    n_orig = fid_orig.size
    n_svd = min(int(use_points), n_orig)
    fid_svd = fid_orig[:n_svd]
    n_nonzero = fid_svd.size

    tacq = n_nonzero / (1000.0 * sw_khz)
    dt = tacq / n_nonzero

    l_max = round(0.4 * n_nonzero)
    m_max = n_nonzero + 1 - l_max
    l_idx = np.arange(l_max)[:, np.newaxis]
    m_idx = np.arange(m_max)[np.newaxis, :]
    hankel = fid_svd[l_idx + m_idx]

    u_mat, _, _ = np.linalg.svd(hankel, full_matrices=False)
    n_comp = min(int(n_hsvd), u_mat.shape[1])
    u_tr = u_mat[:, :n_comp]
    u_up = u_tr[1:, :]
    u_down = u_tr[:-1, :]

    z_mat = np.linalg.pinv(u_down) @ u_up
    q_vals = np.log(np.linalg.eigvals(z_mat))

    frq = np.imag(q_vals) / (2.0 * np.pi * dt)
    decay = np.real(q_vals) / dt

    time_full = np.arange(n_orig, dtype=np.float64) * dt
    basis = np.exp(
        time_full[:, np.newaxis]
        * (decay[np.newaxis, :] + 2j * np.pi * frq[np.newaxis, :])
    )
    amp_complex = np.linalg.pinv(basis) @ fid_orig
    amp = np.abs(amp_complex)
    phs = np.arctan2(np.imag(amp_complex), np.real(amp_complex))

    water_pos = np.where((frq > -1000.0 * hsvd_high) & (frq < -1000.0 * hsvd_low))[0]

    fid_water = np.zeros(n_orig, dtype=np.complex128)
    for idx in water_pos:
        fid_water = fid_water + (
            amp[idx]
            * np.exp(2j * np.pi * frq[idx] * time_full)
            * np.exp(time_full * decay[idx])
            * np.exp(1j * phs[idx])
        )

    result: NDArray[np.complex128] = fid_orig - fid_water
    return result
