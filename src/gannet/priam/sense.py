"""PRIAM SENSE unfolding matrix calculation and dual-voxel reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct


def calc_unfolding_matrix(
    sensitivities: NDArray[np.complex128],
    voxel_centers_vox: list[tuple[int, int, int]],
    voxel_radii_vox: list[tuple[int, int, int]] | None = None,
    noise: NDArray[np.complex128] | None = None,
) -> dict[str, Any]:
    """Compute PRIAM SENSE unfolding matrix ``U`` and g-factors from sensitivity maps.

    Parameters
    ------------
    sensitivities: NDArray[np.complex128]
        Coil sensitivity maps shaped ``(n_coils, nx, ny, nz)``.
    voxel_centers_vox: list[tuple[int, int, int]]
        Center voxel indices ``(ix, iy, iz)`` per simultaneously acquired voxel.
    voxel_radii_vox: list[tuple[int, int, int]] | None
        Optional half-width radii in reference voxels around each center.
    noise: NDArray[np.complex128] | None
        Optional coil noise matrix shaped ``(n_coils, n_noise_samples)``.

    Returns
    --------
        dict[str, Any]:
            Dictionary containing ``U`` (unfolding matrix ``(n_voxels, n_coils)``),
            ``S`` (sensitivity matrix ``(n_coils, n_voxels)``), ``g_factors``,
            and ``psi`` (noise covariance matrix).
    """
    n_coils, nx, ny, nz = sensitivities.shape
    sens_vecs: list[NDArray[np.complex128]] = []
    for k, (cx, cy, cz) in enumerate(voxel_centers_vox):
        rx, ry, rz = voxel_radii_vox[k] if voxel_radii_vox else (1, 1, 1)
        x0, x1 = max(0, cx - rx), min(nx, cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(ny, cy + ry + 1)
        z0, z1 = max(0, cz - rz), min(nz, cz + rz + 1)
        patch = sensitivities[:, x0:x1, y0:y1, z0:z1]
        vec = np.mean(patch, axis=(1, 2, 3))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        sens_vecs.append(vec)

    s_mat = np.column_stack(sens_vecs)
    if noise is not None and noise.shape[0] == n_coils and noise.shape[1] > 1:
        psi = noise @ noise.conj().T
        psi += np.eye(n_coils, dtype=np.complex128) * (
            1e-8 * np.trace(psi).real / n_coils
        )
    else:
        psi = np.eye(n_coils, dtype=np.complex128)

    psi_inv = np.linalg.pinv(psi)
    gram = s_mat.conj().T @ psi_inv @ s_mat
    gram_inv = np.linalg.pinv(gram)
    u_mat = gram_inv @ s_mat.conj().T @ psi_inv
    g_factors = np.sqrt(np.real(np.diag(gram_inv) * np.diag(gram)))

    return {
        "U": u_mat,
        "S": s_mat,
        "psi": psi,
        "g_factors": g_factors,
    }


def sense_recon(mrs_struct: MRSStruct, ii: int) -> MRSStruct:
    """Apply PRIAM SENSE unfolding on loaded multi-coil data in ``MRSStruct``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` populated with PRIAM ``multivoxel`` sensitivities
        and raw coil signals (or ``fids.data``).
    ii: int
        0-based scan index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with unfolded ``multivoxel.allsignals`` and
            updated ``p.gfactors``.
    """
    mv = mrs_struct.get("multivoxel", {})
    if "U" in mv and mv["U"] is not None:
        u_mat = np.asarray(mv["U"], dtype=np.complex128)
        all_sig = mv.get("allsignals")
        if all_sig is not None and np.ndim(all_sig) == 4:
            # shape: (n_coils, n_pts, n_mixes, n_avg)
            _n_coils, n_pts, n_mixes, n_avg = all_sig.shape
            unf = np.zeros((u_mat.shape[0], n_pts, n_mixes, n_avg), dtype=np.complex128)
            for m in range(n_mixes):
                for a in range(n_avg):
                    unf[:, :, m, a] = u_mat @ all_sig[:, :, m, a]
            mv["allsignals"] = unf
            mrs_struct["p"]["NVoxels"] = unf.shape[0]
    return mrs_struct
