"""NIfTI-MRS file reader using nibabel."""

import json

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct


def _extract_nifti_mrs_extension(img: nib.Nifti1Image) -> dict[str, object]:
    """Extract and decode the JSON header extension from a NIfTI-MRS image."""
    for ext in img.header.extensions:
        content = ext.get_content()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore").rstrip("\x00")
        else:
            text = str(content).rstrip("\x00")
        if text.strip().startswith("{"):
            parsed: dict[str, object] = json.loads(text)
            return parsed
    raise ValueError("No JSON header extension found in NIfTI-MRS file.")


def _parse_hdr_ext(hdr_ext: dict[str, object], dims: dict[str, int]) -> dict[str, int]:
    """Map NIfTI-MRS dim_5/dim_6/dim_7 tags to logical dimension indices."""
    dim_map = {
        "DIM_COIL": "coils",
        "DIM_DYN": "averages",
        "DIM_INDIRECT_0": "extras",
        "DIM_INDIRECT_1": "extras",
        "DIM_INDIRECT_2": "extras",
        "DIM_PHASE_CYCLE": "extras",
        "DIM_EDIT": "subSpecs",
        "DIM_MEAS": "extras",
        "DIM_USER_0": "extras",
        "DIM_USER_1": "extras",
        "DIM_USER_2": "extras",
        "DIM_ISIS": "subSpecs",
    }
    for dim_num, key in ((5, "dim_5"), (6, "dim_6"), (7, "dim_7")):
        if key in hdr_ext:
            raw_val = hdr_ext[key]
            tag = raw_val[0] if isinstance(raw_val, list) else str(raw_val)
            if tag not in dim_map:
                raise ValueError(f"Unknown dimension value specified in {key}: {tag}")
            dims[dim_map[tag]] = dim_num
    return dims


def _specify_dims(
    shape: tuple[int, ...],
    hdr_ext: dict[str, object],
    fids: NDArray[np.complex128],
) -> tuple[dict[str, int], NDArray[np.complex128], int]:
    """Squeeze singleton spatial axes and reorder NIfTI-MRS dimensions."""
    dims = {
        "x": 1,
        "y": 2,
        "z": 3,
        "t": 4,
        "coils": 0,
        "averages": 0,
        "subSpecs": 0,
        "extras": 0,
    }
    dims = _parse_hdr_ext(hdr_ext, dims)
    if len(shape) < 4 or int(np.prod(shape[:3])) != 1:
        raise ValueError("Data are not single-voxel data.")

    npoints = int(shape[dims["t"] - 1])
    fids_sqz = np.squeeze(fids)

    sqz_dims: list[str] = []
    for name in ("t", "coils", "averages", "subSpecs", "extras"):
        if dims[name] != 0:
            dims[name] -= 3
            sqz_dims.append(name)
    dims["x"] = dims["y"] = dims["z"] = 0

    order_names = [
        name
        for name in ("t", "coils", "averages", "subSpecs", "extras")
        if dims[name] != 0
    ]
    perm = [dims[name] - 1 for name in order_names]
    if fids_sqz.ndim == len(perm):
        fids_sqz = np.transpose(fids_sqz, axes=perm)
    for new_pos, name in enumerate(order_names, start=1):
        dims[name] = new_pos

    result = (dims, fids_sqz, npoints)
    return result


def _gls_coil_combine(
    fids_3d: NDArray[np.complex128], s_vec: NDArray[np.complex128] | None = None
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Combine multi-coil FIDs of shape ``(nCh, nPts, nReps)`` using GLS."""
    if fids_3d.ndim == 2:
        fids_3d = fids_3d[:, :, np.newaxis]
    n_ch, n_pts, n_reps = fids_3d.shape
    noise_start = int(np.ceil(0.75 * n_pts)) - 1
    noise_pts = np.zeros(n_pts, dtype=bool)
    noise_pts[noise_start:] = True
    noise_mask = np.tile(noise_pts, n_reps)
    tmp = fids_3d.reshape((n_ch, n_pts * n_reps), order="F")
    e_mat = tmp[:, noise_mask]
    psi = e_mat @ e_mat.conj().T
    if s_vec is None:
        s_vec = np.mean(fids_3d, axis=2)[:, 0]
    psi_inv_s = np.linalg.solve(psi, s_vec)
    denom = s_vec.conj().T @ psi_inv_s
    w = (1.0 / denom) * (s_vec.conj().T @ np.linalg.inv(psi))
    combined = np.sum(w[:, np.newaxis, np.newaxis] * fids_3d, axis=0)
    result = (np.squeeze(combined), s_vec)
    return result


def _set_p_val(
    p: MRSStruct, key: str, idx: int, val: float | list[float], cols: int = 0
) -> None:
    """Assign scan-indexed parameter into ``MRS_struct.p``."""
    if cols == 0:
        if key not in p or len(p[key]) <= idx:
            arr = np.zeros(idx + 1, dtype=np.float64)
            if key in p:
                arr[: len(p[key])] = p[key]
            p[key] = arr
        p[key][idx] = float(val)  # type: ignore[arg-type]
    else:
        if key not in p or p[key].shape[0] <= idx:
            mat = np.zeros((idx + 1, cols), dtype=np.float64)
            if key in p:
                mat[: p[key].shape[0], :] = p[key]
            p[key] = mat
        p[key][idx, :] = np.asarray(val, dtype=np.float64)


def nifti_mrs_read(
    mrs_struct: MRSStruct, fname: str, fname_w: str | None = None
) -> MRSStruct:
    """Read MRS data stored in NIfTI-MRS format (``.nii`` / ``.nii.gz``).

    Mirrors ``NIfTIMRSRead.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.
    fname: str
        Path to water-suppressed NIfTI-MRS file.
    fname_w: str | None
        Optional path to unsuppressed water NIfTI-MRS file.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with header fields and ``fids``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    nii = nib.load(fname)
    hdr_ext = _extract_nifti_mrs_extension(nii)
    fids_raw = np.asarray(nii.get_fdata(dtype=np.complex128))
    if np.isrealobj(fids_raw):
        fids_raw = np.asarray(nii.dataobj, dtype=np.complex128)
    zooms = nii.header.get_zooms()

    if "Manufacturer" in hdr_ext:
        nifti_sub = p.ensure_substruct("NIfTI")
        nifti_sub["manufacturer"] = str(hdr_ext["Manufacturer"])

    spec_freq = hdr_ext["SpectrometerFrequency"]
    larmor = float(spec_freq[0] if isinstance(spec_freq, list) else spec_freq)
    _set_p_val(p, "LarmorFreq", ii, larmor)
    _set_p_val(p, "sw", ii, 1.0 / float(zooms[3]))
    _set_p_val(p, "TE", ii, float(hdr_ext["EchoTime"]) * 1e3)
    _set_p_val(p, "TR", ii, float(hdr_ext["RepetitionTime"]) * 1e3)
    _set_p_val(
        p, "voxdim", ii, [float(zooms[0]), float(zooms[1]), float(zooms[2])], cols=3
    )

    dims, fids, npoints = _specify_dims(fids_raw.shape, hdr_ext, fids_raw)
    sz = fids.shape
    _set_p_val(p, "npoints", ii, npoints)
    if dims["averages"] and dims["subSpecs"]:
        n_avg = sz[dims["averages"] - 1] * sz[dims["subSpecs"] - 1]
        if dims["coils"]:
            fids = fids.reshape((sz[0], sz[1], n_avg), order="F")
        else:
            fids = fids.reshape((sz[0], n_avg), order="F")
    elif fids.ndim == 2 and sz[1] == 1:
        n_avg = 1
    elif dims["averages"]:
        n_avg = sz[dims["averages"] - 1]
    else:
        n_avg = sz[1] if fids.ndim > 1 else 1
    _set_p_val(p, "Navg", ii, n_avg)

    if dims["coils"]:
        fids = np.transpose(fids, (1, 0, 2))

    fids_w = None
    dims_w: dict[str, int] = {}
    if fname_w is not None:
        nii_w = nib.load(fname_w)
        hdr_w_ext = _extract_nifti_mrs_extension(nii_w)
        zooms_w = nii_w.header.get_zooms()
        fids_w_raw = np.asarray(nii_w.dataobj, dtype=np.complex128)
        _set_p_val(p, "sw_water", ii, 1.0 / float(zooms_w[3]))
        _set_p_val(p, "TE_water", ii, float(hdr_w_ext["EchoTime"]) * 1e3)
        _set_p_val(p, "TR_water", ii, float(hdr_w_ext["RepetitionTime"]) * 1e3)
        dims_w, fids_w, npts_w = _specify_dims(fids_w_raw.shape, hdr_w_ext, fids_w_raw)
        _set_p_val(p, "npoints_water", ii, npts_w)
        sz_w = fids_w.shape
        if dims_w["averages"] and dims_w["subSpecs"]:
            n_wavg = sz_w[dims_w["averages"] - 1] * sz_w[dims_w["subSpecs"] - 1]
            if dims_w["coils"]:
                fids_w = fids_w.reshape((sz_w[0], sz_w[1], n_wavg), order="F")
            else:
                fids_w = fids_w.reshape((sz_w[0], n_wavg), order="F")
        elif fids_w.ndim == 2 and sz_w[1] == 1:
            n_wavg = 1
        elif dims_w["averages"]:
            n_wavg = sz_w[dims_w["averages"] - 1]
        else:
            n_wavg = 1
        _set_p_val(p, "Nwateravg", ii, n_wavg)
        if dims["coils"] and fids_w.ndim == 3:
            fids_w = np.transpose(fids_w, (1, 0, 2))

    fids_struct = mrs_struct.ensure_substruct("fids")
    s_vec: NDArray[np.complex128] | None = None
    if dims["coils"]:
        if fids_w is not None:
            combined_w, s_vec = _gls_coil_combine(fids_w)
            fids_struct["data_water"] = (
                np.mean(combined_w, axis=1) if combined_w.ndim == 2 else combined_w
            )
        combined_m, _ = _gls_coil_combine(fids, s_vec=s_vec)
        fids_struct["data"] = (
            combined_m[:, np.newaxis] if combined_m.ndim == 1 else combined_m
        )
    else:
        fids_struct["data"] = fids[:, np.newaxis] if fids.ndim == 1 else fids
        if fids_w is not None:
            if dims_w.get("averages", 0) > 0 and fids_w.ndim > 1:
                fids_struct["data_water"] = np.mean(fids_w, axis=1)
            else:
                fids_struct["data_water"] = fids_w.ravel()

    manuf = str(hdr_ext.get("Manufacturer", "")).lower()
    if manuf in ("ge", "siemens") and dims["averages"] and dims["subSpecs"]:
        n_sub_s = sz[dims["subSpecs"] - 1]
        if n_sub_s >= 2:
            n_cols = fids_struct["data"].shape[1]
            ind = np.arange(n_cols).reshape((n_cols // n_sub_s, n_sub_s)).T.ravel()
            fids_struct["data"] = fids_struct["data"][:, ind]
    elif manuf == "philips":
        d_mat = fids_struct["data"]
        n_cols = d_mat.shape[1]
        corrph = np.tile(np.array([-1.0, 1.0]), max(n_cols // 2, 1))[:n_cols]
        d_mat = d_mat * corrph[np.newaxis, :]
        if bool(p.get("HERMES", 0)):
            phi = np.repeat(np.conj(d_mat[0, 1::2]) / np.abs(d_mat[0, 1::2]), 2)[
                :n_cols
            ]
            d_mat = d_mat * phi[np.newaxis, :]
        elif p["target"][0] == "GSH":
            m0 = np.mean(d_mat[0, :])
            d_mat = d_mat * (np.conj(m0) / np.abs(m0))
        else:
            d_mat = d_mat * (np.conj(d_mat[0, :]) / np.abs(d_mat[0, :]))[np.newaxis, :]
        fids_struct["data"] = d_mat
        if "data_water" in fids_struct:
            dw = fids_struct["data_water"]
            fids_struct["data_water"] = dw * np.conj(dw[0]) / np.abs(dw[0])

    return mrs_struct
