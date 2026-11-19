"""Philips SDAT/SPAR file readers."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.io.vax import fread_vaxg


def sdat_read(filename: str, da_xres: int) -> NDArray[np.complex128]:
    """Read a 1D complex FID from a Philips ``.sdat`` file.

    Mirrors ``SDATread.m``.

    Parameters
    ------------
    filename: str
        Path to the ``.sdat`` file.
    da_xres: int
        Number of complex data points.

    Returns
    --------
        NDArray[np.complex128]: 1D complex numpy array
            Complex time-domain FID of length ``da_xres``.
    """
    total_points = int(da_xres) * 2
    with open(filename, "rb") as fid:
        raw_data = fread_vaxg(fid, total_points, "float")
    temp = raw_data.reshape((2, int(da_xres)), order="F")
    fid_complex: NDArray[np.complex128] = temp[0, :] + 1j * temp[1, :]
    return fid_complex


def sdat_read_mega(filename: str, da_xres: int, da_yres: int) -> NDArray[np.complex128]:
    """Read 2D complex FIDs (points x averages) from a Philips ``.sdat`` file.

    Mirrors ``SDATreadMEGA.m``.

    Parameters
    ------------
    filename: str
        Path to the ``.sdat`` file.
    da_xres: int
        Number of points per FID.
    da_yres: int
        Number of averages/rows.

    Returns
    --------
        NDArray[np.complex128]: 2D complex numpy array
            Complex FIDs of shape ``(da_xres, da_yres)``.
    """
    total_points = int(da_xres) * int(da_yres) * 2
    with open(filename, "rb") as fid:
        raw_data = fread_vaxg(fid, total_points, "float")
    temp = raw_data.reshape((2, int(da_xres), int(da_yres)), order="F")
    fids: NDArray[np.complex128] = temp[0, :, :] + 1j * temp[1, :, :]
    if fids.ndim == 1:
        fids = fids[:, np.newaxis]
    return fids


def parse_spar_header(spar_path: str) -> dict[str, str]:
    """Parse key-value pairs from a Philips ``.spar`` text header file.

    Parameters
    ------------
    spar_path: str
        Path to the ``.spar`` or ``.SPAR`` file.

    Returns
    --------
        dict[str, str]: dict object
            Mapping of SPAR parameter names to string values.
    """
    parsed: dict[str, str] = {}
    with open(spar_path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("!") or ":" not in stripped:
                continue
            key, val = stripped.split(":", 1)
            parsed[key.strip()] = val.strip()
    return parsed


def _resolve_spar_path(sdat_filename: str) -> str:
    """Resolve matching ``.spar`` or ``.SPAR`` path for a ``.sdat`` file."""
    path = Path(sdat_filename)
    ext = path.suffix
    spar_ext = ".SPAR" if ext[1:].isupper() else ".spar"
    candidate = path.with_suffix(spar_ext)
    if candidate.exists():
        return str(candidate)
    alt = path.with_suffix(".spar" if spar_ext == ".SPAR" else ".SPAR")
    if alt.exists():
        return str(alt)
    return str(candidate)


def _ensure_param_array(
    p: MRSStruct, key: str, idx: int, value: float | list[float], cols: int = 0
) -> None:
    """Ensure parameter vector or 2D matrix in ``p`` holds ``value`` at ``idx``."""
    if cols == 0:
        if key not in p or len(p[key]) <= idx:
            arr = np.zeros(idx + 1, dtype=np.float64)
            if key in p:
                arr[: len(p[key])] = p[key]
            p[key] = arr
        p[key][idx] = float(value)  # type: ignore[arg-type]
    else:
        if key not in p or p[key].shape[0] <= idx:
            mat = np.zeros((idx + 1, cols), dtype=np.float64)
            if key in p:
                mat[: p[key].shape[0], :] = p[key]
            p[key] = mat
        p[key][idx, :] = np.asarray(value, dtype=np.float64)


def philips_read(
    mrs_struct: MRSStruct, fname: str, fname_water: str | None = None
) -> MRSStruct:
    """Load Philips SDAT/SPAR MRS data and optional water reference.

    Mirrors ``PhilipsRead.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure with current scan index ``ii``.
    fname: str
        Path to the water-suppressed ``.sdat`` file.
    fname_water: str | None
        Optional path to the unsuppressed water ``.sdat`` file.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with header parameters and ``fids``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    spar_path = _resolve_spar_path(fname)
    hdr = parse_spar_header(spar_path)

    npoints = int(float(hdr["samples"]))
    nrows = int(float(hdr["rows"]))
    _ensure_param_array(p, "npoints", ii, npoints)
    _ensure_param_array(p, "nrows", ii, nrows)
    _ensure_param_array(p, "TR", ii, float(hdr["repetition_time"]))
    _ensure_param_array(p, "TE", ii, float(hdr["echo_time"]))
    _ensure_param_array(p, "LarmorFreq", ii, float(hdr["synthesizer_frequency"]) / 1e6)
    _ensure_param_array(p, "sw", ii, float(hdr["sample_frequency"]))

    _ensure_param_array(
        p,
        "voxdim",
        ii,
        [float(hdr["lr_size"]), float(hdr["ap_size"]), float(hdr["cc_size"])],
        cols=3,
    )
    _ensure_param_array(
        p,
        "voxoff",
        ii,
        [
            float(hdr["lr_off_center"]),
            float(hdr["ap_off_center"]),
            float(hdr["cc_off_center"]),
        ],
        cols=3,
    )
    _ensure_param_array(
        p,
        "voxang",
        ii,
        [
            float(hdr["lr_angulation"]),
            float(hdr["ap_angulation"]),
            float(hdr["cc_angulation"]),
        ],
        cols=3,
    )

    data = sdat_read_mega(fname, npoints, nrows)
    n_avg = data.shape[1]
    _ensure_param_array(p, "Navg", ii, n_avg)

    corrph = np.tile(np.array([-1.0, 1.0]), max(n_avg // 2, 1))
    if corrph.size < n_avg:
        corrph = np.resize(corrph, n_avg)
    data = data * corrph[np.newaxis, :]

    if bool(p.get("HERMES", 0)):
        first_even = data[0, 1::2]
        phi_half = np.conj(first_even) / np.abs(first_even)
        phi = np.repeat(phi_half, 2)
        if phi.size < n_avg:
            phi = np.resize(phi, n_avg)
        data = data * phi[np.newaxis, :]
    elif p["target"][0] == "GSH":
        mean_first = np.mean(data[0, :])
        data = data * (np.conj(mean_first) / np.abs(mean_first))
    else:
        data = data * (np.conj(data[0, :]) / np.abs(data[0, :]))[np.newaxis, :]

    fids = mrs_struct.ensure_substruct("fids")
    fids["data"] = np.conj(data)

    if fname_water is not None:
        p["Nwateravg"] = 1
        p["nrows_water"] = 1
        w_fid = sdat_read(fname_water, npoints)
        w_fid = w_fid * np.conj(w_fid[0]) / np.abs(w_fid[0])
        fids["data_water"] = w_fid

    return mrs_struct
