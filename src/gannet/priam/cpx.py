"""Philips CPX, SIN, and LAB/RAW noise reader for PRIAM reconstruction."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SinVoxelInfo:
    """Geometry and acquisition metadata extracted from a Philips ``.sin`` file.

    Parameters
    ------------
    voxdim: list[float]
        Voxel dimensions ``[ap, rl, fh]`` in mm.
    voxoff: list[float]
        Voxel offsets ``[ap, rl, fh]`` in mm (per slice/voxel).
    voxang: list[float]
        Voxel angulations ``[ap, rl, fh]`` in degrees.
    tr: float
        Repetition time in ms.
    te: float
        Echo time in ms.
    samples: int
        Number of samples per FID (after downsampling if paired).
    oversamples: int
        Number of oversampled samples per FID.
    """

    voxdim: list[float]
    voxoff: list[float]
    voxang: list[float]
    tr: float
    te: float
    samples: int
    oversamples: int


def get_sin_info(sin_filename: str, parameter: str) -> list[float]:
    """Read a numeric parameter vector from a Philips ``.sin`` file.

    Parameters
    ------------
    sin_filename: str
        Path to the ``.sin`` file (or prefix with ``.sin``/``.lab``/``.raw``).
    parameter: str
        Parameter key name inside ``.sin`` (e.g., ``voxel_sizes``,
        ``loc_ap_rl_fh_offcentres``, ``loc_ap_rl_fh_angs``).

    Returns
    --------
        list[float]:
            Parsed float values for the matching ``.sin`` entry, or empty list
            if the parameter or file is not found.
    """
    prefix = re.sub(r"\.(sin|lab|raw|cpx)$", "", sin_filename, flags=re.IGNORECASE)
    path = f"{prefix}.sin"
    if not os.path.exists(path):
        return []
    values: list[float] = []
    with open(path, "r", encoding="latin-1") as fid:
        for line in fid:
            if parameter in line and ":" in line:
                rhs = line.split(":", 1)[1].strip()
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", rhs)
                if nums:
                    values = [float(x) for x in nums]
    return values


def read_cpx_header(cpx_path: str) -> NDArray[np.float64]:
    """Read all per-image 512-byte headers from a Philips ``.cpx`` file.

    Parameters
    ------------
    cpx_path: str
        Path to the ``.cpx`` binary file.

    Returns
    --------
        NDArray[np.float64]:
            Array shaped ``(n_images, 19)`` containing stack, slice, coil,
            heart phase, echo, dynamic, segment, data offset, res_x, res_y,
            compression, flip, scaling factors, mix, prep_dir, seq_nr,
            segment2, and syncho number.
    """
    rows: list[list[float]] = []
    with open(cpx_path, "rb") as fid:
        fid.seek(0, os.SEEK_END)
        filesize = fid.tell()
        fid.seek(0, os.SEEK_SET)
        if filesize < 512:
            return np.empty((0, 19), dtype=np.float64)

        h1 = struct.unpack("<15i", fid.read(60))
        _factor = struct.unpack("<2f", fid.read(8))
        h2 = struct.unpack("<111i", fid.read(444))
        matrix_blocks = h1[12]
        offset = h1[9] if h2[25] == 0 else h2[25]
        stride = max(512, matrix_blocks * 512 + offset)

        idx = 0
        while True:
            hdr_offset = stride * idx
            if hdr_offset + 512 > filesize:
                break
            fid.seek(hdr_offset, os.SEEK_SET)
            h1 = struct.unpack("<15i", fid.read(60))
            factor = struct.unpack("<2f", fid.read(8))
            h2 = struct.unpack("<111i", fid.read(444))
            if h1[8] == 0:
                break
            img_offset = h1[9] if h2[25] == 0 else h2[25]
            rows.append(
                [
                    float(h1[1]),
                    float(h1[2]),
                    float(h2[1]),
                    float(h1[5]),
                    float(h1[4]),
                    float(h1[6]),
                    float(h1[7]),
                    float(img_offset),
                    float(h1[10]),
                    float(h1[11]),
                    float(h1[13]),
                    float(h2[110]),
                    float(factor[0]),
                    float(factor[1]),
                    float(h1[0]),
                    float(h1[3]),
                    float(h1[14]),
                    float(h2[0]),
                    float(h2[2]),
                ]
            )
            idx += 1

    if not rows:
        return np.empty((0, 19), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def read_cpx_image(
    cpx_path: str,
    header_row: NDArray[np.float64],
    img_index: int,
    stride: int,
    border: bool = False,
    flip_img: bool = True,
) -> NDArray[np.complex64]:
    """Read a single complex 2D image from a Philips ``.cpx`` file.

    Parameters
    ------------
    cpx_path: str
        Path to ``.cpx`` file.
    header_row: NDArray[np.float64]
        19-element row from ``read_cpx_header``.
    img_index: int
        0-based image record index in the CPX file.
    stride: int
        Record stride in bytes (header + data block size).
    border: bool
        Zero-pad non-square images to square if ``True``.
    flip_img: bool
        Flip along first dimension if ``True`` (scanner convention).

    Returns
    --------
        NDArray[np.complex64]:
            Complex 2D array shaped ``(res_x, res_y)`` (or square if bordered).
    """
    res_x = int(header_row[8])
    res_y = int(header_row[9])
    compression = int(header_row[10])
    scale1 = float(header_row[12])
    scale2 = float(header_row[13])
    data_offset = int(header_row[7])
    byte_pos = stride * img_index + data_offset

    n_vals = res_x * res_y * 2
    with open(cpx_path, "rb") as fid:
        fid.seek(byte_pos, os.SEEK_SET)
        if compression == 1:
            raw = np.fromfile(fid, dtype="<f4", count=n_vals).astype(np.float32)
        elif compression == 2:
            raw = np.fromfile(fid, dtype="<i2", count=n_vals).astype(np.float32)
            raw = scale2 + scale1 * raw
        elif compression == 4:
            raw = np.fromfile(fid, dtype="<i1", count=n_vals).astype(np.float32)
            raw = scale2 + scale1 * raw
        else:
            raw = np.fromfile(fid, dtype="<f4", count=n_vals).astype(np.float32)

    cdata = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    img = cdata.reshape((res_x, res_y), order="F")

    if border and res_x != res_y:
        res = max(res_x, res_y)
        padded = np.zeros((res, res), dtype=np.complex64)
        if res_x > res_y:
            start = (res - res_y) // 2
            padded[:, start : start + res_y] = img
        else:
            start = (res - res_x) // 2
            padded[start : start + res_x, :] = img
        img = padded

    if flip_img:
        img = img[::-1, :]
    return img


def read_cpx(
    cpx_path: str,
    border: bool = False,
    flip_img: bool = True,
) -> tuple[NDArray[np.complex64], NDArray[np.float64]]:
    """Read all stacks, slices, and coils from a Philips ``.cpx`` reference image.

    Parameters
    ------------
    cpx_path: str
        Path to the SENSE reference scan ``.cpx`` file.
    border: bool
        Whether to zero-pad slices to square matrices.
    flip_img: bool
        Whether to apply vertical image flip matching scanner display.

    Returns
    --------
        tuple[NDArray[np.complex64], NDArray[np.float64]]:
            Complex array ``(res_x, res_y, n_stacks, n_slices, n_coils)`` and
            the parsed CPX header matrix.
    """
    hdr = read_cpx_header(cpx_path)
    if hdr.size == 0:
        return (np.empty((0, 0, 0, 0, 0), dtype=np.complex64), hdr)

    with open(cpx_path, "rb") as fid:
        h1 = struct.unpack("<15i", fid.read(60))
        _ = fid.read(8)
        h2 = struct.unpack("<111i", fid.read(444))
        stride = max(512, h1[12] * 512 + (h1[9] if h2[25] == 0 else h2[25]))

    res_x = int(hdr[0, 8])
    res_y = int(hdr[0, 9])
    dim = max(res_x, res_y) if border else res_x
    dim_y = max(res_x, res_y) if border else res_y
    n_stacks = int(np.max(hdr[:, 0])) + 1
    n_slices = int(np.max(hdr[:, 1])) + 1
    n_coils = int(np.max(hdr[:, 2])) + 1

    vol = np.zeros((dim, dim_y, n_stacks, n_slices, n_coils), dtype=np.complex64)
    for idx_img in range(hdr.shape[0]):
        st = int(hdr[idx_img, 0])
        sl = int(hdr[idx_img, 1])
        co = int(hdr[idx_img, 2])
        img = read_cpx_image(
            cpx_path, hdr[idx_img], idx_img, stride, border=border, flip_img=flip_img
        )
        vol[: img.shape[0], : img.shape[1], st, sl, co] = img

    return (vol, hdr)


def read_lab_raw_labels(filename: str) -> dict[str, Any]:
    """Read and decode hexadecimal label records from a Philips ``.lab`` file.

    Parameters
    ------------
    filename: str
        Path prefix or full path to ``.lab``/``.raw``.

    Returns
    --------
        dict[str, Any]:
            Dictionary with decoded label arrays, byte offsets, and index sets
            (``NORMAL_DATA``, ``FRC_NOISE_DATA``, ``n_coils``, ``n_kx``, etc.).
    """
    prefix = re.sub(r"\.(lab|raw)$", "", filename, flags=re.IGNORECASE)
    lab_path = f"{prefix}.lab"
    raw_path = f"{prefix}.raw"
    if not os.path.exists(lab_path):
        lab_path = f"{prefix}.LAB"
    if not os.path.exists(raw_path):
        raw_path = f"{prefix}.RAW"

    raw_labels = np.fromfile(lab_path, dtype="<u4")
    n_labels = raw_labels.size // 16
    labels = raw_labels[: n_labels * 16].reshape((16, n_labels), order="F")

    data_size = labels[0, :].astype(np.int64)
    lead_dum = (labels[1, :] & 0xFFFF).astype(np.int64)
    trail_dum = ((labels[1, :] >> 16) & 0xFFFF).astype(np.int64)
    label_type = ((labels[3, :] >> 16) & 0xFFFF).astype(np.int64)
    ctrl_type = (labels[4, :] & 0xFF).astype(np.int64)
    mix = ((labels[6, :] >> 16) & 0xFFFF).astype(np.int64)
    dyn = (labels[7, :] & 0xFFFF).astype(np.int64)
    card = ((labels[7, :] >> 16) & 0xFFFF).astype(np.int64)
    echo = (labels[8, :] & 0xFFFF).astype(np.int64)
    loca = ((labels[8, :] >> 16) & 0xFFFF).astype(np.int64)
    row = (labels[9, :] & 0xFFFF).astype(np.int64)
    meas = (labels[10, :] & 0xFFFF).astype(np.int64)
    e1 = ((labels[10, :] >> 16) & 0xFFFF).astype(np.int64)
    e2 = (labels[11, :] & 0xFFFF).astype(np.int64)
    e3 = ((labels[11, :] >> 16) & 0xFFFF).astype(np.int64)
    channels_active = labels[15, :]

    offsets = np.zeros(n_labels, dtype=np.int64)
    if n_labels > 0:
        offsets[0] = 512
        for k in range(1, n_labels):
            offsets[k] = (
                offsets[k - 1] + data_size[k - 1] - trail_dum[k - 1] - lead_dum[k - 1]
            )
    offsets[data_size == 0] = -1

    std_mask = label_type == 32513
    normal_idx = np.where((ctrl_type == 0) & std_mask)[0]
    noise_idx = np.where((ctrl_type == 16) & std_mask)[0]

    mask_union = int(np.bitwise_or.reduce(channels_active)) if n_labels > 0 else 0
    n_coils = mask_union.bit_count()

    result: dict[str, Any] = {
        "raw_path": raw_path,
        "lab_path": lab_path,
        "n_labels": n_labels,
        "data_size": data_size,
        "ctrl_type": ctrl_type,
        "label_type": label_type,
        "mix": mix,
        "dyn": dyn,
        "card": card,
        "echo": echo,
        "loca": loca,
        "row": row,
        "meas": meas,
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "channels_active": channels_active,
        "offsets": offsets,
        "normal_idx": normal_idx,
        "noise_idx": noise_idx,
        "n_coils": n_coils,
    }
    return result


def load_ref_scan(
    ref_cpx_path: str,
    raw_noise_path: str | None = None,
) -> dict[str, Any]:
    """Load Philips SENSE coil sensitivity reference maps and noise covariance data.

    Parameters
    ------------
    ref_cpx_path: str
        Path to the ``.cpx`` reference scan file.
    raw_noise_path: str | None
        Optional path to ``.lab/.raw`` file containing ``FRC_NOISE_DATA``.

    Returns
    --------
        dict[str, Any]:
            Dictionary containing ``sensitivities`` (``(n_coils, x, y, z)``),
            ``body_coil``, ``surface_coils``, ``sos``, ``mask``, and ``noise``.
    """
    vol, _hdr = read_cpx(ref_cpx_path, border=True, flip_img=True)
    if vol.size == 0:
        raise ValueError(f"Empty CPX reference scan: {ref_cpx_path}")

    # Stack 0 = surface coils (or mixed), Stack 1 = body coil if dual stack
    if vol.shape[2] >= 2:
        surf = vol[:, :, 0, :, 1:] if vol.shape[4] > 1 else vol[:, :, 0, :, :]
        body = vol[:, :, 1, :, 0]
    else:
        surf = vol[:, :, 0, :, 1:] if vol.shape[4] > 1 else vol[:, :, 0, :, :]
        body = vol[:, :, 0, :, 0]

    # Reorder (x, y, z, coils) -> (coils, x, y, z)
    surf_c = np.moveaxis(surf, -1, 0).astype(np.complex128)
    sos = np.sqrt(np.sum(np.abs(surf_c) ** 2, axis=0))
    max_sos = float(np.max(sos)) if sos.size else 1.0
    mask = (sos / max(max_sos, 1e-12)) > 0.04
    safe_sos = np.where(sos > 0, sos, 1.0)
    sens = (mask[np.newaxis, ...] * surf_c) / safe_sos[np.newaxis, ...]

    noise_arr: NDArray[np.complex128] | None = None
    if raw_noise_path is not None and os.path.exists(raw_noise_path):
        info = read_lab_raw_labels(raw_noise_path)
        n_idx = info["noise_idx"]
        if len(n_idx) > 0:
            chunks: list[NDArray[np.complex128]] = []
            with open(info["raw_path"], "rb") as fid:
                for idx in n_idx:
                    ch_mask = int(info["channels_active"][idx])
                    n_c = ch_mask.bit_count()
                    d_size = int(info["data_size"][idx])
                    n_pts = d_size // (4 * max(1, n_c))
                    fid.seek(int(info["offsets"][idx]), os.SEEK_SET)
                    arr = np.fromfile(fid, dtype="<i2", count=d_size // 2).astype(
                        np.float64
                    )
                    c_arr = (
                        (arr[0::2] + 1j * arr[1::2]).reshape((n_pts, n_c), order="F").T
                    )
                    chunks.append(c_arr)
            if chunks:
                noise_arr = np.concatenate(chunks, axis=1)

    return {
        "sensitivities": sens,
        "surface_coils": surf_c,
        "body_coil": body,
        "sos": sos,
        "mask": mask,
        "noise": noise_arr,
    }
