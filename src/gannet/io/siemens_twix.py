"""Siemens TWIX (.dat) reader implementing VB/VD/VE parsing for Gannet."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct


@dataclass
class TwixMdhConstants:
    """Constants for Siemens TWIX VB/VD/VE MDH parsing.

    Parameters
    ------------
    None
    """

    vb_mdh_size: int = 128
    vd_scan_hdr_size: int = 192
    vd_channel_hdr_size: int = 32
    acqend_bit: int = 0
    rtfeedback_bit: int = 1
    hpfeedback_bit: int = 2
    online_bit: int = 3
    offlineraw_bit: int = 4
    syncdata_bit: int = 5
    rawdatacorrection_bit: int = 10
    patrefscan_bit: int = 22
    patrefandimascan_bit: int = 23
    phasecor_bit: int = 21
    noiseadjscan_bit: int = 25
    signrev_bit: int = 26
    swapped_bit: int = 27
    firstscaninslice_bit: int = 30
    lastscaninslice_bit: int = 31


@dataclass
class TwixMeasurement:
    """Decoded TWIX measurement container mirroring ``mapVBVD`` output.

    Parameters
    ------------
    hdr: dict[str, Any]
        Parsed XPROT and ASCCONV header dictionary.
    image_data: NDArray[np.complex64]
        Spectroscopy/imaging scan array shaped ``(col, cha, ...)``.
    image_dims: list[str]
        Dimension labels for ``image_data``.
    free_param: NDArray[np.int32]
        Free MDH parameters per scan (used for ``pointsBeforeEcho``).
    ice_param: NDArray[np.int32]
        ICE MDH parameters per scan.
    cutoff: NDArray[np.int32]
        Pre/post cutoff values per scan.
    noise_data: NDArray[np.complex64] | None
        Optional noise-adjust scan array ``(col, cha, n_scans)``.
    refscan_data: NDArray[np.complex64] | None
        Optional reference/water scan array.
    phasecor_data: NDArray[np.complex64] | None
        Optional phase-correction scan array.
    """

    hdr: dict[str, Any] = field(default_factory=dict)
    image_data: NDArray[np.complex64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.complex64)
    )
    image_dims: list[str] = field(default_factory=list)
    free_param: NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.int32)
    )
    ice_param: NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.int32)
    )
    cutoff: NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    noise_data: NDArray[np.complex64] | None = None
    refscan_data: NDArray[np.complex64] | None = None
    phasecor_data: NDArray[np.complex64] | None = None


def _try_parse_number(val: str) -> Any:
    """Convert a string header token to int, float, hex int, or stripped string.

    Parameters
    ------------
    val: str
        Raw scalar token string from ASCCONV/XPROT.

    Returns
    --------
        Any:
            Parsed int, float, or cleaned string.
    """
    cleaned = val.strip().strip('"').strip("'")
    if not cleaned:
        return ""
    if cleaned.lower().startswith("0x"):
        try:
            return int(cleaned, 16)
        except ValueError:
            return cleaned
    try:
        if "." in cleaned or "e" in cleaned.lower():
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _assign_nested(root: dict[str, Any], path_parts: list[str], value: Any) -> None:
    """Assign a value into a nested dictionary handling bracketed array indices.

    Parameters
    ------------
    root: dict[str, Any]
        Root header dictionary being populated.
    path_parts: list[str]
        Dot-separated key components (may include ``[idx]``).
    value: Any
        Parsed scalar value to store.

    Returns
    --------
        None:
            Mutates ``root`` in place and returns ``None``.
    """
    curr: Any = root
    idx_pattern = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")
    for ii, part in enumerate(path_parts):
        match = idx_pattern.match(part.strip())
        if match is None:
            key = part.strip()
            idx = None
        else:
            key = match.group(1)
            idx = int(match.group(2)) if match.group(2) is not None else None
        is_last = ii == len(path_parts) - 1
        if idx is None:
            if is_last:
                curr[key] = value
            else:
                if key not in curr or not isinstance(curr[key], dict):
                    curr[key] = {}
                curr = curr[key]
        else:
            if key not in curr or not isinstance(curr[key], list):
                curr[key] = []
            arr: list[Any] = curr[key]
            while len(arr) <= idx:
                arr.append({} if not is_last else None)
            if is_last:
                arr[idx] = value
            else:
                if not isinstance(arr[idx], dict):
                    arr[idx] = {}
                curr = arr[idx]


def parse_ascconv(text: str) -> dict[str, Any]:
    """Parse the Siemens ``### ASCCONV BEGIN ... ### ASCCONV END`` text block.

    Parameters
    ------------
    text: str
        Decoded header text containing one or more ASCCONV blocks.

    Returns
    --------
        dict[str, Any]:
            Nested dictionary of protocol parameters (``sSpecPara``,
            ``sRXSPEC``, ``sProtConsistencyInfo``, ``sWiPMemBlock``, etc.).
    """
    result: dict[str, Any] = {}
    begin_match = re.search(r"###\s*ASCCONV BEGIN.*?###", text)
    end_match = re.search(r"###\s*ASCCONV END\s*###", text)
    if begin_match and end_match and end_match.start() > begin_match.end():
        block = text[begin_match.end() : end_match.start()]
    else:
        block = text

    for line in block.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("#") or "=" not in line_str:
            continue
        lhs, rhs = line_str.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.split("#")[0].strip()
        if not lhs:
            continue
        _assign_nested(result, lhs.split("."), _try_parse_number(rhs))
    return result


def parse_xprot_params(text: str) -> dict[str, Any]:
    """Extract common XPROT XML parameter values from Siemens TWIX headers.

    Parameters
    ------------
    text: str
        Decoded TWIX XPROT buffer text.

    Returns
    --------
        dict[str, Any]:
            Dictionary of extracted XPROT scalar/array parameters such as
            ``lFrequency``, ``alDwellTime``, ``flReadoutOSFactor``, ``tSequence``.
    """
    out: dict[str, Any] = {}
    tag_pattern = re.compile(
        r"<Param(?:Long|Double|String|Bool)\.\"([^\"]+)\">\s*\{(.*?)\}",
        re.DOTALL,
    )
    for match in tag_pattern.finditer(text):
        key = match.group(1)
        body = match.group(2)
        val_lines = [
            ln.strip()
            for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("<")
        ]
        if not val_lines:
            continue
        if len(val_lines) == 1:
            out[key] = _try_parse_number(val_lines[0])
        else:
            out[key] = [_try_parse_number(v) for v in val_lines]
    return out


def parse_twix_header(raw_bytes: bytes) -> dict[str, Any]:
    """Parse a Siemens TWIX measurement header buffer into ``MeasYaps`` and ``Phoenix``.

    Parameters
    ------------
    raw_bytes: bytes
        Raw header bytes read between measurement start and first MDH.

    Returns
    --------
        dict[str, Any]:
            Dictionary with ``MeasYaps`` (ASCCONV), ``Phoenix`` (ASCCONV), and
            ``Meas`` (XPROT flat params).
    """
    text = raw_bytes.decode("latin-1", errors="ignore")
    ascconv = parse_ascconv(text)
    xprot = parse_xprot_params(text)
    header: dict[str, Any] = {
        "MeasYaps": ascconv,
        "Phoenix": ascconv,
        "Meas": xprot,
    }
    return header


def _parse_vb_mdh(buf: bytes) -> dict[str, Any]:
    """Decode one 128-byte Siemens VB-line Measurement Data Header (MDH).

    Parameters
    ------------
    buf: bytes
        128-byte buffer containing the VB MDH.

    Returns
    --------
        dict[str, Any]:
            Decoded MDH fields including ``dma_length``, ``eval_info_mask``,
            ``samples_in_scan``, ``used_channels``, ``channel_id``,
            ``loop_counters``, ``free_param``, ``ice_param``, and ``cutoff``.
    """
    dma_flags = struct.unpack_from("<I", buf, 0)[0]
    dma_length = dma_flags & 0x01FFFFFF
    meas_uid, scan_counter, _time_stamp, _pmu_stamp = struct.unpack_from(
        "<iIII", buf, 4
    )
    eval_mask = struct.unpack_from("<Q", buf, 20)[0]
    samples, used_channels = struct.unpack_from("<HH", buf, 28)
    lc = struct.unpack_from("<14H", buf, 32)
    cutoff_pre, cutoff_post = struct.unpack_from("<HH", buf, 60)
    (
        _kspace_centre_col,
        _coil_select,
        readout_os,
        _kspace_centre_lin,
        _kspace_centre_par,
    ) = struct.unpack_from("<HHfHH", buf, 64)
    ice_param = struct.unpack_from("<4H", buf, 76)
    free_param = struct.unpack_from("<4H", buf, 84)
    channel_id, _ptab_pos_neg = struct.unpack_from("<HH", buf, 124)

    mdh: dict[str, Any] = {
        "dma_length": int(dma_length),
        "meas_uid": int(meas_uid),
        "scan_counter": int(scan_counter),
        "eval_mask": int(eval_mask),
        "samples": int(samples),
        "used_channels": int(used_channels),
        "Lin": int(lc[0]),
        "Ave": int(lc[1]),
        "Sli": int(lc[2]),
        "Par": int(lc[3]),
        "Eco": int(lc[4]),
        "Phs": int(lc[5]),
        "Rep": int(lc[6]),
        "Set": int(lc[7]),
        "Seg": int(lc[8]),
        "Ida": int(lc[9]),
        "Idb": int(lc[10]),
        "Idc": int(lc[11]),
        "Idd": int(lc[12]),
        "Ide": int(lc[13]),
        "cutoff": (int(cutoff_pre), int(cutoff_post)),
        "ice_param": tuple(int(v) for v in ice_param),
        "free_param": tuple(int(v) for v in free_param),
        "channel_id": int(channel_id),
        "readout_os": float(readout_os),
    }
    return mdh


def _parse_vd_scan_hdr(buf: bytes) -> dict[str, Any]:
    """Decode one 192-byte Siemens VD/VE scan header.

    Parameters
    ------------
    buf: bytes
        192-byte buffer containing the VD/VE scan header.

    Returns
    --------
        dict[str, Any]:
            Decoded VD/VE scan header fields matching VB keys where applicable.
    """
    dma_flags = struct.unpack_from("<I", buf, 0)[0]
    dma_length = dma_flags & 0x01FFFFFF
    meas_uid, scan_counter, _time_stamp, _pmu_stamp = struct.unpack_from(
        "<iIII", buf, 4
    )
    samples, used_channels = struct.unpack_from("<HH", buf, 40)
    lc = struct.unpack_from("<14H", buf, 44)
    cutoff_pre, cutoff_post = struct.unpack_from("<HH", buf, 72)
    ice_param = struct.unpack_from("<24H", buf, 112)
    free_param = ice_param[:4]
    eval_mask = struct.unpack_from("<Q", buf, 168)[0]

    mdh: dict[str, Any] = {
        "dma_length": int(dma_length),
        "meas_uid": int(meas_uid),
        "scan_counter": int(scan_counter),
        "eval_mask": int(eval_mask),
        "samples": int(samples),
        "used_channels": int(used_channels),
        "Lin": int(lc[0]),
        "Ave": int(lc[1]),
        "Sli": int(lc[2]),
        "Par": int(lc[3]),
        "Eco": int(lc[4]),
        "Phs": int(lc[5]),
        "Rep": int(lc[6]),
        "Set": int(lc[7]),
        "Seg": int(lc[8]),
        "Ida": int(lc[9]),
        "Idb": int(lc[10]),
        "Idc": int(lc[11]),
        "Idd": int(lc[12]),
        "Ide": int(lc[13]),
        "cutoff": (int(cutoff_pre), int(cutoff_post)),
        "ice_param": tuple(int(v) for v in ice_param[:4]),
        "free_param": tuple(int(v) for v in free_param),
    }
    return mdh


def _is_bit_set(mask: int, bit: int) -> bool:
    """Return True if ``bit`` is set in integer bitmask ``mask``.

    Parameters
    ------------
    mask: int
        64-bit evaluation info mask.
    bit: int
        Bit index (0-based).

    Returns
    --------
        bool:
            Whether the specified flag bit is active.
    """
    return bool((mask >> bit) & 1)


def _assemble_scan_array(
    scans: list[tuple[dict[str, Any], NDArray[np.complex64]]],
    ignore_seg: bool = True,
) -> tuple[
    NDArray[np.complex64],
    list[str],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Assemble parsed scans into a multidimensional NumPy array.

    Parameters
    ------------
    scans: list[tuple[dict[str, Any], NDArray[np.complex64]]]
        List of ``(mdh_dict, complex_data_cha_col)`` pairs.
    ignore_seg: bool
        Whether to collapse the ``Seg`` loop counter (default ``True``).

    Returns
    --------
        tuple[NDArray[np.complex64], list[str], NDArray[np.int32], ...]:
            Tuple ``(data, dim_names, free_params, ice_params, cutoffs)`` where
            ``data`` has shape ``(col, cha, Lin, Ave, Sli, Par, ...)`` with
            singleton trailing dims stripped except ``(col, cha)``.
    """
    if not scans:
        empty = np.empty((0, 0), dtype=np.complex64)
        empty_i = np.empty((0, 4), dtype=np.int32)
        empty_c = np.empty((0, 2), dtype=np.int32)
        return (empty, ["Col", "Cha"], empty_i, empty_i, empty_c)

    loop_names = [
        "Lin",
        "Ave",
        "Sli",
        "Par",
        "Eco",
        "Phs",
        "Rep",
        "Set",
        "Seg",
        "Ida",
        "Idb",
        "Idc",
        "Idd",
        "Ide",
    ]
    n_samples = max(arr.shape[1] for _, arr in scans)
    n_cha = max(arr.shape[0] for _, arr in scans)
    max_indices = {}
    for name in loop_names:
        if name == "Seg" and ignore_seg:
            max_indices[name] = 1
        else:
            max_indices[name] = max(mdh[name] for mdh, _ in scans) + 1

    active_loops = [name for name in loop_names if max_indices[name] > 1]
    full_shape = [n_samples, n_cha] + [max_indices[name] for name in active_loops]
    out = np.zeros(full_shape, dtype=np.complex64)

    free_list = np.zeros((len(scans), 4), dtype=np.int32)
    ice_list = np.zeros((len(scans), 4), dtype=np.int32)
    cut_list = np.zeros((len(scans), 2), dtype=np.int32)

    for idx_s, (mdh, arr) in enumerate(scans):
        free_list[idx_s] = mdh["free_param"][:4]
        ice_list[idx_s] = mdh["ice_param"][:4]
        cut_list[idx_s] = mdh["cutoff"][:2]
        loop_idx = tuple(
            0 if (name == "Seg" and ignore_seg) else mdh[name] for name in active_loops
        )
        c_cha, c_col = arr.shape
        out[(slice(0, c_col), slice(0, c_cha)) + loop_idx] = arr.T

    dim_names = ["Col", "Cha"] + active_loops
    return (out, dim_names, free_list, ice_list, cut_list)


def _read_single_measurement(
    fid: Any,
    meas_offset: int,
    meas_length: int,
    is_vd: bool,
    ignore_seg: bool = True,
) -> TwixMeasurement:
    """Read header and all MDH scan blocks for one TWIX measurement.

    Parameters
    ------------
    fid: Any
        Open binary file handle positioned anywhere.
    meas_offset: int
        Byte offset where this measurement begins.
    meas_length: int
        Byte length of this measurement (0 means until EOF).
    is_vd: bool
        True for VD/VE layout, False for VB layout.
    ignore_seg: bool
        Whether to ignore ``Seg`` index during assembly.

    Returns
    --------
        TwixMeasurement:
            Decoded measurement object.
    """
    const = TwixMdhConstants()
    fid.seek(meas_offset, os.SEEK_SET)
    hdr_len = struct.unpack("<I", fid.read(4))[0]
    raw_hdr = fid.read(max(0, hdr_len - 4))
    hdr = parse_twix_header(raw_hdr)

    cpos = meas_offset + hdr_len
    if cpos % 32 != 0:
        cpos += 32 - (cpos % 32)

    if meas_length > 0:
        end_pos = meas_offset + meas_length
    else:
        fid.seek(0, os.SEEK_END)
        end_pos = fid.tell()

    img_scans: list[tuple[dict[str, Any], NDArray[np.complex64]]] = []
    noise_scans: list[tuple[dict[str, Any], NDArray[np.complex64]]] = []
    ref_scans: list[tuple[dict[str, Any], NDArray[np.complex64]]] = []
    pcor_scans: list[tuple[dict[str, Any], NDArray[np.complex64]]] = []

    vb_grouped: dict[
        tuple[Any, ...], tuple[dict[str, Any], list[tuple[int, NDArray[np.complex64]]]]
    ] = {}

    fid.seek(cpos, os.SEEK_SET)
    while cpos + 64 <= end_pos:
        fid.seek(cpos, os.SEEK_SET)
        if not is_vd:
            mdh_bytes = fid.read(const.vb_mdh_size)
            if len(mdh_bytes) < const.vb_mdh_size:
                break
            mdh = _parse_vb_mdh(mdh_bytes)
            dma_len = mdh["dma_length"]
            if _is_bit_set(mdh["eval_mask"], const.acqend_bit) or dma_len == 0:
                break
            n_samp = mdh["samples"]
            raw_floats = np.fromfile(fid, dtype=np.float32, count=2 * n_samp)
            if raw_floats.size < 2 * n_samp:
                break
            cdata = (raw_floats[0::2] + 1j * raw_floats[1::2]).astype(np.complex64)
            if _is_bit_set(mdh["eval_mask"], const.signrev_bit):
                cdata = -cdata

            key = (
                mdh["eval_mask"],
                mdh["Lin"],
                mdh["Ave"],
                mdh["Sli"],
                mdh["Par"],
                mdh["Eco"],
                mdh["Phs"],
                mdh["Rep"],
                mdh["Set"],
                0 if ignore_seg else mdh["Seg"],
                mdh["Ida"],
                mdh["Idb"],
                mdh["Idc"],
                mdh["Idd"],
                mdh["Ide"],
            )
            if key not in vb_grouped:
                vb_grouped[key] = (mdh, [])
            vb_grouped[key][1].append((mdh["channel_id"], cdata))
            cpos += max(dma_len, const.vb_mdh_size + n_samp * 8)
        else:
            scan_hdr_bytes = fid.read(const.vd_scan_hdr_size)
            if len(scan_hdr_bytes) < const.vd_scan_hdr_size:
                break
            mdh = _parse_vd_scan_hdr(scan_hdr_bytes)
            dma_len = mdh["dma_length"]
            if _is_bit_set(mdh["eval_mask"], const.acqend_bit) or dma_len == 0:
                break
            n_samp = mdh["samples"]
            n_cha = mdh["used_channels"]
            if n_samp == 0 or n_cha == 0:
                cpos += max(dma_len, const.vd_scan_hdr_size)
                continue
            scan_arr = np.zeros((n_cha, n_samp), dtype=np.complex64)
            for c_idx in range(n_cha):
                ch_hdr = fid.read(const.vd_channel_hdr_size)
                if len(ch_hdr) < const.vd_channel_hdr_size:
                    break
                ch_id = struct.unpack_from("<H", ch_hdr, 24)[0]
                raw_floats = np.fromfile(fid, dtype=np.float32, count=2 * n_samp)
                if raw_floats.size < 2 * n_samp:
                    break
                cdata = (raw_floats[0::2] + 1j * raw_floats[1::2]).astype(np.complex64)
                if _is_bit_set(mdh["eval_mask"], const.signrev_bit):
                    cdata = -cdata
                target_c = ch_id if ch_id < n_cha else c_idx
                scan_arr[target_c, :] = cdata

            mask = mdh["eval_mask"]
            if _is_bit_set(mask, const.noiseadjscan_bit):
                noise_scans.append((mdh, scan_arr))
            elif _is_bit_set(mask, const.phasecor_bit):
                pcor_scans.append((mdh, scan_arr))
            elif _is_bit_set(mask, const.patrefscan_bit) and not _is_bit_set(
                mask, const.patrefandimascan_bit
            ):
                ref_scans.append((mdh, scan_arr))
            elif not _is_bit_set(mask, const.rtfeedback_bit) and not _is_bit_set(
                mask, const.hpfeedback_bit
            ):
                img_scans.append((mdh, scan_arr))
            cpos += dma_len

    if not is_vd and vb_grouped:
        for mdh, ch_list in vb_grouped.values():
            n_cha = mdh["used_channels"] or len(ch_list)
            n_samp = mdh["samples"]
            arr = np.zeros((n_cha, n_samp), dtype=np.complex64)
            for idx_pos, (ch_id, cdata) in enumerate(ch_list):
                target = idx_pos if ch_id >= n_cha else ch_id
                arr[target, : len(cdata)] = cdata
            mask = mdh["eval_mask"]
            if _is_bit_set(mask, const.noiseadjscan_bit):
                noise_scans.append((mdh, arr))
            elif _is_bit_set(mask, const.phasecor_bit):
                pcor_scans.append((mdh, arr))
            elif _is_bit_set(mask, const.patrefscan_bit) and not _is_bit_set(
                mask, const.patrefandimascan_bit
            ):
                ref_scans.append((mdh, arr))
            else:
                img_scans.append((mdh, arr))

    img_arr, img_dims, free_p, ice_p, cuts = _assemble_scan_array(
        img_scans, ignore_seg=ignore_seg
    )
    noise_arr = (
        _assemble_scan_array(noise_scans, ignore_seg=True)[0] if noise_scans else None
    )
    ref_arr = _assemble_scan_array(ref_scans, ignore_seg=True)[0] if ref_scans else None
    pcor_arr = (
        _assemble_scan_array(pcor_scans, ignore_seg=True)[0] if pcor_scans else None
    )

    meas = TwixMeasurement(
        hdr=hdr,
        image_data=img_arr,
        image_dims=img_dims,
        free_param=free_p,
        ice_param=ice_p,
        cutoff=cuts,
        noise_data=noise_arr,
        refscan_data=ref_arr,
        phasecor_data=pcor_arr,
    )
    return meas


def map_vbvd(filepath: str, ignore_seg: bool = True) -> list[TwixMeasurement]:
    """Read one or more measurements from a Siemens VB/VD/VE TWIX ``.dat`` file.

    Parameters
    ------------
    filepath: str
        Path to the ``.dat`` TWIX file.
    ignore_seg: bool
        Whether to ignore the ``Seg`` counter when assembling arrays.

    Returns
    --------
        list[TwixMeasurement]:
            List of parsed ``TwixMeasurement`` objects (length 1 for VB files,
            >= 1 for VD/VE multi-RAID files where the last element is primary).
    """
    measurements: list[TwixMeasurement] = []
    with open(filepath, "rb") as fid:
        first_int, second_int = struct.unpack("<II", fid.read(8))
        is_vd = first_int < 10000 and second_int <= 64
        if not is_vd:
            measurements.append(
                _read_single_measurement(
                    fid,
                    meas_offset=0,
                    meas_length=0,
                    is_vd=False,
                    ignore_seg=ignore_seg,
                )
            )
        else:
            n_meas = second_int
            fid.seek(8, os.SEEK_SET)
            entries: list[tuple[int, int]] = []
            for _ in range(n_meas):
                entry = fid.read(152)
                if len(entry) < 152:
                    break
                _meas_id, _file_id, offset, length = struct.unpack_from(
                    "<IIQQ", entry, 0
                )
                entries.append((int(offset), int(length)))
            for offset, length in entries:
                measurements.append(
                    _read_single_measurement(
                        fid,
                        meas_offset=offset,
                        meas_length=length,
                        is_vd=True,
                        ignore_seg=ignore_seg,
                    )
                )
    return measurements


def _nested_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely look up nested dictionary or list elements.

    Parameters
    ------------
    d: dict[str, Any]
        Root mapping.
    *keys: str
        Successive key strings.
    default: Any
        Value returned when any key is missing.

    Returns
    --------
        Any:
            Retrieved parameter or ``default``.
    """
    curr: Any = d
    for key in keys:
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return default
    return curr


def _extract_twix_geometry(
    mrs_struct: MRSStruct,
    meas: TwixMeasurement,
    ii: int,
) -> MRSStruct:
    """Populate voxel dimensions, orientation, and acquisition metadata from TWIX hdr.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` object.
    meas: TwixMeasurement
        Primary ``TwixMeasurement`` object.
    ii: int
        0-based scan index.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct``.
    """
    yaps = meas.hdr.get("MeasYaps", {})
    xprot = meas.hdr.get("Meas", {})

    larmor = _nested_get(yaps, "sTXSPEC", "asNucleusInfo", default=None)
    freq_hz = None
    if isinstance(larmor, list) and larmor and isinstance(larmor[0], dict):
        freq_hz = larmor[0].get("lFrequency")
    if freq_hz is None:
        freq_hz = xprot.get("lFrequency", 123.2e6)
    mrs_struct["p"]["LarmorFreq"][ii] = float(freq_hz) / 1e6

    tr_us = _nested_get(yaps, "alTR")
    if isinstance(tr_us, list) and tr_us:
        mrs_struct["p"]["TR"][ii] = float(tr_us[0]) / 1e3
    te_us = _nested_get(yaps, "alTE")
    if isinstance(te_us, list) and te_us:
        mrs_struct["p"]["TE"][ii] = float(te_us[0]) / 1e3

    dwell_ns = None
    dwell_list = _nested_get(yaps, "sRXSPEC", "alDwellTime")
    if isinstance(dwell_list, list) and dwell_list:
        dwell_ns = dwell_list[0]
    elif isinstance(dwell_list, (int, float)):
        dwell_ns = dwell_list
    elif "alDwellTime" in xprot:
        raw_dt = xprot["alDwellTime"]
        dwell_ns = raw_dt[0] if isinstance(raw_dt, list) else raw_dt
    if dwell_ns:
        mrs_struct["p"]["sw"][ii] = 1e9 / (float(dwell_ns) * 2.0)

    seq_name = str(
        _nested_get(yaps, "tSequenceFileName", default=xprot.get("tSequence", ""))
    )
    mrs_struct["p"]["seq"] = seq_name

    spec_para = yaps.get("sSpecPara", {})
    voi = spec_para.get("sVoI", {}) if isinstance(spec_para, dict) else {}
    pos = voi.get("sPosition", {}) if isinstance(voi, dict) else {}
    norm = voi.get("sNormal", {}) if isinstance(voi, dict) else {}

    ro_fov = float(voi.get("dReadoutFOV", 30.0)) if isinstance(voi, dict) else 30.0
    pe_fov = float(voi.get("dPhaseFOV", 30.0)) if isinstance(voi, dict) else 30.0
    thk = float(voi.get("dThickness", 30.0)) if isinstance(voi, dict) else 30.0
    mrs_struct["p"]["voxdim"][ii] = [pe_fov, ro_fov, thk]
    mrs_struct["p"]["voxoff"][ii] = [
        float(pos.get("dSag", 0.0)),
        float(pos.get("dCor", 0.0)),
        float(pos.get("dTra", 0.0)),
    ]
    mrs_struct["p"]["VoI_InPlaneRot"][ii] = float(voi.get("dInPlaneRot", 0.0))
    mrs_struct["p"]["NormSag"][ii] = float(norm.get("dSag", 0.0))
    mrs_struct["p"]["NormCor"][ii] = float(norm.get("dCor", 0.0))
    mrs_struct["p"]["NormTra"][ii] = float(norm.get("dTra", 1.0))

    table_pos = _nested_get(yaps, "lScanRegionPosSag", default=0.0)
    mrs_struct["p"]["TablePosition"][ii] = [
        float(table_pos or 0.0),
        float(_nested_get(yaps, "lScanRegionPosCor", default=0.0) or 0.0),
        float(_nested_get(yaps, "lScanRegionPosTra", default=0.0) or 0.0),
    ]
    return mrs_struct


def _gls_coil_combine(
    data: NDArray[np.complex64],
    noise: NDArray[np.complex64] | None,
) -> NDArray[np.complex128]:
    """Perform Generalized Least Squares (GLS) or SNR-weighted coil combination.

    Parameters
    ------------
    data: NDArray[np.complex64]
        2D/3D coil data shaped ``(npoints, ncoils, navg)``.
    noise: NDArray[np.complex64] | None
        Optional noise scan shaped ``(npoints, ncoils, ...)``.

    Returns
    --------
        NDArray[np.complex128]:
            Combined complex FID array shaped ``(npoints, navg)``.
    """
    npts, ncha, _navg = data.shape
    if ncha == 1:
        return np.asarray(data[:, 0, :], dtype=np.complex128)

    if noise is not None and noise.size > 0 and noise.shape[1] == ncha:
        noise_2d = noise.reshape(-1, ncha)
        cov = (noise_2d.conj().T @ noise_2d) / max(1, noise_2d.shape[0] - 1)
        cov += np.eye(ncha, dtype=np.complex128) * (1e-8 * np.trace(cov).real / ncha)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(ncha, dtype=np.complex128)
    else:
        inv_cov = np.eye(ncha, dtype=np.complex128)

    sig_ref = np.mean(data[: max(1, min(10, npts)), :, :], axis=(0, 2))
    if np.linalg.norm(sig_ref) == 0:
        sig_ref = np.ones(ncha, dtype=np.complex128)
    weights = inv_cov @ sig_ref
    denom = np.sqrt(np.real(sig_ref.conj() @ inv_cov @ sig_ref))
    if denom > 0:
        weights = weights / denom

    combined = np.tensordot(data.astype(np.complex128), weights.conj(), axes=([1], [0]))
    if combined.ndim == 1:
        combined = combined[:, np.newaxis]
    return np.asarray(combined, dtype=np.complex128)


def siemens_twix_read(
    mrs_struct: MRSStruct,
    fname: str,
    fname_water: str | None = None,
) -> MRSStruct:
    """Read Siemens TWIX ``.dat`` MRS data (and optional water ref) into ``MRSStruct``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` with ``ii`` set to current 0-based scan index.
    fname: str
        Path to metabolite TWIX ``.dat`` file.
    fname_water: str | None
        Optional path to unsuppressed water TWIX ``.dat`` file.

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` containing loaded FIDs and scan parameters.
    """
    ii = int(mrs_struct["ii"])
    measurements = map_vbvd(fname, ignore_seg=True)
    if not measurements:
        raise ValueError(f"No TWIX measurements found in {fname}")
    twix_metab = measurements[-1]

    mrs_struct = _extract_twix_geometry(mrs_struct, twix_metab, ii)

    img = twix_metab.image_data
    if img.size == 0:
        raise ValueError(f"No spectroscopy scan data found in {fname}")

    npts = img.shape[0]
    ncha = img.shape[1] if img.ndim > 1 else 1
    flat_img = img.reshape(npts, ncha, -1)

    if twix_metab.free_param.size > 0:
        pts_before = int(twix_metab.free_param[0, 0])
        if 0 < pts_before < npts // 2:
            flat_img = np.roll(flat_img, -pts_before, axis=0)
            flat_img[-pts_before:, :, :] = 0.0

    # Downsample 2x readout oversampling if present
    if npts >= 4096:
        spec = np.fft.fftshift(np.fft.fft(flat_img, axis=0), axes=0)
        quarter = npts // 4
        spec_crop = spec[quarter : npts - quarter, :, :]
        flat_img = np.fft.ifft(np.fft.ifftshift(spec_crop, axes=0), axis=0) * 0.5
        npts = flat_img.shape[0]

    combined = _gls_coil_combine(flat_img, twix_metab.noise_data)
    mrs_struct["p"]["npoints"][ii] = npts
    mrs_struct["p"]["nrows"][ii] = combined.shape[1]
    mrs_struct["p"]["Navg"][ii] = combined.shape[1]
    mrs_struct["fids"]["data"] = np.conj(combined)

    # Water reference: external file or earlier measurement in VD multi-RAID
    water_arr: NDArray[np.complex128] | None = None
    if fname_water is not None:
        w_meas = map_vbvd(fname_water, ignore_seg=True)[-1]
        w_img = w_meas.image_data
        if w_img.size > 0:
            w_npts = w_img.shape[0]
            w_ncha = w_img.shape[1] if w_img.ndim > 1 else 1
            w_flat = w_img.reshape(w_npts, w_ncha, -1)
            if w_npts >= 4096:
                w_spec = np.fft.fftshift(np.fft.fft(w_flat, axis=0), axes=0)
                q = w_npts // 4
                w_flat = (
                    np.fft.ifft(
                        np.fft.ifftshift(w_spec[q : w_npts - q], axes=0), axis=0
                    )
                    * 0.5
                )
            water_arr = np.conj(_gls_coil_combine(w_flat, w_meas.noise_data))
    elif len(measurements) > 1 and measurements[0].image_data.size > 0:
        w_meas = measurements[0]
        w_img = w_meas.image_data
        w_npts = w_img.shape[0]
        w_ncha = w_img.shape[1] if w_img.ndim > 1 else 1
        w_flat = w_img.reshape(w_npts, w_ncha, -1)
        if w_npts == npts * 2:
            w_spec = np.fft.fftshift(np.fft.fft(w_flat, axis=0), axes=0)
            q = w_npts // 4
            w_flat = (
                np.fft.ifft(np.fft.ifftshift(w_spec[q : w_npts - q], axes=0), axis=0)
                * 0.5
            )
        water_arr = np.conj(_gls_coil_combine(w_flat, w_meas.noise_data))

    if water_arr is not None:
        mrs_struct["p"]["Nwateravg"][ii] = water_arr.shape[1]
        mrs_struct["fids"]["data_water"] = np.mean(water_arr, axis=1)
    return mrs_struct
