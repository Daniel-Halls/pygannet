"""DICOM and Siemens IMA spectroscopy readers."""

import re
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pydicom
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.io.utils import sort_nat


def _set_nested(root: dict[str, object], keys: list[str], value: object) -> None:
    """Assign a value along a nested key path in ``root``."""
    curr = root
    for key in keys[:-1]:
        if key not in curr or not isinstance(curr[key], dict):
            curr[key] = {}
        curr = curr[key]  # type: ignore[assignment]
    curr[keys[-1]] = value


def read_dcm_header(filepath: str) -> MRSStruct:
    """Parse Siemens ASCCONV header block embedded in a DICOM or IMA file.

    Mirrors ``read_dcm_header.m``.

    Parameters
    ------------
    filepath: str
        Path to a ``.dcm`` or ``.ima`` file.

    Returns
    --------
        MRSStruct: MRSStruct object
            Parsed header parameters.
    """
    with open(filepath, "rb") as handle:
        raw_bytes = handle.read()
    text = raw_bytes.decode("latin-1", errors="ignore")
    start_idx = text.find("### ASCCONV BEGIN")
    end_idx = text.find("### ASCCONV END")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"ASCCONV header block not found in '{filepath}'.")
    block = text[start_idx:end_idx]

    dcm_hdr: dict[str, object] = {}
    pattern = re.compile(r'([\w\[\].]+)\s*=\s*([\w.\-"\\/%]+)')
    for line in block.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        raw_key, raw_val = match.group(1), match.group(2)
        cleaned_key = re.sub(r"\[|\]|_", "", raw_key)
        if '"' not in raw_val:
            if raw_val == "0x1":
                val: object = 0.0
            elif raw_val == "1x1":
                val = 1.0
            else:
                try:
                    val = float(raw_val)
                except ValueError:
                    val = raw_val
        else:
            val = raw_val.strip('"')
        parts = [p for p in cleaned_key.split(".") if p]
        if parts:
            _set_nested(dcm_hdr, parts, val)

    header = MRSStruct()
    seq_name = str(dcm_hdr.get("tSequenceFileName", ""))
    header["sequenceFileName"] = seq_name
    if "svs_edit" in seq_name:
        header["seqtype"] = "MEGAPRESS"
        header["seqorig"] = "Universal" if seq_name.endswith("univ") else "WIP"
    elif "JEdit" in seq_name:
        header["seqtype"] = "MEGAPRESS"
        header["seqorig"] = "Utah"
    elif "jn_" in seq_name:
        header["seqtype"] = "MEGAPRESS"
        header["seqorig"] = "JN"
    elif "eja_svs_mpress" in seq_name:
        header["seqtype"] = "MEGAPRESS"
        header["seqorig"] = "CMRR"
    elif "eja_svs_press" in seq_name:
        header["seqtype"] = "PRESS"
        header["seqorig"] = "CMRR"
    elif "svs_se" in seq_name:
        header["seqtype"] = "PRESS"
    elif "svs_slaser" in seq_name:
        header["seqtype"] = "sLASER"
    elif "st_vapor_643" in seq_name or "%SiemensSeq%\\svs_st" in seq_name:
        header["seqtype"] = "STEAM"
    else:
        raise ValueError(f"Unknown sequence: {seq_name}.")

    header["TR"] = float(dcm_hdr.get("alTR0", 0.0)) * 1e-3  # type: ignore[arg-type]
    header["TE"] = float(dcm_hdr.get("alTE0", 0.0)) * 1e-3  # type: ignore[arg-type]
    if "lAverages" in dcm_hdr:
        header["nAverages"] = float(dcm_hdr["lAverages"])  # type: ignore[arg-type]
    else:
        wip = dcm_hdr.get("sWipMemBlock", dcm_hdr.get("sWiPMemBlock", {}))
        if isinstance(wip, dict) and "alFree2" in wip:
            header["nAverages"] = float(wip["alFree2"])  # type: ignore[arg-type]
        else:
            header["nAverages"] = 1.0

    spec_para: dict[str, object] = dcm_hdr.get("sSpecPara", {})  # type: ignore[assignment]
    header["removeOS"] = spec_para.get("ucRemoveOversampling", 0.0)
    header["vectorSize"] = float(spec_para.get("lVectorSize", 1024.0))  # type: ignore[arg-type]

    min_dbl = float(np.finfo(np.float64).tiny)
    svoi: dict[str, object] = spec_para.get("sVoI", {})  # type: ignore[assignment]
    snorm: dict[str, object] = svoi.get("sNormal", {})  # type: ignore[assignment]
    spos: dict[str, object] = svoi.get("sPosition", {})  # type: ignore[assignment]

    header["VoI_InPlaneRot"] = float(svoi.get("dInPlaneRot", min_dbl))  # type: ignore[arg-type]
    header["VoI_RoFOV"] = float(svoi.get("dReadoutFOV", 0.0))  # type: ignore[arg-type]
    header["VoI_PeFOV"] = float(svoi.get("dPhaseFOV", 0.0))  # type: ignore[arg-type]
    header["VoIThickness"] = float(svoi.get("dThickness", 0.0))  # type: ignore[arg-type]
    header["NormCor"] = float(snorm.get("dCor", min_dbl))  # type: ignore[arg-type]
    header["NormSag"] = float(snorm.get("dSag", min_dbl))  # type: ignore[arg-type]
    header["NormTra"] = float(snorm.get("dTra", min_dbl))  # type: ignore[arg-type]
    if spos:
        header["PosCor"] = float(spos.get("dCor", min_dbl))  # type: ignore[arg-type]
        header["PosSag"] = float(spos.get("dSag", min_dbl))  # type: ignore[arg-type]
        header["PosTra"] = float(spos.get("dTra", min_dbl))  # type: ignore[arg-type]

    header["deltaFreq"] = float(spec_para.get("dDeltaFrequency", 0.0))  # type: ignore[arg-type]
    prot_cons: dict[str, object] = dcm_hdr.get("sProtConsistencyInfo", {})  # type: ignore[assignment]
    if "flNominalB0" in prot_cons:
        header["B0"] = float(prot_cons["flNominalB0"])  # type: ignore[arg-type]
    srx: dict[str, object] = dcm_hdr.get("sRXSPEC", {})  # type: ignore[assignment]
    header["dwellTime"] = float(srx.get("alDwellTime0", 1.0))  # type: ignore[arg-type]
    stx: dict[str, object] = dcm_hdr.get("sTXSPEC", {})  # type: ignore[assignment]
    as_nuc: dict[str, object] = stx.get("asNucleusInfo0", {})  # type: ignore[assignment]
    header["tx_freq"] = float(as_nuc.get("lFrequency", 123.0e6))  # type: ignore[arg-type]
    return header


def dicom_open(filename: str) -> BinaryIO:
    """Open a DICOM file and validate the 128-byte preamble + DICM magic.

    Mirrors ``dicom_open.m``.

    Parameters
    ------------
    filename: str
        Path to DICOM file.

    Returns
    --------
        BinaryIO: open file handle
            Handle positioned right after ``'DICM'``.
    """
    handle = open(filename, "rb")  # noqa: SIM115
    handle.read(128)
    magic = handle.read(4)
    if magic != b"DICM":
        handle.close()
        raise ValueError(f"File '{filename}' does not appear to be valid DICOM.")
    return handle


def dicom_move(fd: BinaryIO, str_group: str, str_element: str) -> int:
    """Advance file cursor to target DICOM tag ``(group, element)``.

    Mirrors ``dicom_move.m``.

    Parameters
    ------------
    fd: BinaryIO
        Open DICOM file stream positioned after ``'DICM'``.
    str_group: str
        Hex group string (e.g., ``'7FE1'``).
    str_element: str
        Hex element string (e.g., ``'1010'``).

    Returns
    --------
        int: int object
            Byte length of the matched element, or 0 if EOF reached.
    """
    vr_short = {
        "AE",
        "AS",
        "AT",
        "CS",
        "DA",
        "DS",
        "DT",
        "FL",
        "FD",
        "IS",
        "LO",
        "LT",
        "OF",
        "PN",
        "SH",
        "SL",
        "ST",
        "SS",
        "TM",
        "UI",
        "UL",
        "US",
    }
    all_vrs = vr_short | {"OB", "OW", "SQ", "UN", "UT"}
    target_g = str_group.upper()
    target_e = str_element.upper()

    while True:
        tag_bytes = fd.read(4)
        vr_bytes = fd.read(2)
        if len(tag_bytes) < 4 or len(vr_bytes) < 2:
            return 0
        g_val, e_val = struct.unpack("<HH", tag_bytes)
        g_hex = f"{g_val:X}"
        e_hex = f"{e_val:X}"
        vr_str = vr_bytes.decode("latin-1", errors="ignore")

        if vr_str not in all_vrs:
            fd.seek(-2, 1)
            len_bytes = fd.read(4)
            if len(len_bytes) < 4:
                return 0
            curr_len = struct.unpack("<I", len_bytes)[0]
            if curr_len != 0xFFFFFFFF:
                if g_hex == target_g and e_hex == target_e:
                    return int(curr_len)
                fd.seek(curr_len, 1)
        else:
            if vr_str in vr_short:
                curr_len = struct.unpack("<H", fd.read(2))[0]
            else:
                fd.read(2)
                curr_len = struct.unpack("<I", fd.read(4))[0]
            if g_hex == target_g and e_hex == target_e:
                return int(curr_len)
            if curr_len != 0xFFFFFFFF:
                fd.seek(curr_len, 1)


def dicom_get_spectrum_siemens(fd: BinaryIO) -> NDArray[np.complex128]:
    """Read complex Siemens spectroscopy FID from tag ``(7FE1, 1010)``.

    Mirrors ``dicom_get_spectrum_siemens.m``.

    Parameters
    ------------
    fd: BinaryIO
        Open DICOM file stream positioned after ``'DICM'``.

    Returns
    --------
        NDArray[np.complex128]: 1D complex numpy array
            Complex FID vector.
    """
    field_length = dicom_move(fd, "7FE1", "1010")
    field_size = field_length // 4
    vals = np.fromfile(fd, dtype="<f4", count=field_size).astype(np.float64)
    result: NDArray[np.complex128] = vals[0::2] + 1j * vals[1::2]
    return result


def _set_p_val(
    p: MRSStruct, key: str, idx: int, val: float | list[float], cols: int = 0
) -> None:
    """Assign scan-indexed parameter inside ``MRS_struct.p``."""
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


def _read_single_dicom_fid(filepath: str) -> NDArray[np.complex128]:
    """Read one spectroscopy FID from standard or Siemens private DICOM tag."""
    ds = pydicom.dcmread(filepath, force=True)
    if "SpectroscopyData" in ds:
        raw = np.asarray(ds.SpectroscopyData, dtype=np.float64)
        return np.conj(raw[0::2] + 1j * raw[1::2])
    with dicom_open(filepath) as fd:
        return dicom_get_spectrum_siemens(fd)


def _list_sorted_files(sample_file: str) -> list[str]:
    """List and naturally sort all files sharing the extension in folder."""
    path = Path(sample_file)
    folder = path.parent if str(path.parent) else Path(".")
    ext = path.suffix.lower()
    matches = [
        p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ext
    ]
    sorted_names = sort_nat(matches)
    return [str(folder / name) for name in sorted_names]


def _populate_dicom_header_params(
    mrs_struct: MRSStruct, hdr: MRSStruct, ii: int
) -> None:
    """Copy parsed ASCCONV header values into ``MRS_struct.p``."""
    p = mrs_struct["p"]
    p["seq"] = hdr["sequenceFileName"]
    _set_p_val(p, "TR", ii, float(hdr["TR"]))
    _set_p_val(p, "TE", ii, float(hdr["TE"]))
    _set_p_val(p, "npoints", ii, float(hdr["vectorSize"]))
    _set_p_val(p, "Navg", ii, 2.0 * float(hdr["nAverages"]))
    _set_p_val(p, "nrows", ii, 2.0 * float(hdr["nAverages"]))
    _set_p_val(p, "sw", ii, 1.0 / float(hdr["dwellTime"]) * 1e9 * 0.5)
    _set_p_val(p, "LarmorFreq", ii, float(hdr["tx_freq"]) * 1e-6)
    _set_p_val(
        p,
        "voxdim",
        ii,
        [
            float(hdr["VoI_PeFOV"]),
            float(hdr["VoI_RoFOV"]),
            float(hdr["VoIThickness"]),
        ],
        cols=3,
    )
    _set_p_val(p, "VoI_InPlaneRot", ii, float(hdr["VoI_InPlaneRot"]))
    if "PosSag" in hdr:
        _set_p_val(
            p,
            "voxoff",
            ii,
            [float(hdr["PosSag"]), float(hdr["PosCor"]), float(hdr["PosTra"])],
            cols=3,
        )
    _set_p_val(p, "NormCor", ii, float(hdr["NormCor"]))
    _set_p_val(p, "NormSag", ii, float(hdr["NormSag"]))
    _set_p_val(p, "NormTra", ii, float(hdr["NormTra"]))


def dicom_read(
    mrs_struct: MRSStruct, metabfile: str, waterfile: str | None = None
) -> MRSStruct:
    """Load edited MRS data stored as individual ``.dcm`` files.

    Mirrors ``DICOMRead.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.
    metabfile: str
        Path to a representative ``.dcm`` file in the metabolite folder.
    waterfile: str | None
        Optional path to a representative ``.dcm`` file in the water folder.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    dcm_files = _list_sorted_files(metabfile)
    hdr = read_dcm_header(metabfile)
    _populate_dicom_header_params(mrs_struct, hdr, ii)

    npts = int(p["npoints"][ii])
    data = np.zeros((npts, len(dcm_files)), dtype=np.complex128)
    for jj, fp in enumerate(dcm_files):
        data[:, jj] = _read_single_dicom_fid(fp)

    if data.shape[1] > 1:
        half = data.shape[1] // 2
        idx = np.column_stack([np.arange(half), np.arange(half, 2 * half)]).ravel()
        data = data[:, idx]

    fids = mrs_struct.ensure_substruct("fids")
    fids["data"] = data

    if waterfile is not None:
        w_hdr = read_dcm_header(waterfile)
        _set_p_val(p, "TR_water", ii, float(w_hdr["TR"]))
        _set_p_val(p, "TE_water", ii, float(w_hdr["TE"]))
        w_files = _list_sorted_files(waterfile)
        w_data = np.zeros((npts, len(w_files)), dtype=np.complex128)
        for jj, fp in enumerate(w_files):
            w_data[:, jj] = _read_single_dicom_fid(fp)
        fids["data_water"] = np.mean(w_data, axis=1)

    return mrs_struct


def siemens_dicom_read(
    mrs_struct: MRSStruct, metabfile: str, waterfile: str | None = None
) -> MRSStruct:
    """Load edited MRS data stored as Siemens ``.ima`` DICOM files.

    Mirrors ``SiemensDICOMRead.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.
    metabfile: str
        Path to a representative ``.ima`` file in the metabolite folder.
    waterfile: str | None
        Optional path to a representative ``.ima`` file in the water folder.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct``.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    ima_files = _list_sorted_files(metabfile)
    hdr = read_dcm_header(metabfile)
    _populate_dicom_header_params(mrs_struct, hdr, ii)

    npts = int(p["npoints"][ii])
    data = np.zeros((npts, len(ima_files)), dtype=np.complex128)
    for kk, fp in enumerate(ima_files):
        with dicom_open(fp) as fd:
            data[:, kk] = dicom_get_spectrum_siemens(fd)

    if data.shape[1] > 1:
        half = data.shape[1] // 2
        reordered = np.zeros_like(data)
        reordered[:, 0::2] = data[:, :half]
        reordered[:, 1::2] = data[:, half : 2 * half]
        data = reordered

    fids = mrs_struct.ensure_substruct("fids")
    fids["data"] = data

    if waterfile is not None:
        w_hdr = read_dcm_header(waterfile)
        _set_p_val(p, "TR_water", ii, float(w_hdr["TR"]))
        _set_p_val(p, "TE_water", ii, float(w_hdr["TE"]))
        w_files = _list_sorted_files(waterfile)
        w_data = np.zeros((npts, len(w_files)), dtype=np.complex128)
        for kk, fp in enumerate(w_files):
            with dicom_open(fp) as fd:
                w_data[:, kk] = dicom_get_spectrum_siemens(fd)
        fids["data_water"] = np.mean(w_data, axis=1)

    return mrs_struct
