"""Philips .data/.list MRS reader (3T and 7T) for Gannet."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct
from gannet.io.philips import parse_spar_header


@dataclass
class PhilipsListInfo:
    """Parsed metadata and record index table from a Philips ``.list`` file.

    Parameters
    ------------
    records: list[dict[str, Any]]
        Data/noise/misc records parsed from ``.list``.
    params: dict[str, float | int]
        Scalar summary attributes (``F_resolution``, ``X_resolution``,
        ``kx_range``, ``number_of_mixes``, etc.).
    """

    records: list[dict[str, Any]]
    params: dict[str, float | int]


def parse_list_file(list_path: str) -> PhilipsListInfo:
    """Parse a Philips ``.list`` text index file into records and header scalars.

    Parameters
    ------------
    list_path: str
        Path to the ``.list`` file accompanying a ``.data`` file.

    Returns
    --------
        PhilipsListInfo:
            Structured records and parsed scalar metadata.
    """
    records: list[dict[str, Any]] = []
    params: dict[str, float | int] = {}
    attr_re = re.compile(r"^\.\s+\d+\s+\d+\s+\d+\s+(\S+)\s*:\s*([-+eE0-9.\s]+)$")
    col_names = [
        "typ",
        "mix",
        "dyn",
        "card",
        "echo",
        "loca",
        "chan",
        "extr1",
        "extr2",
        "ky",
        "kz",
        "aver",
        "sign",
        "rf",
        "grad",
        "enc",
        "rtop",
        "rr",
        "size",
        "offset",
    ]
    with open(list_path, "r", encoding="latin-1") as fid:
        for raw_line in fid:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            attr_match = attr_re.match(line)
            if attr_match:
                key = attr_match.group(1)
                vals = attr_match.group(2).split()
                if vals:
                    try:
                        params[key] = (
                            float(vals[-1])
                            if ("." in vals[-1] or "e" in vals[-1].lower())
                            else int(vals[-1])
                        )
                    except ValueError:
                        pass
                continue
            if line.startswith("."):
                continue
            tokens = line.split()
            if len(tokens) >= 20:
                rec: dict[str, Any] = {"typ": tokens[0]}
                for col, tok in zip(col_names[1:], tokens[1:20]):
                    rec[col] = int(tok)
                records.append(rec)
    return PhilipsListInfo(records=records, params=params)


def load_raw_kspace(data_path: str, list_path: str | None = None) -> dict[str, Any]:
    """Read Philips ``.data`` / ``.list`` raw complex FID records.

    Parameters
    ------------
    data_path: str
        Path to the ``.data`` binary file.
    list_path: str | None
        Optional path to ``.list``; inferred from ``data_path`` if omitted.

    Returns
    --------
        dict[str, Any]:
            Dictionary with keys ``std``, ``noise``, and ``list_info``.
    """
    if list_path is None:
        base, _ = os.path.splitext(data_path)
        cand = f"{base}.list"
        if not os.path.exists(cand):
            cand = f"{base}.LIST"
        list_path = cand
    list_info = parse_list_file(list_path)
    std_recs = [r for r in list_info.records if r["typ"] == "STD"]
    noi_recs = [r for r in list_info.records if r["typ"] == "NOI"]

    with open(data_path, "rb") as fid:
        std_data: list[tuple[dict[str, Any], NDArray[np.complex64]]] = []
        for rec in std_recs:
            fid.seek(rec["offset"], os.SEEK_SET)
            n_floats = rec["size"] // 4
            raw = np.fromfile(fid, dtype="<f4", count=n_floats)
            cdata = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
            std_data.append((rec, cdata))

        noise_list: list[tuple[int, NDArray[np.complex64]]] = []
        for rec in noi_recs:
            fid.seek(rec["offset"], os.SEEK_SET)
            n_floats = rec["size"] // 4
            raw = np.fromfile(fid, dtype="<f4", count=n_floats)
            cdata = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
            noise_list.append((rec["chan"], cdata))

    return {
        "std": std_data,
        "noise": noise_list,
        "list_info": list_info,
    }


def _coil_combine_philips_data(
    coil_fid: NDArray[np.complex128],
    noise_records: list[tuple[int, NDArray[np.complex64]]],
    water_fid: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
    """Combine multi-channel Philips ``.data`` FIDs using noise/water weights.

    Parameters
    ------------
    coil_fid: NDArray[np.complex128]
        Array shaped ``(npoints, ncoils, navg)``.
    noise_records: list[tuple[int, NDArray[np.complex64]]]
        Noise scans indexed by coil channel.
    water_fid: NDArray[np.complex128] | None
        Optional unsuppressed water array ``(npoints, ncoils, navg_w)`` used
        for coil phase/amplitude weighting.

    Returns
    --------
        NDArray[np.complex128]:
            Combined complex FID array shaped ``(npoints, navg)``.
    """
    _npts, ncoils, _navg = coil_fid.shape
    if ncoils == 1:
        return coil_fid[:, 0, :]

    noise_var = np.ones(ncoils, dtype=np.float64)
    if noise_records:
        ch_map: dict[int, list[NDArray[np.complex64]]] = {}
        for ch, arr in noise_records:
            ch_map.setdefault(ch, []).append(arr)
        for ch, arrs in ch_map.items():
            if 0 <= ch < ncoils:
                cat = np.concatenate(arrs)
                var = float(np.var(cat))
                if var > 0:
                    noise_var[ch] = var

    ref_src = water_fid if (water_fid is not None and water_fid.size > 0) else coil_fid
    ref_mean = np.mean(ref_src, axis=2)
    sig_amp = (
        np.abs(ref_mean[0, :])
        if np.any(np.abs(ref_mean[0, :]) > 0)
        else np.max(np.abs(ref_mean), axis=0)
    )
    phase_ref = ref_mean[0, :]
    norm_phase = np.where(np.abs(phase_ref) > 0, phase_ref / np.abs(phase_ref), 1.0)
    weights = (sig_amp / noise_var) * np.conj(norm_phase)
    norm_w = np.linalg.norm(weights)
    if norm_w > 0:
        weights /= norm_w
    else:
        weights = np.ones(ncoils, dtype=np.complex128) / np.sqrt(ncoils)

    combined = np.tensordot(coil_fid, weights, axes=([1], [0]))
    if combined.ndim == 1:
        combined = combined[:, np.newaxis]
    return np.asarray(combined, dtype=np.complex128)


def _build_mix_array(
    std_recs: list[tuple[dict[str, Any], NDArray[np.complex64]]],
    mix_val: int,
    npts: int,
    ncoils: int,
    chan_to_idx: dict[int, int],
) -> NDArray[np.complex128]:
    """Group Philips ``STD`` records for one ``mix`` into ``(npts, ncoils, navg)``.

    Parameters
    ------------
    std_recs: list[tuple[dict[str, Any], NDArray[np.complex64]]]
        Loaded STD record metadata and complex FID pairs.
    mix_val: int
        Target mix integer identifier.
    npts: int
        Sample points per FID.
    ncoils: int
        Receiver coil count.
    chan_to_idx: dict[int, int]
        Mapping from scanner channel ID to contiguous array column.

    Returns
    --------
        NDArray[np.complex128]:
            Grouped coil FID array shaped ``(npts, ncoils, navg)``.
    """
    sub = [(rec, arr) for rec, arr in std_recs if rec["mix"] == mix_val]
    keys = sorted(
        {
            (
                r["dyn"],
                r["card"],
                r["echo"],
                r["loca"],
                r["extr1"],
                r["extr2"],
                r["aver"],
            )
            for r, _ in sub
        }
    )
    key_to_avg = {k: idx for idx, k in enumerate(keys)}
    out = np.zeros((npts, ncoils, len(keys)), dtype=np.complex128)
    for r, arr in sub:
        k = (
            r["dyn"],
            r["card"],
            r["echo"],
            r["loca"],
            r["extr1"],
            r["extr2"],
            r["aver"],
        )
        out[: len(arr), chan_to_idx[r["chan"]], key_to_avg[k]] = arr
    return out


def philips_data_read(
    mrs_struct: MRSStruct,
    fname: str,
    fname_water: str | None = None,
) -> MRSStruct:
    """Read Philips ``.data/.list`` (3T or 7T) spectroscopy data into ``MRSStruct``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        Active ``MRSStruct`` with ``ii`` set to current 0-based scan index.
    fname: str
        Path to metabolite ``.data`` file.
    fname_water: str | None
        Optional separate unsuppressed water ``.data`` file (or ``.sdat``).

    Returns
    --------
        MRSStruct:
            Updated ``MRSStruct`` with combined complex FIDs and metadata.
    """
    ii = int(mrs_struct["ii"])
    kspace = load_raw_kspace(fname)
    std_recs: list[tuple[dict[str, Any], NDArray[np.complex64]]] = kspace["std"]
    noise_recs = kspace["noise"]
    if not std_recs:
        raise ValueError(f"No STD records found in {fname}")

    # Parse companion SPAR header if available alongside .data
    base, _ = os.path.splitext(fname)
    for ext in (".spar", ".SPAR"):
        spar_cand = f"{base}{ext}"
        if os.path.exists(spar_cand):
            spar = parse_spar_header(spar_cand)
            if "synthesizer_frequency" in spar:
                mrs_struct["p"]["LarmorFreq"][ii] = (
                    float(spar["synthesizer_frequency"]) / 1e6
                )
            if "sample_frequency" in spar:
                mrs_struct["p"]["sw"][ii] = float(spar["sample_frequency"])
            if "repetition_time" in spar:
                mrs_struct["p"]["TR"][ii] = float(spar["repetition_time"])
            if "echo_time" in spar:
                mrs_struct["p"]["TE"][ii] = float(spar["echo_time"])
            mrs_struct["p"]["voxdim"][ii] = [
                float(spar.get("lr_size", 30.0)),
                float(spar.get("ap_size", 30.0)),
                float(spar.get("cc_size", 30.0)),
            ]
            mrs_struct["p"]["voxoff"][ii] = [
                float(spar.get("lr_off_center", 0.0)),
                float(spar.get("ap_off_center", 0.0)),
                float(spar.get("cc_off_center", 0.0)),
            ]
            mrs_struct["p"]["voxang"][ii] = [
                float(spar.get("lr_angulation", 0.0)),
                float(spar.get("ap_angulation", 0.0)),
                float(spar.get("cc_angulation", 0.0)),
            ]
            break

    mixes = sorted({rec["mix"] for rec, _ in std_recs})
    chans = sorted({rec["chan"] for rec, _ in std_recs})
    chan_to_idx = {ch: idx for idx, ch in enumerate(chans)}
    ncoils = len(chans)
    npts = max(len(arr) for _, arr in std_recs)

    metab_mix = mixes[0]
    metab_coil = _build_mix_array(std_recs, metab_mix, npts, ncoils, chan_to_idx)

    water_coil: NDArray[np.complex128] | None = None
    if len(mixes) > 1:
        water_coil = _build_mix_array(std_recs, mixes[1], npts, ncoils, chan_to_idx)
    elif fname_water is not None:
        w_ks = load_raw_kspace(fname_water)
        w_recs = w_ks["std"]
        if w_recs:
            w_chans = sorted({r["chan"] for r, _ in w_recs})
            w_c2i = {ch: idx for idx, ch in enumerate(w_chans)}
            w_keys = sorted(
                {
                    (
                        r["dyn"],
                        r["card"],
                        r["echo"],
                        r["loca"],
                        r["extr1"],
                        r["extr2"],
                        r["aver"],
                    )
                    for r, _ in w_recs
                }
            )
            w_k2a = {k: idx for idx, k in enumerate(w_keys)}
            w_npts = max(len(arr) for _, arr in w_recs)
            water_coil = np.zeros(
                (w_npts, len(w_chans), len(w_keys)), dtype=np.complex128
            )
            for r, arr in w_recs:
                k = (
                    r["dyn"],
                    r["card"],
                    r["echo"],
                    r["loca"],
                    r["extr1"],
                    r["extr2"],
                    r["aver"],
                )
                water_coil[: len(arr), w_c2i[r["chan"]], w_k2a[k]] = arr

    is_7t = float(mrs_struct["p"]["LarmorFreq"][ii]) > 250.0
    if is_7t and water_coil is not None and water_coil.shape[0] != metab_coil.shape[0]:
        # Match 7T readout length if oversampled differently
        min_pts = min(water_coil.shape[0], metab_coil.shape[0])
        water_use = water_coil[:min_pts]
    else:
        water_use = water_coil

    metab_comb = _coil_combine_philips_data(metab_coil, noise_recs, water_use)
    mrs_struct["p"]["npoints"][ii] = metab_comb.shape[0]
    mrs_struct["p"]["nrows"][ii] = metab_comb.shape[1]
    mrs_struct["p"]["Navg"][ii] = metab_comb.shape[1]
    mrs_struct["fids"]["data"] = metab_comb

    if water_coil is not None:
        water_comb = _coil_combine_philips_data(water_coil, noise_recs, water_coil)
        mrs_struct["p"]["Nwateravg"][ii] = water_comb.shape[1]
        mrs_struct["fids"]["data_water"] = np.mean(water_comb, axis=1)

    return mrs_struct
