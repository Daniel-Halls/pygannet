"""Determine editing subexperiment order (ON/OFF) for MEGA-PRESS and HERMES."""

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct


def _find_max_naa_water(
    spec: NDArray[np.float64],
    water_lim: NDArray[np.bool_],
    naa_lim: NDArray[np.bool_],
) -> tuple[float, float]:
    """Find maximum absolute intensities in water and NAA frequency windows.

    Parameters
    ------------
    spec: NDArray[np.float64]
        1D magnitude-real spectrum.
    water_lim: NDArray[np.bool_]
        Boolean mask for the residual water window (4-5 ppm).
    naa_lim: NDArray[np.bool_]
        Boolean mask for the NAA window (1.7-2.3 ppm).

    Returns
    --------
        tuple[float, float]: tuple of two floats
            Maximum amplitudes ``(max_w, max_naa)``.
    """
    max_w = float(max(abs(np.max(spec[water_lim])), abs(np.min(spec[water_lim]))))
    max_naa = float(max(abs(np.max(spec[naa_lim])), abs(np.min(spec[naa_lim]))))
    result = (max_w, max_naa)
    return result


def _manual_on_off_order(
    order_str: str, hermes: bool, n_transients: int
) -> NDArray[np.int64]:
    """Construct ON_OFF array from an explicit user-specified order string.

    Parameters
    ------------
    order_str: str
        Either ``'offfirst'`` / ``'onfirst'`` or a 4-letter HERMES string.
    hermes: bool
        Whether the acquisition is HERMES/HERCULES.
    n_transients: int
        Total number of transients in ``fids.data``.

    Returns
    --------
        NDArray[np.int64]: 2D numpy array
            ON/OFF indicator array of shape ``(n_targets, n_transients)``.
    """
    if hermes:
        mapping = {
            "A": np.array([1, 1], dtype=np.int64),
            "B": np.array([1, 0], dtype=np.int64),
            "C": np.array([0, 1], dtype=np.int64),
            "D": np.array([0, 0], dtype=np.int64),
        }
        on_off = np.zeros((2, 4), dtype=np.int64)
        for idx, ch in enumerate(order_str.upper()[:4]):
            if ch in mapping:
                on_off[:, idx] = mapping[ch]
        reps = max(n_transients // 4, 1)
        result: NDArray[np.int64] = np.tile(on_off, (1, reps))
        return result

    if order_str.lower() == "offfirst":
        pattern = np.array([[0, 1]], dtype=np.int64)
    else:
        pattern = np.array([[1, 0]], dtype=np.int64)
    reps = max(n_transients // 2, 1)
    result = np.tile(pattern, (1, reps))
    return result


def specify_on_off_order(mrs_struct: MRSStruct) -> MRSStruct:
    """Determine the temporal ON/OFF editing order from ``fids.data``.

    Mirrors ``SpecifyOnOffOrder.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure containing ``fids.data`` and ``p``.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with ``fids.ON_OFF`` populated.
    """
    p = mrs_struct["p"]
    fids_data = np.asarray(mrs_struct["fids"]["data"])
    n_transients = fids_data.shape[1]
    order_str = str(p.get("ON_OFF_order", "") or "")
    hermes = bool(p.get("HERMES", 0))
    targets = list(p["target"])

    if order_str:
        mrs_struct["fids"]["ON_OFF"] = _manual_on_off_order(
            order_str, hermes, n_transients
        )
        return mrs_struct

    spec = np.abs(np.real(np.fft.fftshift(np.fft.fft(fids_data, axis=0), axes=0)))
    npoints = int(p["npoints"][0])
    sw = float(p["sw"][0])
    larmor = float(p["LarmorFreq"][0])
    freq_range = sw / larmor
    freq = (
        (npoints + 1 - np.arange(1, npoints + 1)) / npoints * freq_range
        + 4.68
        - freq_range / 2.0
    )
    naa_lim = (freq <= 2.3) & (freq >= 1.7)
    water_lim = (freq <= 5.0) & (freq >= 4.0)
    water_lim2 = (freq <= 4.25) & (freq >= 3.5)

    if len(targets) == 1:
        ge_meta = p.get("GE", {})
        edit_rf = ge_meta.get("editRF", {}) if isinstance(ge_meta, dict) else {}
        freq_ppm = edit_rf.get("freq_ppm") if isinstance(edit_rf, dict) else None
        if (
            str(p.get("vendor", "")) == "GE"
            and not hermes
            and targets[0] in {"GABA", "Glx", "GABAGlx"}
            and freq_ppm is not None
        ):
            fp_arr = np.asarray(freq_ppm, dtype=np.float64).ravel()
            if fp_arr.size >= 2 and abs(float(fp_arr[0]) - float(fp_arr[1])) > 1.5:
                d0 = abs(float(fp_arr[0]) - 1.9)
                d1 = abs(float(fp_arr[1]) - 1.9)
                if min(d0, d1) < 0.5:
                    pattern = np.array(
                        [[1, 0]] if d0 < d1 else [[0, 1]], dtype=np.int64
                    )
                    mrs_struct["fids"]["ON_OFF"] = np.tile(
                        pattern, (1, n_transients // 2)
                    )
                    return mrs_struct

        spec_c = np.fft.fftshift(np.fft.fft(fids_data, axis=0), axes=0)
        if targets[0] in {"GABA", "Glx", "GABAGlx"}:
            # Phase-correct by Cr/Cho peak and compare narrow NAA core [1.92, 2.10] ppm
            mean_c = np.mean(spec_c, axis=1)
            cr_win = (freq <= 3.12) & (freq >= 2.90)
            phi_ref = (
                float(np.angle(mean_c[cr_win][int(np.argmax(np.abs(mean_c[cr_win])))]))
                if np.any(cr_win)
                else 0.0
            )
            phased = np.real(spec_c * np.exp(-1j * phi_ref))
            naa_core = (freq <= 2.10) & (freq >= 1.92)
            freq_lim = naa_core if np.any(naa_core) else naa_lim
            spec_a = np.mean(phased[freq_lim, 0::2], axis=1)
            spec_b = np.mean(phased[freq_lim, 1::2], axis=1)
        elif targets[0] == "GSH":
            freq_lim = water_lim
            spec_a = np.mean(spec[freq_lim, 0::2], axis=1)
            spec_b = np.mean(spec[freq_lim, 1::2], axis=1)
        else:
            freq_lim = water_lim2
            spec_a = np.mean(spec[freq_lim, 0::2], axis=1)
            spec_b = np.mean(spec[freq_lim, 1::2], axis=1)

        if np.max(spec_a - spec_b) > np.max(spec_b - spec_a):
            pattern = np.array([[0, 1]], dtype=np.int64)
        else:
            pattern = np.array([[1, 0]], dtype=np.int64)
        mrs_struct["fids"]["ON_OFF"] = np.tile(pattern, (1, n_transients // 2))
        return mrs_struct

    max_first = np.zeros(4, dtype=np.float64)
    max_second = np.zeros(4, dtype=np.float64)
    for idx in range(4):
        mean_sub = np.mean(spec[:, idx::4], axis=1)
        max_first[idx], max_second[idx] = _find_max_naa_water(
            mean_sub, water_lim, naa_lim
        )

    order_first = np.argsort(max_first, kind="stable")
    order_second = np.argsort(max_second, kind="stable")

    gaba_on = np.zeros(4, dtype=np.int64)
    gsh_on = np.zeros(4, dtype=np.int64)
    etoh_on = np.zeros(4, dtype=np.int64)

    for ii in range(4):
        idx_first = int(np.where(order_first == ii)[0][0])
        idx_second = int(np.where(order_second == ii)[0][0])
        gaba_on[ii] = 0 if idx_second in (2, 3) else 1
        gsh_on[ii] = 0 if idx_first in (2, 3) else 1
        if len(targets) == 3:
            if (gaba_on[ii] == 1 and gsh_on[ii] == 1) or (
                gaba_on[ii] == 0 and gsh_on[ii] == 0
            ):
                etoh_on[ii] = 0
            else:
                etoh_on[ii] = 1

    reps = max(n_transients // 4, 1)
    if len(targets) == 2:
        mat = np.vstack([gaba_on, gsh_on])
    else:
        mat = np.vstack([etoh_on, gaba_on, gsh_on])
    mrs_struct["fids"]["ON_OFF"] = np.tile(mat, (1, reps))
    return mrs_struct
