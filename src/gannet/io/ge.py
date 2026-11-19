"""GE P-file (.7) reader."""

import numpy as np
from numpy.typing import NDArray

from gannet.config import MRSStruct


def _get_ge_revision_offsets(rev_str: str) -> dict[str, int]:
    """Return 1-based MATLAB header word offsets for a GE P-file revision."""
    if rev_str == "14.3":
        return {
            "rdb_hdr_off_image": 377,
            "rdb_hdr_off_data": 368,
            "rdb_hdr_ps_mps_freq": 107,
            "rdb_hdr_user0": 55,
            "rdb_hdr_user4": 59,
            "rdb_hdr_user19": 74,
            "rdb_hdr_nechoes": 36,
            "rdb_hdr_navs": 37,
            "rdb_hdr_nframes": 38,
            "rdb_hdr_point_size": 42,
            "rdb_hdr_da_xres": 52,
            "rdb_hdr_da_yres": 53,
            "rdb_hdr_dab_start_rcv": 101,
            "rdb_hdr_dab_stop_rcv": 102,
            "image_te": 181,
            "image_tr": 179,
            "image_user8": 38,
            "image_user11": 41,
            "image_user19": 49,
            "image_user20": 50,
            "image_user22": 52,
            "image_user24": 56,
            "tlhc": 121,
            "trhc": 124,
            "brhc": 127,
        }
    if rev_str == "16":
        return {
            "rdb_hdr_off_image": 377,
            "rdb_hdr_off_data": 368,
            "rdb_hdr_ps_mps_freq": 107,
            "rdb_hdr_user0": 55,
            "rdb_hdr_user4": 59,
            "rdb_hdr_user19": 74,
            "rdb_hdr_nechoes": 36,
            "rdb_hdr_navs": 37,
            "rdb_hdr_nframes": 38,
            "rdb_hdr_point_size": 42,
            "rdb_hdr_da_xres": 52,
            "rdb_hdr_da_yres": 53,
            "rdb_hdr_dab_start_rcv": 101,
            "rdb_hdr_dab_stop_rcv": 102,
            "image_te": 193,
            "image_tr": 191,
            "image_user8": 50,
            "image_user11": 53,
            "image_user19": 61,
            "image_user20": 62,
            "image_user22": 64,
            "image_user24": 68,
            "tlhc": 133,
            "trhc": 136,
            "brhc": 139,
        }
    if rev_str in ("20.006", "20.007", "24"):
        return {
            "rdb_hdr_off_image": 377,
            "rdb_hdr_off_data": 368,
            "rdb_hdr_ps_mps_freq": 107,
            "rdb_hdr_user0": 55,
            "rdb_hdr_user4": 59,
            "rdb_hdr_user19": 74,
            "rdb_hdr_nechoes": 36,
            "rdb_hdr_navs": 37,
            "rdb_hdr_nframes": 38,
            "rdb_hdr_point_size": 42,
            "rdb_hdr_da_xres": 52,
            "rdb_hdr_da_yres": 53,
            "rdb_hdr_dab_start_rcv": 101,
            "rdb_hdr_dab_stop_rcv": 102,
            "image_te": 267,
            "image_tr": 265,
            "image_user8": 98,
            "image_user11": 101,
            "image_user19": 109,
            "image_user20": 110,
            "image_user22": 112,
            "image_user24": 116,
            "tlhc": 181,
            "trhc": 184,
            "brhc": 187,
        }
    return {
        "rdb_hdr_off_image": 11,
        "rdb_hdr_off_data": 2,
        "rdb_hdr_ps_mps_freq": 123,
        "rdb_hdr_user0": 71,
        "rdb_hdr_user4": 75,
        "rdb_hdr_user19": 90,
        "rdb_hdr_nechoes": 74,
        "rdb_hdr_navs": 75,
        "rdb_hdr_nframes": 76,
        "rdb_hdr_point_size": 80,
        "rdb_hdr_da_xres": 90,
        "rdb_hdr_da_yres": 91,
        "rdb_hdr_dab_start_rcv": 133,
        "rdb_hdr_dab_stop_rcv": 134,
        "image_te": 267,
        "image_tr": 265,
        "image_user8": 98,
        "image_user11": 101,
        "image_user19": 109,
        "image_user20": 110,
        "image_user22": 112,
        "image_user24": 116,
        "tlhc": 181,
        "trhc": 184,
        "brhc": 187,
    }


def _format_rev(rev_val: float) -> str:
    """Format GE rdbm_rev_num matching MATLAB ``num2str``."""
    return f"{rev_val:g}"


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
        p[key][idx, :] = np.asarray(value_or_arr := val, dtype=np.float64)
        _ = value_or_arr


def _gls_combine(
    water_data: NDArray[np.complex128], metab_data: NDArray[np.complex128]
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Combine GE multi-receiver FIDs using generalized least squares."""
    n_ch, n_pts, n_reps = water_data.shape
    noise_pts = np.zeros(n_pts, dtype=bool)
    noise_pts[int(np.ceil(0.75 * n_pts)) - 1 :] = True
    noise_mask_w = np.tile(noise_pts, n_reps)
    tmp_w = water_data.reshape((n_ch, n_pts * n_reps), order="F")
    e_w = tmp_w[:, noise_mask_w]
    psi_w = e_w @ e_w.conj().T
    s_vec = np.mean(water_data, axis=2)[:, 0]
    psi_w_inv = np.linalg.pinv(psi_w)
    w_w = (1.0 / (s_vec.conj().T @ psi_w_inv @ s_vec)) * (s_vec.conj().T @ psi_w_inv)
    water_comb = np.mean(
        np.sum(w_w[:, np.newaxis, np.newaxis] * water_data, axis=0), axis=1
    )

    n_ch_m, n_pts_m, n_reps_m = metab_data.shape
    noise_pts_m = np.zeros(n_pts_m, dtype=bool)
    noise_pts_m[int(np.ceil(0.75 * n_pts_m)) - 1 :] = True
    noise_mask_m = np.tile(noise_pts_m, n_reps_m)
    tmp_m = metab_data.reshape((n_ch_m, n_pts_m * n_reps_m), order="F")
    e_m = tmp_m[:, noise_mask_m]
    psi_m = e_m @ e_m.conj().T
    psi_m_inv = np.linalg.pinv(psi_m)
    w_m = (1.0 / (s_vec.conj().T @ psi_m_inv @ s_vec)) * (s_vec.conj().T @ psi_m_inv)
    metab_comb = np.sum(w_m[:, np.newaxis, np.newaxis] * metab_data, axis=0)
    if metab_comb.ndim == 1:
        metab_comb = metab_comb[:, np.newaxis]
    result = (metab_comb, water_comb)
    return result


def ge_read(mrs_struct: MRSStruct, fname: str) -> MRSStruct:
    """Load GE P-file (``.7``) raw MRS data and header geometry.

    Mirrors ``GERead.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure.
    fname: str
        Path to GE P-file.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with GE metadata and coil-combined FIDs.
    """
    ii = int(mrs_struct["ii"])
    p = mrs_struct["p"]
    ge_sub = p.ensure_substruct("GE")

    with open(fname, "rb") as fid:
        rdbm_rev_num = float(np.fromfile(fid, dtype=">f4", count=1)[0])
        if rdbm_rev_num == 7.0:
            endian = ">"
            pfile_header_size = 39984
        elif rdbm_rev_num == 8.0:
            endian = ">"
            pfile_header_size = 60464
        elif 5.0 < rdbm_rev_num < 6.0:
            endian = ">"
            pfile_header_size = 39940
        else:
            endian = "<"
            fid.seek(0, 0)
            rdbm_rev_num = float(np.fromfile(fid, dtype="<f4", count=1)[0])
            pfile_header_size = 61464 if rdbm_rev_num == 9.0 else 66072

        rev_str = _format_rev(rdbm_rev_num)
        supported = {
            "14.3",
            "16",
            "20.006",
            "20.007",
            "24",
            "26.002",
            "27",
            "27.001",
            "28.002",
            "28.003",
            "30",
            "30.1",
            "31",
        }
        if rev_str not in supported:
            raise ValueError(f"GERead does not support P-file revision {rev_str}.")
        _set_p_val(ge_sub, "rdbm_rev_num", ii, rdbm_rev_num)
        offs = _get_ge_revision_offsets(rev_str)

        fid.seek(0, 0)
        hdr_short = np.fromfile(
            fid, dtype=f"{endian}i2", count=offs["rdb_hdr_dab_stop_rcv"]
        )
        fid.seek(0, 0)
        hdr_float = np.fromfile(fid, dtype=f"{endian}f4", count=offs["rdb_hdr_user19"])
        n_int = max(offs["rdb_hdr_off_image"], offs["rdb_hdr_ps_mps_freq"])
        fid.seek(0, 0)
        hdr_int = np.fromfile(fid, dtype=f"{endian}i4", count=n_int)
        fid.seek(0, 0)
        hdr_uint = np.fromfile(fid, dtype=f"{endian}u4", count=n_int)

        if rdbm_rev_num > 11.0:
            pfile_header_size = int(hdr_int[offs["rdb_hdr_off_data"] - 1])

        larmor = float(hdr_uint[offs["rdb_hdr_ps_mps_freq"] - 1]) / 1e7
        _set_p_val(p, "LarmorFreq", ii, larmor)
        _set_p_val(p, "sw", ii, float(hdr_float[offs["rdb_hdr_user0"] - 1]))
        nechoes = int(hdr_short[offs["rdb_hdr_nechoes"] - 1])
        _set_p_val(ge_sub, "nechoes", ii, nechoes)
        nex = int(hdr_short[offs["rdb_hdr_navs"] - 1])
        _set_p_val(ge_sub, "NEX", ii, nex)
        nframes = int(hdr_short[offs["rdb_hdr_nframes"] - 1])
        point_size = int(hdr_short[offs["rdb_hdr_point_size"] - 1])
        npoints = int(hdr_short[offs["rdb_hdr_da_xres"] - 1])
        nrows = int(hdr_short[offs["rdb_hdr_da_yres"] - 1])

        start_recv = int(hdr_short[offs["rdb_hdr_dab_start_rcv"] - 1])
        stop_recv = int(hdr_short[offs["rdb_hdr_dab_stop_rcv"] - 1])
        nreceivers = (stop_recv - start_recv) + 1

        dataframes = float(hdr_float[offs["rdb_hdr_user4"] - 1]) / max(nex, 1)
        refframes = float(hdr_float[offs["rdb_hdr_user19"] - 1])

        img_off = int(hdr_int[offs["rdb_hdr_off_image"] - 1])
        fid.seek(img_off, 0)
        t_hdr = np.fromfile(fid, dtype=f"{endian}i4", count=offs["image_te"])
        o_cnt = max(offs["image_user24"], offs["brhc"] + 2)
        fid.seek(img_off, 0)
        o_hdr = np.fromfile(fid, dtype=f"{endian}f4", count=o_cnt)
        _set_p_val(p, "TE", ii, float(t_hdr[offs["image_te"] - 1]) / 1e3)
        _set_p_val(p, "TR", ii, float(t_hdr[offs["image_tr"] - 1]) / 1e3)

        u8_0 = offs["image_user8"] - 1
        u11_0 = offs["image_user11"] - 1
        u20_0 = offs["image_user20"] - 1
        _set_p_val(p, "voxdim", ii, o_hdr[u8_0 : u8_0 + 3].tolist(), cols=3)
        _set_p_val(p, "voxoff", ii, o_hdr[u11_0 : u11_0 + 3].tolist(), cols=3)
        tlhc_0 = offs["tlhc"] - 1
        trhc_0 = offs["trhc"] - 1
        brhc_0 = offs["brhc"] - 1
        if o_hdr.size >= brhc_0 + 3:
            _set_p_val(p, "tlhc", ii, o_hdr[tlhc_0 : tlhc_0 + 3].tolist(), cols=3)
            _set_p_val(p, "trhc", ii, o_hdr[trhc_0 : trhc_0 + 3].tolist(), cols=3)
            _set_p_val(p, "brhc", ii, o_hdr[brhc_0 : brhc_0 + 3].tolist(), cols=3)

        edit_rf = ge_sub.ensure_substruct("editRF")
        _set_p_val(edit_rf, "waveform", ii, float(o_hdr[offs["image_user19"] - 1]))
        freq_hz = o_hdr[u20_0 : u20_0 + 2].astype(np.float64)
        _set_p_val(edit_rf, "freq_Hz", ii, freq_hz.tolist(), cols=2)
        _set_p_val(edit_rf, "freq_ppm", ii, (freq_hz / larmor + 4.68).tolist(), cols=2)
        dur = float(o_hdr[offs["image_user22"] - 1]) / 1e3
        _set_p_val(edit_rf, "dur", ii, 16.0 if dur <= 0 else dur)
        cv24 = float(o_hdr[offs["image_user24"] - 1])
        _set_p_val(ge_sub, "CV24", ii, cv24)

        if npoints == 1 and nrows == 1:
            npoints = 2048
        _set_p_val(p, "npoints", ii, npoints)

        totalframes = nrows * nechoes
        data_elements = npoints * 2 * totalframes * nreceivers
        fid.seek(pfile_header_size, 0)
        raw_dtype = f"{endian}i2" if point_size == 2 else f"{endian}i4"
        raw_data = np.fromfile(fid, dtype=raw_dtype, count=data_elements).astype(
            np.float64
        )

    shape_data = raw_data.reshape((2, npoints, totalframes, nreceivers), order="F")

    if nechoes == 1:
        if (dataframes + refframes) != nframes:
            mult = 1.0
            _set_p_val(ge_sub, "noadd", ii, 1.0)
            df_int = round(dataframes * nex)
            rf_int = round(nframes - df_int)
        else:
            mult = 1.0 / nex
            _set_p_val(ge_sub, "noadd", ii, 0.0)
            df_int = round(dataframes)
            rf_int = round(refframes)

        water_arr = shape_data[:, :, 1 : rf_int + 1, :] * mult
        metab_arr = shape_data[:, :, rf_int + 1 :, :] * mult / 2.0
        totalframes = totalframes - (rf_int + 1)
        _set_p_val(p, "nrows", ii, totalframes)
        _set_p_val(p, "nrows_water", ii, rf_int)
        _set_p_val(p, "Navg", ii, df_int * nex)
        _set_p_val(p, "Nwateravg", ii, rf_int * nex)
    else:
        _set_p_val(p, "Navg", ii, dataframes * nex * nechoes)
        if (dataframes + refframes) != nframes:
            mult = nex / 2.0
            multw = float(nex)
            _set_p_val(ge_sub, "noadd", ii, 1.0)
            df_int = round(dataframes * nex)
            rf_int = round(nframes - df_int)
        else:
            mult = 1.0 / nex
            multw = 1.0
            _set_p_val(ge_sub, "noadd", ii, 0.0)
            df_int = round(dataframes)
            rf_int = round(refframes)

        _set_p_val(p, "Nwateravg", ii, rf_int * nechoes)
        _set_p_val(p, "nrows_water", ii, rf_int)
        if totalframes != (df_int + rf_int + 1) * nechoes:
            raise ValueError(
                "Number of totalframes does not equal "
                "(dataframes + refframes + 1) * nechoes"
            )

        noadd_val = float(ge_sub["noadd"][ii])
        x1_w, x2_w = np.meshgrid(
            np.arange(1, rf_int + 1), np.arange(1, nechoes + 1), indexing="ij"
        )
        x1_w = x1_w.T.ravel(order="F")
        x2_w = x2_w.T.ravel(order="F")
        y1_w = (
            np.ones(x1_w.size) if cv24 >= 16384 else (-1.0) ** (noadd_val * (x1_w - 1))
        )
        y2_w = ((totalframes // nechoes) * (x2_w - 1) + x1_w).astype(np.int64)
        water_arr = (
            y1_w[np.newaxis, np.newaxis, :, np.newaxis]
            * shape_data[:, :, y2_w, :]
            * multw
        )

        x1_m, x2_m = np.meshgrid(
            np.arange(1, df_int + 1), np.arange(1, nechoes + 1), indexing="ij"
        )
        x1_m = x1_m.T.ravel(order="F")
        x2_m = x2_m.T.ravel(order="F")
        y1_m = (
            np.ones(x1_m.size) if cv24 >= 16384 else (-1.0) ** (noadd_val * (x1_m - 1))
        )
        y2_m = (rf_int + (totalframes // nechoes) * (x2_m - 1) + x1_m).astype(np.int64)
        metab_arr = (
            y1_m[np.newaxis, np.newaxis, :, np.newaxis]
            * shape_data[:, :, y2_m, :]
            * mult
        )
        totalframes = totalframes - (rf_int + 1) * nechoes
        _set_p_val(p, "nrows", ii, totalframes)

    metab_c = metab_arr[0, :, :, :] + 1j * metab_arr[1, :, :, :]
    metab_c = np.transpose(metab_c, (2, 0, 1))
    water_c = water_arr[0, :, :, :] + 1j * water_arr[1, :, :, :]
    water_c = np.transpose(water_c, (2, 0, 1))

    p["RFCoilCombination"] = "Generalized least squares"
    comb_metab, comb_water = _gls_combine(water_c, metab_c)
    fids = mrs_struct.ensure_substruct("fids")
    fids["data_water"] = comb_water
    fids["data"] = comb_metab
    return mrs_struct
