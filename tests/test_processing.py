"""Unit tests for Gannet preprocessing, filtering, alignment, and averaging."""

from __future__ import annotations

import numpy as np

from gannet.config import gannet_pre_initialise
from gannet.processing.alignment import (
    align_using_h2o,
    align_using_peak,
    freq_phase_shift_nest,
    robust_spectral_registration,
    spectral_registration,
    spectral_registration_hermes,
)
from gannet.processing.averaging import signal_averaging
from gannet.processing.corrections import eddy_current_correction, phase_correction
from gannet.processing.hsvd import water_removal_hsvd
from gannet.processing.on_off import specify_on_off_order
from gannet.processing.signal_filter import signal_filter
from tests.conftest import synthesize_edited_mrs_fids


def _make_loaded_mrs_for_processing() -> tuple[np.ndarray, dict]:
    """Create a populated MRSStruct + complex spectral frames for processing tests.

    Parameters
    ------------
    None

    Returns
    --------
        tuple[np.ndarray, dict]:
            ``(all_frames_ft, mrs_struct)`` ready for alignment and averaging tests.
    """
    npts, n_avg, sw, larmor = (1024, 16, 2000.0, 127.74)
    zf = 4096
    fids, water = synthesize_edited_mrs_fids(
        npoints=npts, n_avg=n_avg, sw=sw, larmor_mhz=larmor
    )
    mrs = gannet_pre_initialise()
    mrs["ii"] = 0
    mrs["p"]["vendor"] = "Philips"
    mrs["p"]["target"] = ["GABA"]
    mrs["p"]["npoints"] = np.array([npts])
    mrs["p"]["nrows"] = np.array([n_avg])
    mrs["p"]["Navg"] = np.array([n_avg])
    mrs["p"]["Nwateravg"] = np.array([1])
    mrs["p"]["sw"] = np.array([sw])
    mrs["p"]["LarmorFreq"] = np.array([larmor])
    mrs["p"]["ZeroFillTo"] = np.array([zf])
    mrs["p"]["TE"] = np.array([68.0])
    mrs["p"]["TR"] = np.array([2000.0])
    mrs["fids"]["data"] = fids
    mrs["fids"]["data_water"] = water
    mrs = specify_on_off_order(mrs)

    freq = np.linspace(4.68 + sw / larmor / 2.0, 4.68 - sw / larmor / 2.0, zf)
    mrs["spec"]["freq"] = freq
    all_ft = np.fft.fftshift(np.fft.fft(fids, n=zf, axis=0), axes=0)
    mrs["spec"]["AllFramesFT"] = all_ft
    mrs["out"]["reject"] = [np.zeros(n_avg, dtype=int)]
    return (all_ft, mrs)


def test_ecc_phase_correction_and_on_off() -> None:
    """Verify ECC, phase correction, and ON/OFF order assignment.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    npts, n_avg = (512, 8)
    fids, water = synthesize_edited_mrs_fids(npoints=npts, n_avg=n_avg)
    ecc_fids = eddy_current_correction(fids, water)
    assert ecc_fids.shape == fids.shape
    assert np.all(np.isfinite(ecc_fids))

    _, mrs = _make_loaded_mrs_for_processing()
    ph_fids = phase_correction(mrs["fids"]["data"], mrs)
    assert ph_fids.shape == mrs["fids"]["data"].shape
    assert np.all(np.isfinite(ph_fids))
    assert mrs["fids"]["ON_OFF"].size == mrs["p"]["Navg"][0]


def test_hsvd_water_removal_and_signal_filter() -> None:
    """Verify HSVD residual water suppression and wavelet signal filter.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    npts = 1024
    sw_hz = 2000.0
    t = np.arange(npts, dtype=np.float64) / sw_hz
    water = 20.0 * np.exp(-15.0 * t)
    gaba = 0.5 * np.exp((2j * np.pi * (3.01 - 4.68) * 127.74 - 12.0) * t)
    cleaned = water_removal_hsvd(
        water + gaba,
        sw_khz=sw_hz / 1000.0,
        n_hsvd=6,
        hsvd_low=-0.1,
        hsvd_high=0.1,
        use_points=512,
    )
    assert cleaned.shape == (npts,)
    assert np.linalg.norm(cleaned - gaba) < np.linalg.norm((water + gaba) - gaba)

    # Wavelet baseline filter (operates on npoints-length spectrum)
    _, mrs_f = _make_loaded_mrs_for_processing()
    npts_f = int(mrs_f["p"]["npoints"][0])
    sig = np.fft.fftshift(np.fft.fft(mrs_f["fids"]["data"][:, 0]))
    filtered = signal_filter(
        sig, lipid_flag=True, water_flag=True, transient_idx=0, mrs_struct=mrs_f
    )
    assert filtered.shape == (npts_f,)
    assert np.all(np.isfinite(filtered))


def test_alignment_methods_and_averaging() -> None:
    """Verify all alignment methods, shift helper, and signal averaging.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    all_ft, mrs = _make_loaded_mrs_for_processing()
    zf = mrs["p"]["ZeroFillTo"][0]
    fid0 = mrs["fids"]["data"][:, 0]
    shifted = freq_phase_shift_nest(
        [2.0, 5.0],
        {"dwelltime": 1.0 / 2000.0, "data": np.concatenate([fid0.real, fid0.imag])},
    )
    assert shifted.shape == (fid0.size * 2,)

    for method in ("Cr", "Cho", "NAA"):
        mrs["p"]["alignment"] = method
        aligned, mrs = align_using_peak(all_ft, mrs)
        assert aligned.shape == (zf, mrs["p"]["Navg"][0])

    mrs["p"]["alignment"] = "H2O"
    aligned_h2o, mrs = align_using_h2o(all_ft, mrs)
    assert aligned_h2o.shape == all_ft.shape

    aligned_sr, mrs = spectral_registration(mrs, on_what=0, dual=0)
    assert aligned_sr.shape == all_ft.shape

    aligned_rsr, mrs = robust_spectral_registration(mrs)
    assert aligned_rsr.shape == all_ft.shape

    mrs = signal_averaging(mrs, all_ft, aligned_rsr, ii=0, kk=0, vox=["vox1"])
    assert "GABA" in mrs["spec"]["vox1"]
    assert mrs["spec"]["vox1"]["GABA"]["diff_unfilt"].size == zf


def test_hermes_alignment() -> None:
    """Verify 4-step HERMES spectral registration executes cleanly.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    all_ft, mrs = _make_loaded_mrs_for_processing()
    mrs["p"]["HERMES"] = True
    mrs["p"]["target"] = ["GABA", "GSH"]
    mrs = specify_on_off_order(mrs)
    aligned_h, mrs = spectral_registration_hermes(mrs)
    assert aligned_h.shape == all_ft.shape
