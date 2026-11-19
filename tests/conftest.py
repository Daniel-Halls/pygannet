"""Synthetic MRS and structural MRI test fixtures for Gannet tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from numpy.typing import NDArray

from gannet.io.vax import vaxf_to_uint32le


def synthesize_edited_mrs_fids(
    npoints: int = 2048,
    n_avg: int = 32,
    sw: float = 2000.0,
    larmor_mhz: float = 127.74,
    add_freq_drift: bool = True,
    add_phase_drift: bool = True,
    seed: int = 42,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Synthesize realistic MEGA-PRESS ON/OFF FIDs and an unsuppressed water FID.

    Parameters
    ------------
    npoints: int
        Samples per FID.
    n_avg: int
        Total interleaved ON/OFF averages (must be even).
    sw: float
        Spectral bandwidth in Hz.
    larmor_mhz: float
        Scanner Larmor frequency in MHz.
    add_freq_drift: bool
        Apply small frame-to-frame Hz drift to test spectral alignment.
    add_phase_drift: bool
        Apply small frame-to-frame zero-order phase drift.
    seed: int
        RNG seed for reproducible thermal noise.

    Returns
    --------
        tuple[NDArray[np.complex128], NDArray[np.complex128]]:
            ``(metab_fids, water_fid)`` with shapes ``(npoints, n_avg)`` and
            ``(npoints,)``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(npoints, dtype=np.float64) / sw
    f0 = 4.68

    def _peak_fid(ppm: float, amp: float, lw_hz: float) -> NDArray[np.complex128]:
        """Build exponentially damped complex sinusoid for resonance at ``ppm``."""
        df_hz = (ppm - f0) * larmor_mhz
        return amp * np.exp((2j * np.pi * df_hz - np.pi * lw_hz) * t)

    # Water residual + NAA (2.01) + Cr (3.02) + Cho (3.21)
    base_off = (
        _peak_fid(4.68, 0.4, 8.0)
        + _peak_fid(2.01, 1.5, 6.0)
        + _peak_fid(3.02, 1.0, 6.0)
        + _peak_fid(3.21, 0.8, 6.0)
    )
    # ON has GABA pseudo-singlet/triplet at 3.01, Glx at 3.75, inverted NAA at 2.01
    base_on = (
        _peak_fid(4.68, 0.4, 8.0)
        - _peak_fid(2.01, 1.0, 6.0)
        + _peak_fid(3.02, 1.0, 6.0)
        + _peak_fid(3.01, 0.35, 10.0)
        + _peak_fid(3.75, 0.25, 10.0)
        + _peak_fid(3.21, 0.8, 6.0)
    )

    metab = np.zeros((npoints, n_avg), dtype=np.complex128)
    for k in range(n_avg):
        src = base_on if (k % 2 == 0) else base_off
        df = 2.0 * np.sin(2.0 * np.pi * k / max(1, n_avg)) if add_freq_drift else 0.0
        dp = 0.08 * np.cos(k) if add_phase_drift else 0.0
        noise = 0.005 * (
            rng.standard_normal(npoints) + 1j * rng.standard_normal(npoints)
        )
        metab[:, k] = src * np.exp(1j * (2.0 * np.pi * df * t + dp)) + noise

    water = _peak_fid(4.68, 40.0, 7.0) + 0.005 * (
        rng.standard_normal(npoints) + 1j * rng.standard_normal(npoints)
    )
    return (metab, water)


def write_synthetic_philips_sdat_spar(
    out_dir: Path,
    stem: str = "sub01_gaba",
    npoints: int = 2048,
    n_avg: int = 16,
) -> tuple[str, str]:
    """Write matching synthetic Philips ``.sdat``/``.spar`` act + ref files.

    Parameters
    ------------
    out_dir: Path
        Output directory for files.
    stem: str
        Filename stem prefix.
    npoints: int
        Samples per FID.
    n_avg: int
        Number of metabolite averages.

    Returns
    --------
        tuple[str, str]:
            Paths ``(act_sdat_path, ref_sdat_path)``.
    """
    metab, water = synthesize_edited_mrs_fids(npoints=npoints, n_avg=n_avg)
    act_sdat = out_dir / f"{stem}_raw_act.sdat"
    act_spar = out_dir / f"{stem}_raw_act.spar"
    ref_sdat = out_dir / f"{stem}_raw_ref.sdat"
    ref_spar = out_dir / f"{stem}_raw_ref.spar"

    def _write_vax_sdat(path: Path, arr: NDArray[np.complex128]) -> None:
        """Encode complex array into VAX float32 SDAT binary file."""
        flat = arr.reshape(arr.shape[0], -1)
        inter = np.empty((flat.shape[0] * 2, flat.shape[1]), dtype=np.float32)
        inter[0::2, :] = np.real(flat)
        inter[1::2, :] = np.imag(flat)
        u32 = vaxf_to_uint32le(inter.ravel(order="F"))
        u32.astype("<u4").tofile(path)

    _write_vax_sdat(act_sdat, metab)
    _write_vax_sdat(ref_sdat, water[:, np.newaxis])

    spar_template = """examination_name : sub01_test
patient_name : Test Subject
samples : {samples}
rows : {rows}
averages : {avgs}
synthesizer_frequency : 127740000
sample_frequency : 2000
repetition_time : 2000
echo_time : 68
ap_size : 30
lr_size : 30
cc_size : 30
ap_off_center : -10
lr_off_center : 0
cc_off_center : 15
ap_angulation : 5
lr_angulation : 0
cc_angulation : 0
dim2_pnts : {rows}
spec_num_row : {rows}
"""
    act_spar.write_text(
        spar_template.format(samples=npoints, rows=n_avg, avgs=1), encoding="utf-8"
    )
    ref_spar.write_text(
        spar_template.format(samples=npoints, rows=1, avgs=1), encoding="utf-8"
    )
    return (str(act_sdat), str(ref_sdat))


def write_synthetic_t1_nifti(out_dir: Path, filename: str = "T1w.nii.gz") -> str:
    """Create a synthetic 64x64x64 3-tissue T1-weighted structural NIfTI volume.

    Generates distinct air (0), CSF (350), GM (800), and WM (1100) compartments
    matching RAS world coordinates around ``(0, 10, 15)`` mm.

    Parameters
    ------------
    out_dir: Path
        Destination directory.
    filename: str
        Output NIfTI filename.

    Returns
    --------
        str:
            Path to the created NIfTI file.
    """
    shape = (64, 64, 64)
    affine = np.array(
        [
            [-3.0, 0.0, 0.0, 94.5],
            [0.0, 3.0, 0.0, -94.5],
            [0.0, 0.0, 3.0, -94.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ii, jj, kk = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    r = np.sqrt((ii - 31.5) ** 2 + (jj - 31.5) ** 2 + (kk - 31.5) ** 2)
    rng = np.random.default_rng(7)
    vol = rng.normal(5.0, 1.5, shape)
    vol = np.maximum(vol, 0.0)
    # Outer CSF shell, middle GM shell, inner WM core
    vol[r <= 24.0] = 350.0 + rng.normal(0.0, 15.0, np.sum(r <= 24.0))
    vol[r <= 21.0] = 800.0 + rng.normal(0.0, 20.0, np.sum(r <= 21.0))
    vol[r <= 14.0] = 1100.0 + rng.normal(0.0, 20.0, np.sum(r <= 14.0))

    path = out_dir / filename
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(path))
    return str(path)


def write_synthetic_rda(
    out_dir: Path,
    filename: str = "sub01_gaba.rda",
    npoints: int = 1024,
) -> str:
    """Create a synthetic Siemens ``.rda`` spectroscopy file for testing.

    Parameters
    ------------
    out_dir: Path
        Destination directory.
    filename: str
        Output filename.
    npoints: int
        Number of complex spectral samples.

    Returns
    --------
        str:
            Path to the written ``.rda`` file.
    """
    metab, _ = synthesize_edited_mrs_fids(npoints=npoints, n_avg=4)
    hdr_text = f""">>> Begin of header <<<
PatientName: Test Patient
ProtocolName: MEGA_PRESS_68
TR: 2000.0
TE: 68.0
MRFrequency: 123.25
DwellTime: 500
VectorSize: {npoints}
NumberOfAverages: 4
CSIMatrixSize[0]: 1
CSIMatrixSize[1]: 1
CSIMatrixSize[2]: 1
PositionVector[0]: 0.0
PositionVector[1]: 0.0
PositionVector[2]: 0.0
RowVector[0]: 1.0
RowVector[1]: 0.0
RowVector[2]: 0.0
ColumnVector[0]: 0.0
ColumnVector[1]: 1.0
ColumnVector[2]: 0.0
VOIPositionSag: 0.0
VOIPositionCor: -10.0
VOIPositionTra: 15.0
VOINormalSag: 0.0
VOINormalCor: 0.0
VOINormalTra: 1.0
VOIThickness: 30.0
VOIPhaseFOV: 30.0
VOIReadoutFOV: 30.0
>>> End of header <<<
"""
    path = out_dir / filename
    with open(path, "wb") as fid:
        fid.write(hdr_text.encode("latin-1"))
        fid.write(b"\r\n")
        for col in range(metab.shape[1]):
            vec = metab[:, col]
            inter = np.empty(npoints * 2, dtype="<f8")
            inter[0::2] = np.real(vec)
            inter[1::2] = np.imag(vec)
            inter.tofile(fid)
    return str(path)


@pytest.fixture
def synthetic_philips_dataset(tmp_path: Path) -> dict[str, Any]:
    """Pytest fixture providing synthetic Philips SDAT/SPAR act+ref and T1 NIfTI.

    Parameters
    ------------
    tmp_path: Path
        Temporary pytest directory.

    Returns
    --------
        dict[str, Any]:
            Dictionary with ``act``, ``ref``, ``t1``, and ``tmp_path`` keys.
    """
    act, ref = write_synthetic_philips_sdat_spar(tmp_path)
    t1 = write_synthetic_t1_nifti(tmp_path)
    return {"act": act, "ref": ref, "t1": t1, "tmp_path": tmp_path}
