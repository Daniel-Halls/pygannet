"""Integration tests for FSL BET/FAST/FLIRT image segmentation and registration."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from gannet import (
    GannetConfig,
    GannetPipeline,
)
from gannet.segment import (
    call_fsl_segmentation,
    find_existing_tissue_maps,
    find_fsl_bin,
    get_mni152_template,
    normalize_voxel_mask,
)
from tests.conftest import write_synthetic_t1_nifti


def _require_fsl() -> None:
    """Skip test gracefully if FSL binaries or MNI template are not installed.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    if find_fsl_bin("bet") is None or find_fsl_bin("fast") is None:
        pytest.skip("FSL bet/fast not found on PATH or FSLDIR")
    if find_fsl_bin("flirt") is None or get_mni152_template() is None:
        pytest.skip("FSL flirt or MNI152_T1_2mm template not found")


def test_fsl_bet_fast_and_flirt_mask_normalization(tmp_path: Path) -> None:
    """Verify FSL BET skull-strip, FAST segmentation, and FLIRT MNI registration.

    Parameters
    ------------
    tmp_path: Path
        Temporary output directory.

    Returns
    --------
        None
    """
    _require_fsl()
    t1_path = write_synthetic_t1_nifti(tmp_path)

    # 1. Run FSL bet + fast segmentation
    seg_maps = call_fsl_segmentation(t1_path, bet_frac=0.3, backend="fsl")
    assert seg_maps["backend"] == "fsl"
    for key in ("c1", "c2", "c3", "c6", "brain", "brain_mask"):
        assert Path(seg_maps[key]).exists()

    # 2. Existing FSL FAST PVE files are automatically discovered
    found = find_existing_tissue_maps(t1_path)
    assert found["c1"] == seg_maps["c1"]
    assert found["c2"] == seg_maps["c2"]
    assert found["c3"] == seg_maps["c3"]

    # 3. Create synthetic voxel mask matching T1 grid and warp to MNI via FLIRT
    t1_img = nib.load(t1_path)
    mask_arr = np.zeros(t1_img.shape[:3], dtype=np.uint8)
    mask_arr[26:38, 26:38, 26:38] = 1
    mask_path = tmp_path / "sub01_vox_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_arr, t1_img.affine), str(mask_path))

    w_mask = normalize_voxel_mask(
        str(mask_path),
        t1_path=t1_path,
        brain_path=seg_maps["brain"],
        use_fnirt=False,
        backend="fsl",
    )
    assert Path(w_mask).exists()
    w_img = nib.load(w_mask)
    assert w_img.shape[:3] == (91, 109, 91)


def test_python_fallback_segmentation_and_fsl_pipeline(
    synthetic_philips_dataset: dict,
    tmp_path: Path,
) -> None:
    """Verify both explicit python backend fallback and FSL-default pipeline config.

    Parameters
    ------------
    synthetic_philips_dataset: dict
        Pytest fixture with synthetic Philips SDAT/SPAR and T1 NIfTI.
    tmp_path: Path
        Temporary output directory.

    Returns
    --------
        None
    """
    py_dir = tmp_path / "py_t1"
    py_dir.mkdir()
    t1_py = write_synthetic_t1_nifti(py_dir)
    py_maps = call_fsl_segmentation(t1_py, backend="python")
    assert Path(py_maps["c1"]).exists()
    assert Path(py_maps["c2"]).exists()
    assert Path(py_maps["c3"]).exists()

    # Run GannetPipeline using default FSL backend
    pipe_dir = tmp_path / "fsl_pipe"
    cfg = GannetConfig(
        target=["GABA"],
        alignment="RobustSpecReg",
        registration_backend="fsl",
        bet_frac=0.3,
    )
    pipe = GannetPipeline(config=cfg, out_dir=str(pipe_dir))
    res = pipe.run(
        metabfile=synthetic_philips_dataset["act"],
        waterfile=synthetic_philips_dataset["ref"],
        struc=synthetic_philips_dataset["t1"],
    )
    assert Path(res["csv_path"]).exists()
    assert Path(res["html_path"]).exists()
