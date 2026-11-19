"""End-to-end integration tests for readers, pipeline, CSV/HTML reports, and de-id."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from gannet import (
    GannetConfig,
    GannetPipeline,
    co_reg_stand_alone,
    de_identify_disclaimer,
    export_to_csv,
    export_to_html,
    gannet_co_register,
    gannet_fit,
    gannet_fit_phantom,
    gannet_load,
    gannet_pre_initialise,
    gannet_quantify,
    gannet_segment,
    gannet_version,
    paper_plot,
    philips_de_identify,
)
from gannet.io.nifti_mrs import nifti_mrs_read
from gannet.io.siemens_rda import siemens_read
from gannet.io.utils import discern_data_type, sort_nat
from gannet.pipeline.runner import main as cli_main
from tests.conftest import synthesize_edited_mrs_fids, write_synthetic_rda


def test_utilities_and_version() -> None:
    """Verify version metadata, natural sorting, and data-type detection.

    Parameters
    ------------
    None

    Returns
    --------
        None
    """
    info = gannet_version(print_output=False)
    assert info.gannet == "3.5.4"
    assert sort_nat(["sub10.sdat", "sub2.sdat", "sub1.sdat"]) == [
        "sub1.sdat",
        "sub2.sdat",
        "sub10.sdat",
    ]
    mrs = gannet_pre_initialise()
    mrs = discern_data_type("scan01.sdat", mrs)
    assert mrs["p"]["vendor"] == "Philips"
    mrs = discern_data_type("scan01.rda", mrs)
    assert mrs["p"]["vendor"] == "Siemens_rda"
    mrs = discern_data_type("scan01.dat", mrs)
    assert mrs["p"]["vendor"] == "Siemens_twix"


def test_full_end_to_end_pipeline_csv_and_html(
    synthetic_philips_dataset: dict,
    tmp_path: Path,
) -> None:
    """Verify full Gannet pipeline execution with CSV and HTML outputs.

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
    out_dir = tmp_path / "results"
    cfg = GannetConfig(target=["GABA"], alignment="RobustSpecReg")
    pipe = GannetPipeline(config=cfg, out_dir=str(out_dir))
    res = pipe.run(
        metabfile=synthetic_philips_dataset["act"],
        waterfile=synthetic_philips_dataset["ref"],
        struc=synthetic_philips_dataset["t1"],
    )

    mrs = res["mrs_struct"]
    csv_path = Path(res["csv_path"])
    html_path = Path(res["html_path"])

    assert csv_path.exists()
    assert html_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "GABA_ConcIU_TissCorr" in csv_text
    assert "GABA_ConcIU_AlphaTissCorr" in csv_text
    assert "fGM" in csv_text

    html_text = html_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_text
    assert "Gannet MRS Analysis Report" in html_text

    t_out = mrs["out"]["vox1"]["GABA"]
    assert t_out["Area"][0] > 0
    assert t_out["ConcIU"][0] > 0
    assert t_out["ConcIU_CSFcorr"][0] >= t_out["ConcIU"][0] * 0.99
    assert np.isfinite(t_out["ConcIU_TissCorr"][0])
    assert np.isfinite(t_out["ConcIU_AlphaTissCorr"][0])

    tissue = mrs["out"]["vox1"]["tissue"]
    frac_sum = tissue["fGM"][0] + tissue["fWM"][0] + tissue["fCSF"][0]
    np.testing.assert_allclose(frac_sum, 1.0, atol=1e-4)

    # Verify PaperPlot saves PNG cleanly without GUI
    pp_path = out_dir / "group_spec.png"
    fig = paper_plot(mrs, save_path=str(pp_path))
    assert pp_path.exists()
    assert fig is not None


def test_cli_entrypoint_and_phantom_mode(
    synthetic_philips_dataset: dict,
    tmp_path: Path,
) -> None:
    """Verify ``gannet`` CLI entrypoint and phantom multi-Lorentzian pipeline.

    Parameters
    ------------
    synthetic_philips_dataset: dict
        Pytest fixture with synthetic Philips SDAT/SPAR.
    tmp_path: Path
        Temporary output directory.

    Returns
    --------
        None
    """
    cli_out = tmp_path / "cli_out"
    rc = cli_main(
        [
            synthetic_philips_dataset["act"],
            "-w",
            synthetic_philips_dataset["ref"],
            "-s",
            synthetic_philips_dataset["t1"],
            "-o",
            str(cli_out),
        ]
    )
    assert rc == 0
    assert (cli_out / "GannetQuantify_output.csv").exists()
    assert (cli_out / "Gannet_report.html").exists()

    # Phantom fitting path
    phantom_out = tmp_path / "phantom_out"
    cfg = GannetConfig(target=["GABA"], phantom=True)
    mrs_p = gannet_load(
        synthetic_philips_dataset["act"],
        waterfile=synthetic_philips_dataset["ref"],
        config=cfg,
        save_html=False,
    )
    mrs_p = gannet_fit_phantom(
        mrs_p, save_csv=True, save_html=True, out_dir=str(phantom_out)
    )
    assert (phantom_out / "GannetFitPhantom_output.csv").exists()
    assert (phantom_out / "GannetFitPhantom_report.html").exists()
    assert np.isfinite(mrs_p["out"]["vox1"]["GABA"]["ConcIU"][0])


def test_siemens_rda_nifti_mrs_and_coreg_standalone(tmp_path: Path) -> None:
    """Verify Siemens RDA, NIfTI-MRS loading, and CoRegStandAlone execution.

    Parameters
    ------------
    tmp_path: Path
        Temporary directory.

    Returns
    --------
        None
    """
    rda_path = write_synthetic_rda(tmp_path)
    mrs_rda = gannet_pre_initialise()
    mrs_rda["ii"] = 0
    mrs_rda["p"]["vendor"] = "Siemens_rda"
    mrs_rda["p"]["npoints"] = np.array([0])
    mrs_rda["p"]["nrows"] = np.array([0])
    mrs_rda["p"]["Navg"] = np.array([0])
    mrs_rda["p"]["Nwateravg"] = np.array([0])
    mrs_rda["p"]["LarmorFreq"] = np.array([0.0])
    mrs_rda["p"]["sw"] = np.array([0.0])
    mrs_rda["p"]["TR"] = np.array([0.0])
    mrs_rda["p"]["TE"] = np.array([0.0])
    mrs_rda["p"]["voxdim"] = np.zeros((1, 3))
    mrs_rda["p"]["voxoff"] = np.zeros((1, 3))
    mrs_rda["p"]["voxang"] = np.zeros((1, 3))
    mrs_rda = siemens_read(mrs_rda, rda_path)
    assert mrs_rda["fids"]["data"].shape[0] == 1024

    # Test NIfTI-MRS 5D dataset creation & reading (including NIfTI ecode 44 extension)
    metab, _ = synthesize_edited_mrs_fids(npoints=512, n_avg=8)
    data_5d = metab[np.newaxis, np.newaxis, np.newaxis, :, :]
    nii_mrs_path = tmp_path / "sub01_megapress.nii.gz"
    img = nib.Nifti1Image(data_5d.astype(np.complex64), np.eye(4))
    ext_json = (
        '{"SpectrometerFrequency": [127.74], "SpectralWidth": 2000.0, '
        '"EchoTime": 0.068, "RepetitionTime": 2.0, "dim_5": "DIM_DYN"}'
    )
    img.header.extensions.append(
        nib.nifti1.Nifti1Extension(44, ext_json.encode("utf-8"))
    )
    nib.save(img, str(nii_mrs_path))
    mrs_nii = gannet_pre_initialise()
    mrs_nii["ii"] = 0
    mrs_nii["p"]["npoints"] = np.array([0])
    mrs_nii["p"]["nrows"] = np.array([0])
    mrs_nii["p"]["Navg"] = np.array([0])
    mrs_nii["p"]["Nwateravg"] = np.array([0])
    mrs_nii["p"]["LarmorFreq"] = np.array([0.0])
    mrs_nii["p"]["sw"] = np.array([0.0])
    mrs_nii["p"]["TR"] = np.array([0.0])
    mrs_nii["p"]["TE"] = np.array([0.0])
    mrs_nii["p"]["voxdim"] = np.zeros((1, 3))
    mrs_nii["p"]["voxoff"] = np.zeros((1, 3))
    mrs_nii["p"]["voxang"] = np.zeros((1, 3))
    mrs_nii = nifti_mrs_read(mrs_nii, str(nii_mrs_path))
    assert mrs_nii["fids"]["data"].shape[0] == 512

    # CoRegStandAlone on synthetic RDA + T1
    from tests.conftest import write_synthetic_t1_nifti

    t1_path = write_synthetic_t1_nifti(tmp_path)
    csa_out = tmp_path / "csa_out"
    res_csa = co_reg_stand_alone(rda_path, t1_path, out_dir=str(csa_out))
    assert (csa_out / "GannetSegment_output.csv").exists()
    assert (csa_out / "CoRegStandAlone_report.html").exists()
    assert res_csa["out"]["vox1"]["tissue"]["fGM"][0] >= 0.0


def test_deidentification_and_explicit_export(
    synthetic_philips_dataset: dict,
    tmp_path: Path,
) -> None:
    """Verify Philips SPAR/SDAT de-identification and modular CSV/HTML export.

    Parameters
    ------------
    synthetic_philips_dataset: dict
        Pytest fixture with synthetic Philips SDAT/SPAR.
    tmp_path: Path
        Temporary directory.

    Returns
    --------
        None
    """
    assert "DISCLAIMER" in de_identify_disclaimer()
    deid_dir = tmp_path / "deid"
    out_sdat, out_spar = philips_de_identify(
        synthetic_philips_dataset["act"], out_dir=str(deid_dir), subject_id="anon-99"
    )
    assert Path(out_sdat).exists()
    assert "anon-99" in Path(out_spar).read_text(encoding="utf-8")
    assert "Test Subject" not in Path(out_spar).read_text(encoding="utf-8")

    # Individual stage calls -> export_to_csv & export_to_html
    mrs = gannet_load(
        synthetic_philips_dataset["act"],
        waterfile=synthetic_philips_dataset["ref"],
    )
    mrs = gannet_fit(mrs, save_csv=False, save_html=False)
    mrs = gannet_co_register(mrs, synthetic_philips_dataset["t1"], save_html=False)
    mrs = gannet_segment(mrs, save_csv=False, save_html=False)
    mrs = gannet_quantify(mrs, save_csv=False, save_html=False)

    csv_p = export_to_csv(mrs, caller="quantify", out_dir=str(tmp_path))
    html_p = export_to_html(mrs, out_path=str(tmp_path / "custom.html"))
    assert Path(csv_p).exists()
    assert Path(html_p).exists()
