"""High-level headless pipeline runner and CLI entrypoint for Gannet."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

from gannet.config import GannetConfig, MRSStruct
from gannet.pipeline.coregister import gannet_co_register
from gannet.pipeline.fit import gannet_fit
from gannet.pipeline.fit_phantom import gannet_fit_phantom
from gannet.pipeline.load import gannet_load
from gannet.pipeline.quantify import gannet_quantify
from gannet.pipeline.segment import gannet_segment
from gannet.utils.export import export_to_csv, export_to_html


@dataclass
class GannetPipeline:
    """Orchestrates headless Gannet MRS processing, CSV export, and HTML report.

    Parameters
    ------------
    config: GannetConfig
        Gannet parameter configuration object.
    out_dir: str
        Output directory where CSV results and self-contained HTML reports are written.
    """

    config: GannetConfig = field(default_factory=GannetConfig)
    out_dir: str = "."

    def run(
        self,
        metabfile: str | list[str],
        waterfile: str | list[str] | None = None,
        struc: str | list[str] | None = None,
        phantom: bool = False,
        trim_avgs: int = 0,
    ) -> dict[str, Any]:
        """Execute GannetLoad -> Fit/FitPhantom (-> CoRegister -> Segment -> Quantify).

        Automatically exports results to a CSV file and a self-contained HTML
        report in ``self.out_dir`` without launching any interactive GUI.

        Parameters
        ------------
        metabfile: str | list[str]
            Input metabolite MRS file(s) or BIDS folder.
        waterfile: str | list[str] | None
            Optional water reference MRS file(s).
        struc: str | list[str] | None
            Optional T1-weighted structural MRI path(s) for tissue correction.
        phantom: bool
            Run ``gannet_fit_phantom`` instead of in-vivo fitting/quantification.
        trim_avgs: int
            Optional maximum transient count to keep per scan.

        Returns
        --------
            dict[str, Any]:
                Dictionary containing ``mrs_struct``, ``csv_path``, and ``html_path``.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        if phantom:
            self.config.phantom = True

        mrs_struct: MRSStruct = gannet_load(
            metabfile=metabfile,
            waterfile=waterfile,
            trim_avgs=trim_avgs,
            config=self.config,
            save_html=False,
            out_dir=self.out_dir,
        )

        if phantom:
            mrs_struct = gannet_fit_phantom(
                mrs_struct, save_csv=True, save_html=False, out_dir=self.out_dir
            )
            csv_path = export_to_csv(mrs_struct, caller="phantom", out_dir=self.out_dir)
        else:
            mrs_struct = gannet_fit(
                mrs_struct, save_csv=True, save_html=False, out_dir=self.out_dir
            )
            csv_path = export_to_csv(mrs_struct, caller="fit", out_dir=self.out_dir)

            if struc is not None:
                mrs_struct = gannet_co_register(
                    mrs_struct, struc, save_html=False, out_dir=self.out_dir
                )
                mrs_struct = gannet_segment(
                    mrs_struct, save_csv=False, save_html=False, out_dir=self.out_dir
                )
                mrs_struct = gannet_quantify(
                    mrs_struct, save_csv=True, save_html=False, out_dir=self.out_dir
                )
                csv_path = export_to_csv(
                    mrs_struct, caller="quantify", out_dir=self.out_dir
                )

        html_path = export_to_html(
            mrs_struct,
            out_path=os.path.join(self.out_dir, "Gannet_report.html"),
            title="Gannet MRS Analysis Report",
        )
        return {
            "mrs_struct": mrs_struct,
            "csv_path": csv_path,
            "html_path": html_path,
        }


def build_cli_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the ``gannet`` executable.

    Parameters
    ------------
    None

    Returns
    --------
        argparse.ArgumentParser:
            Configured CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="gannet",
        description="Gannet Python MRS processing pipeline (outputs CSV + HTML).",
    )
    parser.add_argument(
        "metab",
        nargs="+",
        help="Input metabolite MRS file(s) or BIDS directory.",
    )
    parser.add_argument(
        "-w",
        "--water",
        nargs="+",
        default=None,
        help="Optional unsuppressed water reference file(s).",
    )
    parser.add_argument(
        "-s",
        "--struc",
        nargs="+",
        default=None,
        help="Optional structural T1 NIfTI file(s) or DICOM folder(s).",
    )
    parser.add_argument(
        "-t",
        "--target",
        nargs="+",
        default=["GABA"],
        help="Target metabolite(s): GABA, Glx, GABAGlx, GSH, Lac, EtOH.",
    )
    parser.add_argument(
        "-a",
        "--alignment",
        default="RobustSpecReg",
        help="Alignment algorithm (RobustSpecReg, SpecReg, Cr, Cho, NAA, H2O, none).",
    )
    parser.add_argument(
        "--phantom",
        action="store_true",
        help="Enable phantom multi-Lorentzian fitting mode.",
    )
    parser.add_argument(
        "--trim-avgs",
        type=int,
        default=0,
        help="Trim number of transients to retain per scan.",
    )
    parser.add_argument(
        "--reg-backend",
        choices=["fsl", "auto", "python"],
        default="fsl",
        help="T1 segmentation/MNI registration backend (default: fsl).",
    )
    parser.add_argument(
        "--fnirt",
        action="store_true",
        help="Use nonlinear FSL FNIRT + applywarp after FLIRT for MNI masking.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default=".",
        help="Directory for output CSV and HTML reports (default: current dir).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Gannet CLI pipeline and write CSV and HTML outputs.

    Parameters
    ------------
    argv: list[str] | None
        Optional CLI argument list (defaults to ``sys.argv[1:]``).

    Returns
    --------
        int:
            Exit status code (0 on success).
    """
    args = build_cli_parser().parse_args(argv)
    cfg = GannetConfig(
        target=args.target,
        alignment=args.alignment,
        phantom=bool(args.phantom),
        csv=True,
        registration_backend=str(args.reg_backend),
        use_fnirt=bool(args.fnirt),
    )
    metab_arg: str | list[str] = args.metab[0] if len(args.metab) == 1 else args.metab
    water_arg: str | list[str] | None = None
    if args.water is not None:
        water_arg = args.water[0] if len(args.water) == 1 else args.water
    struc_arg: str | list[str] | None = None
    if args.struc is not None:
        struc_arg = args.struc[0] if len(args.struc) == 1 else args.struc

    pipe = GannetPipeline(config=cfg, out_dir=args.out_dir)
    result = pipe.run(
        metabfile=metab_arg,
        waterfile=water_arg,
        struc=struc_arg,
        phantom=bool(args.phantom),
        trim_avgs=int(args.trim_avgs),
    )
    print(f"CSV saved to: {result['csv_path']}")
    print(f"HTML report saved to: {result['html_path']}")
    return 0
