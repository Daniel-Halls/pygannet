"""Configuration structures and pre-initialization for Gannet."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


class MRSStruct(dict):
    """Dictionary-backed structure supporting MATLAB-style attribute access.

    Allows nested access such as ``mrs.p.target`` or ``mrs.out.vox1.GABA.Area``
    while preserving full Python dictionary semantics and easy serialization.
    """

    def __getattr__(self, name: str) -> Any:
        """Retrieve a field from the structure as an attribute.

        Parameters
        ------------
        name: str
            Key name to retrieve from the underlying dictionary.

        Returns
        --------
            Any: any object
                Value stored under ``name``.

        Raises
        ------
        AttributeError
            If ``name`` does not exist in the structure.
        """
        if name in self:
            return self[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign a field value in the structure as an attribute.

        Parameters
        ------------
        name: str
            Key name to set.
        value: Any
            Value to store; plain dicts are converted to ``MRSStruct``.

        Returns
        --------
            None: None
                Modifies the structure in place.
        """
        if isinstance(value, dict) and not isinstance(value, MRSStruct):
            value = MRSStruct.from_dict(value)
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete a field from the structure.

        Parameters
        ------------
        name: str
            Key name to remove.

        Returns
        --------
            None: None
                Modifies the structure in place.
        """
        if name in self:
            del self[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MRSStruct":
        """Recursively convert a plain dictionary into an ``MRSStruct``.

        Parameters
        ------------
        data: dict[str, Any]
            Input dictionary to convert.

        Returns
        --------
            MRSStruct: MRSStruct object
                Converted nested ``MRSStruct``.
        """
        result = cls()
        for key, val in data.items():
            if isinstance(val, dict) and not isinstance(val, MRSStruct):
                result[key] = cls.from_dict(val)
            else:
                result[key] = val
        return result

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert an ``MRSStruct`` back into plain dicts.

        Returns
        --------
            dict[str, Any]: dict object
                Plain dictionary representation suitable for serialization.
        """
        plain: dict[str, Any] = {}
        for key, val in self.items():
            if isinstance(val, MRSStruct):
                plain[key] = val.to_dict()
            else:
                plain[key] = val
        return plain

    def ensure_substruct(self, *keys: str) -> "MRSStruct":
        """Ensure nested substructs exist along the given key path.

        Parameters
        ------------
        *keys: str
            Sequence of nested key names.

        Returns
        --------
            MRSStruct: MRSStruct object
                The deepest nested ``MRSStruct`` at the end of ``keys``.
        """
        curr: MRSStruct = self
        for key in keys:
            if key not in curr or not isinstance(curr[key], MRSStruct):
                curr[key] = MRSStruct()
            curr = curr[key]
        return curr


@dataclass
class GannetConfig:
    """Typed configuration parameters matching ``GannetPreInitialise.m``.

    Attributes
    ----------
    target: list[str]
        Edited metabolite(s) of interest, e.g., ``['GABAGlx']``.
    on_off_order: str | None
        Manual editing order (``'offfirst'``, ``'onfirst'``, or 4-letter HERMES
        code such as ``'ABCD'``), or None for automatic detection.
    seqorig: str
        Sequence origin (``'JHU'``, ``'Philips'``, ``'Lythgoe'``, etc.).
    lb: float
        Exponential line-broadening factor in Hz. Defaults to 3.0.
    water_ecc: bool
        Perform eddy current correction on water data. Defaults to True.
    metab_ecc: bool
        Perform eddy current correction on metabolite data. Defaults to False.
    water_removal: bool
        Remove residual water signal in DIFF spectrum via HSVD. Defaults to True.
    alignment: str
        Frequency-and-phase alignment method. Defaults to ``'RobustSpecReg'``.
    use_prealign_ref: bool
        Use pre-aligned subspectra as references in RobustSpecReg.
    vox: list[str]
        Voxel names, e.g., ``['vox1']``.
    fit_resid_water: bool
        Fit residual water in OFF spectrum to compute suppression factor.
    weighted_averaging: bool
        Use weighted averaging for subspectra. Defaults to True.
    hermes: bool
        Whether data were acquired using HERMES. Defaults to False.
    hercules: bool
        Whether data were acquired using HERCULES. Defaults to False.
    priam: bool
        Whether data were acquired using PRIAM. Defaults to False.
    phantom: bool
        Whether data are from an in vitro phantom. Defaults to False.
    join: bool
        Join multiple files per subject. Defaults to False.
    mat: bool
        Save ``MRS_struct`` to a ``.mat`` file. Defaults to False.
    csv: bool
        Export output metrics to a ``.csv`` file. Defaults to False.
    normalize: bool
        Normalize voxel masks to MNI space during segmentation.
    append: bool
        Append PDF figures into a single multi-page PDF per module.
    hide: bool
        Suppress interactive display of output figures. Defaults to True.
    spm_path: str | None
        Optional path to SPM12 / MATLAB executable for external segmentation.
    vox_sep: float | None
        Optional PRIAM voxel separation in mm to avoid interactive prompt.
    """

    target: list[str] = field(default_factory=lambda: ["GABAGlx"])
    on_off_order: str | None = None
    seqorig: str = "JHU"
    lb: float = 3.0
    water_ecc: bool = True
    metab_ecc: bool = False
    water_removal: bool = True
    alignment: str = "RobustSpecReg"
    use_prealign_ref: bool = False
    vox: list[str] = field(default_factory=lambda: ["vox1"])
    fit_resid_water: bool = False
    weighted_averaging: bool = True
    hermes: bool = False
    hercules: bool = False
    priam: bool = False
    phantom: bool = False
    join: bool = False
    mat: bool = False
    csv: bool = False
    normalize: bool = False
    append: bool = False
    hide: bool = True
    spm_path: str | None = None
    vox_sep: float | None = None
    registration_backend: str = "fsl"
    use_fnirt: bool = False
    bet_frac: float = 0.5

    def to_p_struct(self) -> MRSStruct:
        """Convert this configuration into a Gannet ``MRS_struct.p`` object.

        Returns
        --------
            MRSStruct: MRSStruct object
                Parameter substruct using canonical Gannet field names.
        """
        p = MRSStruct()
        p["target"] = list(self.target)
        p["ON_OFF_order"] = self.on_off_order or ""
        p["seqorig"] = self.seqorig
        p["LB"] = float(self.lb)
        p["water_ECC"] = int(self.water_ecc)
        p["metab_ECC"] = int(self.metab_ecc)
        p["water_removal"] = int(self.water_removal)
        p["alignment"] = self.alignment
        p["use_prealign_ref"] = int(self.use_prealign_ref)
        p["vox"] = list(self.vox)
        p["Vox"] = list(self.vox)
        p["numVox"] = len(self.vox)
        p["numScans"] = 1
        p["fit_resid_water"] = int(self.fit_resid_water)
        p["weighted_averaging"] = int(self.weighted_averaging)
        p["HERMES"] = int(self.hermes)
        p["HERCULES"] = int(self.hercules)
        p["PRIAM"] = int(self.priam)
        p["phantom"] = int(self.phantom)
        p["join"] = int(self.join)
        p["mat"] = int(self.mat)
        p["csv"] = int(self.csv)
        p["normalize"] = int(self.normalize)
        p["append"] = int(self.append)
        p["hide"] = int(self.hide)
        p["registration_backend"] = self.registration_backend
        p["use_fnirt"] = int(self.use_fnirt)
        p["bet_frac"] = float(self.bet_frac)
        if self.spm_path is not None:
            p["spm_path"] = self.spm_path
        if self.vox_sep is not None:
            p["vox_sep"] = float(self.vox_sep)
        return p


def gannet_pre_initialise(
    mrs_struct: MRSStruct | None = None,
    config: GannetConfig | None = None,
) -> MRSStruct:
    """Set up default parameters and options for the Gannet workflow.

    Mirrors ``GannetPreInitialise.m``. Any parameters already present in
    ``mrs_struct.p`` are preserved unless overridden by an explicit ``config``.

    Parameters
    ------------
    mrs_struct: MRSStruct | None
        Existing MRS data structure, or None to create a new one.
    config: GannetConfig | None
        Optional typed configuration overriding defaults.

    Returns
    --------
        MRSStruct: MRSStruct object
            Initialized ``MRSStruct`` with populated ``p`` parameters.
    """
    if mrs_struct is None:
        mrs_struct = MRSStruct()
    else:
        mrs_struct = (
            mrs_struct
            if isinstance(mrs_struct, MRSStruct)
            else MRSStruct.from_dict(mrs_struct)
        )

    defaults = (config or GannetConfig()).to_p_struct()
    existing_p = mrs_struct.get("p")
    if isinstance(existing_p, dict):
        merged = deepcopy(defaults)
        for key, val in existing_p.items():
            merged[key] = val
        if config is not None:
            for key, val in config.to_p_struct().items():
                merged[key] = val
        mrs_struct["p"] = merged
    else:
        mrs_struct["p"] = defaults

    for sub in ("fids", "spec", "out", "mask", "info"):
        mrs_struct.ensure_substruct(sub)
    return mrs_struct


def check_targets(mrs_struct: MRSStruct) -> bool:
    """Validate target metabolites and HERMES/phantom flags in ``mrs_struct.p``.

    Mirrors ``CheckTargets.m``.

    Parameters
    ------------
    mrs_struct: MRSStruct
        MRS data structure containing ``p.target``, ``p.HERMES``, and
        ``p.phantom``.

    Returns
    --------
        bool: boolean object
            True if the target configuration is valid.

    Raises
    ------
    ValueError
        If target names, counts, or HERMES/phantom combinations are invalid.
    """
    p = mrs_struct["p"]
    targets = list(p["target"])
    hermes = bool(p.get("HERMES", 0))
    phantom = bool(p.get("phantom", 0))
    msg = (
        "Incorrect targets entered in GannetPreInitialise. "
        "Check spelling, allowable options, and order of targets."
    )

    if len(targets) == 1:
        allowed_single = {"GABA", "Glx", "GABAGlx", "GSH", "Lac", "EtOH"}
        if targets[0] not in allowed_single:
            raise ValueError(msg)
        if phantom and targets[0] == "GABAGlx":
            raise ValueError(msg)
        if hermes:
            raise ValueError(
                "MRS_struct.p.HERMES is set to 1. Add at least a second "
                "target metabolite or set HERMES to 0."
            )
    elif len(targets) == 2:
        allowed_pairs = [
            ["GABAGlx", "GSH"],
            ["GABA", "GSH"],
            ["Glx", "GSH"],
            ["Lac", "GSH"],
        ]
        if targets not in allowed_pairs:
            raise ValueError(msg)
        if phantom and "GABAGlx" in targets:
            raise ValueError(msg)
        if not hermes:
            raise ValueError(
                "Two target metabolites detected. MRS_struct.p.HERMES must be set to 1."
            )
    elif len(targets) == 3:
        if targets != ["EtOH", "GABA", "GSH"]:
            raise ValueError(msg)
        if not hermes:
            raise ValueError(
                "Three target metabolites detected. "
                "MRS_struct.p.HERMES must be set to 1."
            )
    else:
        raise ValueError(msg)

    return True
