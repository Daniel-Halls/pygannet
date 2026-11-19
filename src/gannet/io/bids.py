"""Lightweight BIDS-MRS layout, query, and derivatives helper."""

import json
import os
from pathlib import Path

from gannet.config import MRSStruct
from gannet.io.utils import sort_nat


class BIDSFile:
    """Parse and construct BIDS-compliant filenames and entities.

    Attributes
    ----------
    path: Path
        Original file path.
    entities: dict[str, str]
        Parsed BIDS key-value entities.
    suffix: str
        BIDS suffix (e.g., ``'svs'``, ``'T1w'``, ``'probseg'``).
    extension: str
        File extension (including ``.nii.gz``).
    bids_path: str
        Relative ``sub-.../ses-.../datatype`` path.
    """

    def __init__(self, filepath: str) -> None:
        """Parse BIDS entities, suffix, extension, and subpath from ``filepath``.

        Parameters
        ------------
        filepath: str
            Path to a BIDS file.
        """
        self.path = Path(filepath)
        name = self.path.name
        if name.endswith(".nii.gz"):
            self.extension = ".nii.gz"
            stem = name[: -len(".nii.gz")]
        else:
            self.extension = self.path.suffix
            stem = self.path.stem

        tokens = stem.split("_")
        self.entities: dict[str, str] = {}
        self.suffix = ""
        for tok in tokens:
            if "-" in tok:
                k, v = tok.split("-", 1)
                self.entities[k] = v
            else:
                self.suffix = tok

        parts = self.path.parts
        sub_parts: list[str] = []
        for idx, part in enumerate(parts[:-1]):
            if part.startswith("sub-"):
                sub_parts = list(parts[idx:-1])
                break
        self.bids_path = os.path.join(*sub_parts) if sub_parts else ""

    @property
    def filename(self) -> str:
        """Reconstruct BIDS filename from current entities, suffix, and extension.

        Returns
        --------
            str: str object
                Formatted BIDS filename.
        """
        ordered_keys = [
            "sub",
            "ses",
            "task",
            "acq",
            "ce",
            "rec",
            "run",
            "space",
            "label",
            "desc",
        ]
        parts: list[str] = []
        seen: set[str] = set()
        for key in ordered_keys:
            if self.entities.get(key):
                parts.append(f"{key}-{self.entities[key]}")
                seen.add(key)
        for key, val in self.entities.items():
            if key not in seen and val:
                parts.append(f"{key}-{val}")
        if self.suffix:
            parts.append(self.suffix)
        return "_".join(parts) + self.extension


def bids_layout(bids_dir: str) -> MRSStruct:
    """Index files and participants inside a BIDS directory.

    Parameters
    ------------
    bids_dir: str
        Root path of the BIDS dataset.

    Returns
    --------
        MRSStruct: MRSStruct object
            BIDS layout summary with ``pth``, ``participants``, and ``files``.
    """
    root = Path(bids_dir).resolve()
    layout = MRSStruct()
    layout["pth"] = str(root)
    participants_tsv = root / "participants.tsv"
    if participants_tsv.exists():
        with open(participants_tsv, encoding="utf-8") as handle:
            lines = [ln.strip() for ln in handle if ln.strip()]
        layout["participants"] = lines
    else:
        layout["participants"] = []

    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d != "derivatives" and not d.startswith(".")
        ]
        for fname in filenames:
            if not fname.startswith("."):
                files.append(os.path.join(dirpath, fname))
    layout["files"] = sort_nat(files)
    return layout


def bids_query(
    layout: MRSStruct,
    query_type: str = "data",
    suffix: str = "svs",
    acq: str | None = None,
) -> list[str]:
    """Query NIfTI MRS files in a BIDS layout matching suffix and acq entity.

    Parameters
    ------------
    layout: MRSStruct
        Output from ``bids_layout``.
    query_type: str
        Query type (``'data'``). Defaults to ``'data'``.
    suffix: str
        Target BIDS suffix. Defaults to ``'svs'``.
    acq: str | None
        Optional acquisition entity filter. Defaults to None.

    Returns
    --------
        list[str]: list of str
            Sorted list of matching file paths.
    """
    _ = query_type
    matches: list[str] = []
    for fp in layout.get("files", []):
        if not (fp.endswith((".nii", ".nii.gz"))):
            continue
        bf = BIDSFile(fp)
        if bf.suffix != suffix:
            continue
        if acq is not None and bf.entities.get("acq") != acq:
            continue
        matches.append(fp)
    return sort_nat(matches)


def bids_get_metadata(filepath: str) -> dict[str, object]:
    """Load merged BIDS JSON sidecar metadata for ``filepath``.

    Parameters
    ------------
    filepath: str
        Path to a BIDS NIfTI file.

    Returns
    --------
        dict[str, object]: dict object
            JSON sidecar dictionary (empty if no sidecar found).
    """
    p = Path(filepath)
    if p.name.endswith(".nii.gz"):
        json_path = p.with_name(p.name[: -len(".nii.gz")] + ".json")
    else:
        json_path = p.with_suffix(".json")
    if json_path.exists():
        with open(json_path, encoding="utf-8") as handle:
            data: dict[str, object] = json.load(handle)
            return data
    return {}


def bids_resolve_uri(uri: str, layout: MRSStruct) -> str:
    """Resolve a ``bids::`` URI or relative path against a BIDS layout.

    Parameters
    ------------
    uri: str
        URI string (e.g., ``'bids::sub-01/mrs/sub-01_ref.nii.gz'``).
    layout: MRSStruct
        BIDS layout object containing ``pth``.

    Returns
    --------
        str: str object
            Resolved absolute filesystem path.
    """
    cleaned = uri
    cleaned = cleaned.removeprefix("bids::")
    candidate = Path(layout["pth"]) / cleaned
    return str(candidate)
