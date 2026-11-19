"""General I/O, path, sorting, and structure utilities for Gannet."""

import os
import re
import warnings
from pathlib import Path

from gannet.config import MRSStruct


def discern_data_type(filename: str, mrs_struct: MRSStruct) -> MRSStruct:
    """Deduce scanner vendor and MRS format from the input filename extension.

    Mirrors ``DiscernDataType.m``.

    Parameters
    ------------
    filename: str
        Path to the metabolite MRS file.
    mrs_struct: MRSStruct
        MRS data structure whose ``p.vendor`` field will be populated.

    Returns
    --------
        MRSStruct: MRSStruct object
            Updated ``MRS_struct`` with ``p.vendor`` set.

    Raises
    ------
    ValueError
        If the file extension is not a supported MRS format.
    """
    ext = Path(filename).suffix.lower()
    mapping = {
        ".7": "GE",
        ".dat": "Siemens_twix",
        ".data": "Philips_data",
        ".dcm": "DICOM",
        ".gz": "NIfTI",
        ".nii": "NIfTI",
        ".ima": "Siemens_dicom",
        ".raw": "Philips_raw",
        ".rda": "Siemens_rda",
        ".sdat": "Philips",
    }
    if ext not in mapping:
        raise ValueError(
            "Unrecognized file type! Extension must be .7, .dat, .data, "
            ".dcm, .gz, .ima, .nii, .raw, .rda, or .sdat."
        )
    vendor = mapping[ext]
    mrs_struct["p"]["vendor"] = vendor
    if vendor == "Siemens_rda":
        warnings.warn(
            "The Siemens .rda format is NOT recommended for use in Gannet. "
            "If possible, please re-export your data in the TWIX (.dat) format.",
            UserWarning,
            stacklevel=2,
        )
    return mrs_struct


def get_full_path(paths: str | list[str]) -> str | list[str]:
    """Resolve one or more file paths to absolute canonical paths.

    Mirrors ``GetFullPath.m``.

    Parameters
    ------------
    paths: str | list[str]
        Single path string or list of path strings.

    Returns
    --------
        str | list[str]: resolved path(s)
            Absolute path string or list of absolute path strings.
    """
    if isinstance(paths, str):
        return os.path.abspath(os.path.expanduser(paths))
    return [os.path.abspath(os.path.expanduser(p)) for p in paths]


def sort_nat(items: list[str], mode: str = "ascend") -> list[str]:
    """Sort strings in natural alphanumeric order.

    Mirrors ``sort_nat.m``.

    Parameters
    ------------
    items: list[str]
        List of strings to sort.
    mode: str
        Sorting direction: ``'ascend'`` or ``'descend'``.

    Returns
    --------
        list[str]: list of str
            Naturally sorted copy of ``items``.

    Raises
    ------
    ValueError
        If ``mode`` is neither ``'ascend'`` nor ``'descend'``.
    """
    mode_low = mode.lower()
    if mode_low not in ("ascend", "descend"):
        raise ValueError("Sorting direction must be 'ascend' or 'descend'.")

    def nat_key(text: str) -> list[tuple[int, object]]:
        """Build natural sort key splitting digits and text."""
        parts = re.split(r"(\d+)", text)
        key: list[tuple[int, object]] = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append((0, int(part)))
            else:
                key.append((1, part))
        return key

    sorted_items = sorted(items, key=nat_key, reverse=(mode_low == "descend"))
    return sorted_items


def merge_structs(s1: dict[str, object], s2: dict[str, object]) -> MRSStruct:
    """Merge two dictionaries/structures into a single ``MRSStruct``.

    Mirrors ``mergestructs.m``.

    Parameters
    ------------
    s1: dict[str, object]
        Base mapping.
    s2: dict[str, object]
        Mapping whose fields overwrite or extend ``s1``.

    Returns
    --------
        MRSStruct: MRSStruct object
            Combined ``MRSStruct``.
    """
    out = MRSStruct.from_dict(dict(s1))
    for key, val in s2.items():
        out[key] = val
    return out


def write_xxx(text: str, old: str) -> str:
    """Replace occurrences of ``old`` within ``text`` by ``'X'`` characters.

    Mirrors ``write_xxx.m``.

    Parameters
    ------------
    text: str
        Source text buffer.
    old: str
        Substring to mask out (quotes and surrounding whitespace stripped).

    Returns
    --------
        str: str object
            Masked text string of identical length.
    """
    cleaned = re.sub(r'\s*"\s*', "", old)
    if not cleaned:
        return text
    return text.replace(cleaned, "X" * len(cleaned))
