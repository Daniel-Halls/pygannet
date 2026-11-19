"""De-identification tools for DICOM, GE P-file, Philips SDAT/SPAR, and Siemens TWIX."""

from __future__ import annotations

import os
import re
import shutil
import struct

import pydicom


def de_identify_disclaimer() -> str:
    """Return Gannet's standard MRS data de-identification disclaimer string.

    Parameters
    ------------
    None

    Returns
    --------
        str:
            Warning text explaining user responsibility to verify PHI removal.
    """
    return (
        "DISCLAIMER: Gannet de-identification removes common patient-identifying "
        "header fields (name, ID, DOB, accession, dates, site strings), but users "
        "remain responsible for verifying that all protected health information "
        "has been scrubbed before sharing."
    )


def _deid_output_path(in_path: str, out_path: str | None = None) -> str:
    """Build output path for a de-identified copy beside the input file.

    Parameters
    ------------
    in_path: str
        Original file path.
    out_path: str | None
        Explicit destination path, or ``None`` to append ``_deid``.

    Returns
    --------
        str:
            Destination file path.
    """
    if out_path is not None:
        return out_path
    stem, ext = os.path.splitext(in_path)
    return f"{stem}_deid{ext}"


def dicom_de_identify(
    in_path: str,
    out_path: str | None = None,
    subject_id: str = "sub-01",
) -> str:
    """Scrub patient-identifying DICOM tags and save a de-identified copy.

    Parameters
    ------------
    in_path: str
        Path to input DICOM (``.dcm`` / ``.ima``) file.
    out_path: str | None
        Optional output path; defaults to ``<stem>_deid<ext>``.
    subject_id: str
        Replacement anonymous PatientName / PatientID string.

    Returns
    --------
        str:
            Path to the written de-identified DICOM file.
    """
    ds = pydicom.dcmread(in_path, force=True)
    for tag_name in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "PatientWeight",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "StationName",
        "StudyID",
        "AccessionNumber",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
    ):
        if tag_name in ds:
            if tag_name in ("PatientName", "PatientID"):
                setattr(ds, tag_name, subject_id)
            elif tag_name.endswith("Date"):
                setattr(ds, tag_name, "19000101")
            else:
                setattr(ds, tag_name, "ANONYMIZED")

    dest = _deid_output_path(in_path, out_path)
    ds.save_as(dest)
    return dest


def ge_de_identify(
    in_path: str,
    out_path: str | None = None,
    subject_id: str = "sub-01",
) -> str:
    """Anonymize patient/exam text fields inside a GE ``.7`` P-file binary header.

    Parameters
    ------------
    in_path: str
        Path to input GE ``.7`` P-file.
    out_path: str | None
        Optional destination path; defaults to ``<stem>_deid.7``.
    subject_id: str
        Replacement anonymous patient identifier (truncated to fit field bytes).

    Returns
    --------
        str:
            Path to the de-identified GE P-file.
    """
    dest = _deid_output_path(in_path, out_path)
    shutil.copy2(in_path, dest)
    with open(dest, "r+b") as fid:
        header = bytearray(fid.read(150000))
        rev = struct.unpack_from("<f", header, 0)[0]
        # Zero patient name/ID slots in RDB_HEADER_REC (offset varies by rev)
        if rev >= 24.0:
            # Exam/Series patient string slots at fixed RDB header byte ranges
            offsets = [(142600, 65), (142665, 97), (143000, 64)]
        else:
            offsets = [(128000, 64), (128100, 64)]
        id_bytes = subject_id.encode("ascii", errors="ignore")
        for pos, length in offsets:
            if pos + length <= len(header):
                header[pos : pos + length] = b"\x00" * length
                if id_bytes:
                    n = min(len(id_bytes), length - 1)
                    header[pos : pos + n] = id_bytes[:n]
        fid.seek(0, os.SEEK_SET)
        fid.write(header[: len(header)])
    return dest


def philips_de_identify(
    spar_or_sdat_path: str,
    out_dir: str | None = None,
    subject_id: str = "sub-01",
) -> tuple[str, str]:
    """De-identify a Philips ``.sdat/.spar`` file pair by anonymizing ``.spar`` text.

    Parameters
    ------------
    spar_or_sdat_path: str
        Path to either the ``.spar`` or ``.sdat`` file.
    out_dir: str | None
        Destination directory (defaults to same directory with ``_deid`` suffix).
    subject_id: str
        Replacement patient/examination name written into ``.spar``.

    Returns
    --------
        tuple[str, str]:
            Paths ``(out_sdat_path, out_spar_path)`` to the written files.
    """
    stem = re.sub(r"\.(sdat|spar)$", "", spar_or_sdat_path, flags=re.IGNORECASE)
    in_sdat = f"{stem}.sdat" if os.path.exists(f"{stem}.sdat") else f"{stem}.SDAT"
    in_spar = f"{stem}.spar" if os.path.exists(f"{stem}.spar") else f"{stem}.SPAR"

    base_name = os.path.basename(stem)
    dest_dir = out_dir or os.path.dirname(os.path.abspath(stem))
    os.makedirs(dest_dir, exist_ok=True)
    tag = "" if out_dir else "_deid"
    out_sdat = os.path.join(dest_dir, f"{base_name}{tag}.sdat")
    out_spar = os.path.join(dest_dir, f"{base_name}{tag}.spar")

    if os.path.exists(in_sdat):
        shutil.copy2(in_sdat, out_sdat)

    if os.path.exists(in_spar):
        phi_keys = (
            "examination_name",
            "patient_name",
            "patient_birth_date",
            "scan_id",
            "scan_date",
            "equipment_sw_verions",
        )
        lines_out: list[str] = []
        with open(in_spar, "r", encoding="latin-1") as fid:
            for line in fid:
                replaced = False
                for key in phi_keys:
                    if line.strip().startswith(key) and ":" in line:
                        lhs = line.split(":", 1)[0]
                        val = (
                            subject_id
                            if key in ("examination_name", "patient_name", "scan_id")
                            else "ANONYMIZED"
                        )
                        lines_out.append(f"{lhs}: {val}\n")
                        replaced = True
                        break
                if not replaced:
                    lines_out.append(line)
        with open(out_spar, "w", encoding="latin-1") as fid:
            fid.writelines(lines_out)

    return (out_sdat, out_spar)


def twix_de_identify(
    in_path: str,
    out_path: str | None = None,
    subject_id: str = "sub-01",
) -> str:
    """Anonymize PHI tags inside the text header of a Siemens TWIX ``.dat`` file.

    Preserves exact header byte length so all binary MDH scan offsets remain valid.

    Parameters
    ------------
    in_path: str
        Path to input Siemens ``.dat`` TWIX file.
    out_path: str | None
        Optional destination path; defaults to ``<stem>_deid.dat``.
    subject_id: str
        Replacement patient name string.

    Returns
    --------
        str:
            Path to the de-identified TWIX ``.dat`` file.
    """
    dest = _deid_output_path(in_path, out_path)
    shutil.copy2(in_path, dest)
    first_u, second_u = (0, 0)
    with open(dest, "r+b") as fid:
        first_u, second_u = struct.unpack("<II", fid.read(8))
        is_vd = first_u < 10000 and second_u <= 64
        offsets: list[int] = []
        if not is_vd:
            offsets = [0]
        else:
            fid.seek(8, os.SEEK_SET)
            for _ in range(second_u):
                entry = fid.read(152)
                if len(entry) < 152:
                    break
                offsets.append(int(struct.unpack_from("<Q", entry, 8)[0]))
                # Scrub patName (offset 24, 64 bytes) inside VD MeasInfo table
                fid.seek(-152 + 24, os.SEEK_CUR)
                pat_buf = bytearray(64)
                id_b = subject_id.encode("ascii", errors="ignore")[:63]
                pat_buf[: len(id_b)] = id_b
                fid.write(pat_buf)
                fid.seek(152 - 24 - 64, os.SEEK_CUR)

        phi_patterns: list[tuple[re.Pattern[bytes], bytes]] = [
            (
                re.compile(
                    rb'(<ParamString\."(?:tPatientName|PatientName|PatientID|PatientBirthDay)">\s*\{\s*")([^"]*)("\s*\})'
                ),
                subject_id.encode("ascii", errors="ignore"),
            ),
            (
                re.compile(rb'(tPatientName\s*=\s*")([^"]*)(")'),
                subject_id.encode("ascii", errors="ignore"),
            ),
        ]
        for m_off in offsets:
            fid.seek(m_off, os.SEEK_SET)
            hdr_len = struct.unpack("<I", fid.read(4))[0]
            if hdr_len <= 4 or hdr_len > 20_000_000:
                continue
            hdr_bytes = bytearray(fid.read(hdr_len - 4))
            for pat, rep_id in phi_patterns:
                for m in pat.finditer(hdr_bytes):
                    s, e = m.span(2)
                    field_len = e - s
                    if field_len > 0:
                        padded = (rep_id[:field_len]).ljust(field_len, b" ")
                        hdr_bytes[s:e] = padded
            fid.seek(m_off + 4, os.SEEK_SET)
            fid.write(hdr_bytes)
    return dest
