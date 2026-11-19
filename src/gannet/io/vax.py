"""DEC VAX floating-point conversion and raw binary reading utilities."""

from typing import BinaryIO

import numpy as np
from numpy.typing import ArrayLike, NDArray


def uint32le_to_vaxf(uint32le: ArrayLike) -> NDArray[np.float64]:
    """Convert little-endian uint32 words to VAXF single-precision floats.

    Mirrors ``uint32le_to_VAXF.m``.

    Parameters
    ------------
    uint32le: ArrayLike
        Array of raw 32-bit unsigned integers in little-endian order.

    Returns
    --------
        NDArray[np.float64]: numpy array
            Decoded floating-point values.
    """
    raw = np.asarray(uint32le, dtype=np.uint32)
    word2 = raw >> np.uint32(16)
    word1 = raw & np.uint32(0xFFFF)
    vax_int = (word1 << np.uint32(16)) | word2

    sign = (vax_int >> np.uint32(31)).astype(np.float64)
    exp = ((vax_int >> np.uint32(23)) & np.uint32(0xFF)).astype(np.float64)
    frac = (vax_int & np.uint32(0x7FFFFF)).astype(np.float64)

    mantissa = 0.5 + frac / 16777216.0
    result = ((-1.0) ** sign) * mantissa * (2.0 ** (exp - 128.0))
    zero_mask = raw == np.uint32(0)
    if np.any(zero_mask):
        result = np.where(zero_mask, 0.0, result)
    return result


def uint64le_to_vaxg(uint32le: ArrayLike) -> NDArray[np.float64]:
    """Convert pairs of little-endian uint32 words to VAXG double floats.

    Mirrors ``uint64le_to_VAXG.m``.

    Parameters
    ------------
    uint32le: ArrayLike
        Array of raw 32-bit unsigned integers (length must be even).

    Returns
    --------
        NDArray[np.float64]: numpy array
            Decoded double-precision floating-point values.
    """
    arr = np.asarray(uint32le, dtype=np.uint32)
    if arr.ndim == 2 and arr.shape[0] == 2:
        raw = arr.T
    else:
        raw = arr.reshape(-1, 2)
    raw_a = raw[:, 0]
    raw_b = raw[:, 1]

    vax_a = ((raw_a & np.uint32(0xFFFF)) << np.uint32(16)) | (raw_a >> np.uint32(16))
    vax_b = ((raw_b & np.uint32(0xFFFF)) << np.uint32(16)) | (raw_b >> np.uint32(16))
    vax_int = (vax_a.astype(np.uint64) << np.uint64(32)) | vax_b.astype(np.uint64)

    sign = (vax_int >> np.uint64(63)).astype(np.float64)
    exp = ((vax_int >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.float64)
    frac = (vax_int & np.uint64(0xFFFFFFFFFFFFF)).astype(np.float64)

    mantissa = 0.5 + frac / 9007199254740992.0
    result = ((-1.0) ** sign) * mantissa * (2.0 ** (exp - 1024.0))
    zero_mask = (raw_a == np.uint32(0)) & (raw_b == np.uint32(0))
    if np.any(zero_mask):
        result = np.where(zero_mask, 0.0, result)
    return result


def vaxf_to_uint32le(values: ArrayLike) -> NDArray[np.uint32]:
    """Convert floating-point values to VAXF little-endian uint32 words.

    Mirrors ``VAXF_to_uint32le.m``.

    Parameters
    ------------
    values: ArrayLike
        Floating-point array to encode.

    Returns
    --------
        NDArray[np.uint32]: numpy array
            Encoded uint32 words.
    """
    arr = np.asarray(values, dtype=np.float64)
    orig_shape = arr.shape
    flat = arr.ravel().copy()
    zero_mask = flat == 0.0
    sign = (flat < 0.0).astype(np.uint32)
    abs_val = np.where(zero_mask, 1.0, np.abs(flat))

    exp = np.floor(np.log2(abs_val) + 1.0 + 128.0)
    frac = (abs_val / (2.0 ** (exp - 128.0))) - 0.5
    frac_int = np.floor(frac * 16777216.0).astype(np.uint32)
    exp_int = exp.astype(np.uint32)

    vax_int = (
        (sign << np.uint32(31))
        | ((exp_int & np.uint32(0xFF)) << np.uint32(23))
        | (frac_int & np.uint32(0x7FFFFF))
    )
    uint32le = ((vax_int & np.uint32(0xFFFF)) << np.uint32(16)) | (
        vax_int >> np.uint32(16)
    )
    uint32le = np.where(zero_mask, np.uint32(0), uint32le)
    return uint32le.reshape(orig_shape)


def _encode_vax64(values: ArrayLike, bias: int) -> NDArray[np.uint32]:
    """Shared encoder for 64-bit VAXD (bias=128) and VAXG (bias=1024)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    zero_mask = arr == 0.0
    sign = (arr < 0.0).astype(np.uint64)
    abs_val = np.where(zero_mask, 1.0, np.abs(arr))

    exp = np.floor(np.log2(abs_val) + 1.0 + bias)
    frac = (abs_val / (2.0 ** (exp - bias))) - 0.5
    if bias == 128:
        frac_int = np.round(frac * 72057594037927936.0).astype(np.uint64)
        exp_shift = 55
        frac_mask = np.uint64(0x7FFFFFFFFFFFFF)
    else:
        frac_int = np.round(frac * 9007199254740992.0).astype(np.uint64)
        exp_shift = 52
        frac_mask = np.uint64(0xFFFFFFFFFFFFF)

    vax_int = (
        (sign << np.uint64(63))
        | (exp.astype(np.uint64) << np.uint64(exp_shift))
        | (frac_int & frac_mask)
    )
    vax_int = np.where(zero_mask, np.uint64(0), vax_int)

    vax_a = (vax_int >> np.uint64(32)).astype(np.uint32)
    vax_b = (vax_int & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    le_a = ((vax_a & np.uint32(0xFFFF)) << np.uint32(16)) | (vax_a >> np.uint32(16))
    le_b = ((vax_b & np.uint32(0xFFFF)) << np.uint32(16)) | (vax_b >> np.uint32(16))
    paired: NDArray[np.uint32] = np.column_stack([le_a, le_b]).T
    return paired


def vaxd_to_uint64le(values: ArrayLike) -> NDArray[np.uint32]:
    """Convert double-precision floats to VAXD little-endian uint32 pairs.

    Mirrors ``VAXD_to_uint64le.m``.

    Parameters
    ------------
    values: ArrayLike
        Double-precision values to encode.

    Returns
    --------
        NDArray[np.uint32]: 2D numpy array
            Array of shape ``(2, N)`` containing little-endian uint32 pairs.
    """
    return _encode_vax64(values, bias=128)


def vaxg_to_uint64le(values: ArrayLike) -> NDArray[np.uint32]:
    """Convert double-precision floats to VAXG little-endian uint32 pairs.

    Mirrors ``VAXG_to_uint64le.m``.

    Parameters
    ------------
    values: ArrayLike
        Double-precision values to encode.

    Returns
    --------
        NDArray[np.uint32]: 2D numpy array
            Array of shape ``(2, N)`` containing little-endian uint32 pairs.
    """
    return _encode_vax64(values, bias=1024)


def fread_vaxg(fid: BinaryIO, number: int, precision: str) -> NDArray[np.float64]:
    """Read numbers from a binary stream encoded in VAXF/VAXG format.

    Mirrors ``freadVAXG.m`` (on 64-bit platforms ``'float'`` maps to VAXF
    when single-precision 32-bit SDAT is read — note: in MATLAB 64-bit, wait:
    in MATLAB ``intmax`` is 2147483647 for int32! So ``freadVAXG`` with
    ``'float'`` reads ``uint32`` and calls ``uint32le_to_VAXF``!).

    Parameters
    ------------
    fid: BinaryIO
        Open binary file object positioned at start of data.
    number: int
        Number of elements to read.
    precision: str
        Precision string (``'float32'``, ``'single'``, ``'float'``,
        ``'float64'``, ``'double'``).

    Returns
    --------
        NDArray[np.float64]: 1D numpy array
            Decoded floating-point values.
    """
    prec = precision.lower()
    if prec in ("float32", "single", "float"):
        raw = np.fromfile(fid, dtype="<u4", count=number)
        return uint32le_to_vaxf(raw)
    if prec in ("float64", "double"):
        raw = np.fromfile(fid, dtype="<u4", count=2 * number)
        return uint64le_to_vaxg(raw)
    return np.fromfile(fid, dtype=np.dtype(precision), count=number).astype(np.float64)


def fwrite_vaxd(fid: BinaryIO, values: ArrayLike, precision: str) -> int:
    """Write values to an open binary file in VAXF or VAXD format.

    Mirrors ``fwriteVAXD.m``.

    Parameters
    ------------
    fid: BinaryIO
        Open writable binary file object.
    values: ArrayLike
        Numeric array to write.
    precision: str
        Precision string (``'float32'``, ``'single'``, ``'float'``,
        ``'float64'``, ``'double'``).

    Returns
    --------
        int: int object
            Number of elements written.
    """
    prec = precision.lower()
    if prec in ("float32", "single", "float"):
        encoded = vaxf_to_uint32le(values).ravel(order="F")
        encoded.astype("<u4").tofile(fid)
        return int(encoded.size)
    if prec in ("float64", "double"):
        encoded = vaxd_to_uint64le(values)
        encoded.T.ravel().astype("<u4").tofile(fid)
        return int(encoded.shape[1])
    arr = np.asarray(values)
    arr.tofile(fid)
    return int(arr.size)


def fwrite_vaxg(fid: BinaryIO, values: ArrayLike, precision: str) -> int:
    """Write values to an open binary file in VAXF or VAXG format.

    Mirrors ``PRIAM/fwriteVAXG.m``.

    Parameters
    ------------
    fid: BinaryIO
        Open writable binary file object.
    values: ArrayLike
        Numeric array to write.
    precision: str
        Precision string (``'float32'``, ``'single'``, ``'float'``,
        ``'float64'``, ``'double'``).

    Returns
    --------
        int: int object
            Number of elements written.
    """
    prec = precision.lower()
    if prec in ("float32", "single", "float"):
        encoded = vaxf_to_uint32le(values).ravel(order="F")
        encoded.astype("<u4").tofile(fid)
        return int(encoded.size)
    if prec in ("float64", "double"):
        encoded = vaxg_to_uint64le(values)
        encoded.T.ravel().astype("<u4").tofile(fid)
        return int(encoded.shape[1])
    arr = np.asarray(values)
    arr.tofile(fid)
    return int(arr.size)


def read_raw_gannet(
    filename: str,
    dtype_str: str,
    dims: tuple[int, ...],
    endian: str = "b",
    offset: int = 0,
) -> NDArray[np.float64]:
    """Read a shaped binary array from a raw file at a byte offset.

    Mirrors ``readraw_Gannet.m``.

    Parameters
    ------------
    filename: str
        Path to binary raw file.
    dtype_str: str
        Data type name (e.g., ``'float'``, ``'float32'``, ``'int16'``).
    dims: tuple[int, ...]
        Target shape using Fortran (column-major) ordering.
    endian: str
        Byte order: ``'b'`` for big-endian or ``'l'`` for little-endian.
    offset: int
        Byte offset from start of file. Defaults to 0.

    Returns
    --------
        NDArray[np.float64]: numpy array
            Reshaped array of requested dimensions.
    """
    type_map = {
        "float": "f4",
        "float32": "f4",
        "single": "f4",
        "double": "f8",
        "float64": "f8",
        "int16": "i2",
        "integer*2": "i2",
        "int32": "i4",
        "integer*4": "i4",
        "uint8": "u1",
    }
    prefix = ">" if endian.lower().startswith("b") else "<"
    np_dtype = np.dtype(prefix + type_map.get(dtype_str, dtype_str))
    count = int(np.prod(dims))
    with open(filename, "rb") as fid:
        fid.seek(offset, 0)
        raw = np.fromfile(fid, dtype=np_dtype, count=count)
    result: NDArray[np.float64] = raw.reshape(dims, order="F").astype(np.float64)
    return result
