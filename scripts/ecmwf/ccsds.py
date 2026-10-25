"""The leading rows of a GRIB2 field from the start of its message alone.

ECMWF open data packs fields with CCSDS (GRIB2 template 5.42), a sequential coder: the values come out in grid order,
north to south on the regular latitude-longitude grid, so the first rows of a field decode from the first bytes of its
message. A product that needs only high northern latitudes (the 65N circle of the polar cap's edge is row 99 of 721 at
0.25 deg, ~12 % of a message's bytes) fetches a prefix of each message instead of the whole field (rangefetch.fetch_rows,
prefix_share).

leading_rows() returns None for anything it does not handle - another packing, a bitmap, another grid or scanning
order, a prefix too short or no libaec - and the caller decodes the whole message with eccodes instead.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import sys
from pathlib import Path

import numpy as np

NJ = 721  # rows of the 0.25-deg global grid the prefix is sized for


def prefix_share(nrows: int, nj: int = NJ) -> float:
    """The share of a message fetched for its first nrows rows: the rows' share of the grid with a margin. The polar
    rows compress best, so a field needs less than its rows' share (rows 0-100 of v and z at 0.25 deg: at most 12.3 %
    of the message for 14 % of the rows); a prefix that falls short is refetched whole."""
    return min(1.0, 1.1 * nrows / nj + 0.01)


class _Stream(ctypes.Structure):  # libaec's struct aec_stream
    _fields_ = [
        ("next_in", ctypes.c_void_p),
        ("avail_in", ctypes.c_size_t),
        ("total_in", ctypes.c_size_t),
        ("next_out", ctypes.c_void_p),
        ("avail_out", ctypes.c_size_t),
        ("total_out", ctypes.c_size_t),
        ("bits_per_sample", ctypes.c_uint),
        ("block_size", ctypes.c_uint),
        ("rsi", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("state", ctypes.c_void_p),
    ]


def _load():
    names = [ctypes.util.find_library("aec")] + [
        str(Path(sys.prefix) / "lib" / n) for n in ("libaec.so.0", "libaec.dylib")
    ]
    for name in filter(None, names):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


AEC = _load()
AEC_DATA_3BYTE, AEC_DATA_MSB = 2, 4


def _signed(b: bytes) -> int:
    """A GRIB2 sign-and-magnitude integer."""
    v = int.from_bytes(b, "big")
    top = 1 << (8 * len(b) - 1)
    return -(v & (top - 1)) if v & top else v


def _sections(buf: bytes) -> dict:
    """{section number: (offset, length)} for sections 1-7 of the message's first field."""
    out, off = {}, 16
    while off + 5 <= len(buf) and buf[off : off + 4] != b"7777":
        n, num = struct.unpack(">IB", buf[off : off + 5])
        out[num] = (off, n)
        if num == 7 or n < 5:
            break
        off += n
    return out


def leading_rows(buf: bytes, nrows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(values (nrows, Ni) float32, latitudes (nrows,), longitudes (Ni,)) of the first nrows rows of the field in buf, a
    whole message or its first bytes; None when they cannot be had from it (see the module docstring). Longitudes run
    from the grid's first, unwrapped (ECMWF's 0.25-deg grid: 180 to 539.75)."""
    s = _sections(buf)
    if AEC is None or not {3, 5, 6, 7} <= s.keys():
        return None
    o3, _ = s[3]
    if struct.unpack(">H", buf[o3 + 12 : o3 + 14])[0] != 0 or buf[o3 + 71] != 0:
        return None  # not a regular lat-lon grid scanned west to east, north to south
    ni = struct.unpack(">I", buf[o3 + 30 : o3 + 34])[0]
    la1, dj = _signed(buf[o3 + 46 : o3 + 50]) * 1e-6, struct.unpack(">I", buf[o3 + 67 : o3 + 71])[0] * 1e-6
    lo1, di = (
        struct.unpack(">I", buf[o3 + 50 : o3 + 54])[0] * 1e-6,
        struct.unpack(">I", buf[o3 + 63 : o3 + 67])[0] * 1e-6,
    )
    o5, _ = s[5]
    if struct.unpack(">H", buf[o5 + 9 : o5 + 11])[0] != 42 or buf[s[6][0] + 5] != 255:
        return None  # not CCSDS, or a bitmap
    ref = struct.unpack(">f", buf[o5 + 11 : o5 + 15])[0]
    e, d = _signed(buf[o5 + 15 : o5 + 17]), _signed(buf[o5 + 17 : o5 + 19])
    nbits, flags, block = buf[o5 + 19], buf[o5 + 21], buf[o5 + 22]
    rsi = struct.unpack(">H", buf[o5 + 23 : o5 + 25])[0]
    want = nrows * ni
    if nbits == 0:  # a constant field
        vals = np.full(want, ref, "float64")
    else:
        width = 1 if nbits <= 8 else 2 if nbits <= 16 else 4
        data = buf[s[7][0] + 5 :]
        inp = ctypes.create_string_buffer(bytes(data), len(data))
        out = (ctypes.c_ubyte * (want * width))()
        st = _Stream(bits_per_sample=nbits, block_size=block, rsi=rsi, flags=flags & ~(AEC_DATA_3BYTE | AEC_DATA_MSB))
        if AEC.aec_decode_init(ctypes.byref(st)) != 0:
            return None
        st.next_in, st.avail_in = ctypes.cast(inp, ctypes.c_void_p), len(data)
        st.next_out, st.avail_out = ctypes.cast(out, ctypes.c_void_p), want * width
        AEC.aec_decode(ctypes.byref(st), 0)  # AEC_NO_FLUSH: as far as the bytes go
        got = st.total_out // width
        AEC.aec_decode_end(ctypes.byref(st))
        if got < want:
            return None
        dtype = {1: "<u1", 2: "<u2", 4: "<u4"}[width]  # native little-endian samples (MSB flag cleared)
        vals = (ref + np.frombuffer(bytes(out), dtype=dtype).astype("float64") * 2.0**e) / 10.0**d
    return vals.reshape(nrows, ni).astype("float32"), la1 - dj * np.arange(nrows), lo1 + di * np.arange(ni)
