#!/usr/bin/env python3
"""
HRRR forecast soundings for the short-range model viewer (models.html -> click anywhere -> the sounding explorer).

The ECAPE job (ecape.yml) already holds every forecast hour's native-level cube (fetch_hrrr.py: PRES, HGT, TMP, SPFH,
UGRD, VGRD on the lowest 42 hybrid levels, ~70 hPa top) plus, with --surface, the surface layer (surface P and z, 2 m
T and Td, 10 m wind). This script keeps every 4th column (12 km) and writes them as TILES: one file per 24 x 24 columns
holding ALL forecast hours, so a click costs one download and stepping through the hours is instant.

  encode STEM --out DIR --fxx F   one hour -> DIR/snd/_parts/{ty}_{tx}_{F}.z  (run inside the per-hour loop)
  pack DIR --hours "0 1 ..."      parts -> DIR/snd/{ty}_{tx}.bin + DIR/snd/index.json; removes _parts
  check STEM DIR --fxx F          decode one hour back (the page's algorithm, in numpy) and report the errors

What is stored per column and hour (43 levels: surface + 42 hybrid):
  surface: p_s (2 Pa), z_s (1 m), T2 (0.1 K), Td2 (0.2 K), u10/v10 (0.25 m/s)
  levels : p residual from A_k + B_k*p_s (5 Pa), T (0.1 K), Td (0.2 K, from SPFH), u/v (0.25 m/s)
Winds are EARTH-relative (HRRR's are grid-relative; rotated here). Heights are NOT stored: the page rebuilds them
hypsometrically from z_s upward with the virtual temperature, plus a fitted per-level correction zc (see fit_coef).
A_k, B_k (and zc) are fitted on the cycle's first encoded hour and reused for every hour (the residual absorbs the rest; HRRR's
coordinate follows dry pressure, so p is not exactly A + B p_s - rms 3 Pa near the ground, ~35 Pa mid-column).

Tile file (.bin): b"SND1", uint16 nhours, uint16 0, nhours x uint32 byte lengths, then one zlib stream per hour.
A stream = the hour's planes in order [surface 6 vars][level vars 5 x 42], each plane ty x tx, predicted
(left neighbour; first column from the row above), taken modulo 2**16, byte-shuffled (all low bytes, then all high
bytes) over the whole stream, zlib level 6. Decoding = inflate, unshuffle, cumulative sum mod 2**16, then signed
int16 for everything except p_s (unsigned).
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

D = 4                         # keep every 4th column: 12 km
TS = 24                       # columns per tile side
NLEV = 42
SFC_SCALE = [2.0, 1.0, 0.1, 0.2, 0.25, 0.25]       # p_s Pa, z_s m, T2 K, Td2 K, u10, v10 m/s
LEV_SCALE = [5.0, 0.1, 0.2, 0.25, 0.25]            # p residual Pa, T K, Td K, u, v m/s
R_EARTH = 6371229.0


def load(stem: Path):
    meta = json.loads(stem.with_suffix(".json").read_text())
    nv, nl, ny, nx = meta["shape"]
    cube = np.memmap(stem.with_suffix(".f32"), np.float32, "r", shape=(nv, nl, ny, nx))
    sfc = np.fromfile(str(stem) + "_sfc.f32", np.float32).reshape(6, ny, nx)
    return meta, cube, sfc


def lcc_latlon(g, ny, nx):
    """Lat/lon of every native grid point (spherical LCC, tangent at latin1 = latin2)."""
    phi1 = np.deg2rad(g["latin1"]); n = np.sin(phi1)
    F = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n
    rho0 = R_EARTH * F / np.tan(np.pi / 4 + phi1 / 2) ** n
    lov = np.deg2rad(((g["lov"] + 180) % 360) - 180)
    la1, lo1 = np.deg2rad(g["lat1"]), np.deg2rad(((g["lon1"] + 180) % 360) - 180)
    rho = R_EARTH * F / np.tan(np.pi / 4 + la1 / 2) ** n
    x0, y0 = rho * np.sin(n * (lo1 - lov)), rho0 - rho * np.cos(n * (lo1 - lov))
    x = x0 + np.arange(nx) * g["dx"]; y = y0 + np.arange(ny) * g["dy"]
    X, Y = np.meshgrid(x, y)
    rho = np.sign(n) * np.hypot(X, rho0 - Y)
    lon = np.rad2deg(lov + np.arctan2(X, rho0 - Y) / n)
    lat = np.rad2deg(2 * np.arctan((R_EARTH * F / rho) ** (1 / n)) - np.pi / 2)
    return lat, lon


def alpha_of(lat, lon):
    """Angle of the grid's +i axis counter-clockwise from east (same as scripts/models/models.py)."""
    dlon = (np.gradient(lon, axis=1) + 180) % 360 - 180
    return np.arctan2(np.gradient(lat, axis=1), dlon * np.cos(np.deg2rad(lat)))


def dewpoint(q, p):
    """Dewpoint (K) from specific humidity (kg/kg) and pressure (Pa), Bolton over water."""
    e = np.maximum(q * p / (0.622 + 0.378 * q), 1e-3) / 100.0
    l = np.log(e / 6.112)
    return 243.5 * l / (17.67 - l) + 273.15


def quantise(meta, cube, sfc, coef):
    """(6 + 5*NLEV, ny4, nx4) int64 codes for one hour on the 12 km grid."""
    g = meta["grid"]; ny, nx = g["ny"], g["nx"]
    sl = np.s_[::D, ::D]
    lat, lon = lcc_latlon(g, ny, nx)
    al = alpha_of(lat[sl], lon[sl])
    ca, sa = np.cos(al), np.sin(al)
    s = sfc[(slice(None),) + sl].astype(np.float64)
    ps = s[0]
    u10, v10 = s[4] * ca - s[5] * sa, s[4] * sa + s[5] * ca
    planes = [np.round(x / k) for x, k in zip([ps, s[1], s[2], s[3], u10, v10], SFC_SCALE)]
    A, B = np.asarray(coef["A"]), np.asarray(coef["B"])
    for k in range(NLEV):
        p = np.asarray(cube[0, k][sl], np.float64)
        t = np.asarray(cube[2, k][sl], np.float64); q = np.asarray(cube[3, k][sl], np.float64)
        u = np.asarray(cube[4, k][sl], np.float64); v = np.asarray(cube[5, k][sl], np.float64)
        ue, ve = u * ca - v * sa, u * sa + v * ca
        pr = p - (A[k] + B[k] * ps)
        planes += [np.round(x / sc) for x, sc in zip([pr, t, dewpoint(q, p), ue, ve], LEV_SCALE)]
    return np.stack(planes).astype(np.int64)


def fit_coef(cube, sfc):
    """Per-level pressure coefficients (p ~ A + B p_s) and height corrections (HRRR's HGT minus the hypsometric
    height rebuilt from the surface: a near-constant offset per level, -2 m at level 1 to +11 m mid-column, spread
    <= 2 m - HGT and PRES are not located at the same point of the layer). Fitted on every 8th column of one hour."""
    sl = np.s_[::2 * D, ::2 * D]
    ps = sfc[0][sl].astype(np.float64)
    A, B = [], []
    for k in range(NLEV):
        p = np.asarray(cube[0, k][sl], np.float64).ravel()
        b, a = np.polyfit(ps.ravel(), p, 1)
        A.append(round(float(a), 3)); B.append(round(float(b), 7))
    P = np.asarray(cube[0][(slice(None),) + sl], np.float64)
    p = np.concatenate([ps[None], P])
    t = np.concatenate([sfc[2][sl][None].astype(np.float64), np.asarray(cube[2][(slice(None),) + sl], np.float64)])
    td = np.concatenate([sfc[3][sl][None].astype(np.float64),
                         dewpoint(np.asarray(cube[3][(slice(None),) + sl], np.float64), P)])
    z = heights(ps, sfc[1][sl].astype(np.float64), t, td, p)
    H = np.asarray(cube[1][(slice(None),) + sl], np.float64)
    zc = [round(float(np.median(H[k] - z[k + 1])), 2) for k in range(NLEV)]
    return dict(A=A, B=B, zc=zc)


def predict(x):
    """Left-neighbour residuals per plane; the first column from the row above."""
    r = x.copy()
    r[..., 1:] = x[..., 1:] - x[..., :-1]
    r[..., 1:, 0] = x[..., 1:, 0] - x[..., :-1, 0]
    return r


def pack_stream(codes):
    b = (predict(codes) % 65536).astype("<u2")
    u8 = b.view(np.uint8).reshape(-1, 2)
    return zlib.compress(np.concatenate([u8[:, 0], u8[:, 1]]).tobytes(), 6)   # 9 is 14x slower for 3 %


def unpack_stream(buf, nplanes, ty, tx):
    """The page's decoder, in numpy (for `check`)."""
    raw = np.frombuffer(zlib.decompress(buf), np.uint8)
    n = raw.size // 2
    r = (raw[:n].astype(np.int64) | (raw[n:].astype(np.int64) << 8)).reshape(nplanes, ty, tx)
    r[..., :, 0] = np.cumsum(r[..., :, 0], axis=-1)            # first column down the rows
    x = np.cumsum(r, axis=-1) % 65536                           # then along each row
    x = x.astype(np.int64)
    signed = np.where(x >= 32768, x - 65536, x)
    signed[0] = x[0]                                            # p_s is unsigned
    return signed


def tiles(ny4, nx4):
    for ty, j0 in enumerate(range(0, ny4, TS)):
        for tx, i0 in enumerate(range(0, nx4, TS)):
            yield ty, tx, np.s_[:, j0:j0 + TS, i0:i0 + TS]


def cmd_encode(a):
    stem, out = Path(a.stem), Path(a.out) / "snd"
    parts = out / "_parts"; parts.mkdir(parents=True, exist_ok=True)
    meta, cube, sfc = load(stem)
    cf = parts / "coef.json"
    if cf.exists():
        coef = json.loads(cf.read_text())
    else:
        coef = fit_coef(cube, sfc); coef["grid"] = meta["grid"]; coef["fit_fxx"] = a.fxx
        cf.write_text(json.dumps(coef))
    codes = quantise(meta, cube, sfc, coef)
    n = 0
    for ty, tx, sl in tiles(*codes.shape[1:]):
        buf = pack_stream(codes[sl])
        (parts / f"{ty}_{tx}_{a.fxx}.z").write_bytes(buf); n += len(buf)
    print(f"  soundings f{a.fxx:02d}: {n / 1e6:.1f} MB in {sum(1 for _ in tiles(*codes.shape[1:]))} tiles", flush=True)


def cmd_pack(a):
    out = Path(a.out) / "snd"; parts = out / "_parts"
    hours = [int(h) for h in a.hours.split()]
    coef = json.loads((parts / "coef.json").read_text())
    g = coef["grid"]; ny4, nx4 = -(-g["ny"] // D), -(-g["nx"] // D)
    total = nt = 0
    for ty, tx, _ in tiles(ny4, nx4):
        bufs = []
        for h in hours:
            p = parts / f"{ty}_{tx}_{h}.z"
            if not p.exists():
                raise SystemExit(f"missing part {p.name}")
            bufs.append(p.read_bytes())
        head = b"SND1" + struct.pack("<HH", len(bufs), 0) + struct.pack(f"<{len(bufs)}I", *map(len, bufs))
        f = out / f"{ty}_{tx}.bin"
        f.write_bytes(head + b"".join(bufs)); total += f.stat().st_size; nt += 1
    idx = dict(v=1, model="hrrr", D=D, ts=TS, nx=nx4, ny=ny4, nlev=NLEV + 1, hours=hours,
               sfc_scale=SFC_SCALE, lev_scale=LEV_SCALE, A=coef["A"], B=coef["B"], zc=coef["zc"], grid=g,
               bytes=total, tiles=nt,
               note="12 km columns of the 3 km HRRR (every 4th point); level 0 = surface (2 m T/Td, 10 m wind)")
    (out / "index.json").write_text(json.dumps(idx, separators=(",", ":")))
    shutil.rmtree(parts)
    print(f"soundings: {nt} tiles, {len(hours)} hours, {total / 1e6:.1f} MB", flush=True)


def heights(ps, zs, t, td, p):
    """Hypsometric heights (m) of each level from the surface up; t, td, p are (nlev+1, ...) with level 0 = surface."""
    def tv(T, Td, P):
        e = 6.112 * np.exp(17.67 * (Td - 273.15) / (Td - 29.65)) * 100.0
        q = 0.622 * e / (P - 0.378 * e)
        return T * (1 + 0.608 * q)
    z = np.empty_like(t); z[0] = zs
    for k in range(1, t.shape[0]):
        tm = 0.5 * (tv(t[k], td[k], p[k]) + tv(t[k - 1], td[k - 1], p[k - 1]))
        z[k] = z[k - 1] + 287.04 * tm / 9.80665 * np.log(p[k - 1] / p[k])
    return z


def cmd_check(a):
    """Re-encode one hour in memory, decode it with the page's algorithm, and report errors against the source."""
    stem = Path(a.stem)
    meta, cube, sfc = load(stem)
    coef = fit_coef(cube, sfc)
    codes = quantise(meta, cube, sfc, coef)
    ny4, nx4 = codes.shape[1:]
    dec = np.empty_like(codes); nbytes = 0
    for ty, tx, sl in tiles(ny4, nx4):
        buf = pack_stream(codes[sl]); nbytes += len(buf)
        sub = codes[sl]
        dec[sl] = unpack_stream(buf, sub.shape[0], sub.shape[1], sub.shape[2])
    assert np.array_equal(dec, codes), "lossless stage is not lossless"
    A, B = np.asarray(coef["A"]), np.asarray(coef["B"])
    ps = dec[0] * SFC_SCALE[0]
    lev = dec[6:].reshape(NLEV, 5, ny4, nx4).astype(np.float64)
    p = np.concatenate([ps[None], A[:, None, None] + B[:, None, None] * ps + lev[:, 0] * LEV_SCALE[0]])
    t = np.concatenate([(dec[2] * SFC_SCALE[2])[None], lev[:, 1] * LEV_SCALE[1]])
    td = np.concatenate([(dec[3] * SFC_SCALE[3])[None], lev[:, 2] * LEV_SCALE[2]])
    z = heights(ps, dec[1] * SFC_SCALE[1], t, td, p)
    z[1:] += np.asarray(coef["zc"])[:, None, None]
    sl = np.s_[::D, ::D]
    P0 = np.asarray(cube[0][(slice(None),) + sl], np.float64); H0 = np.asarray(cube[1][(slice(None),) + sl], np.float64)
    T0 = np.asarray(cube[2][(slice(None),) + sl], np.float64)
    TD0 = dewpoint(np.asarray(cube[3][(slice(None),) + sl], np.float64), P0)
    rep = dict(
        bytes_per_hour=nbytes,
        p_err_pa=float(np.abs(p[1:] - P0).max()), t_err_k=float(np.abs(t[1:] - T0).max()),
        td_err_k=float(np.abs(td[1:] - TD0).max()),
        z_err_m_p50=float(np.median(np.abs(z[1:] - H0))), z_err_m_p99=float(np.percentile(np.abs(z[1:] - H0), 99)),
        z_err_m_max=float(np.abs(z[1:] - H0).max()),
        z_err_m_top_p99=float(np.percentile(np.abs(z[-1] - H0[-1]), 99)))
    print(json.dumps(rep, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("encode"); e.add_argument("stem"); e.add_argument("--out", required=True)
    e.add_argument("--fxx", type=int, required=True)
    p = sp.add_parser("pack"); p.add_argument("out"); p.add_argument("--hours", required=True)
    c = sp.add_parser("check"); c.add_argument("stem")
    a = ap.parse_args(argv)
    {"encode": cmd_encode, "pack": cmd_pack, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
