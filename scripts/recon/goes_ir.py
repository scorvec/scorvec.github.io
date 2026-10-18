#!/usr/bin/env python3
"""GOES-East infrared (ABI band 13, 10.3 um) under the reconnaissance track, one cut-out per centre pass, for recon.html.

For each mission in the recon mirror's manifest that has centre passes (manifest `legs`, see recon_mirror.legs), the
ABI Level 2 cloud and moisture imagery (CMIP) of band 13 nearest the pass time (within 10 min) is read from NOAA's open
data on AWS (noaa-goes19 from 7 Apr 2025, when GOES-19 became GOES-East; noaa-goes16 before), choosing the smallest
sector that holds the whole cut-out: a mesoscale sector (1-min, when one is over the storm), else CONUS (5-min), else
the full disk (10-min). The brightness temperature is resampled (nearest neighbour, from the GOES fixed-grid
projection) onto a Web Mercator grid of +-3 deg longitude around the centre, so the page lays it under the map with
Leaflet's imageOverlay, and coloured with a stepped enhancement (grey above -30 C, colours for colder cloud tops).

Output on the tdr-data branch (shared with tdr_mirror.py):  sat/<year>/<mission key>/<YYYYmmddHHMM>.webp, and in
index.json  "sat": {<mission key>: [[t_pass, t_image, sector, S, W, N, E, path], ...]}.  state.json "sat" remembers
passes already done (or found to have no imagery). Runs in recon.yml after the TDR step; numpy/h5py/Pillow are
installed only when there is something to make.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tdr_mirror as TM  # (Branch, git helpers, get)

NOW = datetime.now(UTC)
HW = 3.0  # half-width of the cut-out, degrees of longitude (legs reach 150 n mi = 2.5 deg)
PX = 360  # cut-out size (px), square in Web Mercator (~1.8 km per pixel, ABI band 13 is 2 km)
MAX_PER_MISSION = 8
TOL = 600  # s: nearest image within 10 min of the pass
G19_FROM = datetime(2025, 4, 7, 15, tzinfo=UTC)
CREDIT = "NOAA GOES-East ABI band 13 (10.3 um) cloud and moisture imagery, NOAA Open Data Dissemination (AWS)"


def log(*a):
    print(*a, flush=True)


def need_libs():
    try:
        import numpy, h5py, PIL  # noqa: F401,E401  availability probe: installed below when missing
    except ImportError:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--user",
                "--break-system-packages",
                "numpy",
                "h5py",
                "pillow",
            ],
            check=True,
        )
        import site

        sys.path.insert(0, site.getusersitepackages())


def bucket(t):
    return ("noaa-goes19", "G19") if datetime.fromtimestamp(t, UTC) >= G19_FROM else ("noaa-goes16", "G16")


KEY_RX = re.compile(r"<Key>([^<]+)</Key>")


def s3_list(bkt, prefix):
    out, tok = [], None
    for _ in range(5):
        url = f"https://{bkt}.s3.amazonaws.com/?list-type=2&prefix={prefix}" + (
            f"&continuation-token={urllib.request.quote(tok)}" if tok else ""
        )
        x = TM.get(url) or ""
        out += KEY_RX.findall(x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            break
        tok = m.group(1)
    return out


def start_of(key):
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    d = datetime(int(m.group(1)), 1, 1, tzinfo=UTC) + timedelta(
        days=int(m.group(2)) - 1, hours=int(m.group(3)), minutes=int(m.group(4)), seconds=int(m.group(5))
    )
    return int(d.timestamp())


def candidates(t, product, sector_tag):
    """Band-13 files of one product within TOL of t, nearest first."""
    bkt, sat = bucket(t)
    keys = []
    for dt in (-TOL, 0, TOL):
        d = datetime.fromtimestamp(t + dt, UTC)
        pre = f"{product}/{d:%Y}/{d:%j}/{d:%H}/OR_{product}{sector_tag}-M6C13_{sat}_s{d:%Y%j%H}"
        keys += s3_list(bkt, pre)
    keys = sorted(set(keys), key=lambda k: abs(start_of(k) - t))
    return bkt, [k for k in keys if abs(start_of(k) - t) <= TOL]


def fetch(bkt, key):
    data = TM.get(f"https://{bkt}.s3.amazonaws.com/{key}", binary=True)
    if not data:
        return None
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        f.write(data)
        return f.name


def target_grid(la, lo):
    import numpy as np

    mlat = math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
    hw = math.radians(HW)
    xs = np.linspace(lo - HW, lo + HW, PX)
    ys = np.linspace(mlat + hw, mlat - hw, PX)  # top row first
    lats = np.degrees(np.arctan(np.sinh(ys)))
    lon2, lat2 = np.meshgrid(xs, lats)
    bounds = [float(lats[-1]), lo - HW, float(lats[0]), lo + HW]  # S, W, N, E
    return lat2, lon2, bounds


def scan_angles(h5, lat, lon):
    """GOES-R PUG forward projection (lat/lon -> fixed-grid scan angles x, y); NaN where not visible."""
    import numpy as np

    pj = h5["goes_imager_projection"].attrs
    req, rpol = float(pj["semi_major_axis"][0]), float(pj["semi_minor_axis"][0])
    H = float(pj["perspective_point_height"][0]) + req
    l0 = math.radians(float(pj["longitude_of_projection_origin"][0]))
    phi, lam = np.radians(lat), np.radians(lon)
    e2 = (req**2 - rpol**2) / req**2
    phic = np.arctan((rpol**2 / req**2) * np.tan(phi))
    rc = rpol / np.sqrt(1 - e2 * np.cos(phic) ** 2)
    sx = H - rc * np.cos(phic) * np.cos(lam - l0)
    sy = -rc * np.cos(phic) * np.sin(lam - l0)
    sz = rc * np.sin(phic)
    vis = H * (H - sx) >= sy**2 + (req**2 / rpol**2) * sz**2
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    y = np.arctan(sz / sx)
    x[~vis] = np.nan
    y[~vis] = np.nan
    return x, y


def coord(h5, name):
    v = h5[name]
    return float(v.attrs["scale_factor"][0]), float(v.attrs["add_offset"][0]), v.shape[0]


def sample(path, lat, lon, need):
    """-> brightness temperature (K) on the target grid, or None when the file holds less than `need` of the cut-out
    (or less than `need` of it has a valid value)."""
    import numpy as np
    import h5py

    with h5py.File(path, "r") as h5:
        x, y = scan_angles(h5, lat, lon)
        sx, ox, nx = coord(h5, "x")
        sy, oy, ny = coord(h5, "y")
        i = np.round((x - ox) / sx)
        j = np.round((y - oy) / sy)
        ok = np.isfinite(i) & np.isfinite(j) & (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        if ok.mean() < need:
            return None
        ii, jj = i[ok].astype(int), j[ok].astype(int)
        i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
        cmi = h5["CMI"]
        raw = cmi[j0:j1, i0:i1]
        fill = cmi.attrs.get("_FillValue")
        sc, of = float(cmi.attrs["scale_factor"][0]), float(cmi.attrs["add_offset"][0])
        bt = raw.astype("float32") * sc + of
        if fill is not None:
            bt[raw == fill[0]] = np.nan
        dqf = h5["DQF"][j0:j1, i0:i1] if "DQF" in h5 else None
        if dqf is not None:
            bt[dqf > 1] = np.nan  # 0 good, 1 conditionally usable; 2+ out of range / no value
        out = np.full(lat.shape, np.nan, dtype="float32")
        out[ok] = bt[jj - j0, ii - i0]
    good = np.isfinite(out) & (out > 160) & (out < 340)
    if good.mean() < need:
        return None
    out[~good] = np.nan
    return out


# stepped enhancement: (upper bound degC, RGB); above -30 C a grey ramp
STEPS = [
    (-80, (255, 255, 255)),
    (-75, (180, 40, 200)),
    (-70, (200, 20, 30)),
    (-65, (240, 110, 20)),
    (-60, (245, 200, 30)),
    (-50, (40, 170, 60)),
    (-40, (40, 110, 210)),
    (-30, (90, 200, 230)),
]


def colour(bt):
    import numpy as np
    from PIL import Image

    t = bt - 273.15
    g = np.clip(np.nan_to_num((30 - t) / 60.0, nan=0.0), 0, 1)  # 30 C black-grey .. -30 C light grey
    v = (40 + g * 175).astype("uint8")
    rgb = np.stack([v, v, v], -1)
    lo = -1e9
    for hi, c in STEPS:
        m = (t > lo) & (t <= hi)
        rgb[m] = c
        lo = hi
    rgb[~np.isfinite(t)] = (0, 0, 0)
    a = np.where(np.isfinite(t), 255, 0).astype("uint8")
    im = Image.fromarray(np.dstack([rgb, a]), "RGBA")
    b = io.BytesIO()
    im.save(b, "WEBP", quality=80, method=5)
    return b.getvalue()


def make(t, la, lo):
    """-> (webp bytes, t_image, sector, bounds) or (None, reason)."""
    lat, lon, bounds = target_grid(la, lo)
    for product, tags, need in (
        ("ABI-L2-CMIPM", ("1", "2"), 0.97),
        ("ABI-L2-CMIPC", ("",), 0.99),
        ("ABI-L2-CMIPF", ("",), 0.9),
    ):
        for tag in tags:
            bkt, keys = candidates(t, product, tag)
            for k in keys[:2]:
                p = fetch(bkt, k)
                if not p:
                    continue
                try:
                    bt = sample(p, lat, lon, need)
                except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
                    log(f"    ! {k}: {type(e).__name__}: {e}")
                    bt = None
                finally:
                    os.unlink(p)
                if bt is not None:
                    sector = {"ABI-L2-CMIPM": f"mesoscale {tag}", "ABI-L2-CMIPC": "CONUS", "ABI-L2-CMIPF": "full disk"}[
                        product
                    ]
                    return colour(bt), start_of(k), f"{bkt.split('-')[1].upper()} {sector}", bounds
                if product == "ABI-L2-CMIPM":
                    break  # this mesoscale sector is not over the storm: next sector
    return None, "no GOES-East band-13 image within 10 min that covers the cut-out"


def passes(m):
    """Centre-pass times of a mission (manifest legs: [dir, tc, ...]), one per pass, at most MAX_PER_MISSION."""
    ts = sorted({g[1] for g in (m.get("legs") or [])})
    out = []
    for t in ts:
        if not out or t - out[-1] > 1800:
            out.append(t)
    return out[:MAX_PER_MISSION]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo", default="/tmp/tdr-git")
    ap.add_argument("--storm", default="", help="only this storm (backfill), e.g. AL132025")
    ap.add_argument("--max", type=int, default=40, help="at most this many new images per run")
    a = ap.parse_args()
    man_p = Path(a.manifest) / "manifest.json"
    if not man_p.exists():
        return 0
    man = json.loads(man_p.read_text())
    tok, repo = os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "scorvec/scorvec.github.io")
    url = f"https://x-access-token:{tok}@github.com/{repo}.git" if tok else f"https://github.com/{repo}.git"
    br = TM.Branch(a.repo, url)
    state = json.loads(br.read("state.json") or b"{}")
    index = json.loads(br.read("index.json") or b"{}")
    index.setdefault("missions", {})
    sat = index.setdefault("sat", {})
    done = state.setdefault("sat", {})  # "<key>|<t>" -> 1 (made) or reason
    changed, made, t0 = False, 0, time.time()
    sid_only = a.storm.upper()
    todo = []
    for key, m in man["missions"].items():
        if not m.get("storm") or (sid_only and m["storm"] != sid_only):
            continue
        # live: passes of the last 3 days only (older ones were done when they were new, or are backfills)
        for t in passes(m):
            if f"{key}|{t}" in done or (not sid_only and NOW.timestamp() - t > 3 * 86400):
                continue
            todo.append((key, m, t))
    if todo:
        need_libs()
    for key, m, t in todo[: a.max]:
        fixes = m.get("fixes") or []
        # centre at the pass: the mission's fix nearest in time, else its median position
        c = min(fixes, key=lambda f: abs(f[0] - t))[1:3] if fixes else (m.get("med") or [None, None])
        if c[0] is None:
            done[f"{key}|{t}"] = "no centre"
            continue
        if c[1] > -25:
            done[f"{key}|{t}"] = "east of GOES-East's useful view"
            continue
        log(f"  {key} pass {datetime.fromtimestamp(t, UTC):%d %H:%MZ} at {c[0]:.1f},{c[1]:.1f}")
        res = make(t, c[0], c[1])
        if res[0] is None:
            done[f"{key}|{t}"] = res[1]
            log(f"    - {res[1]}")
            changed = True
            continue
        img, ti, sector, b = res
        path = f"sat/{key[:4]}/{key}/{datetime.fromtimestamp(t, UTC):%Y%m%d%H%M}.webp"
        br.write(path, img)
        lst = [x for x in sat.get(key, []) if x[0] != t] + [
            [t, ti, sector, round(b[0], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3), path, len(img)]
        ]
        sat[key] = sorted(lst)
        done[f"{key}|{t}"] = 1
        made += 1
        changed = True
        log(f"    {sector} {datetime.fromtimestamp(ti, UTC):%H:%M:%SZ}, {len(img) / 1e3:.0f} KB")
    # prune missions that left the mirror
    for key in [k for k in sat if k not in man["missions"]]:
        for x in sat.pop(key):
            br.remove(x[7])
        changed = True
    for k in [k for k in done if k.split("|")[0] not in man["missions"]]:
        done.pop(k)
        changed = True
    if not changed:
        log("satellite: nothing new")
        return 0
    tot = sum(x[8] for v in sat.values() for x in v)
    index["sat_bytes"], index["sat_credit"] = tot, CREDIT
    br.write("index.json", TM.jbytes(index))
    br.write("state.json", TM.jbytes(state))
    log(f"satellite: {made} new image(s) in {time.time() - t0:.0f}s; {tot / 1e6:.2f} MB over {len(sat)} mission(s)")
    br.commit_push(f"GOES IR under the recon track: {NOW:%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
