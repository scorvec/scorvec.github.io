#!/usr/bin/env python3
"""IMERG V07 daily rainfall over the Americas on the bias-field 1° grid (grid.py) - the observed half (2026-10-13).

Two products, kept apart:
  late   GPM_3IMERGDL  ~1 day latency, satellite-only with a climatological gauge calibration: the record that keeps
                       up with the forecasts, so the bias field is measured against it;
  final  GPM_3IMERGDF  monthly GPCC gauge adjustment over land, but it lags (V07 Final ended 2025-09-30 on 2026-10-13):
                       used over the overlap to measure Late's own error, which the bias page shows beside the model bias.
Days are UTC days, as the 00Z model accumulations are. Access: GES DISC OPeNDAP, one request per day for `precipitation`
over 60°S-60°N, 170°W-30°W at 0.1° (~4 MB), and when OPeNDAP answers 5xx (it did for a whole month on 2026-10-13) the
whole daily file over HTTPS (~27 MB) cut to the same box; 1° box means. Earthdata login from a .netrc file or
EARTHDATA_USERNAME / EARTHDATA_PASSWORD (earthaccess), over IPv4 only: on the Actions runner the login host resolved to an
IPv6 route that is not there ("Network is unreachable", 2026-10-13).

    python scripts/precip_bias/imerg_americas.py --product late --month 2024-03 --out DIR
-> DIR/imerg_{product}_{YYYYMM}.npz: dates (YYYYMMDD), pr (day, lat, lon) float32 mm/day, missing
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G

PRODUCTS = {"late": ("GPM_3IMERGDL", "3B-DAY-L."), "final": ("GPM_3IMERGDF", "3B-DAY.")}
DATA = "https://data.gesdisc.earthdata.nasa.gov/data/"
OPENDAP = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/"


def query() -> str:
    return f"precipitation[0:0][{G.IM_LON0}:1:{G.IM_LON1}][{G.IM_LAT0}:1:{G.IM_LAT1}]"  # read at call time (--domain)


def _ipv4_only():
    import urllib3.util.connection as uc

    uc.HAS_IPV6 = False  # urllib3 then resolves A records only (allowed_gai_family -> AF_INET)


@functools.cache
def _login():
    _ipv4_only()
    import earthaccess

    for k in range(4):
        try:
            if earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc"):
                break
        except Exception as ex:  # noqa: BLE001
            if k == 3:
                raise
            print(f"  Earthdata login failed ({repr(ex)[:60]}); retrying", flush=True)
        time.sleep(10 * (k + 1))
    return earthaccess


@functools.cache
def _session():
    return _login().get_requests_https_session()


def granule_urls(product: str, d0: date, d1: date) -> dict:
    """{date: (OPeNDAP base url, HTTPS file url)} from one CMR search (retried)."""
    short, stem = PRODUCTS[product]
    ea = _login()
    res = []
    for k in range(5):
        try:
            res = ea.search_data(short_name=short, version="07", temporal=(d0.isoformat(), d1.isoformat()))
            break
        except Exception as ex:  # noqa: BLE001 - CMR 500s under load
            if k == 4:
                print(f"  CMR {short} {d0}..{d1} failed: {repr(ex)[:80]}", flush=True)
            time.sleep(3 * (k + 1))
    out = {}
    for r in res:
        for link in r.data_links():
            name = link.rsplit("/", 1)[-1]
            if link.startswith("https") and link.endswith(".nc4") and name.startswith(stem):
                out[datetime.strptime(name.split(".3IMERG.")[1][:8], "%Y%m%d").date()] = (
                    link.replace(DATA, OPENDAP),
                    link,
                )
    return out


def _read(content: bytes, subset: bool) -> np.ndarray:
    import netCDF4  # not h5py - two HDF5 libraries in one process abort on the runner

    with netCDF4.Dataset("imerg.nc4", mode="r", memory=content) as f:
        v = f.variables["precipitation"]
        v.set_auto_maskandscale(False)
        a = np.asarray(v[0] if subset else v[0, G.IM_LON0 : G.IM_LON1 + 1, G.IM_LAT0 : G.IM_LAT1 + 1]).astype(
            "float32"
        )  # (lon, lat)
    if a.shape != (G.IM_LON1 - G.IM_LON0 + 1, G.IM_LAT1 - G.IM_LAT0 + 1):
        raise ValueError(f"shape {a.shape}")
    return a


def fetch_day(urls, tries: int = 4):
    """(NY, NX) mm/day, validated: domain mean 0.8-8 mm/day, >= 95 % valid cells, max < 1500 mm/day.
    OPeNDAP subset first; on a 5xx or a timeout the whole daily file."""
    import requests

    dap, full = urls
    w = G.area_weights()
    use_full = False
    for k in range(tries):
        try:
            if not use_full:
                try:
                    r = _session().get(dap + ".nc4?" + query(), timeout=120)
                    r.raise_for_status()
                    a = _read(r.content, subset=True)
                except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as ex:
                    code = getattr(getattr(ex, "response", None), "status_code", None)
                    if code is not None and code < 500:
                        raise
                    use_full = True
            if use_full:
                r = _session().get(full, timeout=300)
                r.raise_for_status()
                a = _read(r.content, subset=False)
            g = G.imerg_to_1deg(a)
            ok = np.isfinite(g)
            mu = float(np.nansum(g * w) / (w * ok).sum())
            mx = float(np.nanmax(g))
            if ok.mean() < 0.95 or not (0.8 < mu < 8.0) or mx > 1500:
                raise ValueError(f"implausible: valid {ok.mean():.2f} mean {mu:.2f} max {mx:.0f}")
            return g
        except Exception as ex:  # noqa: BLE001
            if k == tries - 1:
                print(f"  {full.rsplit('/', 1)[-1][:40]} failed: {repr(ex)[:90]}", flush=True)
                return None
            time.sleep(4 * (k + 1))
    return None


def month(product: str, ym: str, out: Path, last: date | None = None) -> Path | None:
    y, m = map(int, ym.split("-"))
    d0 = date(y, m, 1)
    d1 = (d0 + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    if last:
        d1 = min(d1, last)
    if d1 < d0:
        return None
    t0 = time.time()
    urls = granule_urls(product, d0, d1)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    got, missing = [], []
    for d in days:
        g = fetch_day(urls[d]) if d in urls else None
        if g is None:
            missing.append(f"{d:%Y%m%d}: {'no granule' if d not in urls else 'fetch failed'}")
        else:
            got.append((d, g))
    if not got:
        print(f"IMERG {product} {ym}: nothing fetched ({len(urls)} granules)", flush=True)
        return None
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"imerg_{product}_{y:04d}{m:02d}.npz"
    np.savez_compressed(
        f,
        dates=np.array([f"{d:%Y%m%d}" for d, _ in got]),
        pr=np.stack([g for _, g in got]),
        lat=G.LAT,
        lon=G.LON,
        missing=np.array(missing, dtype=str),
    )
    print(f"IMERG {product} {ym}: {len(got)}/{len(days)} days, {(time.time() - t0) / 60:.1f} min", flush=True)
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, choices=sorted(PRODUCTS))
    ap.add_argument("--month", required=True, help="YYYY-MM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--last", help="last date to include (YYYY-MM-DD)")
    ap.add_argument(
        "--domain",
        default="americas",
        choices=["americas", "global"],
        help="global = 60°S-60°N at all longitudes (2026-10-14, tropical rainfall forecast)",
    )
    a = ap.parse_args()
    G.use_domain(a.domain)
    return 0 if month(a.product, a.month, Path(a.out), date.fromisoformat(a.last) if a.last else None) else 1


if __name__ == "__main__":
    sys.exit(main())
