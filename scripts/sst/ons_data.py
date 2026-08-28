#!/usr/bin/env python3
"""Brazil ONS open data: ENA + EAR daily by basin and subsystem, 2000->.

Source: public S3 bucket ons-aws-prod-opendata (no auth). Yearly CSVs,
semicolon-separated. ENA arrives with ONS's own %-of-MLT norms built in;
EAR as % of maximum storage. Caches are full mirrors (historical years
immutable; current year re-fetched when older than a day).

Also builds the SIN basin geometry: Bacias_Hidrograficas_SIN shapefile
-> geojson with ENA-name normalization and macro-basin grouping
(AMAZONAS = Madeira+Tapajos+Xingu+Curua-Una+Uatuama+Jari,
PARAGUAI = Manso+Itiquira+Jauru+Correntes, ANTAS -> JACUI).

Outputs:
  ~/brazil_hydro/raw/ena_bacia_daily.json.gz    {basin: {date: [mwmed, pctmlt]}}
  ~/brazil_hydro/raw/ear_bacia_daily.json.gz    {basin: {date: pct}}
  ~/brazil_hydro/raw/ena_subsistema_daily.json.gz / ear_subsistema_daily.json.gz
  ~/brazil_hydro/out/brazil_basins.geojson      (grouped, ENA names)
  brazil_hydro/ena_norms.webp + ear_norms.webp  (site, hidden page)
  brazil_hydro/data/ons_daily.json

    python scripts/sst/ons_data.py [--backfill]
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRIV = Path.home() / "brazil_hydro"
RAW = PRIV / "raw"
OUTD = PRIV / "out"
SITE = REPO / "brazil_hydro"
BASE = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"
Y0 = 2000

GROUP = {"MADEIRA": "AMAZONAS", "TAPAJOS": "AMAZONAS", "XINGU": "AMAZONAS",
         "CURUA-UNA": "AMAZONAS", "UATUAMA": "AMAZONAS", "JARI": "AMAZONAS",
         "MANSO": "PARAGUAI", "ITIQUIRA": "PARAGUAI", "JAURU": "PARAGUAI",
         "CORRENTES": "PARAGUAI", "ITAJAI-ACU": "ITAJAI", "ANTAS": "JACUI"}
MAJORS = ["GRANDE", "PARANAIBA", "TIETE", "PARANAPANEMA", "PARANA", "IGUACU",
          "URUGUAI", "JACUI", "SAO FRANCISCO", "TOCANTINS", "AMAZONAS",
          "PARAIBA DO SUL"]
SUBS = ["SUDESTE/CENTRO-OESTE", "SUL", "NORDESTE", "NORTE"]


def norm(s: str) -> str:
    return (unicodedata.normalize("NFD", s).encode("ascii", "ignore")
            .decode().upper().strip())


def get_csv(url: str):
    with urllib.request.urlopen(url, timeout=180) as r:
        txt = r.read().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(txt), delimiter=";"))


def fetch_dataset(prefix: str, fname: str, keyf, valf, cache: Path,
                  backfill: bool) -> dict:
    data = {}
    if cache.exists():
        with gzip.open(cache, "rt") as f:
            data = json.load(f)
    thisyear = datetime.now().year
    years = (range(Y0, thisyear + 1) if backfill or not data
             else [thisyear - 1, thisyear])
    for y in years:
        done = data.get("_years", [])
        if y in done and y < thisyear:
            continue
        try:
            rows = get_csv(f"{BASE}/{prefix}/{fname}_{y}.csv")
        except Exception as e:                          # noqa: BLE001
            print(f"  {fname} {y}: {repr(e)[:60]}", flush=True)
            continue
        n = 0
        for row in rows:
            k, d, v = keyf(row), None, None
            try:
                d = row.get("ena_data") or row.get("ear_data")
                v = valf(row)
            except (ValueError, TypeError):
                continue
            if k and d and v is not None:
                data.setdefault(k, {})[d] = v
                n += 1
        data.setdefault("_years", [])
        if y not in data["_years"]:
            data["_years"].append(y)
        print(f"  {fname} {y}: {n} rows", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def build_geojson() -> None:
    out = OUTD / "brazil_basins.geojson"
    if out.exists():
        return
    import shapefile
    sf = shapefile.Reader(str(RAW / "bacias_sin" / "Bacias_Hidrograficas_SIN"))
    feats = {}
    for rec, shp in zip(sf.records(), sf.shapes()):
        name = norm(rec[1])
        name = GROUP.get(name, name)
        gj = shp.__geo_interface__
        polys = (gj["coordinates"] if gj["type"] == "MultiPolygon"
                 else [gj["coordinates"]])
        feats.setdefault(name, []).extend(polys)
    OUTD.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"basin": k},
         "geometry": {"type": "MultiPolygon", "coordinates": v}}
        for k, v in feats.items()]}))
    print(f"wrote {out} ({len(feats)} grouped basins)", flush=True)


def main() -> int:
    backfill = "--backfill" in sys.argv[1:]
    print("ENA by basin …", flush=True)
    ena_b = fetch_dataset(
        "ena_bacia_di", "ENA_DIARIO_BACIAS",
        lambda r: norm(r["nom_bacia"]),
        lambda r: [round(float(r["ena_bruta_bacia_mwmed"]), 1),
                   round(float(r["ena_bruta_bacia_percentualmlt"]), 1)],
        RAW / "ena_bacia_daily.json.gz", backfill)
    print("EAR by basin …", flush=True)
    ear_b = fetch_dataset(
        "ear_bacia_di", "EAR_DIARIO_BACIAS",
        lambda r: norm(r["nomecurto"]),
        lambda r: round(float(r["ear_verif_bacia_percentual"]), 2),
        RAW / "ear_bacia_daily.json.gz", backfill)
    print("ENA by subsystem …", flush=True)
    ena_s = fetch_dataset(
        "ena_subsistema_di", "ENA_DIARIO_SUBSISTEMA",
        lambda r: norm(r["nom_subsistema"]),
        lambda r: [round(float(r["ena_bruta_regiao_mwmed"]), 1),
                   round(float(r["ena_bruta_regiao_percentualmlt"]), 1)],
        RAW / "ena_subsistema_daily.json.gz", backfill)
    print("EAR by subsystem …", flush=True)
    ear_s = fetch_dataset(
        "ear_subsistema_di", "EAR_DIARIO_SUBSISTEMA",
        lambda r: norm(r["nom_subsistema"]),
        lambda r: round(float(r["ear_verif_subsistema_percentual"]), 2),
        RAW / "ear_subsistema_daily.json.gz", backfill)
    build_geojson()

    # ── charts: ENA % of MLT + EAR % for the majors, last 2 years ──────────
    t0 = np.datetime64(datetime.now().strftime("%Y-%m-%d")) - np.timedelta64(730, "D")

    def series(d, key, idx=None):
        if key not in d:
            return [], []
        days = sorted(k for k in d[key] if np.datetime64(k) >= t0)
        v = [d[key][k][idx] if idx is not None else d[key][k] for k in days]
        return [datetime.strptime(k, "%Y-%m-%d") for k in days], v

    for which, dat, ylab, fname, ttl in (
            ("ena", ena_b, "ENA, % of MLT", "ena_norms.webp",
             "Brazil ENA (natural inflow energy) — % of long-term mean by basin"),
            ("ear", ear_b, "EAR, % of max", "ear_norms.webp",
             "Brazil EAR (stored energy) — % of maximum by basin")):
        fig, axes = plt.subplots(4, 3, figsize=(14.5, 12.0), sharex=True)
        for ax, b in zip(axes.flat, MAJORS):
            td, v = series(dat, b, 1 if which == "ena" else None)
            if td:
                ax.plot(td, v, color="#1f4e8c", lw=0.9)
                k30 = np.ones(30) / 30
                vv = np.asarray(v, float)
                sm = np.convolve(np.nan_to_num(vv, nan=np.nanmean(vv)),
                                 k30, "full")[:len(vv)]
                sm[:29] = np.nan                    # spin-up masked
                ax.plot(td, sm, color="#c62828", lw=1.6)
            ax.axhline(100 if which == "ena" else 50, color="0.6", lw=0.7,
                       ls="--")
            ax.set_title(b, fontsize=10, fontweight="bold", loc="left")
            ax.grid(lw=0.25, alpha=0.5)
            ax.tick_params(labelsize=7.5)
            ax.set_ylabel(ylab, fontsize=7.5)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        fig.suptitle(ttl + " — daily (blue) + 30-day mean (red), last 2 years",
                     fontsize=13, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        SITE.mkdir(parents=True, exist_ok=True)
        fig.savefig(SITE / fname, dpi=110)
        plt.close(fig)
        print(f"wrote brazil_hydro/{fname}", flush=True)

    # compact JSON feed (subsystems + majors, 2 years)
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    feed = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "ena_pct_mlt": {}, "ear_pct": {}}
    for b in MAJORS + [norm(s) for s in SUBS]:
        for src, key in ((ena_b, b), (ena_s, b)):
            td, v = series(src, key, 1)
            if td:
                feed["ena_pct_mlt"][b] = {"dates": [f"{t:%Y-%m-%d}" for t in td],
                                          "pct": v}
                break
        for src, key in ((ear_b, b), (ear_s, b)):
            td, v = series(src, key)
            if td:
                feed["ear_pct"][b] = {"dates": [f"{t:%Y-%m-%d}" for t in td],
                                      "pct": v}
                break
    (SITE / "data" / "ons_daily.json").write_text(
        json.dumps(feed, separators=(",", ":")))
    print("wrote brazil_hydro/data/ons_daily.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
