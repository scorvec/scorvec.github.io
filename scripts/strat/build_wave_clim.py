#!/usr/bin/env python3
"""Climatological stationary waves 1 and 2 in geopotential height, by calendar day: reference/wave_clim.nc.

wave_maps.py compares each map's wave-1 or wave-2 with it: the amplitude as a multiple of the day's normal amplitude
("× normal", and the amplified / weak / near-normal label) and the normal ridges' longitudes (the open markers), the
ridges left out where the normal has no standing wave to speak of.

TWO NORMALS. The ridges come from the mean COMPLEX coefficient (below). The amplitude a day is measured against is
the MEAN AMPLITUDE of the days (..._amp): a day's wave is the standing wave plus that day's travelling part, so its
amplitude is on average larger than the standing wave's, and dividing by the standing wave alone read a typical day
as 1.3 to 2 times normal.

WHY A COMPLEX COEFFICIENT, NOT AN AMPLITUDE. The question is partly phase: a forecast ridge sitting over the normal
ridge and one sitting over the normal trough can have the same amplitude. So the climatology stored here is the mean
COMPLEX coefficient of each wavenumber per calendar day: averaging complex coefficients keeps the phase (where the
normal ridges are), averaging amplitudes would throw it away.

  band mean of z over 55-65 deg (cos-weighted, per hemisphere)
    -> rfft along longitude -> coefficients at k=1 and k=2  -> complex, per day
    -> mean over 1991-2020 per slot of the 366-slot calendar (quantmet.degdays.doy366: 29 Feb is slot 59 and
       1 March 60 in every year; 29 Feb, sampled in leap years only, is the mean of its neighbours)
    -> smoothed by keeping the first 3 annual harmonics

Sources
  100, 500 hPa  WeatherBench2's ERA5 (1.5 deg, 240x121, anonymous GCS), 00 UTC: k=1 and k=2 are planetary waves, which
                240 longitudes resolve with a wide margin, and the stationary waves have no meaningful diurnal cycle.
                The mean amplitude comes from the same days.
  10 hPa        WeatherBench2's levels stop at 50 hPa, so 10 hPa comes from the ERA5 10/100 hPa day-of-year MEAN height
                field, 1991-2020 (ARCO ERA5 00 UTC every 5th day, +/-7-day window, 1 deg; strat/strat_clim_1991-2020.nc
                in the GEPS reference set, release geps-ref-v3). The FFT is linear, so each coefficient of the mean
                field is the mean complex coefficient, as above. Its 73 sampled days (common-year days of year) are put on
                the 366-slot calendar, interpolated periodically onto all 366 slots and smoothed to the same 3
                harmonics. Its 100 hPa level is reduced the same way purely as a check against WeatherBench2's.
                A mean field has no daily amplitudes, so the 10 hPa mean amplitude is read from ARCO ERA5 itself
                (0.25 deg, 00 UTC every 5th day, 1991-2020): the same harmonics fitted to the sampled days.

Built in Actions (.github/workflows/wave-clim.yml), which commits the file:

    python scripts/strat/build_wave_clim.py --strat-clim PATH/strat_clim_1991-2020.nc
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from quantmet.degdays import doy366

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "reference" / "wave_clim.nc"

WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_EVERY = 5  # days between the 10 hPa samples
G = 9.80665
Y0, Y1 = 1991, 2020  # the period of the 10 hPa normal too
LEVELS = (100, 500)  # from WeatherBench2; 10 hPa from the strat climatology
BAND = (55.0, 65.0)  # the wave-driving latitudes, as in wave_maps
WAVES = (1, 2)  # zonal wavenumbers kept
NHARM = 3  # annual harmonics kept when smoothing over the calendar
HEMIS = (("nh", 1.0), ("sh", -1.0))


def band_coeff(z: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """(time, lat, lon) heights in gpm -> (time, len(WAVES)) complex coefficients of the cos-weighted band mean over
    BAND, scaled so |coeff| is the wave AMPLITUDE in gpm (rfft returns a sum, not a mean, so it needs 2/N). Longitudes
    run east from 0 E, as on wave_maps' climatology origin."""
    rows = (np.abs(lat) >= BAND[0]) & (np.abs(lat) <= BAND[1])
    w = np.cos(np.deg2rad(lat[rows]))
    prof = np.tensordot(z[:, rows, :], w / w.sum(), axes=([1], [0]))  # (time, lon)
    return np.fft.rfft(prof, axis=-1)[:, list(WAVES)] * 2.0 / prof.shape[-1]


def smooth_doy(c: np.ndarray, nharm: int = NHARM) -> np.ndarray:
    """Keep the first `nharm` annual harmonics of a 366-long complex series.

    A raw day-of-year mean over 30 years still carries visible sampling noise, and the maps divide by this
    climatology: noise in the denominator would show up as day-to-day jitter in the published ratio.
    """
    s = np.fft.fft(c)
    keep = np.zeros_like(s)
    keep[0] = s[0]
    for k in range(1, nharm + 1):
        keep[k] = s[k]
        keep[-k] = s[-k]
    return np.fft.ifft(keep)


def harmonic_fit(slot: np.ndarray, y: np.ndarray, nharm: int = NHARM) -> np.ndarray:
    """The first `nharm` annual harmonics fitted by least squares to values y on calendar slots (0..365): (366,)."""

    def design(s):
        a = 2 * np.pi * np.asarray(s, float) / 366.0
        return np.column_stack([np.ones(len(a))] + [f(k * a) for k in range(1, nharm + 1) for f in (np.cos, np.sin)])

    return design(np.arange(366)) @ np.linalg.lstsq(design(slot), y, rcond=None)[0]


def arco_amplitude(level: int = 10) -> dict:
    """{(level, hemi, k): mean amplitude (366,), gpm} from ARCO ERA5 at 00 UTC every ARCO_EVERY days, Y0-Y1."""
    import dask

    dask.config.set(scheduler="threads", num_workers=16)
    z = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)["geopotential"].sel(level=level)
    days = np.arange(f"{Y0}-01-01", f"{Y1 + 1}-01-01", ARCO_EVERY, dtype="datetime64[D]").astype("datetime64[ns]")
    lat = z.latitude.values
    rows = np.flatnonzero((np.abs(lat) >= BAND[0]) & (np.abs(lat) <= BAND[1]))
    amp = {h: [] for h, _ in HEMIS}
    for i in range(0, len(days), 64):
        zz = z.sel(time=days[i : i + 64]).isel(latitude=rows).chunk({"time": 1}).compute()
        la = zz.latitude.values
        for h, sign in HEMIS:
            side = np.sign(la) == sign
            amp[h].append(np.abs(band_coeff(zz.values[:, side] / G, la[side])))
        print(f"  10 hPa: {min(i + 64, len(days))} of {len(days)} days", flush=True)
    slot = doy366(days)
    return {
        (level, h, k): harmonic_fit(slot, np.concatenate(amp[h])[:, j]) for h, _ in HEMIS for j, k in enumerate(WAVES)
    }


def wb2_normals() -> tuple[dict, dict]:
    """({(level, hemi, k): smoothed complex (366,) normal}, {the same keys: the smoothed mean amplitude}) from
    WeatherBench2 at 00 UTC, Y0-Y1, read a year at a time."""
    import dask

    dask.config.set(scheduler="threads", num_workers=16)  # the chunks stream in parallel
    z = xr.open_zarr(WB2, storage_options={"token": "anon"})["geopotential"].sel(level=list(LEVELS))
    lat = z.latitude.values
    sums = {(lev, h): np.zeros((366, len(WAVES)), complex) for lev in LEVELS for h, _ in HEMIS}
    asum = {key: np.zeros((366, len(WAVES))) for key in sums}
    n = np.zeros(366)
    for y in range(Y0, Y1 + 1):
        zy = z.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
        zy = zy.isel(time=zy.time.dt.hour.values == 0).transpose("time", "level", "latitude", "longitude").compute()
        slot = doy366(zy.time.values)
        np.add.at(n, slot, 1)
        for i, lev in enumerate(zy.level.values):
            for h, sign in HEMIS:
                side = np.sign(lat) == sign
                c = band_coeff(zy.values[:, i][:, side] / G, lat[side])
                np.add.at(sums[(int(lev), h)], slot, c)
                np.add.at(asum[(int(lev), h)], slot, np.abs(c))
        print(f"  {y}: {zy.sizes['time']} fields", flush=True)
    out, mean_amp = {}, {}
    for (lev, h), s in sums.items():
        c, a = s / n[:, None], asum[(lev, h)] / n[:, None]
        for x in (c, a):
            x[59] = 0.5 * (x[58] + x[60])  # 29 Feb has ~1 year in 4: take its neighbours before the harmonic fit
        for j, k in enumerate(WAVES):
            out[(lev, h, k)] = smooth_doy(c[:, j])
            mean_amp[(lev, h, k)] = smooth_doy(a[:, j].astype(complex)).real
            amp = np.abs(out[(lev, h, k)])
            print(
                f"  {lev} hPa {h.upper()}: normal wave-{k} {amp.min():.0f}..{amp.max():.0f} gpm, mean amplitude "
                f"{mean_amp[(lev, h, k)].min():.0f}..{mean_amp[(lev, h, k)].max():.0f} gpm",
                flush=True,
            )
    return out, mean_amp


def strat_normals(path: Path) -> dict:
    """{(level, hemi, k): smoothed complex (366,) normal} at 10 and 100 hPa from the ERA5 day-of-year mean height
    field."""
    c = xr.open_dataset(path)
    lat = c.lat.values
    d0 = c.doy.values.astype(float)
    d0 = np.where(d0 > 59, d0 + 1, d0)  # common-year day of year -> 366-slot day (29 Feb is interpolated)
    xs = np.concatenate([d0 - 366, d0, d0 + 366])
    slots = np.arange(1, 367)
    out = {}
    for lev in (10, 100):
        zl = c["Z"].sel(level=lev).transpose("doy", "lat", "lon").values
        for h, sign in HEMIS:
            side = np.sign(lat) == sign
            cc = band_coeff(zl[:, side], lat[side])
            for j, k in enumerate(WAVES):
                re, im = (np.interp(slots, xs, np.tile(part, 3)) for part in (cc[:, j].real, cc[:, j].imag))
                out[(lev, h, k)] = smooth_doy(re + 1j * im)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strat-clim",
        default=str(HERE / "data" / "strat_clim_1991-2020.nc"),
        help="the ERA5 10/100 hPa day-of-year climatology (strat/strat_clim_1991-2020.nc of release geps-ref-v3)",
    )
    a = ap.parse_args()
    print(f"  WeatherBench2 ERA5 {Y0}-{Y1}, 00 UTC, {LEVELS} hPa", flush=True)
    wb2, mean_amp = wb2_normals()
    mean_amp |= arco_amplitude()
    st = strat_normals(Path(a.strat_clim))
    for (h, _), k in itertools.product(HEMIS, WAVES):
        ref, sm = wb2[(100, h, k)], st[(100, h, k)]
        dph = np.degrees(np.angle(sm * np.conj(ref)))
        win = np.abs(ref) > 0.25 * np.abs(ref).max()
        print(
            f"  check 100 hPa {h.upper()} wave-{k}: amplitude ratio (strat climatology / WeatherBench2) median "
            f"{np.median(np.abs(sm)[win] / np.abs(ref)[win]):.2f}, phase difference median "
            f"{np.median(np.abs(dph[win])):.0f} deg where the standing wave exists",
            flush=True,
        )
        amp = np.abs(st[(10, h, k)])
        print(f"  10 hPa {h.upper()}: normal wave-{k} {amp.min():.0f}..{amp.max():.0f} gpm", flush=True)
    normals = {**wb2, **{key: v for key, v in st.items() if key[0] == 10}}
    out = {}
    for (lev, h, k), c in sorted(normals.items(), key=lambda kv: (kv[0][2], -kv[0][0], kv[0][1])):
        out[f"{h}{lev}_k{k}_re"] = ("doy", c.real.astype("float32"))
        out[f"{h}{lev}_k{k}_im"] = ("doy", c.imag.astype("float32"))
        # a mean of amplitudes cannot be below the amplitude of the mean; the two are smoothed separately (and at
        # 10 hPa sampled from different days), which can leave it a little under: floored at the standing wave's
        amp, stand = mean_amp[(lev, h, k)], np.abs(c)
        big = stand >= 0.25 * stand.max()  # where a standing wave exists: a few gpm of summer wave says nothing
        short = float(np.max(1 - amp[big] / stand[big]))
        print(
            f"  {lev} hPa {h.upper()} wave-{k}: mean amplitude {amp.min():.0f}..{amp.max():.0f} gpm, at most "
            f"{100 * max(short, 0):.0f} % under the standing wave (where one exists) before the floor",
            flush=True,
        )
        if short > 0.15:
            raise SystemExit(f"{h}{lev} wave-{k}: the mean amplitude is {100 * short:.0f} % below the standing wave's")
        out[f"{h}{lev}_k{k}_amp"] = ("doy", np.maximum(amp, stand).astype("float32"))
    d = xr.Dataset(out, coords={"doy": np.arange(1, 367)})
    d.attrs["source"] = f"WeatherBench2 ERA5 {Y0}-{Y1}, 00 UTC (100, 500 hPa)"
    d.attrs["source_10hPa"] = (
        f"ERA5 (ARCO) {Y0}-{Y1} day-of-year mean height, 00 UTC every 5th day +/-7 d (release geps-ref-v3)"
    )
    d.attrs["band"] = f"{BAND[0]}-{BAND[1]} deg cos-weighted, both hemispheres"
    d.attrs["note"] = (
        f"mean COMPLEX k=1 and k=2 coefficients per calendar day ({{hemi}}{{level}}_k{{k}}_re/_im), smoothed to {NHARM} "
        "annual harmonics; |coeff| is amplitude in gpm, longitudes from 0 E; ..._amp the mean AMPLITUDE of the days, "
        f"gpm (10 hPa: ARCO ERA5 00 UTC every {ARCO_EVERY}th day); doy 1..366 on a 366-slot calendar "
        "(29 Feb = 60, 1 March = 61 in every year)"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_netcdf(OUT, encoding={k: {"zlib": True, "complevel": 6} for k in out})
    print(f"  wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
