# ERA5 day-of-year normals (ensemble-anomaly monitor)

**These files are committed to the repo on purpose.** They are the one-time, expensive
product of reading *every day* of ERA5 over a 30- or 10-year period from the ARCO-ERA5
zarr store and taking the per-day-of-year mean. Building one is a 30–45 min,
network-bound download (~11k global fields for the 30-yr periods). **Do not delete
them and do not re-download unless the method or grid changes** — committing them means
this laptop, the GitHub Action fallback, and any future project all read the cache
instead of rebuilding.

## Files

| file | what it is | reload with |
|------|-----------|-------------|
| `ens_clim_<var>_<period>.nc` | **smoothed** normal — per-doy mean + periodic smoothing spline. This is what `render.py` reads to form anomalies. | `xr.open_dataarray(...)` |
| `ens_clim_raw_<var>_<period>.nc` | **raw** per-doy mean, no smoothing. The re-fit source: lets us change the smoothing (or anything downstream of the mean) **without re-downloading ERA5**. | `xr.open_dataarray(...)` |

`<var>` ∈ {`z500`, `t2m`} · `<period>` ∈ {`30yr` (1991–2020), `10yr` (2014–2023)}

Grids (see `src/common.py`): **z500 = full Northern Hemisphere** (0–90 N, all lon, 0.5°);
**t2m = North America** (15–75 N, 170–50 W, 0.5°). Dims: `(dayofyear 1..366, latitude, longitude)`.
z500 in **dam**, t2m in **K**. Both written zlib-compressed.

## Rebuilding (only if method/grid changes)

```bash
# mjo env has gcsfs + scipy; ~30–45 min per 30-yr var, network-bound
PY=/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python
$PY src/build_ens_clim.py --var z500 --period 30yr --workers 8
```

If only the *smoothing* needs to change (not the mean), re-fit from the cached
`ens_clim_raw_*.nc` instead — no download required.
