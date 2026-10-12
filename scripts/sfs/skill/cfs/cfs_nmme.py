"""CFSv2 seasonal hindcasts from the IRI NMME archive (anonymous data.nc), Sep / Oct starts 1991-2025:
hindcast 1991-2010 (.HINDCAST.MONTHLY, 24 members), real-time 2011-2025 (.FORECAST.PENTAD_SAMPLES.MONTHLY: 28 slots,
24 filled - sampled like the hindcast). L = 0.5 is the start month.
  Nino-3.4: server-side box mean (5S-5N, 190-240E) per member and lead 0-9.
  t2m (tref, K, global) and precipitation (prec, mm/day, 0-90N): server-side 3-month means L 1.5-3.5 and 2.5-4.5, per
  member, bilinear to the ERA5 1.5 deg grid."""
import sys, os, time, subprocess, numpy as np, xarray as xr
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import OUT

B = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2"
YRS = list(range(1991, 2026)); MON = {9: "Sep", 10: "Oct"}
GLAT = np.arange(-90.0, 90.0 + 1e-6, 1.5); GLON = np.arange(0.0, 360.0 - 1e-6, 1.5)
RAW = OUT + "raw_nmme/"


def src(y):
    return f"{B}/.HINDCAST/.MONTHLY" if y <= 2010 else f"{B}/.FORECAST/.PENTAD_SAMPLES/.MONTHLY"


def fetch(url, f):
    for k in range(6):
        r = subprocess.run(["curl", "-s", "-m", "600", "-o", f, "-w", "%{http_code}", url], capture_output=True, text=True)
        if r.stdout == "200":
            try:
                return xr.open_dataset(f, decode_times=False)
            except Exception:                                           # noqa: BLE001
                pass
        time.sleep(15 * (k + 1))
    raise RuntimeError(f"failed {url}")


def regrid_to(field, lat, lon, glat, glon):
    from scipy.interpolate import RegularGridInterpolator
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if lat[0] > lat[-1]: lat = lat[::-1]; field = field[..., ::-1, :]
    lon2 = np.concatenate([lon, [lon[0] + 360.0]]); f2 = np.concatenate([field, field[..., :1]], -1)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    flat = f2.reshape(-1, f2.shape[-2], f2.shape[-1]); out = np.empty((flat.shape[0], len(glat), len(glon)), np.float32)
    for k in range(flat.shape[0]):
        out[k] = RegularGridInterpolator((lat, lon2), flat[k], bounds_error=False, fill_value=None)((LA, LO))
    return out.reshape(*f2.shape[:-2], len(glat), len(glon))


def n34(y):
    f = f"{RAW}n34_{y}.nc"
    ds = fetch(f"{src(y)}/.sst/X/190/240/RANGEEDGES/Y/-5/5/RANGEEDGES/%5BX/Y%5Daverage/"
               f"S/(0000%201%20Sep%20{y})/(0000%201%20Oct%20{y})/RANGEEDGES/data.nc", f)
    a = ds["sst"].transpose("S", "M", "L").values.astype(np.float64)
    out = np.full((2, 28, 10), np.nan); out[:, :a.shape[1]] = a
    ds.close(); os.remove(f)
    return y, out


def seas(job):
    var, y, m, k = job
    npy = f"{RAW}{var}_{y}{m:02d}_{k}.npy"
    if os.path.exists(npy): return job, np.load(npy)
    l0, l1 = ((1.5, 3.5), (2.5, 4.5))[k]
    ysel = "" if var == "tref" else "Y/-1.5/90/RANGEEDGES/"
    f = f"{RAW}{var}_{y}{m:02d}_{k}.nc"
    ds = fetch(f"{src(y)}/.{var}/{ysel}S/(0000%201%20{MON[m]}%20{y})/VALUE/L/{l0}/{l1}/RANGEEDGES/%5BL%5Daverage/data.nc", f)
    da = ds[var].squeeze("S", drop=True).transpose("M", "Y", "X")
    a = da.values.astype(np.float32)
    a = np.where(np.abs(a) > 1e10, np.nan, a)
    glat = GLAT if var == "tref" else GLAT[GLAT >= 0]
    good = np.isfinite(a).any((1, 2))
    out = np.full((28, len(glat), len(GLON)), np.nan, np.float32)
    out[:a.shape[0]][good] = regrid_to(a[good], ds["Y"].values, ds["X"].values, glat, GLON)
    ds.close(); os.remove(f); np.save(npy, out)
    return job, out


def main():
    t0 = time.time()
    N = np.full((2, len(YRS), 28, 10), np.nan)
    with ThreadPoolExecutor(3) as ex:
        for y, a in ex.map(n34, YRS):
            N[:, YRS.index(y)] = a
    np.save(OUT + "cfs_n34.npy", N)
    print(f"n34 done {time.time() - t0:.0f} s; members per start {np.isfinite(N[..., 0]).sum(-1)[:, [0, 19, 20, 34]]}", flush=True)
    for var in ("tref", "prec"):
        jobs = [(var, y, m, k) for y in YRS for m in MON for k in (0, 1)]
        with ThreadPoolExecutor(3) as ex:
            for n, (job, a) in enumerate(ex.map(seas, jobs)):
                if (n + 1) % 20 == 0: print(f"  {var} {n + 1}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
        print(var, "done", flush=True)


if __name__ == "__main__":
    main()
