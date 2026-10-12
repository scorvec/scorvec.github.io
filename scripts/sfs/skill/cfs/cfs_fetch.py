"""CFSv2 weekly means for the Sep 1 / Oct 1 starts from IRI SubX (server-side 7-day box means), on the ERA5 NH 1.5 grid.
Ensemble: the four cycles 00/06/12/18Z of the start day, member 1 of each (hindcast has only member 1; real-time
2018-2025 member 1 too, so the construction is identical across years). Days counted from each cycle's own init."""
import sys, time, subprocess, os, numpy as np, xarray as xr
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import YEARS, MONTHS, OUT, regrid

B = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.NCEP/.CFSv2"
VARS = {"t2m": (".tas", "", 1.0), "z500": (".zg", "P/500/VALUE/", 1.0), "pr": (".pr", "", 86400.0)}
MON = {9: "Sep", 10: "Oct"}


def one(job):
    t, m, y = job
    npy = f"{OUT}raw/cfs_{t}_{y}{m:02d}.npy"
    if os.path.exists(npy):
        return job, (np.load(npy), 4, 0), None
    if t == "z500" and y > 2016:
        return s3_z500(job)
    v, extra, fac = VARS[t]
    seg = "hindcast" if y <= 2016 else "forecast"
    msel = "" if seg == "hindcast" else "M/1/1/RANGEEDGES/"
    url = (f"{B}/.{seg}/{v}/{extra}{msel}Y/0/90/RANGEEDGES/S/(0000%201%20{MON[m]}%20{y})/(1800%201%20{MON[m]}%20{y})/RANGEEDGES/"
           f"L/0.5/34.5/RANGEEDGES/L/7/boxAverage/data.nc")
    f = f"{OUT}raw/cfs_{t}_{y}{m:02d}.nc"
    for k in range(5):
        r = subprocess.run(["curl", "-s", "-m", "600", "-o", f, "-w", "%{http_code}", url], capture_output=True, text=True)
        if r.stdout == "200":
            try:
                ds = xr.open_dataset(f, decode_times=False); break
            except Exception as e:                                      # noqa: BLE001
                err = str(e)
        else:
            err = r.stdout
        time.sleep(10 * (k + 1))
    else:
        return job, None, err
    da = ds[v[1:]]
    if "P" in da.dims: da = da.isel(P=0)
    da = da.isel(M=0).transpose("S", "L", "Y", "X")
    a = da.values.astype(np.float32) * fac                              # (4 cycles, 5 weeks, Y, X)
    a = np.where(np.abs(a) > 1e19, np.nan, a)
    ncyc = da.sizes["S"]
    wk = np.stack([a[:, 0], a[:, 1], 0.5 * (a[:, 2] + a[:, 3]), a[:, 4]], axis=1)   # wk1, wk2, wk34, wk5
    out = regrid(wk, ds["Y"].values, ds["X"].values)
    ds.close(); os.remove(f)
    if ncyc == 4: np.save(npy, out)
    return job, (out, ncyc, int(np.isnan(a).sum())), None


def s3_z500(job):
    """IRI's real-time SubX zg returns empty files, so z500 2019-2025 comes from the operational CFSv2 on S3
    (noaa-cfs-pds, time_grib_01 = member 01 of each cycle, the same member SubX carries): the first 140 six-hourly
    steps (6..840 h) by byte range from the .idx; day k = mean of steps 24(k-1)+6 .. 24k h."""
    import urllib.request, eccodes
    t, m, y = job
    npy = f"{OUT}raw/cfs_{t}_{y}{m:02d}.npy"
    outs = []
    for c in ("00", "06", "12", "18"):
        base = f"https://noaa-cfs-pds.s3.amazonaws.com/cfs.{y}{m:02d}01/{c}/time_grib_01/z500.01.{y}{m:02d}01{c}.daily.grb2"
        try:
            idx = urllib.request.urlopen(base + ".idx", timeout=120).read().decode().splitlines()
        except Exception as e:                                          # noqa: BLE001
            return job, None, f"{c}: {e}"
        offs = [int(l.split(":")[1]) for l in idx]
        hrs = [l.split(":")[5] for l in idx[:141]]
        assert hrs[0] == "6 hour fcst" and hrs[139] == "840 hour fcst", hrs[:2]
        req = urllib.request.Request(base, headers={"Range": f"bytes=0-{offs[140] - 1}"})
        raw = urllib.request.urlopen(req, timeout=300).read()
        fields = []
        for k in range(140):
            gid = eccodes.codes_new_from_message(raw[offs[k]:offs[k + 1]])
            ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
            v = eccodes.codes_get_values(gid).reshape(nj, ni)
            if k == 0:
                la0, la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                lo0, dlo = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"), eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
            eccodes.codes_release(gid)
            fields.append(v)
        a = np.stack(fields).reshape(35, 4, nj, ni).mean(1)                # (35 days, lat, lon)
        lat = np.linspace(la0, la1, nj); lon = lo0 + dlo * np.arange(ni)
        nh = lat >= -1.5
        wk = np.stack([a[0:7].mean(0), a[7:14].mean(0), a[14:28].mean(0), a[28:35].mean(0)])
        outs.append(regrid(wk[:, nh], lat[nh], lon))
    out = np.stack(outs)
    np.save(npy, out)
    return job, (out, 4, 0), None


def main():
    jobs = [(t, m, y) for t in VARS for m in MONTHS for y in YEARS]
    res = {t: np.full((2, len(YEARS), 4, 4, 61, 240), np.nan, np.float32) for t in VARS}
    t0 = time.time(); fails = []; done = 0
    with ThreadPoolExecutor(3) as ex:
        for job, r, err in ex.map(one, jobs):
            t, m, y = job
            if r is None:
                fails.append((job, err)); continue
            out, ncyc, nnan = r
            if ncyc != 4 or nnan:
                print(f"  NOTE {job}: cycles {ncyc}, NaNs {nnan}", flush=True)
            res[t][MONTHS.index(m), YEARS.index(y), :ncyc] = out
            done += 1
            if done % 20 == 0: print(f"  {done}/{len(jobs)}, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUT + "cfs_weekly.npz", years=np.array(YEARS), **res)
    print(f"done {done}/{len(jobs)} in {time.time() - t0:.0f} s; failures {fails}", flush=True)


if __name__ == "__main__":
    main()
