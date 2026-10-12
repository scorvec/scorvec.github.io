"""SFS beta2 MONTHLY reforecast (Sep 1 / Oct 1 starts 1991-2025, 11 members, leads 0-11; lead 0 = the start month),
chunk by chunk: Nino-3.4 per member and lead from ocn_monthly SST; 3-month t2m and precipitation means on a global
1.5 deg grid (3x3 box mean of the 0.5 deg field around each 1.5 deg point) from atm_monthly tmp2m / pratesfc.
Seasons: Sep start OND (leads 1-3), NDJ (2-4); Oct start NDJ (1-3), DJF (2-4)."""
import sys, os, time, urllib.request, numpy as np, numcodecs
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import OUT

B = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta2/reforecast/{m:02d}/{s}.zarr/{v}/{i}.0.{l}.0.0"
YRS = list(range(1991, 2026)); MON = (9, 10)
BLOSC = numcodecs.Blosc()
GLAT = np.arange(-90.0, 90.0 + 1e-6, 1.5); GLON = np.arange(0.0, 360.0 - 1e-6, 1.5)


def get(url, shape):
    for k in range(5):
        try:
            raw = urllib.request.urlopen(url, timeout=300).read(); break
        except Exception:                                              # noqa: BLE001
            if k == 4: raise
            time.sleep(5 * (k + 1))
    return np.frombuffer(BLOSC.decode(raw), "<f4").reshape(shape)


def n34(job):
    m, i, l = job
    a = get(B.format(m=m, s="ocn_monthly", v="SST", i=i, l=l), (11, 181, 360)).astype(np.float64)
    a = np.where(np.abs(a) > 1e10, np.nan, a)
    box = a[:, 85:96, 190:241]                                          # lat -5..5, lon 190..240 (1 deg, from -90 / 0)
    w = np.cos(np.deg2rad(np.arange(-5, 6)))[:, None] * np.ones((1, 51))
    w = w[None] * np.isfinite(box)
    return job, np.nansum(np.nan_to_num(box) * w, (1, 2)) / w.sum((1, 2))


def coarsen(a):
    """(..., 361, 720) at 0.5 deg from -90 / 0 -> (..., 121, 240): 3x3 mean around every 3rd point."""
    a = np.concatenate([a[..., -1:], a, a[..., :1]], -1)                # wrap lon
    a = np.concatenate([a[..., :1, :], a, a[..., -1:, :]], -2)          # edge-pad lat
    s = sum(a[..., 1 + di:1 + di + 361:3, 1 + dj:1 + dj + 720:3] for di in (-1, 0, 1) for dj in (-1, 0, 1))
    return s / 9.0


def atm(job):
    v, m, i = job
    name, fac = {"t2m": ("tmp2m", 1.0), "pr": ("pratesfc", 86400.0)}[v]
    L = []
    for l in (1, 2, 3, 4):
        a = get(B.format(m=m, s="atm_monthly", v=name, i=i, l=l), (11, 361, 720)).astype(np.float32)
        assert np.abs(a).max() < 1e10
        L.append(coarsen(a) * fac)
    L = np.stack(L, 1)                                                  # (11, 4 leads, 121, 240)
    return job, np.stack([L[:, 0:3].mean(1), L[:, 1:4].mean(1)], 1)     # (11, 2 seasons, 121, 240)


def main():
    t0 = time.time()
    N = np.full((2, len(YRS), 11, 12), np.nan)
    jobs = [(m, i, l) for m in MON for i in range(len(YRS)) for l in range(12)]
    with ThreadPoolExecutor(12) as ex:
        for (m, i, l), v in ex.map(n34, jobs):
            N[MON.index(m), i, :, l] = v
    print(f"n34 done {time.time() - t0:.0f} s; mean {np.nanmean(N):.2f} C", flush=True)
    np.save(OUT + "sfs_n34.npy", N)
    for var in ("t2m", "pr"):
        R = np.full((2, len(YRS), 11, 2, 121, 240), np.nan, np.float32)
        jobs = [(var, m, i) for m in MON for i in range(len(YRS))]
        with ThreadPoolExecutor(6) as ex:
            for k, ((v, m, i), s) in enumerate(ex.map(atm, jobs)):
                R[MON.index(m), i] = s
                if (k + 1) % 10 == 0: print(f"  {var} {k + 1}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
        np.save(OUT + f"sfs_seas_{var}.npy", R)
        print(f"{var} done; mean {np.nanmean(R):.2f}", flush=True)


if __name__ == "__main__":
    main()
