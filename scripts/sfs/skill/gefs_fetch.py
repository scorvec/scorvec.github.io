"""GEFSv12 reforecast (Wednesday 35-day starts nearest Sep 1 / Oct 1, 2001-2019, members c00 p01 p02 p03) -> weekly
means on the ERA5 NH 1.5 deg grid. Byte-ranged per message via the repo's gefs_reforecast helpers (get/index/decode/
coarsen). t2m: the two samples ending each day (12Z, 00Z next) as gefs_reforecast; z500: 12Z only (smooth field)."""
import sys, time, datetime as dt, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "gefs"))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import gefs_reforecast as GR                                   # noqa: E402
from common import YEARS, MONTHS, OUT, weekly                   # noqa: E402

S3 = GR.S3
MEMBERS = ("c00", "p01", "p02", "p03")
SPEC = {"t2m": ("tmp_2m", "tmp_2m", "2 m above ground", ("12", "24")),
        "z500": ("hgt_pres_abv700mb", "hgt_pres", "500 mb", ("12",))}


def nearest_wed(y, m):
    d = dt.date(y, m, 1)
    c = [d + dt.timedelta(days=k) for k in range(-3, 4)]
    return min((x for x in c if x.weekday() == 2), key=lambda x: abs((x - d).days))


def member(start, mem):
    base = f"{S3}/{start[:4]}/{start}00/{mem}"
    out = {}
    for tag, (f1, f2, lev, hrs) in SPEC.items():
        jobs, cache = [], {}
        for d in range(1, 36):
            for hh in hrs:
                h = 24 * d - 12 if hh == "12" else 24 * d
                f = f"{base}/Days:{'1-10' if h <= 240 else '10-35'}/{f1 if h <= 240 else f2}_{start}00_{mem}.grib2"
                if f not in cache:
                    cache[f] = GR.index(f)
                hit = [e for e in cache[f] if e[2] == lev and e[3] == f"{h} hour fcst"]
                if not hit:
                    raise RuntimeError(f"missing {tag} {h} h in {f}")
                jobs.append((d, f, hit[0][0], hit[0][1]))
        with ThreadPoolExecutor(16) as ex:
            bufs = list(ex.map(lambda j: GR.get(j[1], (j[2], j[3])), jobs))
        daily = np.zeros((35, len(GR.LAT), len(GR.LON))); cnt = np.zeros(35)
        for (d, *_), b in zip(jobs, bufs):
            daily[d - 1] += GR.coarsen(GR.decode(b)); cnt[d - 1] += 1
        daily /= cnt[:, None, None]
        nh = daily[:, GR.LAT >= -1e-6][:, ::-1]                         # 0..90 ascending, like ERA5
        out[tag] = weekly(nh).astype(np.float32)
    return out


def main():
    res = {t: np.full((2, len(YEARS), len(MEMBERS), 4, 61, 240), np.nan, np.float32) for t in SPEC}
    starts = {}
    jobs = [(m, y, k) for m in MONTHS for y in YEARS for k in range(len(MEMBERS))]
    t0 = time.time()

    def run(j):
        m, y, k = j
        s = nearest_wed(y, m).strftime("%Y%m%d")
        for attempt in range(3):
            try:
                return j, s, member(s, MEMBERS[k]), None
            except Exception as e:                                       # noqa: BLE001
                err = str(e)[:150]; time.sleep(5)
        return j, s, None, err

    with ThreadPoolExecutor(4) as ex:
        for n, (j, s, r, err) in enumerate(ex.map(run, jobs), 1):
            m, y, k = j; starts[(m, y)] = s
            if r is None:
                print(f"  FAIL {s} {MEMBERS[k]}: {err}", flush=True); continue
            for t in SPEC:
                res[t][MONTHS.index(m), YEARS.index(y), k] = r[t]
            if n % 10 == 0: print(f"  {n}/{len(jobs)} member-starts, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUT + "gefs_weekly.npz", **res,
                        starts=np.array([[starts.get((m, y), "") for y in YEARS] for m in MONTHS]))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
