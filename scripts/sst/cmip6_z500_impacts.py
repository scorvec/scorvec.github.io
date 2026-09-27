#!/usr/bin/env python3
"""ENSO and 500 hPa height over the Northern Hemisphere (2026-09-27; user: "can you also show 500mb height anomalies
(actually just a show a flat northern hemisphere view)"). LAPTOP analysis: the CMIP6 members with z500 extracts
(cmip6_z500_extract.py; <= 10 per model, 147 members of the 16 study models) and ERA5, with exactly the tas/pr study's
definitions (cmip6_enso_impacts.py) and the E/C and ENSO-free PDO indices of cmip6_enso_modes.py:

  regression   per K of the same season's Nino-3.4 (member's own quadratic trend removed from index and field)
  classes      El Nino / strong / super / La Nina / strong La Nina minus neutral; members event-weighted within a model,
               models equal; asymmetry; super El Nino lifecycle (JJA before -> MAM after) and beyond-linear
  E/C          partial regression on Takahashi's E and C (per sd), east- minus central-based at matched Nino-3.4
  PDO          partial regression on [Nino-3.4, ENSO-free PDO] (per sd), raw PDO; El Nino / La Nina by ENSO-free PDO phase,
               the interaction
Tests: CMIP6 across-model t-test, BH-FDR 10 % over the map (0-90N) AND >= 80 % of models agreeing on the sign. ERA5
(wb2_1p5_daily z500, metres despite the m2/s2 attribute; 1959-2026) on CPC's ONI: OLS / Welch t-tests per grid point, FDR
10 %; observed classes with < 8 seasons are not tested.
Wave-train position per season: pattern r (20-90N, cos-weighted) of the observed and CMIP6 regression maps, and the
longitudes of the Aleutian low (min, 30-65N 150E-130W) and Canadian ridge (max, 40-70N 130W-60W) centres - area-weighted
centroid of the cells within 20 % of the extreme; per model and observed; and across models, the Aleutian-low longitude
of the super El Nino composite against the model's super El Nino rain centroid (cmip6_nino_rain_position.py).

    python scripts/sst/cmip6_z500_impacts.py          -> reference/enso_z500_site.{npz,json} + a printed summary
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmip6_enso_impacts as E                                               # noqa: E402
import cmip6_enso_modes as MO                                                # noqa: E402

ZD = E.SRC / "z500"
ERA5Z = Path.home() / "era5_store" / "wb2_1p5_daily" / "z500"
REF = HERE / "reference"
LAT = np.arange(0.0, 90.01, 2.5)
LON = np.arange(0.0, 357.51, 2.5)
AL_BOX = ((30, 65), (150, 230))
CR_BOX = ((40, 70), (230, 300))
SEAS = E.SEASONS


def seasonal_fields(z, months):
    """(T, lat, lon) monthly -> {season: {year: field}} for complete seasons (DJF by its Jan year)."""
    acc = defaultdict(list)
    for i, m in enumerate(months):
        y, mm = int(m[:4]), int(m[5:7])
        for s, ms in E.SMON.items():
            if mm in ms:
                acc[(s, y + 1 if (s == "DJF" and mm == 12) else y)].append(i)
    out = {s: {} for s in SEAS}
    for (s, y), ii in acc.items():
        if len(ii) == 3:
            out[s][y] = z[ii].mean(0)
    return out


def centre_lon(field, box, sign):
    """Area-weighted longitude centroid of the cells within 20 % of the box extreme (sign -1: a low, +1: a high)."""
    (a, b), (c, d) = box
    la = (LAT >= a) & (LAT <= b); lo = (LON >= c) & (LON <= d)
    f = sign * field[np.ix_(la, lo)]
    if not np.isfinite(f).any() or np.nanmax(f) <= 0:
        return None
    w = np.cos(np.deg2rad(LAT[la]))[:, None] * (f >= 0.8 * np.nanmax(f))
    return float(np.sum(w * LON[lo][None, :]) / np.sum(w))


def pattern_r(a, b, lat0=20.0):
    m = (LAT >= lat0)[:, None] & np.isfinite(a) & np.isfinite(b)
    w = np.broadcast_to(np.cos(np.deg2rad(LAT))[:, None], a.shape)[m]
    x, y = a[m] - np.average(a[m], weights=w), b[m] - np.average(b[m], weights=w)
    return float(np.sum(w * x * y) / np.sqrt(np.sum(w * x * x) * np.sum(w * y * y)))


class Acc:
    def __init__(self):
        self.d = defaultdict(dict)

    def add(self, key, model, C, w):
        e = self.d[key].setdefault(model, [0.0, 0.0])
        e[0] = e[0] + w * C; e[1] += w

    def stack(self, key):
        names = sorted(self.d.get(key, {}))
        return names, (np.stack([self.d[key][m][0] / self.d[key][m][1] for m in names]) if names else None)


def process(name, ref, acc, Sm, n_ev):
    ds = xr.open_dataset(E.SRC / f"{name}.nc")
    model = ds.attrs["source_id"]
    idx34, _ = E.season_index(ds); ds.close()
    z = np.load(ZD / f"{name}.npz")
    Z = z["z"].astype("float64") + 5000.0
    SF = seasonal_fields(Z, [str(m) for m in z["months"]])
    mi = MO.member_indices(MO.PAC / f"{name}.npz", ref)
    sea = {k: MO.seasonal(mi[k], mi["months"]) for k in ("E", "C", "pdo", "pdo_free")}
    per = {}
    for s in SEAS:
        yrs = np.array(sorted(y for y in SF[s] if y in idx34[s] and 1950 <= y <= 2014))
        x = E.detrend(np.array([idx34[s][y] for y in yrs]), yrs.astype(float)).astype(float)
        A = E.detrend(np.stack([SF[s][y] for y in yrs]), yrs.astype(float)).astype("float64")
        per[s] = (yrs, x, A)
        acc.add(("clim", s), model, np.mean([SF[s][y] for y in yrs], axis=0), 1.0)
        # regression per K
        acc.add(("reg", s), model, MO.ols(x[:, None], A)[0][0], 1.0)
        neu = np.abs(x) < E.NEUTRAL
        cn = A[neu].mean(0)
        comps = {}
        for c, _, _, cond in E.CLASSES:
            sel = cond(x)
            if sel.any() and neu.sum() >= E.MIN_NEU:
                C = A[sel].mean(0) - cn
                comps[c] = (C, int(sel.sum()), float(x[sel].mean()))
                acc.add(("x", s, c), model, C, sel.sum())
                n_ev[(s, c)][model] += int(sel.sum())
        for key, _, pair in E.DERIVED:
            if pair is None:
                if "super" in comps:
                    xc = x - x.mean(); slope = np.tensordot(xc, A - A.mean(0), axes=(0, 0)) / (xc @ xc)
                    C, n, xb = comps["super"]; acc.add(("x", s, key), model, C - slope * xb, n)
            elif pair[0] in comps and pair[1] in comps:
                (Cp, npos, xp), (Cn, nneg, xn) = comps[pair[0]], comps[pair[1]]
                m = 0.5 * (xp + abs(xn)); acc.add(("x", s, key), model, Cp * m / xp + Cn * m / abs(xn), min(npos, nneg))
        # E/C, PDO: years with the Pacific indices
        ok = np.array([all(y in sea[k][s] for k in sea) for y in yrs])
        yy, xx, AA = yrs[ok], x[ok], A[ok]
        I = {k: np.array([sea[k][s][y] for y in yy]) for k in sea}
        B = MO.ols(np.column_stack([I["E"], I["C"]]), AA)[0]
        acc.add(("E", s), model, B[0], 1.0); acc.add(("C", s), model, B[1], 1.0)
        acc.add(("free", s), model, MO.ols(np.column_stack([xx, I["pdo_free"]]), AA)[0][1], 1.0)
        acc.add(("raw", s), model, MO.ols(I["pdo"][:, None], AA)[0][0], 1.0)
        neu2 = np.abs(xx) < E.NEUTRAL
        if neu2.sum() < E.MIN_NEU:
            continue
        An = AA - AA[neu2].mean(0)
        en, ln = xx >= 0.5, xx <= -0.5
        ep, cp = en & (I["E"] > I["C"]), en & (I["E"] <= I["C"])
        pp, pm = I["pdo_free"] >= MO.PHASE, I["pdo_free"] <= -MO.PHASE
        b = MO.bin_of(xx)
        for k in range(4):
            bk = b == k
            for nm, m in (("ep", ep), ("cp", cp), ("en+", en & pp), ("en-", en & pm), ("ln+", ln & pp), ("ln-", ln & pm)):
                Sm.add(model, ("z", s, f"{nm}{k}"), An[m & bk], xx[m & bk])
        Sm.add(model, ("z", s, "neu+0"), An[neu2 & pp], xx[neu2 & pp]); Sm.add(model, ("z", s, "neu-0"), An[neu2 & pm], xx[neu2 & pm])
    # super El Nino lifecycle, keyed on DJF
    yD, xD, _ = per["DJF"]
    for ph, s, off, _ in E.LIFE:
        ys, _, As = per[s]; row = {y: i for i, y in enumerate(ys)}
        keep = np.array([(y + off) in row for y in yD])
        if not keep.any():
            continue
        xk = xD[keep]; Ak = As[[row[y + off] for y in yD[keep]]]
        neu, sup = np.abs(xk) < E.NEUTRAL, xk >= 2.0
        if neu.sum() < E.MIN_NEU or not sup.any():
            continue
        C = Ak[sup].mean(0) - Ak[neu].mean(0)
        xc = xk - xk.mean(); slope = np.tensordot(xc, Ak - Ak.mean(0), axes=(0, 0)) / (xc @ xc)
        acc.add(("life", ph, "super"), model, C, sup.sum()); acc.add(("life", ph, "nonlin"), model, C - slope * xk[sup].mean(), sup.sum())
    return model


def era5_monthly():
    """{(y, m): z500 on the 2.5-degree NH grid}, complete months; metres (the attribute says m2/s2 and is wrong)."""
    out = {}
    for f in sorted(ERA5Z.glob("z500_*.nc")):
        d = xr.open_dataset(f); v = d[list(d.data_vars)[0]].transpose("time", "latitude", "longitude").sortby("latitude")
        v = xr.concat([v, v.isel(longitude=0).assign_coords(longitude=v.longitude[0] + 360.0)], "longitude").load()
        for m in range(1, 13):
            vm = v.sel(time=v.time.dt.month == m)
            if vm.sizes["time"] < 28 or not np.isfinite(vm.values).all():
                continue
            a = vm.mean("time").interp(latitude=LAT, longitude=LON).values.astype("float64")
            if not (4000 < np.nanmean(a) < 6500):
                raise ValueError(f"{f.name} month {m}: mean {np.nanmean(a):.0f} is not metres")
            out[(int(vm.time.dt.year.values[0]), m)] = a
        d.close()
    return out


def main() -> int:
    t0 = time.time()
    ref = MO.obs_indices()
    names = sorted(Path(f).stem for f in glob.glob(str(E.SRC / "*.nc")) if ".part" not in f)
    names = [n for n in names if (ZD / f"{n}.npz").exists() and (MO.PAC / f"{n}.npz").exists()]
    acc = Acc(); Sm = MO.Sums(); n_ev = defaultdict(lambda: defaultdict(int)); members = defaultdict(int)
    for i, n in enumerate(names):
        members[process(n, ref, acc, Sm, n_ev)] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(names)} members, {time.time() - t0:.0f} s", flush=True)
    print(f"{len(names)} members, {len(members)} models: {dict(members)}", flush=True)
    npz = {"lat": LAT.astype("float32"), "lon": LON.astype("float32")}
    meta = {"models": len(members), "members": int(sum(members.values())), "per_model": dict(members), "maps": {}}

    def put(key, mm, sg, info):
        npz[key + "|f"] = np.asarray(mm, "float32"); npz[key + "|s"] = np.asarray(sg, bool)
        info = dict(info, sig_frac=round(float(sg[LAT >= 20].mean()), 3)); meta["maps"][key] = info

    per_model = {}
    for key in list(acc.d):
        if key[0] == "clim":
            continue
        names_, S = acc.stack(key)
        if len(names_) < E.MIN_MODELS:
            meta["maps"]["|".join(key)] = dict(tested=False, models=len(names_)); continue
        mm, sg, _, _ = E.robust(S)
        per_model[key] = (names_, S)
        info = dict(models=len(names_), tested=True)
        if key[0] == "x":
            info["events"] = int(sum(n_ev[(key[1], key[2])].values())) if (key[1], key[2]) in n_ev else None
        if key[0] == "life":
            info["events"] = int(sum(acc.d[key][m][1] for m in names_))
        put("cmip6|" + "|".join(key), mm, sg, info)
    for s in SEAS:
        for nm, a, b, bins in (("epcp", "ep{k}", "cp{k}", range(4)), ("en_pdo", "en+{k}", "en-{k}", range(4)),
                               ("ln_pdo", "ln+{k}", "ln-{k}", range(4)), ("neu_pdo", "neu+0", "neu-0", (0,))):
            rows = {m: MO.matched(Sm, m, "z", s, a, b, bins) for m in members}
            rows = {m: r for m, r in rows.items() if r is not None}
            S = np.stack([rows[m]["diff"] for m in sorted(rows)])
            mm, sg, _, _ = E.robust(S)
            per_model[("cmp", s, nm)] = (sorted(rows), S)
            if nm != "neu_pdo":
                put(f"cmip6|cmp|{s}|{nm}", mm, sg, dict(models=len(rows), tested=True,
                                                        n_a=int(sum(r["na"] for r in rows.values())), n_b=int(sum(r["nb"] for r in rows.values())),
                                                        x_a=round(float(np.mean([r["xa"] for r in rows.values()])), 3),
                                                        x_b=round(float(np.mean([r["xb"] for r in rows.values()])), 3)))
        en, ne = per_model[("cmp", s, "en_pdo")], per_model[("cmp", s, "neu_pdo")]
        common = [m for m in en[0] if m in ne[0]]
        S = np.stack([en[1][en[0].index(m)] - ne[1][ne[0].index(m)] for m in common])
        mm, sg, _, _ = E.robust(S)
        put(f"cmip6|cmp|{s}|int", mm, sg, dict(models=len(common), tested=True))
        names_, C = acc.stack(("clim", s)); npz[f"clim_cmip6|{s}"] = C.mean(0).astype("float32")
    print(f"model maps {time.time() - t0:.0f} s; ERA5 ...", flush=True)
    # observed
    em = era5_monthly()
    oni = E.oni_table()
    OS = E.obs_seasons(em, oni)
    osea = {k: MO.seasonal(np.asarray(ref[k], float), ref["months"]) for k in ("E", "C", "pdo", "pdo_free")}
    obs = {}
    for s in SEAS:
        yrs, x, A, clim = OS[s]
        npz[f"clim_obs|{s}"] = clim.astype("float32")
        B, t, dof = MO.ols(x[:, None], A)
        p = 2 * stats.t.sf(np.abs(t[0]), dof)
        put(f"obs|reg|{s}", B[0], E.fdr(p), dict(n=len(yrs), span=[int(yrs[0]), int(yrs[-1])]))
        obs[("reg", s)] = B[0]
        neu = np.abs(x) < E.NEUTRAL
        for c, _, _, cond in E.CLASSES:
            sel = cond(x)
            if sel.sum() >= E.OBS_TEST_N:
                _, p = stats.ttest_ind(A[sel], A[neu], axis=0, equal_var=False)
                put(f"obs|x|{s}|{c}", A[sel].mean(0) - A[neu].mean(0), E.fdr(p), dict(n=int(sel.sum()), span=[int(yrs[0]), int(yrs[-1])]))
            else:
                meta["maps"][f"obs|x|{s}|{c}"] = dict(tested=False, n=int(sel.sum()), years=[int(y) for y in yrs[sel]])
            if c == "super" and sel.any():
                obs[("super", s)] = A[sel].mean(0) - A[neu].mean(0)
        ok = np.array([all(y in osea[k][s] for k in osea) for y in yrs])
        yy, xx, AA = yrs[ok], x[ok], A[ok]
        I = {k: np.array([osea[k][s][y] for y in yy]) for k in osea}
        for key, X, j in (("E", np.column_stack([I["E"], I["C"]]), 0), ("C", np.column_stack([I["E"], I["C"]]), 1),
                          ("free", np.column_stack([xx, I["pdo_free"]]), 1), ("raw", I["pdo"][:, None], 0)):
            B, t, dof = MO.ols(X, AA); p = 2 * stats.t.sf(np.abs(t[j]), dof)
            put(f"obs|{key}|{s}", B[j], E.fdr(p), dict(n=len(yy), span=[int(yy[0]), int(yy[-1])]))
    # wave-train position: observed vs CMIP6, per season (regression per K), and super El Nino DJF
    pos = {}
    for s in SEAS:
        names_, S = per_model[("reg", s)]
        mm = S.mean(0); o = obs[("reg", s)]
        e = dict(pattern_r=round(pattern_r(o, mm), 3),
                 model_pattern_r=[round(float(q), 3) for q in np.percentile([pattern_r(o, S[i]) for i in range(len(S))], [10, 50, 90])])
        for nm, box, sg in (("aleutian_low", AL_BOX, -1), ("canadian_ridge", CR_BOX, +1)):
            ml = [centre_lon(S[i], box, sg) for i in range(len(S))]; ml = [v for v in ml if v is not None]
            ol = centre_lon(o, box, sg)
            e[nm] = dict(obs=None if ol is None else round(ol, 1), cmip6_mean_map=None if centre_lon(mm, box, sg) is None else round(centre_lon(mm, box, sg), 1),
                         models_p10_50_90=[round(float(q), 1) for q in np.percentile(ml, [10, 50, 90])] if ml else None,
                         obs_minus_models_median=None if (ol is None or not ml) else round(ol - float(np.median(ml)), 1),
                         obs_pctile=None if (ol is None or not ml) else round(float(100 * np.mean(np.array(ml) < ol)), 1),
                         obs_value=None if ol is None else round(float(o[np.argmin(np.abs(LAT - 50))][0]), 1))
            # value of the extreme (gpm per K) for both
            (a, b), (c, d) = box
            la = (LAT >= a) & (LAT <= b); lo = (LON >= c) & (LON <= d)
            e[nm]["obs_extreme"] = round(float((sg * o[np.ix_(la, lo)]).max() * sg), 1)
            e[nm]["cmip6_extreme"] = round(float((sg * mm[np.ix_(la, lo)]).max() * sg), 1)
            e[nm].pop("obs_value")
        pos[s] = e
    # super El Nino (DJF) Aleutian low vs the model's rain centroid
    rp = json.loads((E.OUT / "rain_position.json").read_text())
    names_, S = per_model[("x", "DJF", "super")]
    al = {m: centre_lon(S[i], AL_BOX, -1) for i, m in enumerate(names_)}
    cr = {m: centre_lon(S[i], CR_BOX, +1) for i, m in enumerate(names_)}
    both = [m for m in names_ if m in rp["models"] and al[m] is not None]
    rain = np.array([rp["models"][m]["centroid_mean"] for m in both]); alv = np.array([al[m] for m in both])
    crv = np.array([cr[m] for m in both if cr[m] is not None]); rain_cr = np.array([rp["models"][m]["centroid_mean"] for m in both if cr[m] is not None])
    r1, p1 = stats.pearsonr(rain, alv); r2, p2 = stats.pearsonr(rain_cr, crv)
    osup = obs.get(("super", "DJF"))
    pos["super_DJF"] = dict(
        models=len(both), rain_centroid_models=[round(float(v), 1) for v in rain], aleutian_low_models=[round(float(v), 1) for v in alv],
        r_rain_vs_aleutian_low=round(float(r1), 3), p_rain_vs_aleutian_low=round(float(p1), 4),
        slope_deg_per_deg=round(float(np.polyfit(rain, alv, 1)[0]), 2),
        r_rain_vs_canadian_ridge=round(float(r2), 3), p_rain_vs_canadian_ridge=round(float(p2), 4),
        obs_rain_centroid=rp["observed"]["composite_centroid"],
        obs_aleutian_low=None if osup is None else round(centre_lon(osup, AL_BOX, -1), 1),
        obs_canadian_ridge=None if osup is None else round(centre_lon(osup, CR_BOX, +1), 1),
        cmip6_aleutian_low_mean_map=round(centre_lon(S.mean(0), AL_BOX, -1), 1),
        cmip6_canadian_ridge_mean_map=round(centre_lon(S.mean(0), CR_BOX, +1), 1),
        pattern_r_obs_vs_cmip6=None if osup is None else round(pattern_r(osup, S.mean(0)), 3))
    meta["position"] = pos
    np.savez_compressed(REF / "enso_z500_site.npz", **npz)
    (REF / "enso_z500_site.json").write_text(json.dumps(meta, separators=(",", ":"), default=float))
    print(json.dumps(pos, indent=1))
    print(f"wrote enso_z500_site.npz ({(REF / 'enso_z500_site.npz').stat().st_size / 1e6:.1f} MB, {len(meta['maps'])} maps) "
          f"in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
