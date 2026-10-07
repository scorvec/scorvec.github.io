"""Scoring for verify_fog.py: contingency scores with station-block bootstrap CIs, parameter sweeps
(fit 2010-2019, evaluate 2020-2025), hybrids with the fog-model probability, ceiling categories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_fog import BARS, DEPTHS, FIT_LAST, NEARS, SATS, SHIP

B = 2000
RNG = np.random.default_rng(20261007)
VIS_FOG = 0.63           # statute miles: <= 5/8 SM == <= 1 km
THETAS = [round(x, 2) for x in np.arange(0.05, 0.91, 0.05)]


# ------------------------------------------------------------ helpers ---
def counts(pred, obs):
    pred, obs = np.asarray(pred, bool), np.asarray(obs, bool)
    return np.array([(pred & obs).sum(), (pred & ~obs).sum(), (~pred & obs).sum(), (~pred & ~obs).sum()], float)


def metrics(c):
    a, b, cc, d = (c[..., k] for k in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        pod = a / (a + cc)
        far = b / (a + b)
        csi = a / (a + b + cc)
        bias = (a + b) / (a + cc)
        pss = pod - b / (b + d)
    return {"POD": pod, "FAR": far, "CSI": csi, "bias": bias, "PSS": pss}


class Boot:
    """Station-block bootstrap: resample stations with replacement; the same draws for every forecast,
    so differences between forecasts are paired."""

    def __init__(self, groups):
        self.keys, self.idx = np.unique(np.asarray(groups), return_inverse=True)
        S = len(self.keys)
        draws = RNG.integers(0, S, size=(B, S))
        self.W = np.zeros((B, S))
        for k in range(B):
            np.add.at(self.W[k], draws[k], 1)

    def per_group(self, pred, obs):
        pred, obs = np.asarray(pred, bool), np.asarray(obs, bool)
        S = len(self.keys)
        out = np.zeros((S, 4))
        for j, m in enumerate([pred & obs, pred & ~obs, ~pred & obs, ~pred & ~obs]):
            out[:, j] = np.bincount(self.idx, weights=m.astype(float), minlength=S)
        return out

    def score(self, pred, obs):
        g = self.per_group(pred, obs)
        point = metrics(g.sum(0))
        boot = metrics(self.W @ g)
        return {k: (float(point[k]), *map(float, np.nanpercentile(boot[k], [2.5, 97.5]))) for k in point}, g

    def diff(self, pred1, pred2, obs, key="CSI"):
        g1, g2 = self.per_group(pred1, obs), self.per_group(pred2, obs)
        p = metrics(g1.sum(0))[key] - metrics(g2.sum(0))[key]
        bd = metrics(self.W @ g1)[key] - metrics(self.W @ g2)[key]
        lo, hi = np.nanpercentile(bd, [2.5, 97.5])
        return float(p), float(lo), float(hi)


def fmt(t, d=3):
    p, lo, hi = t
    if not np.isfinite(p):
        return "-"
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def ft_of(zb):
    return np.where(np.isnan(zb), np.nan, np.round(zb * 3.28084 / 100) * 100)


def cat_ft(ft):
    """0 LIFR (<500 ft or obscured), 1 IFR (<1000), 2 MVFR (<=3000), 3 VFR or no ceiling."""
    ft = np.asarray(ft, float)
    c = np.full(ft.shape, 3)
    c[ft <= 3000] = 2
    c[ft < 1000] = 1
    c[ft < 500] = 0
    return c


CATN = ["LIFR", "IFR", "MVFR", "VFR/none"]


def contingency_md(est, obs):
    m = np.zeros((4, 4), int)
    np.add.at(m, (est, obs), 1)
    lines = ["| est \\ obs | " + " | ".join(CATN) + " | total |", "|---" * 6 + "|"]
    for i in range(4):
        lines.append(f"| {CATN[i]} | " + " | ".join(str(v) for v in m[i]) + f" | {m[i].sum()} |")
    lines.append("| total | " + " | ".join(str(v) for v in m.sum(0)) + f" | {m.sum()} |")
    return "\n".join(lines)


# -------------------------------------------------------------- score ---
def score(out: Path) -> None:
    df = pd.read_parquet(out / "pairs.parquet")
    st = {s["wmo"]: s for s in json.loads((out / "stations.json").read_text())}
    qc = json.loads((out / "qc.json").read_text())
    R, J = [], {}
    df["hires"] = (df["dz1"] <= 50) & (df["n500"] >= 5)
    df["train"] = df["year"] <= FIT_LAST
    fog_ok = df["vis"].notna() | df["fg"]
    df["fog_obs"] = (df["vis"] <= VIS_FOG).fillna(False) | df["fg"]
    df["lowvis_nofg"] = (df["vis"] <= VIS_FOG).fillna(False) & ~df["fg"]
    F = df[fog_ok].reset_index(drop=True)
    R.append("# Skew-T fog / ceiling verification vs METAR\n")
    R.append(f"IGRA v2 00/12Z {int(df.year.min())}-{int(df.year.max())}, IEM routine METAR in [nominal-90, nominal] "
             f"min closest to nominal-45. Fog truth: vis <= 1 km (<= 5/8 SM) or FG/FZFG. Station-block bootstrap "
             f"(B={B}) 95% CIs in brackets. High-res = first level above the surface <= 50 m and >= 5 levels in the "
             f"lowest 500 m.\n")

    # ---- counts by station
    R.append("## Data by station\n")
    R.append("| WMO | ICAO | km | records | no sfc flag | no sfc T/Td | no METAR | pairs | fog-scored | fog obs | "
             "FG | vis<=1km no FG | BR | high-res % | years |")
    R.append("|---" * 15 + "|")
    for w, g in df.groupby("wmo"):
        s, q = st[w], qc[w]
        fg_ = F[F.wmo == w]
        R.append(f"| {w} | {s['icao']} | {s['km']} | {q['records']} | {q['no_sfc_flag']} | {q['missing_sfc_TTd']} | "
                 f"{q['no_metar']} | {len(g)} | {len(fg_)} | {int(fg_.fog_obs.sum())} | {int(fg_.fg.sum())} | "
                 f"{int(fg_.lowvis_nofg.sum())} | {int(fg_.br.sum())} | {100 * g.hires.mean():.0f} | "
                 f"{g.year.min()}-{g.year.max()} |")
    R.append(f"\nTotal pairs {len(df)}; fog-scored {len(F)}; fog events {int(F.fog_obs.sum())} "
             f"({100 * F.fog_obs.mean():.2f}%); FG reported {int(F.fg.sum())}; vis <= 1 km without FG "
             f"{int(F.lowvis_nofg.sum())}; shallow/patchy/vicinity FG only {int((F.fg_shallow & ~F.fog_obs).sum())}; "
             f"BR {int(F.br.sum())}. METAR offset (min) median {F.m_off.median():.0f}.\n")
    wxl = F[F.lowvis_nofg].wx.str.replace(r"[+-]", "", regex=True).value_counts().head(12)
    R.append("vis <= 1 km without FG, present weather: " + ", ".join(f"{k or 'none'} {v}" for k, v in wxl.items()) + "\n")
    R.append("Resolution by data source: " + ", ".join(
        f"{k}: {len(g)} ({100 * g.hires.mean():.0f}% high-res, median levels <=500 m {g.n500.median():.0f})"
        for k, g in df.groupby("psrc")) + "\n")

    # ---- fog scores
    subsets = {"all": np.ones(len(F), bool), "high-res": F.hires.values, "coarse": ~F.hires.values}
    obs = F.fog_obs.values
    shipped = F[f"fog_{SHIP[0]}_{SHIP[1]}"].values
    pf = F.pfog.values
    fc = {"old rule": F.old_obsc.values, "new rule (97/30)": shipped, "fog model p>=0.5": pf >= 0.5}

    def table(title, preds, mask_sets, obsv=obs):
        R.append(f"### {title}\n")
        R.append("| subset | forecast | n | events | POD | FAR | CSI | bias | PSS |")
        R.append("|---" * 9 + "|")
        for sn, msk in mask_sets.items():
            if msk.sum() == 0:
                continue
            bt = Boot(F.wmo.values[msk])
            for fn, p in preds.items():
                sc, _ = bt.score(p[msk], obsv[msk])
                R.append(f"| {sn} | {fn} | {msk.sum()} | {int(obsv[msk].sum())} | " +
                         " | ".join(fmt(sc[k]) for k in ("POD", "FAR", "CSI", "bias", "PSS")) + " |")
                J.setdefault(title, {}).setdefault(sn, {})[fn] = sc
        R.append("")

    R.append("## Fog / obscured\n")
    table("Fog, all years", fc, subsets)
    R.append("Secondary truth (FG/FZFG reported only):\n")
    table("Fog (FG-only truth), all years", fc, {"all": subsets["all"]}, obsv=F.fg.values)

    # ---- sweep: fit on train, evaluate on test
    tr, te = F.train.values, ~F.train.values
    sweep = []
    for sat in SATS:
        for dep in DEPTHS:
            p = F[f"fog_{sat}_{dep}"].values
            for sn, msk in subsets.items():
                m_tr = metrics(counts(p[msk & tr], obs[msk & tr]))
                m_te = metrics(counts(p[msk & te], obs[msk & te]))
                sweep.append({"sat": sat, "depth": dep, "subset": sn, "train_csi": float(m_tr["CSI"]),
                              "train_bias": float(m_tr["bias"]), "test_csi": float(m_te["CSI"]),
                              "test_bias": float(m_te["bias"])})
    SW = pd.DataFrame(sweep)
    J["sweep"] = sweep
    R.append("### Sweep of the new rule (fog): CSI / bias, train 2010-2019 | test 2020-2025\n")
    for sn in subsets:
        s = SW[SW.subset == sn].copy()
        piv = s.assign(cell=s.apply(lambda r: f"{r.train_csi:.3f}/{r.train_bias:.2f} \\| "
                                              f"{r.test_csi:.3f}/{r.test_bias:.2f}", axis=1)) \
            .pivot(index="sat", columns="depth", values="cell")
        R.append(f"**{sn}** (rows SAT %, columns DEPTH m)\n")
        R.append("| SAT | " + " | ".join(str(c) for c in piv.columns) + " |")
        R.append("|---" * (len(piv.columns) + 1) + "|")
        for sat, row in piv.iterrows():
            R.append(f"| {sat} | " + " | ".join(row.values) + " |")
        R.append("")
    best = {}
    for sn in subsets:
        s = SW[SW.subset == sn].sort_values("train_csi", ascending=False).iloc[0]
        best[sn] = (int(s.sat), int(s.depth))
    R.append("Best on train (max CSI): " + ", ".join(f"{k}: SAT {v[0]} / DEPTH {v[1]}" for k, v in best.items()) + "\n")

    # held-out comparison with paired differences
    R.append("### Held-out 2020-2025: shipped vs old vs tuned (train-selected) vs fog model\n")
    preds_te = {"old rule": F.old_obsc.values, "new rule (97/30)": shipped}
    for sn, (sat, dep) in best.items():
        preds_te[f"tuned on {sn} ({sat}/{dep})"] = F[f"fog_{sat}_{dep}"].values
    # hybrids: theta chosen on train (all) by CSI
    hyb = {}
    for name, f in {"model p>=t": lambda t: pf >= t,
                    "new AND p>=t": lambda t: shipped & (pf >= t),
                    "new OR p>=t": lambda t: shipped | (pf >= t),
                    "old AND p>=t": lambda t: F.old_obsc.values & (pf >= t)}.items():
        csis = [(float(metrics(counts(f(t)[tr], obs[tr]))["CSI"]), t) for t in THETAS]
        _, t = max(csis)
        hyb[name] = t
        preds_te[f"{name} (t={t:.2f})"] = f(t)
    J["hybrid_theta"] = hyb
    table("Fog, held-out 2020-2025", preds_te, {k: v & te for k, v in subsets.items()})
    R.append("Paired CSI differences on 2020-2025 (station-block bootstrap):\n")
    bt = Boot(F.wmo.values[te])
    base = shipped[te]
    for fn, p in preds_te.items():
        if fn == "new rule (97/30)":
            continue
        d = bt.diff(p[te], base, obs[te], "CSI")
        db = bt.diff(p[te], base, obs[te], "bias")
        R.append(f"- {fn} minus new rule: dCSI {fmt(d)}; dbias {fmt(db, 2)}")
    R.append("")
    for sn in ("high-res", "coarse"):
        msk = subsets[sn] & te
        bt2 = Boot(F.wmo.values[msk])
        sat, dep = best[sn]
        d = bt2.diff(F[f"fog_{sat}_{dep}"].values[msk], shipped[msk], obs[msk])
        d2 = bt2.diff(F.old_obsc.values[msk], shipped[msk], obs[msk])
        R.append(f"- {sn} test: tuned ({sat}/{dep}) minus new dCSI {fmt(d)}; old minus new dCSI {fmt(d2)}")
    R.append("")

    # ---- bias-constrained selection (train): max CSI among configs with train bias in [0.8, 1.25]
    bc = {}
    for sn in subsets:
        s = SW[(SW.subset == sn) & SW.train_bias.between(0.8, 1.25)].sort_values("train_csi", ascending=False)
        if len(s):
            bc[sn] = (int(s.iloc[0].sat), int(s.iloc[0].depth))
    J["bias_constrained"] = {k: list(v) for k, v in bc.items()}
    R.append("### Bias-constrained selection (train: max CSI with bias 0.8-1.25), held-out 2020-2025\n")
    R.append("Selected: " + ", ".join(f"{k}: SAT {v[0]} / DEPTH {v[1]}" for k, v in bc.items()) +
             ". 'resolution-aware' = the high-res pick on high-res profiles, the coarse pick on coarse ones.\n")
    cand = {"new rule (97/30)": shipped}
    if "all" in bc:
        cand[f"bias-constrained all ({bc['all'][0]}/{bc['all'][1]})"] = F[f"fog_{bc['all'][0]}_{bc['all'][1]}"].values
    if "high-res" in bc and "coarse" in bc:
        h, c_ = bc["high-res"], bc["coarse"]
        cand[f"resolution-aware ({h[0]}/{h[1]} | {c_[0]}/{c_[1]})"] = np.where(
            F.hires.values, F[f"fog_{h[0]}_{h[1]}"].values, F[f"fog_{c_[0]}_{c_[1]}"].values)
    table("Fog, held-out 2020-2025, bias-constrained candidates", cand, {k: v & te for k, v in subsets.items()})
    R.append("Paired differences vs the shipped rule, 2020-2025 (dCSI; dbias; dPSS):\n")
    for sn, msk0 in subsets.items():
        msk = msk0 & te
        btc = Boot(F.wmo.values[msk])
        for fn, p in cand.items():
            if fn == "new rule (97/30)":
                continue
            R.append(f"- {sn}, {fn}: {fmt(btc.diff(p[msk], shipped[msk], obs[msk], 'CSI'))}; "
                     f"{fmt(btc.diff(p[msk], shipped[msk], obs[msk], 'bias'), 2)}; "
                     f"{fmt(btc.diff(p[msk], shipped[msk], obs[msk], 'PSS'))}")
    R.append("")

    # ---- per station (all years), year-block bootstrap within station
    R.append("### Per station, all years (CI = year-block bootstrap within the station)\n")
    R.append("| WMO | ICAO | n | fog obs | old CSI | old bias | new CSI | new bias | model p>=0.5 CSI |")
    R.append("|---" * 9 + "|")
    for w, g in F.groupby("wmo"):
        bt = Boot(g.year.values)
        o = g.fog_obs.values
        so, _ = bt.score(g.old_obsc.values, o)
        sn_, _ = bt.score(g[f"fog_{SHIP[0]}_{SHIP[1]}"].values, o)
        sm, _ = bt.score(g.pfog.values >= 0.5, o)
        R.append(f"| {w} | {st[w]['icao']} | {len(g)} | {int(o.sum())} | {fmt(so['CSI'])} | {fmt(so['bias'], 2)} | "
                 f"{fmt(sn_['CSI'])} | {fmt(sn_['bias'], 2)} | {fmt(sm['CSI'])} |")
    R.append("")

    # ---- ceilings
    C = df[df.sky_ok].reset_index(drop=True)
    oc = cat_ft(C.m_ceil.values)
    cats = {"old rule": cat_ft(ft_of(np.where(C.old_obsc, 0.0, C.old_zb.values))),
            "new rule (97/30/300)": cat_ft(ft_of(C[f"zb_{SHIP[0]}_{SHIP[1]}_{SHIP[2]}"].values))}
    R.append("## Ceilings (flight categories; obscured = LIFR)\n")
    R.append(f"{len(C)} pairs with a usable sky report. Observed: " +
             ", ".join(f"{CATN[k]} {int((oc == k).sum())}" for k in range(4)) + "\n")
    for n, ec in cats.items():
        R.append(f"**{n}**, all years\n")
        R.append(contingency_md(ec, oc) + "\n")

    def ceil_table(title, preds, msets):
        R.append(f"### {title}\n")
        R.append("| subset | forecast | n | ceil<1000 obs | CSI <1000 ft | bias <1000 | CSI <=3000 ft | exact cat | within 1 cat |")
        R.append("|---" * 9 + "|")
        for sn, msk in msets.items():
            bt = Boot(C.wmo.values[msk])
            for fn, ec in preds.items():
                s1, _ = bt.score(ec[msk] <= 1, oc[msk] <= 1)
                s3, _ = bt.score(ec[msk] <= 2, oc[msk] <= 2)
                ex = (ec[msk] == oc[msk]).astype(float)
                w1 = (np.abs(ec[msk] - oc[msk]) <= 1).astype(float)
                gidx = bt.idx
                def rate(v):
                    num = np.bincount(gidx, weights=v, minlength=len(bt.keys))
                    den = np.bincount(gidx, minlength=len(bt.keys)).astype(float)
                    bo = (bt.W @ num) / (bt.W @ den)
                    return (float(v.mean()), *np.percentile(bo, [2.5, 97.5]))
                R.append(f"| {sn} | {fn} | {msk.sum()} | {int((oc[msk] <= 1).sum())} | {fmt(s1['CSI'])} | "
                         f"{fmt(s1['bias'], 2)} | {fmt(s3['CSI'])} | {fmt(rate(ex))} | {fmt(rate(w1))} |")
                J.setdefault(title, {}).setdefault(sn, {})[fn] = {"csi1000": s1["CSI"], "csi3000": s3["CSI"]}
        R.append("")

    csub = {"all": np.ones(len(C), bool), "high-res": C.hires.values, "coarse": ~C.hires.values}
    ceil_table("Ceilings, all years", cats, csub)
    # NEAR sweep at the train-best fog params and at the shipped ones; BAR sweep
    trc, tec = (C.year <= FIT_LAST).values, (C.year > FIT_LAST).values
    R.append("### NEAR-zone sweep (ceiling CSI <1000 ft, train | test)\n")
    R.append("| SAT/DEPTH | " + " | ".join(f"NEAR {n}" for n in NEARS) + " |")
    R.append("|---" * (len(NEARS) + 1) + "|")
    near_best = {}
    for sat, dep in sorted({SHIP[:2], best["all"]}):
        cells = []
        for near in NEARS:
            ec = cat_ft(ft_of(C[f"zb_{sat}_{dep}_{near}"].values))
            a = metrics(counts(ec[trc] <= 1, oc[trc] <= 1))["CSI"]
            b = metrics(counts(ec[tec] <= 1, oc[tec] <= 1))["CSI"]
            cells.append(f"{a:.3f} \\| {b:.3f}")
            near_best.setdefault((sat, dep), []).append((a, near))
        R.append(f"| {sat}/{dep} | " + " | ".join(cells) + " |")
    R.append("")
    sat, dep = best["all"]
    nb = max(near_best[(sat, dep)])[1]
    tuned_ceil = cat_ft(ft_of(C[f"zb_{sat}_{dep}_{nb}"].values))
    preds_c = dict(cats)
    preds_c[f"tuned ({sat}/{dep}/{nb})"] = tuned_ceil
    for bar in BARS:
        preds_c[f"new, aloft bar {bar}"] = cat_ft(ft_of(C[f"zbbar_{bar}"].values))
    ceil_table("Ceilings, held-out 2020-2025", preds_c, {k: v & tec for k, v in csub.items()})
    bt = Boot(C.wmo.values[tec])
    base = cats["new rule (97/30/300)"][tec]
    R.append("Paired differences vs the shipped rule on 2020-2025 (CSI <1000 ft / CSI <=3000 ft):\n")
    for fn, ec in preds_c.items():
        if fn == "new rule (97/30/300)":
            continue
        d1 = bt.diff(ec[tec] <= 1, base <= 1, oc[tec] <= 1)
        d3 = bt.diff(ec[tec] <= 2, base <= 2, oc[tec] <= 2)
        R.append(f"- {fn}: {fmt(d1)} / {fmt(d3)}")
    R.append("")

    # ---- the page's "~X% of similar layers verify as ceilings"
    R.append("## 'Verify as ceilings' confidence check (new rule, estimated base <= 3000 m AGL, not obscured)\n")
    zb = C[f"zb_{SHIP[0]}_{SHIP[1]}_{SHIP[2]}"].values
    m = np.isfinite(zb) & (zb >= 50) & (zb <= 3000) & C.new_conf.notna().values
    R.append("| page says | n | METAR BKN/OVC/VV <= 12 kft | METAR ceiling within x2 of the estimate |")
    R.append("|---|---|---|---|")
    for conf in (62, 65, 69, 77):
        mm = m & (C.new_conf.values == conf)
        if mm.sum() < 30:
            continue
        sub = C[mm]
        v1 = (sub.m_ceil <= 12000).fillna(False).values.astype(float)
        est_ft = zb[mm] * 3.28084
        v2 = (np.abs(np.log(sub.m_ceil.values / est_ft)) <= np.log(2))
        v2 = np.where(np.isnan(sub.m_ceil.values), False, v2).astype(float)
        bt = Boot(sub.wmo.values)

        def rate(v):
            num = np.bincount(bt.idx, weights=v, minlength=len(bt.keys))
            den = np.bincount(bt.idx, minlength=len(bt.keys)).astype(float)
            bo = (bt.W @ num) / (bt.W @ den)
            return (100 * float(v.mean()), *(100 * np.percentile(bo, [2.5, 97.5])))
        R.append(f"| ~{conf}% | {mm.sum()} | {fmt(rate(v1), 1)} | {fmt(rate(v2), 1)} |")
    R.append("")
    (out / "summary.md").write_text("\n".join(R) + "\n")
    (out / "summary.json").write_text(json.dumps(J, default=float))
    print("\n".join(R), flush=True)
