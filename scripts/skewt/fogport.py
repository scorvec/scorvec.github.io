"""Faithful Python port of the Skew-T explorer's profile handling, cloud layers, ceiling and fog model
(skewt/app.js). Checked level-for-level against the real JS under node by verify_fog.py jstest.

Ported: parseIGRA, iemProfile (IEM RAOB levels), thin, sanitizeHeights, fillDewpoints, fillWinds,
cloudLayers, estCeiling (new, c53ce78, generalised to SAT/DEPTH/NEAR/BAR), estCeiling (old, c53ce78^),
bulkRi, fogModelFeatures, fogModelProbs, doyOf.
"""

from __future__ import annotations

import calendar
import math
from datetime import date

NAN = float("nan")
KT = 1.9438


def isf(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


# ---------------------------------------------------------------- IGRA ---
def _iv(line: str, a: int, b: int):
    s = line[a:b].strip()
    # JS parseInt: leading integer, NaN when none
    j = 0
    if j < len(s) and s[j] in "+-":
        j += 1
    while j < len(s) and s[j].isdigit():
        j += 1
    if j == 0 or (j == 1 and s[0] in "+-"):
        return None
    v = int(s[:j])
    return None if v in (-9999, -8888) else v


def parse_igra_record(lines: list[str], elev: float):
    """lines[0] = '#' header, lines[1:] = levels. Mirrors parseIGRA after the record is chosen.
    Returns (prof, info) or None. info carries the surface-flag QC the JS does not do."""
    nlev = int(lines[0][32:36])

    # JS Math.pow(negative, 5.25588) is NaN (gph > 44.3 km); NaN < 2000 is false, so JS keeps the level
    def stdP(h):
        return (101325 * math.pow(1 - 2.25577e-5 * h, 5.25588)) if 1 - 2.25577e-5 * h > 0 else NAN

    thermo, winds = [], []
    for i in range(1, min(nlev, len(lines) - 1) + 1):
        L = lines[i]
        p = _iv(L, 9, 15)
        gph, tt, dpdp = _iv(L, 16, 21), _iv(L, 22, 27), _iv(L, 34, 39)
        rh = _iv(L, 28, 33)
        wd, ws = _iv(L, 40, 45), _iv(L, 46, 51)
        if p is None and gph is not None and gph > -900:
            p = stdP(gph)
        if p is None or p < 2000:
            continue
        if wd is not None and ws is not None and ws >= 0:
            winds.append(
                {
                    "p": p,
                    "h": gph,
                    "u": -(ws / 10) * math.sin(wd * math.pi / 180),
                    "v": -(ws / 10) * math.cos(wd * math.pi / 180),
                }
            )
        if tt is not None and tt > -8888:
            D = NAN
            if dpdp is not None:
                D = tt / 10 + 273.15 - dpdp / 10
            elif rh is not None and rh > 0:
                tc, RH = tt / 10, min(100, rh / 10)
                gamma = math.log(RH / 100) + 17.67 * tc / (tc + 243.5)
                D = 243.5 * gamma / (17.67 - gamma) + 273.15
            thermo.append(
                {"p": p, "gph": gph, "T": tt / 10 + 273.15, "D": D, "sfc": L[1:2] == "1", "dpdp": dpdp, "rh": rh}
            )
    thermo.sort(key=lambda a: -a["p"])  # stable, like Array.prototype.sort
    winds.sort(key=lambda a: -a["p"])
    if len(thermo) < 3:
        return None  # wind-only: no thermo profile (not usable here)

    def windAtP(p):
        if not winds:
            return {"u": 0, "v": 0}
        if p >= winds[0]["p"]:
            return winds[0]
        for j in range(1, len(winds)):
            if winds[j]["p"] <= p:
                f = math.log(winds[j - 1]["p"] / p) / math.log(winds[j - 1]["p"] / winds[j]["p"])
                return {
                    "u": winds[j - 1]["u"] + f * (winds[j]["u"] - winds[j - 1]["u"]),
                    "v": winds[j - 1]["v"] + f * (winds[j]["v"] - winds[j - 1]["v"]),
                }
        return winds[-1]

    out = {k: [] for k in "PHTDUV"}
    lastP, lastH, lastT = 1e9, elev, None
    first = None
    for lv in thermo:
        if lv["p"] >= lastP:
            continue
        if lv["gph"] is not None:
            Hm = lv["gph"]
        else:
            Tbar = lv["T"] if lastT is None else (lv["T"] + lastT) / 2
            Hm = lastH + (287.05 * Tbar / 9.80665) * math.log(lastP / lv["p"])
        if not out["P"] and lv["gph"] is None:
            Hm = elev
        if first is None:
            first = lv
        lastP, lastH, lastT = lv["p"], Hm, lv["T"]
        w = windAtP(lv["p"])
        out["P"].append(lv["p"])
        out["H"].append(Hm)
        out["T"].append(lv["T"])
        out["D"].append(lv["D"])
        out["U"].append(w["u"])
        out["V"].append(w["v"])
    if len(out["P"]) < 4:
        return None
    nan_p = any(math.isnan(lv["p"]) for lv in thermo) or any(math.isnan(w["p"]) for w in winds)
    info = {
        "nan_p": nan_p,
        "sfc_flag": bool(first and first["sfc"]),
        "raw_t0": first["T"],
        "raw_d0": first["D"],
        "nlev": len(out["P"]),
        "nwind": len(winds),
    }
    return out, info


def iem_profile(levels: list[dict]):
    """iemProfile(): IEM RAOB levels with keys pres (hPa), hght, tmpc, dwpc, drct, sknt."""
    o = {k: [] for k in "PHTDUV"}
    for L in levels:
        if L.get("pres") is None or L.get("tmpc") is None or L["pres"] < 20:
            continue
        o["P"].append(L["pres"] * 100)
        o["H"].append(NAN if L.get("hght") is None else L["hght"])
        o["T"].append(L["tmpc"] + 273.15)
        o["D"].append(NAN if L.get("dwpc") is None else L["dwpc"] + 273.15)
        if L.get("sknt") is not None and L.get("drct") is not None:
            ws, wd = L["sknt"] / KT, L["drct"]
            o["U"].append(-ws * math.sin(wd * math.pi / 180))
            o["V"].append(-ws * math.cos(wd * math.pi / 180))
        else:
            o["U"].append(NAN)
            o["V"].append(NAN)
    return o


# ------------------------------------------------------- profile prep ---
def thin(prof, target=350):
    N = len(prof["P"])
    if N <= target:  # noqa: SIM300  variable N (maths notation), not a constant
        return prof
    k = math.ceil(N / target)
    out = {key: [] for key in prof}
    for i in range(N):
        if i == 0 or i == N - 1 or i % k == 0:
            for key in prof:
                out[key].append(prof[key][i])
    return out


def sanitize_heights(prof):
    Rd, g = 287.05, 9.80665
    P, T, H = prof["P"], prof["T"], prof["H"]
    for i in range(1, len(P)):
        bad = not isf(H[i]) or H[i] <= H[i - 1]
        if not bad:
            continue
        if isf(T[i]) and isf(T[i - 1]) and P[i] > 0 and P[i - 1] > P[i]:
            tbar = 0.5 * (T[i] + T[i - 1])
            H[i] = H[i - 1] + Rd * tbar / g * math.log(P[i - 1] / P[i])
        else:
            H[i] = H[i - 1] + 1


def fill_dewpoints(prof):
    P, T, D = prof["P"], prof["T"], prof["D"]
    ok = [i for i in range(len(P)) if isf(D[i])]
    if not ok:
        for i in range(len(P)):
            D[i] = T[i] - 30
        prof["dewSynth"] = True
        return
    for i in range(len(P)):
        if isf(D[i]):
            continue
        lo = hi = None
        for j in ok:
            if j < i:
                lo = j
            else:
                hi = j
                break
        if lo is not None and hi is not None:
            f = math.log(P[lo] / P[i]) / math.log(P[lo] / P[hi])
            D[i] = D[lo] + f * (D[hi] - D[lo])
        else:
            ref = lo if lo is not None else hi
            D[i] = min(D[ref], T[i] - 2)
        if D[i] > T[i]:
            D[i] = T[i]


def fill_winds(o):
    n = len(o["P"])
    ok = [i for i in range(n) if isf(o["U"][i]) and isf(o["V"][i])]
    if not ok:
        for i in range(n):
            o["U"][i] = 0
            o["V"][i] = 0
        return
    for i in range(n):
        if isf(o["U"][i]) and isf(o["V"][i]):
            continue
        lo = hi = None
        for j in ok:
            if j < i:
                lo = j
            else:
                hi = j
                break
        if lo is None:
            o["U"][i] = o["U"][ok[0]]
            o["V"][i] = o["V"][ok[0]]
        elif hi is None:
            o["U"][i] = o["U"][lo]
            o["V"][i] = o["V"][lo]
        else:
            f = math.log(o["P"][lo] / o["P"][i]) / math.log(o["P"][lo] / o["P"][hi])
            o["U"][i] = o["U"][lo] + f * (o["U"][hi] - o["U"][lo])
            o["V"][i] = o["V"][lo] + f * (o["V"][hi] - o["V"][lo])


def prepare(prof):
    """render(thin(prof)) preprocessing, in place on a thinned copy."""
    p = thin(prof)
    if p is prof:
        p = {k: list(v) for k, v in prof.items()}
    sanitize_heights(p)
    fill_dewpoints(p)
    fill_winds(p)
    return p


# -------------------------------------------------------- cloud layers ---
def cloud_layers(prof):
    z0 = prof["H"][0]
    levs = []
    for i in range(len(prof["P"])):
        tC, dC = prof["T"][i] - 273.15, prof["D"][i] - 273.15
        z = prof["H"][i] - z0
        if z > 15000:
            break
        if not isf(tC) or not isf(dC):
            continue
        ew = 6.112 * math.exp(17.67 * dC / (dC + 243.5))
        rh = 100 * ew / (6.112 * math.exp(17.67 * tC / (tC + 243.5)))
        rhw = rh
        if tC < 0:
            rh = max(rh, 100 * ew / (6.112 * math.exp(22.46 * tC / (tC + 272.62))))
        levs.append((z, tC, min(rh, 105), rhw))
    raw, cur = [], None
    for z, _tC, rh, _rhw in levs:
        if rh >= (85 if z < 2000 else 80):
            if cur is None:
                cur = {"base": z, "top": z, "maxRh": 0}
            cur["top"] = z
            cur["maxRh"] = max(cur["maxRh"], rh)
        elif cur is not None:
            raw.append(cur)
            cur = None
    if cur is not None:
        raw.append(cur)
    layers = []
    for l in raw:
        if layers and l["base"] - layers[-1]["top"] < 300:
            layers[-1]["top"] = l["top"]
            layers[-1]["maxRh"] = max(layers[-1]["maxRh"], l["maxRh"])
        else:
            layers.append(dict(l))
    return {"layers": layers, "levs": levs}


def p_ceil(rh):
    """app.js estCeiling's pCeil (kept identical: fogport_jstest checks the port against the app)."""
    return 85 if rh >= 100 else 74 if rh >= 98 else 65 if rh >= 96 else 56 if rh >= 94 else 54


def est_ceiling_old(res):
    """c53ce78^: returns None (undefined), {'none':True} (null), or dict(zb, obscured, conf, maxRh)."""
    L = res["levs"]
    if len(L) < 3:
        return None
    ceil = next((l for l in res["layers"] if l["maxRh"] >= 94), None)
    if ceil is None:
        return {"none": True}
    lv = next((v for v in L if v[0] >= ceil["base"] - 1 and v[0] <= ceil["top"] + 1 and v[2] >= 94), None)
    zb = lv[0] if lv else ceil["base"]
    return {"zb": zb, "obscured": zb < 50, "conf": p_ceil(ceil["maxRh"]), "maxRh": ceil["maxRh"]}


def est_ceiling_new(res, SAT=97.0, DEPTH=30.0, NEAR=300.0, BAR=94.0):
    """c53ce78 estCeiling with its constants exposed (defaults = the shipped rule)."""
    L = res["levs"]
    n = len(L)
    if n < 3:
        return None

    def run_top(i):
        j = i
        while j + 1 < n and L[j + 1][3] >= SAT:
            j += 1
        return j

    fog = L[0][0] < 15 and L[0][3] >= SAT and L[run_top(0)][0] >= DEPTH
    mist = L[0][0] < 15 and L[0][2] >= BAR and not fog
    for l in res["layers"]:
        if l["maxRh"] < BAR:
            continue
        sfc = l["base"] < 50
        if sfc and fog:
            return {
                "zb": 0.0,
                "obscured": True,
                "satTop": L[run_top(0)][0],
                "conf": p_ceil(l["maxRh"]),
                "maxRh": l["maxRh"],
            }
        i = 0
        lo, hi = max(50, l["base"] - 1), l["top"] + 1
        while i < n:
            z, _t, rh, rhw = L[i]
            if z < lo or z > hi:
                i += 1
                continue
            if not sfc or z >= NEAR:
                if rh >= BAR:
                    return {"zb": z, "obscured": False, "conf": p_ceil(l["maxRh"]), "mist": mist, "maxRh": l["maxRh"]}
                i += 1
                continue
            if rhw < SAT:
                i += 1
                continue
            j = run_top(i)
            if L[j][0] - z >= DEPTH:
                return {"zb": z, "obscured": False, "conf": p_ceil(l["maxRh"]), "mist": mist, "maxRh": l["maxRh"]}
            i = j + 1
    return {"none": True, "mist": mist}


# ------------------------------------------------------------ fog model ---
FOG_MODEL = {
    "fog": {
        "mu": [
            3.5493,
            80.5304,
            4.1064,
            2.2102,
            1.8378,
            114.3392,
            41.5371,
            142.164,
            82.3222,
            72.5089,
            1.4016,
            0.996,
            443.6591,
            7.3728,
            10.2455,
            -0.0083,
            0.0013,
            0.9926,
        ],
        "sd": [
            3.6834,
            16.3206,
            3.9419,
            5.665,
            2.6478,
            154.3423,
            250.1182,
            471.2011,
            16.6885,
            23.5149,
            1.5727,
            7.1132,
            460.4216,
            4.2844,
            8.8232,
            0.7064,
            0.7077,
            0.0854,
        ],
        "w": [
            -7.1094,
            -0.043,
            1.9138,
            -0.6283,
            0.0184,
            -0.535,
            0.1701,
            0.017,
            -0.0878,
            2.6623,
            -0.056,
            0.1444,
            0.68,
            -0.043,
            -0.3321,
            0.111,
            -0.161,
            0.3622,
            -0.0711,
        ],
    }
}
FOG_FEATS = [
    "dep_sfc",
    "rh_sfc",
    "spd_sfc",
    "ri",
    "inv_dt",
    "inv_top",
    "sat95_dep",
    "rh90_dep",
    "rhmax_lo",
    "rhmax_mid",
    "dq_mix",
    "lapse500",
    "lcl_m",
    "q_sfc",
    "t_sfc",
    "sin_doy",
    "cos_doy",
    "is12z",
]


def bulk_ri(prof):
    z0 = prof["H"][0]
    P, H, T, U, V, D = (prof[k] for k in "PHTUVD")
    i = 1
    while i < len(P) and (H[i] - z0 < 30 or not isf(T[i]) or not isf(U[i])):
        i += 1
    if i >= len(P) or not isf(T[0]) or not isf(U[0]):
        return None

    def thv(k):
        dC = D[k] - 273.15
        e = 6.112 * math.exp(17.67 * dC / (dC + 243.5)) if isf(dC) else 0
        r = 0.622 * e / (P[k] / 100 - e)
        return T[k] * (1 + 0.61 * r) * math.pow(100000 / P[k], 0.2854)

    dz = H[i] - z0
    du, dv = U[i] - U[0], V[i] - V[0]
    s2 = max(du * du + dv * dv, 1e-4)
    return 9.80665 / (0.5 * (thv(0) + thv(i))) * (thv(i) - thv(0)) * dz / s2


def doy_of(ymd: str) -> int:
    """1..366 on a 366-slot calendar: 29 Feb = 60 and 1 March = 61 in every year (app.js doyOf)."""
    d = date.fromisoformat(ymd[:10])
    doy = (d - date(d.year, 1, 1)).days + 1
    return doy + 1 if doy >= 60 and not calendar.isleap(d.year) else doy


def fog_model_features(prof, doy, hour):
    def esw(tc):
        return 6.112 * math.exp(17.67 * tc / (tc + 243.5))

    P, H, T, D, U, V = (prof[k] for k in "PHTDUV")
    t0, d0 = T[0] - 273.15, D[0] - 273.15
    if not isf(t0) or not isf(d0):
        return None
    f = {
        "dep_sfc": t0 - d0,
        "rh_sfc": 100 * esw(d0) / esw(t0),
        "lcl_m": 125 * (t0 - d0),
        "t_sfc": t0,
        "q_sfc": 622 * esw(d0) / (P[0] / 100 - esw(d0)),
    }
    f["spd_sfc"] = math.hypot(U[0], V[0]) * KT if isf(U[0]) else None
    rb = bulk_ri(prof)
    # JS: Math.max(-25, Math.min(25, NaN)) is NaN (Python's min/max would return 25)
    f["ri"] = None if rb is None else (rb if math.isnan(rb) else max(-25, min(25, rb)))
    z0 = H[0]
    invDt, invTop, tmax = 0.0, 0.0, t0
    sat95 = rh90 = 0.0
    b95 = b90 = False
    rhLo = rhMid = 0.0
    qs = []
    for i in range(len(P)):
        za, tC, dC = H[i] - z0, T[i] - 273.15, D[i] - 273.15
        if za <= 1000 and isf(tC) and i > 0:
            if tC >= tmax - 0.05 and invTop >= 0:
                tmax = max(tmax, tC)
                invDt = tmax - t0
                invTop = za
            else:
                invTop = -1
        if za > 3000 or not isf(tC) or not isf(dC):
            continue
        rh = 100 * esw(dC) / esw(tC)
        if rh >= 95 and not b95:
            sat95 = za
        elif rh < 95:
            b95 = True
        if rh >= 90 and not b90:
            rh90 = za
        elif rh < 90:
            b90 = True
        if za <= 500:
            rhLo = max(rhLo, rh)
        elif za <= 1500:
            rhMid = max(rhMid, rh)
        if za <= 1500:
            qs.append((za, 622 * esw(dC) / (P[i] / 100 - esw(dC))))
    f["inv_dt"] = invDt
    f["inv_top"] = max(0, invTop)
    f["sat95_dep"] = sat95
    f["rh90_dep"] = rh90
    f["rhmax_lo"] = rhLo
    f["rhmax_mid"] = rhMid
    qlo = [q[1] for q in qs if q[0] <= 300]
    qhi = [q[1] for q in qs if 600 <= q[0] <= 1200]
    f["dq_mix"] = (sum(qlo) / len(qlo) - sum(qhi) / len(qhi)) if qlo and qhi else None
    t500 = None
    for i in range(1, len(P)):
        za = H[i] - z0
        if za >= 500 and isf(T[i]) and isf(T[i - 1]):
            zp = H[i - 1] - z0
            fr = (500 - zp) / max(1, za - zp)
            t500 = (T[i - 1] - 273.15) + fr * (T[i] - T[i - 1])
            break
    f["lapse500"] = (t500 - t0) / 0.5 if t500 is not None else None
    f["sin_doy"] = math.sin(2 * math.pi * doy / 366)
    f["cos_doy"] = math.cos(2 * math.pi * doy / 366)
    f["is12z"] = 1 if hour == 12 else 0
    return f


def fog_prob(prof, doy, hour):
    f = fog_model_features(prof, doy, hour)
    if f is None:
        return None
    m = FOG_MODEL["fog"]
    z = m["w"][0]
    for i, name in enumerate(FOG_FEATS):
        v = f[name]
        if v is not None and isf(v):
            z += m["w"][i + 1] * (v - m["mu"][i]) / m["sd"][i]
    return 1 / (1 + math.exp(-max(-30, min(30, z))))
