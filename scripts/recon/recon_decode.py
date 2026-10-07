#!/usr/bin/env python3
"""Decoders for the NHC reconnaissance text products (Atlantic: AHONT1, REPNT1/2/3, REPRPD).

Formats follow the National Hurricane Operations Plan (NHOP) Appendix G and NHC's HDOB format note
(https://www.nhc.noaa.gov/abouthdobs_2007.shtml):

  URNT15  HDOB, high-density observations: one 30-s line per ob.
          hhmmss LLLLH NNNNNW PPPP GGGGG XXXX sTTT sddd wwwSSS MMM KKK ppp FF
  UZNT13  TEMP DROP (FM 37): parts XXAA (mandatory levels) and XXBB (significant T levels + 21212 winds),
          31313 launch time, 61616 mission, 62626 remarks (REL/SPG/SPL, MBL/DLM/WL150 winds, EYE/EYEWALL...).
  URNT12  Vortex data message (2018+ lettered A.-U. layout).
  URNT11  RECCO (intermediate flight-level observation).
  REPRPD  Tropical cyclone plan of the day (kept as text).

Quality control (strict by design: a flagged value is never published as good):
  * the HDOB FF flags mask the fields they name (first digit: position / pressure-height; second digit:
    T-Td, flight-level wind, SFMR, as in the NHC table) - the value is dropped, the flag code is kept;
  * "///" (and the legacy 999 codes) are missing;
  * physical range screens on every field, Td > T + 0.5 C dropped, positions that imply > 450 kt between
    consecutive obs dropped (screened in the mirror where the whole track is known).
Only stdlib; every decoder returns plain dicts ready for JSON.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
MISS = re.compile(r"^/+$")


def _int(s):
    if s is None or MISS.match(s) or not s.strip("/"):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _hdr(text):
    """WMO abbreviated heading 'URNT15 KNHC 071257' -> (ttaaii, cccc, ddhhmm)."""
    m = re.search(r"\b([A-Z]{4}\d{2})\s+([A-Z]{4})\s+(\d{6})(?:\s+(CC[A-Z]|RR[A-Z]|AA[A-Z]))?", text)
    return (m.group(1), m.group(2), m.group(3), m.group(4)) if m else (None, None, None, None)


def mission_of(s):
    """'AF304 0209A ISAIAS' / 'NOAA9 01BBA SURV' -> dict(aircraft, mission, storm_no, basin, name)."""
    # mission: 2-digit mission no + storm no (or CARCAH letters) + basin, e.g. 0209A / 01BBA; WXWXA = non-tasked weather recon
    m = re.search(r"\b((?:AF|NOAA|NASA|NAVY|USN|CARCAH)\s?\d{1,3}[A-Z]?)\s+(\d{2}[A-Z0-9]{2}[A-Z]|WXWX[A-Z])\b\s*([A-Z0-9-]+)?", s)
    if not m:
        return None
    ac, mi, nm = m.group(1).replace(" ", ""), m.group(2), (m.group(3) or "").strip()
    sn = "WX" if mi.startswith("WX") else mi[2:4]
    return dict(aircraft=ac, mission=mi, storm_no=int(sn) if sn.isdigit() else None, tag=sn,
                basin=mi[4], name=nm if nm not in ("HDOB", "OB") and not nm.isdigit() else "")


def aircraft_type(ac):
    """Platform from the call sign (NHOP): AF3xx = USAF Reserve 53rd WRS WC-130J; NOAA2/3 = WP-3D Orion
    (N42RF 'Kermit' / N43RF 'Miss Piggy'); NOAA9 = G-IV SP (N49RF); NOAA6/7 = other AOC aircraft."""
    if not ac:
        return ""
    if ac.startswith("AF"):
        return "USAF WC-130J (53rd WRS)"
    return {"NOAA2": "NOAA WP-3D N42RF", "NOAA3": "NOAA WP-3D N43RF", "NOAA9": "NOAA G-IV N49RF",
            "NOAA7": "NOAA aircraft N47RF", "NOAA6": "NOAA aircraft"}.get(ac, ac)


# ───────────────────────────────────────────────────────────── HDOB (URNT15) ──
HDOB_LINE = re.compile(
    r"^(\d{6}) (\d{4}[NS]) (\d{5}[EW]) (\S{4}) (\S{5}) (\S{4}) ([+-]\S{3}|/{4}) ([+-]\S{3}|/{4}) (\S{6}) (\S{3}) (\S{3}) (\S{3}) (\S{2})\s*$")

# second-digit flag -> fields it marks questionable (NHC HDOB table)
MET_FLAG = {0: (), 1: ("t", "td"), 2: ("wd", "ws", "wp"), 3: ("sfmr", "rr"), 4: ("t", "td", "wd", "ws", "wp"),
            5: ("t", "td", "sfmr", "rr"), 6: ("wd", "ws", "wp", "sfmr", "rr"),
            9: ("t", "td", "wd", "ws", "wp", "sfmr", "rr")}
POS_FLAG = {0: (), 1: ("pos",), 2: ("p", "z", "x"), 3: ("pos", "p", "z", "x")}


def _lat(s):
    v = int(s[:2]) + int(s[2:4]) / 60.0
    return round(-v if s[4] == "S" else v, 4)


def _lon(s):
    v = int(s[:3]) + int(s[3:5]) / 60.0
    return round(-v if s[5] == "W" else v, 4)


def _signed10(s):
    if s is None or "/" in s:
        return None
    try:
        return int(s) / 10.0
    except ValueError:
        return None


def decode_hdob(text, file_dt):
    """-> dict(mission..., obnum, date, obs=[...]) or None. file_dt = datetime of the file name stamp (UTC)."""
    tt, cccc, ddhhmm, bbb = _hdr(text)
    if not tt or not tt.startswith("URNT15") and not tt.startswith("URPN15") and not tt.startswith("URPA15"):
        return None
    lines = [ln.rstrip() for ln in text.splitlines()]
    head = next((ln for ln in lines if " HDOB " in ln), None)
    if not head:
        return None
    mi = mission_of(head) or {}
    m = re.search(r"HDOB\s+(\d+)\s+(\d{8})", head)
    obnum = int(m.group(1)) if m else None
    base = datetime.strptime(m.group(2), "%Y%m%d").replace(tzinfo=UTC) if m else file_dt.replace(hour=0, minute=0, second=0)
    obs, prev = [], None
    rej = dict(flagged=0, range=0)
    for ln in lines:
        g = HDOB_LINE.match(ln)
        if not g:
            continue
        hh, la, lo, pp, gz, xx, tt_, td, wnd, mx, sf, rr, ff = g.groups()
        t = base + timedelta(hours=int(hh[:2]), minutes=int(hh[2:4]), seconds=int(hh[4:6]))
        if prev is not None and t < prev - timedelta(hours=12):
            t += timedelta(days=1)          # crossed 00 UTC inside the message (the header date is the first ob's)
        prev = t
        o = {"t": int(t.timestamp()), "la": _lat(la), "lo": _lon(lo)}
        pv = _int(pp)
        if pv is not None:
            p = pv / 10.0
            o["p"] = round(p + 1000.0 if p < 100.0 else p, 1)          # leading 1 dropped at >= 1000 hPa
        o["z"] = _int(gz)
        xv = _int(xx)
        if xv is not None and o.get("p") is not None:
            if o["p"] >= 550.0:                                          # extrapolated surface pressure
                x = xv / 10.0
                o["x"] = round(x + 1000.0 if x < 200.0 else x, 1)
            else:                                                        # D-value, +5000 = negative
                o["dv"] = xv - 5000 if xv >= 5000 else xv
        o["t_"] = _signed10(tt_)
        o["td"] = _signed10(td)
        wd, ws = _int(wnd[:3]), _int(wnd[3:])
        if wd is not None and ws is not None and wd <= 360 and ws != 999:
            o["wd"], o["ws"] = wd % 360, ws
        wp = _int(mx)
        if wp is not None and wp != 999:
            o["wp"] = wp
        s = _int(sf)
        if s is not None and s != 999:
            o["sfmr"] = s
        r = _int(rr)
        if r is not None and r != 999:
            o["rr"] = r
        f1, f2 = (_int(ff[0]), _int(ff[1])) if len(ff) == 2 else (None, None)
        o["qc"] = ff if (f1 is not None and f2 is not None) else "//"
        bad = set(POS_FLAG.get(f1, ("pos", "p", "z", "x"))) | set(MET_FLAG.get(f2, MET_FLAG[9]))
        if "x" in bad:
            bad.add("dv")
        if "t" in bad:
            o.pop("t_", None)
        for k in list(o):
            if k in bad and k not in ("t", "la", "lo", "qc"):
                o.pop(k)
                rej["flagged"] += 1
        if "pos" in bad:
            o["la"] = o["lo"] = None
            rej["flagged"] += 1
        # physical range screens
        for k, lo_, hi_ in (("p", 100.0, 1080.0), ("z", -100, 16000), ("x", 870.0, 1060.0), ("dv", -1500, 1500),
                            ("t_", -85.0, 45.0), ("td", -95.0, 40.0), ("ws", 0, 260), ("wp", 0, 270),
                            ("sfmr", 0, 220), ("rr", 0, 200)):
            v = o.get(k)
            if v is not None and not (lo_ <= v <= hi_):
                o.pop(k)
                rej["range"] += 1
        if o.get("td") is not None and o.get("t_") is not None and o["td"] > o["t_"] + 0.5:
            o.pop("td")
            rej["range"] += 1
        if "ws" not in o:
            o.pop("wd", None)
        o = {k: v for k, v in o.items() if v is not None or k in ("la", "lo")}
        obs.append(o)
    if not obs:
        return None
    return dict(kind="hdob", hdr=f"{tt} {cccc} {ddhhmm}", cor=bbb, obnum=obnum, obs=obs, rej=rej, **mi)


# ───────────────────────────────────────────────────── TEMP DROP (UZNT13) ──
STD_LEVELS = {"00": 1000, "92": 925, "85": 850, "70": 700, "50": 500, "40": 400, "30": 300, "25": 250,
              "20": 200, "15": 150, "10": 100}
ID_LEVEL = {"0": 1000, "9": 925, "8": 850, "7": 700, "5": 500, "4": 400, "3": 300, "2": 200, "1": 100}
SECTION = {"21212", "31313", "41414", "51515", "61616", "62626"}


def _temp_dd(g):
    """TTTDD -> (T C, Td C): tenths digit even = positive, odd = negative; DD 00-50 tenths, 56-99 = DD-50."""
    if g is None or len(g) != 5:
        return None, None
    tt, dd = g[:3], g[3:]
    if "/" in tt:
        return None, None
    v = int(tt)
    t = v / 10.0 if v % 2 == 0 else -v / 10.0
    if "/" in dd:
        return t, None
    d = int(dd)
    if d <= 50:
        dep = d / 10.0
    elif d >= 56:
        dep = float(d - 50)
    else:
        return t, None
    return t, round(t - dep, 1)


def _wind(g):
    """ddfff -> (dir, kt). Direction to 5 deg: 500 added to fff when it ends in 5."""
    if g is None or len(g) != 5 or "/" in g:
        return None, None
    dd, fff = int(g[:2]), int(g[2:])
    d = dd * 10 + (5 if fff >= 500 else 0)
    f = fff % 500
    if d > 360 or (d == 0 and f != 0):
        return None, None
    return d, f                       # 360 = from the north; 0 only with calm


def _pos(s):
    """'2171N09470W' -> (lat, lon) in degrees (tenths... encoded as DDdd hundredths of a degree)."""
    m = re.match(r"(\d{4})([NS])(\d{5})([EW])", s)
    if not m:
        return None
    la = int(m.group(1)) / 100.0
    lo = int(m.group(3)) / 100.0
    return (-la if m.group(2) == "S" else la, -lo if m.group(4) == "W" else lo)


def _split_parts(text):
    """Return {'A': tokens, 'B': tokens, 'rmk61': str, 'rmk62': str} for one UZNT13 message."""
    lines = text.splitlines()
    parts, cur, buf = {}, None, []
    for ln in lines:
        s = ln.strip()
        if s.startswith("XXAA") or s.startswith("XXBB") or s.startswith("XXCC") or s.startswith("XXDD"):
            if cur:
                parts[cur] = buf
            cur, buf = s[:4], [ln.rstrip()]
        elif cur:
            buf.append(ln.rstrip())
            if s.endswith("="):
                parts[cur] = buf
                cur, buf = None, []
    if cur:
        parts[cur] = buf
    return parts


def _sections(lines):
    """Split a part's lines into the coded body and the 61616/62626 remarks. Coded lines end on a group
    boundary; remark lines are hard-wrapped at 65 characters, so a 65-character remark line is joined to the
    next one WITHOUT a space (the WL150 layer height and REL fields are often split by the wrap)."""
    coded, r61, r62, mode, r62_last_len = [], "", "", "c", 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("61616"):
            mode, r61 = "61", s[5:].strip()
            continue
        if s.startswith("62626"):
            mode, r62 = "62", s[5:].strip()
            r62_last_len = len(ln.rstrip())
            continue
        if mode == "c":
            coded.append(s)
        elif mode == "61":
            r61 += " " + s
        else:
            r62 = r62 + ("" if r62_last_len >= 65 else " ") + s
            r62_last_len = len(ln.rstrip())
    return " ".join(coded).replace("=", " ").split(), r61.strip(" ="), r62.strip(" =")


def _std_height(level, hhh, ref=None):
    """Mandatory-level height (m) from its 3-digit code; ambiguous ranges resolved by `ref` (a hypsometric
    estimate) when given, else by the WMO conventions."""
    if hhh is None:
        return None
    if level == 1000:
        return -(hhh - 500) if hhh >= 500 else hhh
    if level in (925,):
        cands = [hhh, hhh + 1000]
    elif level == 850:
        cands = [hhh + 1000, hhh]
    elif level == 700:
        cands = [hhh + 3000, hhh + 2000] if hhh < 500 else [hhh + 2000, hhh + 3000]
    elif level in (500, 400):
        cands = [hhh * 10]
    elif level in (300, 250):
        cands = [hhh * 10 + 10000, hhh * 10] if hhh < 500 else [hhh * 10, hhh * 10 + 10000]
    else:                                    # 200, 150, 100 hPa: decametres above 10 km
        cands = [hhh * 10 + 10000]
    if ref is not None:
        return min(cands, key=lambda c: abs(c - ref))
    return cands[0]


def _hypso(levels):
    """Fill heights by hypsometric integration with virtual temperature, upward from the lowest level with a
    known height (the splash level = 0 m over the sea) and downward when only an upper height is known."""
    lv = sorted([l for l in levels if l.get("p")], key=lambda l: -l["p"])

    def tv(l):
        t = l.get("t")
        if t is None:
            return None
        td = l.get("td")
        tk = t + 273.15
        if td is None:
            return tk
        e = 6.112 * math.exp(17.67 * td / (td + 243.5))
        q = 0.622 * e / (l["p"] - 0.378 * e)
        return tk * (1 + 0.608 * q)

    # interpolate missing Tv in log-p
    tvs = [tv(l) for l in lv]
    known = [i for i, v in enumerate(tvs) if v is not None]
    if not known:
        return lv
    for i, v in enumerate(tvs):
        if v is None:
            lo = max([k for k in known if k < i], default=None)
            hi = min([k for k in known if k > i], default=None)
            if lo is None:
                tvs[i] = tvs[hi]
            elif hi is None:
                tvs[i] = tvs[lo]
            else:
                a = math.log(lv[lo]["p"] / lv[i]["p"]) / math.log(lv[lo]["p"] / lv[hi]["p"])
                tvs[i] = tvs[lo] + a * (tvs[hi] - tvs[lo])
    anchor = next((i for i, l in enumerate(lv) if l.get("z") is not None), None)
    if anchor is None:
        return lv
    zz = [None] * len(lv)
    zz[anchor] = lv[anchor]["z"]
    for i in range(anchor + 1, len(lv)):
        tm = 0.5 * (tvs[i] + tvs[i - 1])
        zz[i] = zz[i - 1] + 287.04 * tm / 9.80665 * math.log(lv[i - 1]["p"] / lv[i]["p"])
    for i in range(anchor - 1, -1, -1):
        tm = 0.5 * (tvs[i] + tvs[i + 1])
        zz[i] = zz[i + 1] - 287.04 * tm / 9.80665 * math.log(lv[i]["p"] / lv[i + 1]["p"])
    for l, z in zip(lv, zz):
        l["zh"] = z
    return lv


def decode_drop(text, file_dt):
    """One UZNT13 TEMP DROP message -> dict (profile, release / splash, boundary-layer winds) or None."""
    tt, cccc, ddhhmm, bbb = _hdr(text)
    if not tt or not tt.startswith(("UZNT13", "UZPN13", "UZPA13")):
        return None
    parts = _split_parts(text)
    if not parts:
        return None
    out = dict(kind="drop", hdr=f"{tt} {cccc} {ddhhmm}", cor=bbb, parts=sorted(parts), raw=text.strip())
    levels = {}      # p (hPa, 1 decimal) -> level dict

    def lev(p):
        k = round(float(p), 1)
        return levels.setdefault(k, {"p": k})

    sfc_p = None
    rmk61 = rmk62 = ""
    for name, lines in parts.items():
        toks, r61, r62 = _sections(lines)
        rmk61, rmk62 = rmk61 or r61, rmk62 or r62
        if len(toks) < 5:
            continue
        yygg = toks[1]
        if not out.get("day") and yygg[:2].isdigit():
            yy = int(yygg[:2])
            out["day"], out["kt"] = (yy - 50, True) if yy > 50 else (yy, False)
        if toks[2].startswith("99") and toks[2][2:].isdigit():
            la = int(toks[2][2:]) / 10.0
            q, lo = toks[3][0], int(toks[3][1:]) / 10.0
            # quadrant Qc: 1 = N/E, 3 = S/E, 5 = S/W, 7 = N/W
            out["obs_pos"] = (-la if q in "35" else la, -lo if q in "57" else lo)
        i = 5
        if name == "XXAA":
            idl = ID_LEVEL.get(yygg[4])          # last standard level carrying a wind group
            while i < len(toks):
                g = toks[i]
                if g in SECTION or g.startswith("88") or g.startswith("77") or g.startswith("66"):
                    break
                if g.startswith("99") and i == 5:          # surface 99PPP TTTDD ddfff
                    pv = _int(g[2:])
                    if pv is not None:
                        sfc_p = pv + 1000 if pv < 100 else pv
                        L = lev(sfc_p)
                        L["sfc"] = True
                        L["z"] = 0.0                       # splash level = the sea surface
                        L["t"], L["td"] = _temp_dd(toks[i + 1] if i + 1 < len(toks) else None)
                        L["wd"], L["ws"] = _wind(toks[i + 2] if i + 2 < len(toks) else None)
                    i += 3
                    continue
                code = g[:2]
                if code not in STD_LEVELS:
                    i += 1
                    continue
                p = STD_LEVELS[code]
                hhh = _int(g[2:])
                has_w = idl is not None and p >= idl
                t, td = _temp_dd(toks[i + 1] if i + 1 < len(toks) else None)
                wd, ws = _wind(toks[i + 2]) if has_w and i + 2 < len(toks) else (None, None)
                i += 3 if has_w else 2
                if sfc_p is not None and p > sfc_p:        # below the sea surface: extrapolated, not observed
                    continue
                L = lev(p)
                L["std"] = True
                L["hhh"] = hhh
                if t is not None:
                    L["t"], L["td"] = t, td
                if ws is not None:
                    L["wd"], L["ws"] = wd, ws
            # tropopause / max wind (88PPP TTTDD ddfff / 77PPP ddfff (4vbvb))
            while i < len(toks) and toks[i] not in SECTION:
                g = toks[i]
                if g.startswith("77") or g.startswith("66"):
                    if g[2:] != "999":
                        pv = _int(g[2:])
                        wd, ws = _wind(toks[i + 1] if i + 1 < len(toks) else None)
                        if pv and ws is not None:
                            out["maxwind"] = dict(p=pv, wd=wd, ws=ws)
                        i += 2
                        if i < len(toks) and toks[i].startswith("4"):
                            i += 1
                        continue
                if g.startswith("88") and g[2:] != "999":
                    i += 3
                    continue
                i += 1
        elif name == "XXBB":
            sect = "t"
            while i < len(toks):
                g = toks[i]
                if g == "21212":
                    sect, i = "w", i + 1
                    continue
                if g in SECTION:
                    if g == "31313":
                        sect = "x"
                    else:
                        sect = "x"
                    i += 1
                    continue
                if sect == "x":
                    i += 1
                    continue
                pv = _int(g[2:]) if len(g) == 5 else None
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                i += 2
                if pv is None or nxt is None:
                    continue
                p = pv + 1000 if pv < 100 else pv
                if sect == "t":
                    t, td = _temp_dd(nxt)
                    if t is not None:
                        L = lev(p)
                        L["t"], L["td"] = t, td
                        if g[:2] == "00":
                            L["sfc"] = True
                            if sfc_p is None:
                                sfc_p = p
                                L["z"] = 0.0
                else:
                    wd, ws = _wind(nxt)
                    if ws is not None:
                        L = lev(p)
                        L["wd"], L["ws"] = wd, ws
        # 31313 sr rara sasa 8GGgg -> launch time
        if "31313" in toks:
            j = toks.index("31313")
            if j + 2 < len(toks) and toks[j + 2].startswith("8") and toks[j + 2][1:].isdigit():
                out["launch_hhmm"] = toks[j + 2][1:]
    if not levels:
        return None
    mi = mission_of(rmk61) or {}
    out.update(mi)
    m = re.search(r"\bOB\s+(\d+)", rmk61)
    out["obnum"] = int(m.group(1)) if m else None
    out["rmk"] = rmk62
    r = rmk62
    for key, rx in (("rel", r"REL\s*(\d{4}[NS])\s*(\d{5}[EW])\s*(\d{6})"),
                    ("spg", r"SPG\s*(\d{4}[NS])\s*(\d{5}[EW])\s*(\d{6})"),
                    ("spl", r"SPL\s*(\d{4}[NS])\s*(\d{5}[EW])\s*(\d{4,6})")):
        m = re.search(rx, r)
        if m:
            pos = _pos(m.group(1) + m.group(2))
            out[key] = dict(la=pos[0], lo=pos[1], hms=m.group(3))
    m = re.search(r"MBL WND\s*(\d{5})", r)
    if m:
        d, f = _wind(m.group(1))
        if f is not None:
            out["mbl"] = dict(wd=d, ws=f)
    m = re.search(r"WL150\s*(\d{5})\s*(\d{3})", r)
    if m:
        d, f = _wind(m.group(1))
        if f is not None:
            out["wl150"] = dict(wd=d, ws=f, zm=int(m.group(2)))
    m = re.search(r"DLM WND\s*(\d{5})\s*(\d{3})\s*(\d{3})", r)
    if m:
        d, f = _wind(m.group(1))
        b, t = int(m.group(2)), int(m.group(3))
        if f is not None:
            out["dlm"] = dict(wd=d, ws=f, pbot=b + 1000 if b < 100 else b, ptop=t + 1000 if t < 100 else t)
    m = re.search(r"LST WND\s*(\d{3})", r)
    if m:
        v = int(m.group(1))
        out["lst_wnd_p"] = v + 1000 if v < 100 else v
    tags = [w for w in ("EYEWALL", "EYE", "CENTER", "CNTR", "RAINBAND", "SONDE FAILED", "NO SPLASH", "FAST FALL")
            if re.search(r"\b" + w + r"\b", r)]
    out["tags"] = tags
    # profile: sorted surface -> top, heights from hypsometric integration anchored at the splash level
    lv = _hypso(list(levels.values()))
    for L in lv:
        if L.get("std") and L.get("hhh") is not None:
            L["z_std"] = _std_height(int(L["p"]), L["hhh"], L.get("zh"))
    if not any(L.get("z") is not None for L in lv):     # no splash: anchor on the reported mandatory heights
        a = next((L for L in lv if L.get("z_std") is not None), None)
        if a is not None:
            a["z"] = a["z_std"]
            lv = _hypso(lv)
    # tie the integrated heights to the coded mandatory-level heights (the integration drifts where T is
    # missing near the top): offsets at the mandatory levels, interpolated in log p, held beyond the ends
    ties = [(math.log(L["p"]), L["z_std"] - L["zh"]) for L in lv
            if L.get("z_std") is not None and L.get("zh") is not None and abs(L["z_std"] - L["zh"]) < 400]
    if ties:
        ties.sort()
        for L in lv:
            if L.get("zh") is None or L.get("sfc"):
                continue
            x = math.log(L["p"])
            if x <= ties[0][0]:
                off = ties[0][1]
            elif x >= ties[-1][0]:
                off = ties[-1][1]
            else:
                j = next(k for k in range(1, len(ties)) if ties[k][0] >= x)
                (x0, o0), (x1, o1) = ties[j - 1], ties[j]
                off = o0 + (o1 - o0) * (x - x0) / (x1 - x0)
            # below the lowest tie the offset fades to zero at the splash level
            sfcL = next((S for S in lv if S.get("sfc")), None)
            if sfcL is not None and x > ties[-1][0]:
                xs = math.log(sfcL["p"])
                off *= (xs - x) / (xs - ties[-1][0]) if xs > ties[-1][0] else 0.0
            L["zh"] += off
    prof = []
    for L in lv:
        if L.get("t") is not None and not (-90 <= L["t"] <= 45):
            L["t"] = L["td"] = None
        if L.get("td") is not None and L.get("t") is not None and L["td"] > L["t"] + 0.5:
            L["td"] = None
        if L.get("ws") is not None and L["ws"] > 250:
            L["wd"] = L["ws"] = None
        z = L.get("zh")
        prof.append([L["p"], None if z is None else round(z), L.get("t"), L.get("td"), L.get("wd"), L.get("ws"),
                     1 if L.get("std") else 0])
    out["prof"] = prof            # [p hPa, z m, T C, Td C, dir, kt, mandatory?]
    out["sfc_p"] = sfc_p
    sfc = next((L for L in lv if L.get("sfc")), None)
    if sfc is not None and sfc.get("ws") is not None:
        out["sfc_wind"] = dict(wd=sfc["wd"], ws=sfc["ws"])
    # surface wind estimates (Franklin et al. 2003): 0.83 x WL150, 0.80 x MBL (0-500 m mean)
    est = {}
    if "wl150" in out:
        est["wl150"] = round(0.83 * out["wl150"]["ws"])
    if "mbl" in out:
        est["mbl"] = round(0.80 * out["mbl"]["ws"])
    out["sfc_est"] = est
    # timestamps: release / splash hhmmss on the day of the message
    out["rel_t"] = _stamp(file_dt, out.get("day"), (out.get("rel") or {}).get("hms"))
    sp = out.get("spg") or out.get("spl")
    out["spl_t"] = _stamp(file_dt, out.get("day"), sp.get("hms") if sp else None)
    if out["rel_t"] and out["spl_t"] and out["spl_t"] < out["rel_t"]:
        out["spl_t"] += 86400
    return out


def _stamp(file_dt, day, hms):
    if not hms:
        return None
    hms = (hms + "00")[:6] if len(hms) == 4 else hms
    try:
        h, m, s = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
    except ValueError:
        return None
    if h > 23 or m > 59 or s > 59:
        return None
    t = file_dt.replace(hour=h, minute=m, second=s, microsecond=0)
    if day and day != file_dt.day:          # message day differs (sonde released before 00 UTC)
        cand = t - timedelta(days=1)
        if cand.day == day:
            t = cand
    if t > file_dt + timedelta(hours=2):
        t -= timedelta(days=1)
    return int(t.timestamp())


# ──────────────────────────────────────────────── vortex data message (URNT12) ──
def _brg_rng(s):
    m = re.search(r"(\d{1,3}) deg (\d{1,3}) nm(?: (\d{2}:\d{2}:\d{2})Z)?", s or "")
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def destination(lat, lon, brg, nm):
    """Great-circle point `nm` nautical miles from (lat, lon) on bearing `brg` (deg true)."""
    R = 3440.065
    d = nm / R
    b = math.radians(brg)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return round(math.degrees(p2), 3), round((math.degrees(l2) + 540) % 360 - 180, 3)


def decode_vdm(text, file_dt):
    tt, cccc, ddhhmm, bbb = _hdr(text)
    if not tt or not tt.startswith(("URNT12", "URPN12", "URPA12")):
        return None
    f = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*([A-U])\.\s*(.*)$", ln)
        if m and m.group(1) not in f:
            f[m.group(1)] = m.group(2).strip()
    m = re.search(r"VORTEX DATA MESSAGE\s+([A-Z]{2}\d{6})", text)
    out = dict(kind="vdm", hdr=f"{tt} {cccc} {ddhhmm}", cor=bbb, sid=m.group(1) if m else None, raw=text.strip())
    m = re.match(r"(\d{2})/(\d{2}):(\d{2}):(\d{2})Z", f.get("A", ""))
    if not m:
        return None
    day, h, mi, s = map(int, m.groups())
    t = file_dt.replace(hour=h, minute=mi, second=s, microsecond=0)
    if day != t.day:
        t = t - timedelta(days=1) if (t - timedelta(days=1)).day == day else t
    if t > file_dt + timedelta(hours=2):
        t -= timedelta(days=1)
    out["t"] = int(t.timestamp())
    b = f.get("B", "")
    m = re.match(r"(\d+\.\d+) deg ([NS]) (\d+\.\d+) deg ([EW])", b)
    if m:
        la, lo = float(m.group(1)), float(m.group(3))
        out["la"], out["lo"] = (-la if m.group(2) == "S" else la), (-lo if m.group(4) == "W" else lo)
    else:
        m = re.match(r"(\d+) deg (\d+) min ([NS]) (\d+) deg (\d+) min ([EW])", b)     # pre-2018 layout
        if m:
            la = int(m.group(1)) + int(m.group(2)) / 60
            lo = int(m.group(4)) + int(m.group(5)) / 60
            out["la"], out["lo"] = round(-la if m.group(3) == "S" else la, 3), round(-lo if m.group(6) == "W" else lo, 3)
    if "la" not in out:
        return None
    m = re.match(r"(\d+) mb (\d+) m", f.get("C", ""))
    if m:
        out["lvl"], out["lvl_z"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d{3,4}) mb", f.get("D", ""))
    if m:
        out["mslp"] = int(m.group(1))
        out["mslp_extrap"] = "EXTRAP" in f.get("D", "").upper()
    m = re.match(r"(\d{3}) deg (\d+) kt", f.get("E", ""))
    if m:
        out["ctr_sfc_wind"] = dict(wd=int(m.group(1)), ws=int(m.group(2)))
    out["eye"] = f.get("F") if f.get("F") not in (None, "NA") else None
    out["eye_shape"] = f.get("G") if f.get("G") not in (None, "NA") else None
    for key, val, loc in (("in_sfc", "H", "I"), ("out_sfc", "L", "M")):
        m = re.match(r"(\d+) kt", f.get(val, ""))
        br = _brg_rng(f.get(loc))
        if m:
            d = dict(ws=int(m.group(1)))
            if br:
                d.update(brg=br[0], rng=br[1], hms=br[2])
                d["la"], d["lo"] = destination(out["la"], out["lo"], br[0], br[1])
            out[key] = d
    for key, val, loc in (("in_fl", "J", "K"), ("out_fl", "N", "O")):
        m = re.match(r"(\d{3}) deg (\d+) kt", f.get(val, ""))
        br = _brg_rng(f.get(loc))
        if m:
            d = dict(wd=int(m.group(1)), ws=int(m.group(2)))
            if br:
                d.update(brg=br[0], rng=br[1], hms=br[2])
                d["la"], d["lo"] = destination(out["la"], out["lo"], br[0], br[1])
            out[key] = d
    for key, val in (("t_out", "P"), ("t_in", "Q")):
        m = re.match(r"(-?\d+) C / (\d+|NA)", f.get(val, ""))
        if m:
            out[key] = dict(t=int(m.group(1)), z=None if m.group(2) == "NA" else int(m.group(2)))
    m = re.match(r"(-?\d+|NA) C / (-?\d+|NA)", f.get("R", ""))
    if m:
        out["td_in"] = None if m.group(1) == "NA" else int(m.group(1))
        out["sst"] = None if m.group(2) == "NA" else int(m.group(2))
    out["fix_by"] = f.get("S")
    out["acc"] = f.get("T")
    mi = mission_of(f.get("U", "")) or {}
    out.update(mi)
    m = re.search(r"\bOB\s+(\d+)", f.get("U", ""))
    out["obnum"] = int(m.group(1)) if m else None
    # free-text remarks after U. (MAX FL WIND, SLP EXTRAP FROM ..., etc.)
    tail = text.split(f.get("U", "@@"), 1)[-1] if f.get("U") else ""
    rm = " ".join(ln.strip() for ln in tail.splitlines() if ln.strip() and ln.strip() not in (";", "$$")).strip()
    out["rmk"] = rm
    m = re.search(r"MAX FL WIND (\d+) KT (\d{3}) / (\d+) NM (\d{2}:\d{2}:\d{2})Z", rm)
    if m:
        d = dict(ws=int(m.group(1)), brg=int(m.group(2)), rng=int(m.group(3)), hms=m.group(4))
        d["la"], d["lo"] = destination(out["la"], out["lo"], d["brg"], d["rng"])
        out["max_fl"] = d
    return out


# ────────────────────────────────────────────────────────────── RECCO (URNT11) ──
def decode_recco(text, file_dt):
    """URNT11/URNT10 RECCO, section one only: 9XXX9 GGggi YQLaLaLa LoLoLoBf hhhdt ddfff TTTdTdw /jHHH."""
    tt, cccc, ddhhmm, bbb = _hdr(text)
    if not tt or not tt.startswith(("URNT11", "URNT10", "URPN11", "URPN10")):
        return None
    toks = text.split()
    try:
        k = next(i for i, g in enumerate(toks) if re.fullmatch(r"9\d{3}9", g))
    except StopIteration:
        return None
    g = toks[k:k + 8]
    if len(g) < 7:
        return None
    try:
        hh, mm = int(g[1][:2]), int(g[1][2:4])
        q, la = g[2][1], int(g[2][2:]) / 10.0
        lo = int(g[3][:3]) / 10.0
    except ValueError:
        return None
    # octant Q: 0 = 0-90W, 1 = 90-180W, 2 = 180-90E, 3 = 90-0E (north); 5-8 the same in the south.
    # LoLoLo is tenths of a degree with the hundreds digit dropped beyond 100 deg.
    if q not in "01235678":
        return None
    if q in "1267" and lo < 90:
        lo += 100.0
    if q in "0156":
        lo = -lo
    if q in "5678":
        la = -la
    t = file_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t > file_dt + timedelta(hours=2):
        t -= timedelta(days=1)
    out = dict(kind="recco", hdr=f"{tt} {cccc} {ddhhmm}", t=int(t.timestamp()), la=la, lo=round(lo, 1), raw=text.strip())
    hv = _int(g[4][:3])
    if hv is not None:
        out["alt_m"] = hv * 10
    wd, ws = _wind(g[5])
    if ws is not None:
        out["wd"], out["ws"] = wd, ws
    tv = _int(g[6][:2])
    if tv is not None:
        out["t_"] = tv if tv < 50 else -(tv - 50)
    dv = _int(g[6][2:4])
    if dv is not None:
        out["td"] = dv if dv < 50 else -(dv - 50)
    m = re.search(r"RMK\s+(.*)", text, re.S)
    rm = " ".join(m.group(1).split()).strip(" ;") if m else ""
    out["rmk"] = rm
    out.update(mission_of(rm) or {})
    return out


# ─────────────────────────────────────────────────────── plan of the day ──
def decode_pod(text, file_dt):
    if "REPRPD" not in text and "PLAN OF THE DAY" not in text:
        return None
    m = re.search(r"VALID\s+(\d{2})/(\d{4})Z TO (\d{2})/(\d{4})Z", text)
    body = text.strip()
    body = re.sub(r"^\s*\d{3}\s*\n", "", body)
    return dict(kind="pod", t=int(file_dt.timestamp()), valid=m.group(0) if m else None,
                text=body.replace("\r", ""))


DECODERS = {"AHONT1": decode_hdob, "AHOPN1": decode_hdob, "REPNT3": decode_drop, "REPNT2": decode_vdm,
            "REPNT1": decode_recco, "REPNT0": decode_recco, "REPRPD": decode_pod}


def file_time(name):
    """'AHONT1-KNHC.202610071257.txt' / 'REPRPD.202610061722.txt' -> aware datetime."""
    m = re.search(r"\.(\d{12})\.txt$", name)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC) if m else None
