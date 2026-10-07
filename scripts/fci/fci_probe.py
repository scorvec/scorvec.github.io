#!/usr/bin/env python3
"""Temporary source probe for the FCI extension (runs in Actions only): FRED ids + copyright status, the Fed's EBP csv,
Shiller's ie_data.xls, the FFIEC CDR bulk Call Report download and the FDIC API variable list."""
import io, re, sys, time, urllib.request, zipfile, json, http.cookiejar, urllib.parse

UA = {"User-Agent": "scorvec.com credit conditions index (+https://scorvec.com/fci.html)", "Accept": "*/*"}


def get(url, data=None, headers=None, timeout=60, opener=None):
    h = dict(UA); h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with (opener.open if opener else urllib.request.urlopen)(req, timeout=timeout) as r:
        return r.read(), r.headers


def fred():
    ids = ["DFII10", "THREEFYTP10", "WRESBAL", "WLRRAL", "RRPONTSYD", "SOFR", "IORB", "IOER", "TLAACBW027SBOG", "GDP",
           "DRTSCREL", "SUBLPDRCSC", "SUBLPDRCSN", "SUBLPDRCSM", "DRTSCLCC", "DRTSCLNG", "DRBLACBS", "DRCCLACBS",
           "DRCRELEXFACBS", "BAMLH0A0HYM2", "BAMLH0A3HYC", "EXPINF10YR", "REAINTRATREARAT10Y", "GS10", "DGS10", "TOTRESNS"]
    for sid in ids:
        try:
            b, _ = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
            lines = [l for l in b.decode().splitlines()[1:] if l and not l.endswith(",") and not l.endswith(".")]
            rng = f"{lines[0]} .. {lines[-1]} n={len(lines)}" if lines else "EMPTY"
        except Exception as e:
            rng = f"csv ERR {e}"
        try:
            b, _ = get(f"https://fred.stlouisfed.org/series/{sid}")
            t = b.decode("utf-8", "replace")
            st = sorted(set(re.findall(r"(Public Domain[^<\"]{0,60}|Copyrighted: [A-Za-z\- ]{0,40})", t)))
            title = re.search(r"<title>([^<]+)</title>", t)
            title = title.group(1).strip() if title else "?"
        except Exception as e:
            st, title = [f"page ERR {e}"], "?"
        print(f"FRED {sid}: {rng} | {st} | {title[:110]}", flush=True)
        time.sleep(0.6)


def ebp():
    b, _ = get("https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv")
    L = b.decode().splitlines()
    print("EBP rows", len(L)); print("\n".join(L[:4])); print("..."); print("\n".join(L[-4:]))


def shiller():
    b, _ = get("https://shillerdata.com/")
    links = re.findall(r'href="([^"]*ie_data\.xls[^"]*)"', b.decode("utf-8", "replace"))
    print("Shiller links", links)
    u = links[0]; u = "https:" + u if u.startswith("//") else u
    x, h = get(u, timeout=120)
    print("xls bytes", len(x), h.get("Last-Modified"))
    import pandas as pd
    xl = pd.ExcelFile(io.BytesIO(x))
    print("sheets", xl.sheet_names)
    d = pd.read_excel(xl, "Data", header=None)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(d.iloc[:10, :16].to_string()); print(d.iloc[-14:, :16].to_string())


def fdic():
    try:
        b, _ = get("https://api.fdic.gov/banks/docs/risview_properties.yaml", timeout=90)
        t = b.decode("utf-8", "replace")
        L = t.splitlines()
        print("FDIC yaml lines", len(L))
        for i, l in enumerate(L):
            if re.search(r"nondepository|non-depository|intermediar|private equity", l, re.I):
                print("\n".join(L[max(0, i - 4):i + 2])); print("---")
    except Exception as e:
        print("FDIC ERR", e)


def ffiec():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    url = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
    b, h = get(url, opener=op, headers={"User-Agent": "Mozilla/5.0 (compatible; scorvec.com FCI)"})
    t = b.decode("utf-8", "replace")
    print("FFIEC page bytes", len(t))
    for sel in re.findall(r'<select[^>]*name="([^"]+)"[^>]*>(.*?)</select>', t, re.S):
        opts = re.findall(r'<option[^>]*value="([^"]*)"[^>]*>([^<]*)<', sel[1])
        print("SELECT", sel[0], opts[:12], "... n", len(opts))
    hid = dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', t))
    print("hidden", {k: len(v) for k, v in hid.items()})
    for inp in re.findall(r'<input[^>]*>', t):
        if "hidden" not in inp: print("INPUT", inp[:200])
    return t, hid, op


def ffiec_download(t, hid, op):
    url = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
    lb = re.search(r'<select[^>]*name="([^"]*ListBox1[^"]*)"', t).group(1)
    dd = re.search(r'<select[^>]*name="([^"]*Dates[^"]*)"', t)
    form = dict(hid)
    form[lb] = "ReportingSeriesSinglePeriod"
    form["__EVENTTARGET"] = lb; form["__EVENTARGUMENT"] = ""
    b, _ = get(url, data=urllib.parse.urlencode(form).encode(), opener=op,
               headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0 (compatible; scorvec.com FCI)"})
    t2 = b.decode("utf-8", "replace")
    hid2 = dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', t2))
    sel = re.search(r'<select[^>]*name="([^"]*Dates[^"]*)"[^>]*>(.*?)</select>', t2, re.S)
    opts = re.findall(r'<option[^>]*value="([^"]*)"[^>]*>([^<]*)<', sel.group(2))
    print("dates", opts[:10])
    radios = re.findall(r'<input[^>]*type="radio"[^>]*>', t2); print("radios", radios)
    btns = re.findall(r'<input[^>]*type="submit"[^>]*>', t2); print("buttons", btns)
    form = dict(hid2); form[lb] = "ReportingSeriesSinglePeriod"; form[sel.group(1)] = opts[0][0]
    for r in radios:
        if "TSV" in r or "Tab" in r:
            nm = re.search(r'name="([^"]+)"', r).group(1); vl = re.search(r'value="([^"]+)"', r).group(1); form[nm] = vl
    for bt in btns:
        if "Download" in bt:
            nm = re.search(r'name="([^"]+)"', bt).group(1); vl = re.search(r'value="([^"]+)"', bt).group(1); form[nm] = vl
    b, h = get(url, data=urllib.parse.urlencode(form).encode(), opener=op, timeout=300,
               headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0 (compatible; scorvec.com FCI)"})
    print("download bytes", len(b), h.get("Content-Type"), h.get("Content-Disposition"))
    if not b.startswith(b"PK"):
        tt = b.decode("utf-8", "replace")
        print("FORM SENT", {k: (v if len(v) < 60 else len(v)) for k, v in form.items()})
        for m in re.findall(r'<span[^>]*(?:Label|lbl|Error|Message)[^>]*>([^<]{3,300})</span>', tt): print("MSG", m)
        sel2 = re.search(r'<select[^>]*name="([^"]*Dates[^"]*)"[^>]*>(.*?)</select>', tt, re.S)
        print("selected after", re.findall(r'<option[^>]*selected[^>]*>([^<]*)<', tt))
        print(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", tt))[:3000])
        return
    z = zipfile.ZipFile(io.BytesIO(b))
    names = z.namelist(); print("zip n", len(names)); print([n for n in names if "RCCI" in n or "RCC" in n][:10])
    for n in names:
        if "RCCI" in n and "II" not in n.split("RCCI")[1][:2]:
            with z.open(n) as f:
                head = [next(f).decode("latin-1") for _ in range(2)]
            codes = head[0].rstrip("\n").split("\t"); desc = head[1].rstrip("\n").split("\t")
            print("file", n, "cols", len(codes))
            for c, dsc in zip(codes, desc):
                if re.search(r"nondepository|intermediar|private equity|NDFI", dsc, re.I) or c.strip('"') in ("RCON1563", "RCFD1563", "RCONJ454", "RCFDJ454"):
                    print("  ", c, "|", dsc)


if __name__ == "__main__":
    for f in (fdic,):
        try:
            print(f"===== {f.__name__}", flush=True); f()
        except Exception as e:
            print(f"{f.__name__} FAILED {e!r}", flush=True)
    try:
        print("===== ffiec", flush=True)
        t, hid, op = ffiec(); ffiec_download(t, hid, op)
    except Exception as e:
        import traceback; traceback.print_exc()
