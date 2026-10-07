#!/usr/bin/env python3
"""Offline checks of the recon decoders + mirror merge against the sample bulletins in samples/.

    python scripts/recon/test_recon.py

Values asserted here were read off the raw messages by hand (see the comments).
"""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recon_decode as D  # noqa: E402
import recon_mirror as M  # noqa: E402

S = HERE / "samples"
UTC = timezone.utc


def dec(fname):
    name = fname.replace("2024_", "")
    prod = name.split("-")[0].split(".")[0]
    return D.DECODERS[prod]((S / fname).read_text(), D.file_time(name))


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp())


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# HDOB: "124800 2149N 09436W 8428 01531 0042 +180 +172 342027 028 /// /// 03"
h = dec("AHONT1-KNHC.202610071257.txt")
o = h["obs"][0]
check(h["aircraft"] == "AF304" and h["mission"] == "0209A" and h["storm_no"] == 9 and h["obnum"] == 20, h)
check(o["t"] == ts("2026-10-07 12:48:00"), o)
check(abs(o["la"] - (21 + 49 / 60)) < 1e-3 and abs(o["lo"] + (94 + 36 / 60)) < 1e-3, o)
check(o["p"] == 842.8 and o["z"] == 1531 and o["x"] == 1004.2, o)
check(o["t_"] == 18.0 and o["td"] == 17.2 and o["wd"] == 342 and o["ws"] == 27 and o["wp"] == 28, o)
check("sfmr" not in o and o["qc"] == "03", o)
# "124900 ... +175 //// 348026 027 /// /// 05": flag 5 = T/Td and SFMR questionable -> T dropped
o = h["obs"][2]
check("t_" not in o and "td" not in o and o["ws"] == 26, o)

# G-IV above 550 hPa: XXXX is a D-value, "002230 2800N 08546W 1484 14392 0720 -710 ////"
g = dec("AHONT1-KWBC.202610070032.txt")["obs"][0]
check(g["p"] == 148.4 and g["z"] == 14392 and g["dv"] == 720 and "x" not in g, g)
check("t_" not in g, g)            # flag 05: temperature questionable

# Milton AF302 "233230 2217N 09105W 6963 03073 9955 +091 //// 018046 047 048 005 01"
m = dec("2024_AHONT1-KNHC.202410072340.txt")["obs"][0]
check(m["x"] == 995.5 and m["sfmr"] == 48 and m["rr"] == 5 and m["wd"] == 18 and m["ws"] == 46, m)
check("t_" not in m, m)            # flag 01: T/Td questionable

# VDM Isaias: A. 07/12:33:50Z  B. 22.40 N 093.97 W  D. 1004 mb  J. 122 deg 43 kt  K. 061 deg 32 nm 12:23:30Z
v = dec("REPNT2-KNHC.202610071255.txt")
check(v["t"] == ts("2026-10-07 12:33:50") and v["la"] == 22.40 and v["lo"] == -93.97 and v["mslp"] == 1004, v)
check(v["in_fl"]["ws"] == 43 and v["in_fl"]["brg"] == 61 and v["in_fl"]["rng"] == 32, v)
check(v["lvl"] == 850 and v["lvl_z"] == 1446 and v["sid"] == "AL092026", v)
# Milton: F. OPEN S, G. CO5-24, H. 142 kt
v = dec("2024_REPNT2-KNHC.202410080004.txt")
check(v["mslp"] == 906 and v["eye"] == "OPEN S" and v["eye_shape"] == "CO5-24" and v["in_sfc"]["ws"] == 142, v)
check(v["max_fl"]["ws"] == 138, v)

# TEMP DROP Isaias AF304 OB 06: 99005 28027 36031 / 00044 27430 35027 / 92731 22008 35528 / 85465 /////
d = dec("REPNT3-KNHC.202610071259.txt")
pr = {round(r[0]): r for r in d["prof"]}
check(d["sfc_p"] == 1005 and pr[1005][2] == 28.0 and pr[1005][3] == 25.3, pr[1005])      # DD 27 -> 2.7
check(d["sfc_wind"] == {"wd": 360, "ws": 31}, d["sfc_wind"])
check(pr[1000][1] == 44 and pr[1000][2] == 27.4 and pr[1000][3] == 24.4 and pr[1000][4:6] == [350, 27], pr[1000])
check(pr[925][1] == 731 and pr[925][2] == 22.0 and pr[925][3] == 21.2 and pr[925][4:6] == [355, 28], pr[925])
check(pr[850][1] == 1465 and pr[850][2] is None and pr[850][5] is None, pr[850])   # Id=9: no 850 wind group
check(d["rel"] == {"la": 21.71, "lo": -94.70, "hms": "125043"} and d["spg"]["hms"] == "125231", d)
check(d["wl150"] == {"wd": 350, "ws": 26, "zm": 81} and d["sfc_est"]["wl150"] == 22, d)   # 0.83 x 26
check(d["dlm"]["pbot"] == 1004 and d["dlm"]["ptop"] == 856, d)
check(d["rel_t"] == ts("2026-10-07 12:50:43") and d["spl_t"] == ts("2026-10-07 12:52:31"), d)
# G-IV sonde: mandatory heights match the coded ones (500 hPa 586 dam, 200 hPa 247 dam, 150 hPa 426 dam)
d = dec("REPNT3-KWBC.202610070029.txt")
pr = {round(r[0]): r for r in d["prof"]}
check(pr[500][1] == 5860 and pr[200][1] == 12470 and pr[150][1] == 14260, (pr[500], pr[200], pr[150]))
check(pr[500][2] == -5.1 and pr[500][3] == -5.3, pr[500])       # 05102: odd tenth -> negative, DD 02
check(pr[150][4:6] == [285, 33], pr[150])                        # 28533 -> 285 deg 33 kt
check(d["rel_t"] == ts("2026-10-07 00:11:49") and d["spl_t"] == ts("2026-10-07 00:26:04"), d)
# Milton KNHC sonde: 1000 hPa group 00501 = -1 m (sfc 1000 hPa); 21212 66709 01034 = 709 hPa 010/34
d = dec("2024_REPNT3-KNHC.202410072324.txt")
pr = {round(r[0]): r for r in d["prof"]}
check(d["sfc_p"] == 1000 and pr[709][4:6] == [10, 34], pr.get(709))

# RECCO "97779 12500 41217 94708 15300 35031 17//8 /2465"
r = dec("REPNT1-KNHC.202610071259.txt")
check(r["la"] == 21.7 and r["lo"] == -94.7 and r["alt_m"] == 1530 and r["ws"] == 31 and r["t_"] == 17, r)

# mirror merge + manifest, offline
with tempfile.TemporaryDirectory() as tmp:
    mx = M.Mirror(tmp, None)
    for f in sorted(S.glob("*.txt")):
        name = f.name.replace("2024_", "")
        prod = name.split("-")[0].split(".")[0]
        mx.ingest(int(name.split(".")[1][:4]), prod, [(name, f.read_text())])
    # a b-deck for the proximity assignment of the G-IV surveillance mission
    btk = (HERE / "samples" / "bal092026.dat")
    if btk.exists():
        mx.write_storm("AL092026", M.parse_atcf(btk.read_text()))
    mx.write_missions()
    mx.build_manifest([{"id": "AL092026", "name": "Isaias"}], set())
    man = json.loads((Path(tmp) / "manifest.json").read_text())
    ms = man["missions"]
    check(ms["2026_AF304_0209A"]["storm"] == "AL092026", ms["2026_AF304_0209A"])
    check(ms["2024_AF302_1114A"]["storm"] == "AL142024", ms["2024_AF302_1114A"])
    if btk.exists():
        check(ms["2026_NOAA9_01BBA"]["storm"] == "AL092026", ms["2026_NOAA9_01BBA"])
    check(ms["2026_AF304_0209A"]["n_drop"] == 1 and ms["2026_AF304_0209A"]["n_vdm"] == 1, ms["2026_AF304_0209A"])
    check(len(man["pods"]) == 1, man.get("pods"))
    print(json.dumps({k: {kk: v[kk] for kk in ("storm", "how", "n_hdob", "n_drop", "max_fl", "min_x")}
                      for k, v in ms.items()}, indent=1))
print("all recon checks passed")
