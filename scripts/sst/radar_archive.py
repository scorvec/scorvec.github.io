#!/usr/bin/env python3
"""Report and control the size of the published radar archive.

Frame archives grow fast: one radar-day of 15-min reflectivity is ~7 MB, and
the cost is multiplicative in radars x products x days. A git repo is a bad
place to discover that late, so this reports growth and prunes on a policy.

    python scripts/sst/radar_archive.py                 # report only
    python scripts/sst/radar_archive.py --prune         # apply retention
    python scripts/sst/radar_archive.py --prune --keep-days 14

Retention is two weeks: anything older can be re-derived from the IDEAM
bucket on demand, so there is no reason to carry it in the repo.

Retention: keep the most recent --keep-days per radar, always keep --pin
dates (case studies), delete the rest from the published tree. The local
archive under ~/colombia_hydro/radar is left alone.
"""
import argparse, json, shutil, os
from collections import defaultdict
from pathlib import Path

SITE=Path(os.environ.get("RADAR_SITE") or
          Path(__file__).resolve().parents[2]/"radar")
LOCAL=Path.home()/"colombia_hydro"/"radar"
GIT_SOFT_MB=900          # GitHub warns past ~1 GB; stay clear of it

def mb(p): return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/1048576

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prune",action="store_true")
    ap.add_argument("--keep-days",type=int,default=14)
    ap.add_argument("--pin",nargs="*",default=["20260816","20260817","20260818"])
    a=ap.parse_args()
    fr=SITE/"frames"
    if not fr.exists(): raise SystemExit("no published frames yet")
    rows=[]; per_rad=defaultdict(list)
    for rad in sorted(p.name for p in fr.iterdir() if p.is_dir()):
        for d in sorted((p.name for p in (fr/rad).iterdir() if p.is_dir()),reverse=True):
            size=mb(fr/rad/d)
            prods=sorted(p.name for p in (fr/rad/d).iterdir() if p.is_dir())
            n=sum(len(list((fr/rad/d/p).glob("*.webp"))) for p in prods)
            rows.append((rad,d,size,n,prods)); per_rad[rad].append(d)
    tot=sum(r[2] for r in rows)
    print(f"{'radar':20s}{'date':10s}{'MB':>8s}{'frames':>8s}  products")
    for rad,d,size,n,prods in rows:
        print(f"{rad:20s}{d:10s}{size:8.1f}{n:8d}  {','.join(prods)}")
    days=len({r[1] for r in rows}); rads=len(per_rad)
    per_day=tot/max(days,1)
    print(f"\ntotal {tot:.1f} MB across {days} day(s), {rads} radar(s)"
          f"  ->  {per_day:.1f} MB per day published")
    print(f"projection at this rate: {per_day*30:.0f} MB/month, "
          f"{per_day*365/1024:.1f} GB/year")
    hdr=mb(SITE)-tot
    print(f"repo headroom: {GIT_SOFT_MB} MB soft budget, "
          f"{(GIT_SOFT_MB-tot-hdr)/max(per_day,.01):.0f} days at this rate")
    if not a.prune:
        print("\n(report only; pass --prune to apply retention)")
        return
    killed=0.0
    for rad,ds in per_rad.items():
        keep=set(ds[:a.keep_days]) | set(a.pin)
        for d in ds:
            if d in keep: continue
            p=fr/rad/d; killed+=mb(p); shutil.rmtree(p)
            print(f"  pruned {rad}/{d}")
    # index.json must not advertise what was removed
    ix=SITE/"index.json"
    if ix.exists():
        j=json.loads(ix.read_text())
        for rad in list(j.get("radars",{})):
            for d in list(j["radars"][rad]):
                if not (fr/rad/d).exists(): del j["radars"][rad][d]
            if not j["radars"][rad]: del j["radars"][rad]
        ix.write_text(json.dumps(j,indent=1))
    print(f"\npruned {killed:.1f} MB; published archive now {mb(SITE):.1f} MB")

if __name__=="__main__": main()
