#!/usr/bin/env python3
"""Publish the local radar archive into the public site (unlisted).

Downscales frames to webp, writes one index.json describing every
radar/date/product available, and leaves the page to read it.

    python scripts/sst/radar_publish.py [--max-days 6]
"""
import argparse, json, shutil, os
from pathlib import Path
from PIL import Image

ARCH=Path(os.environ.get("RADAR_ARCH") or
          Path.home()/"colombia_hydro"/"radar")
SITE=Path(os.environ.get("RADAR_SITE") or
          Path(__file__).resolve().parents[2]/"radar")
SITES={"Barrancabermeja":(6.933,-73.763),"Munchique":(2.845,-76.995),
       "santa_elena":(6.199,-75.500),"Guaviare":(2.573,-72.639),
       "Carimagua":(4.560,-71.336),"Tablazo":(5.000,-74.000),
       "Bogota":(4.658,-74.094),"San_Andres":(12.583,-81.700)}
NICE={"ppi":"Reflectivity","qpe":"Rain rate","vel":"Radial velocity"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--max-days",type=int,default=8)
    ap.add_argument("--width",type=int,default=1400)
    a=ap.parse_args()
    idx={"radars":{},"sites":SITES,"products":NICE}
    for rad in sorted(p.name for p in ARCH.iterdir() if p.is_dir()):
        days=sorted((p.name for p in (ARCH/rad).iterdir() if p.is_dir()),reverse=True)[:a.max_days]
        for d in days:
            for man in sorted((ARCH/rad/d).glob("*_manifest.json")):
                prod=man.name.replace("_manifest.json","")
                m=json.loads(man.read_text())
                if not m.get("frames"): continue
                dst=SITE/"frames"/rad/d/prod; dst.mkdir(parents=True,exist_ok=True)
                for f in dst.glob("*.webp"): f.unlink()
                for fr in m["frames"]:
                    src=ARCH/rad/d/prod/fr["file"]
                    if not src.exists(): continue
                    im=Image.open(src).convert("RGB"); im.thumbnail((a.width,a.width))
                    im.save(dst/fr["file"],"WEBP",quality=82,method=5)
                (dst/"manifest.json").write_text(json.dumps(m))
                print(f"  {rad} {d} {prod}: {m['n_frames']} frames")
    # Build the index from what is PUBLISHED, not from the local archive. In CI
    # the local archive holds only the day just ingested, so indexing it would
    # silently unpublish every date already committed to the site.
    fr=SITE/"frames"
    if fr.exists():
        for rad in sorted(p.name for p in fr.iterdir() if p.is_dir()):
            for d in sorted((p.name for p in (fr/rad).iterdir() if p.is_dir()),reverse=True):
                prods=[p.name for p in (fr/rad/d).iterdir()
                       if p.is_dir() and (p/"manifest.json").exists()]
                if prods: idx["radars"].setdefault(rad,{})[d]=sorted(prods)
    (SITE/"index.json").write_text(json.dumps(idx,indent=1))
    mb=sum(f.stat().st_size for f in SITE.rglob("*.webp"))/1048576
    print(f"\nindex.json written · {mb:.1f} MB of frames")

if __name__=="__main__": main()
