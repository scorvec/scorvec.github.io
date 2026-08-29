#!/usr/bin/env python3
"""IDEAM radar -> web products for the Colombia radar page.

The bucket (s3-radaresideam) carries ONLY raw SIGMET volumes: no QPE, no
CAPPI, no composite, no level-3 of any kind. Everything here is derived.

Scan strategy (Barrancabermeja, and Munchique similarly): one volume is
split across FOUR files on a ~5 min cycle —
    1.3            (long-PRF surveillance, 300 km)
    1.5 2.4 3.1 5.1
    6.4 8.0
    10 12.5 15
so a "frame" must select ONE task, or merge all four. Mixing them silently
produces a loop where the beam tilts rather than the weather moving.

Products
    ppi   lowest surveillance sweep (1.3 deg) reflectivity
    qpe   rain rate from the same sweep, polarimetric where possible
    vel   radial velocity (1.3 deg)

    python radar_ingest.py --date 20260827 --radar Barrancabermeja
    python radar_ingest.py --recent 3            # poll the last N days

Archive: ~/colombia_hydro/radar/{radar}/{date}/{product}/NNN.webp + manifest
"""
from __future__ import annotations
import argparse, json, urllib.parse, urllib.request, warnings, os
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

B="https://s3.amazonaws.com/s3-radaresideam"
NS={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
ARCH=Path(os.environ.get("RADAR_ARCH") or
          Path.home()/"colombia_hydro"/"radar")
# The basin overlay is optional: it lives in the private ~/wrf tree, so a CI
# runner has no copy and those frames come out without the yellow outlines.
# Say so loudly — a silent skip makes CI and laptop frames diverge inside the
# same loop, which reads as a rendering bug rather than a missing input.
_warned=set()
def _no_basins(path):
    if "x" in _warned: return
    _warned.add("x")
    print(f"    note: no basin overlay ({path} absent) — frames omit region outlines")

SITES={"Barrancabermeja":(6.933,-73.763),"Munchique":(2.845,-76.995),
       "santa_elena":(6.199,-75.500),"Guaviare":(2.573,-72.639),
       "Carimagua":(4.560,-71.336),"Tablazo":(5.000,-74.000)}

def s3_keys(prefix, with_size=False):
    out=[]; tok=None
    while True:
        u=f"{B}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if tok: u+="&continuation-token="+urllib.parse.quote(tok)
        x=ET.fromstring(urllib.request.urlopen(u,timeout=120).read())
        for c in x.findall(".//s:Contents",NS):
            k=c.find("s:Key",NS).text; sz=int(c.find("s:Size",NS).text)
            out.append((k,sz) if with_size else k)
        t=x.find(".//s:NextContinuationToken",NS)
        if t is None: return out
        tok=t.text

def sites_on(d):
    p=f"l2_data/{d[:4]}/{d[4:6]}/{d[6:8]}/"
    u=f"{B}/?list-type=2&prefix={p}&delimiter=/"
    x=ET.fromstring(urllib.request.urlopen(u,timeout=90).read())
    return [e.text[len(p):].strip("/") for e in x.findall(".//s:CommonPrefixes/s:Prefix",NS)]


def qc(sw, fld):
    """Dual-pol gate filter. Raw sweeps are ~80% clutter/biota at range."""
    d = sw.fields[fld]["data"]
    if "cross_correlation_ratio" in sw.fields:
        d = np.ma.masked_where(np.ma.filled(sw.fields["cross_correlation_ratio"]["data"],0) < 0.85, d)
    if "normalized_coherent_power" in sw.fields:
        d = np.ma.masked_where(np.ma.filled(sw.fields["normalized_coherent_power"]["data"],0) < 0.30, d)
    from scipy.ndimage import convolve
    ok = (~np.ma.getmaskarray(d)).astype(float)
    d = np.ma.masked_where(convolve(ok, np.ones((3,3))/9.0, mode="nearest") < 0.45, d)
    return d


def rain_rate(sw):
    """Rain rate (mm/h). Polarimetric where KDP is trustworthy, else Z-R.

    Marshall-Palmer Z=200R^1.6 underestimates convective rain; KDP-based
    rates are far less sensitive to hail contamination and attenuation, so
    use R(KDP) where KDP is well determined and blend to R(Z) elsewhere.
    """
    z = qc(sw, "reflectivity")
    R = np.ma.power(np.ma.power(10.0, z/10.0)/200.0, 1/1.6)          # R(Z)
    if "specific_differential_phase" in sw.fields:
        kdp = sw.fields["specific_differential_phase"]["data"]
        kdp = np.ma.masked_where(np.ma.getmaskarray(z), kdp)
        Rk = 44.0*np.ma.power(np.ma.abs(kdp), 0.822)*np.sign(np.ma.filled(kdp,0))
        use = np.ma.filled(z,0) > 40                                  # heavy echo only
        R = np.ma.where(use & (np.ma.filled(Rk,0) > 0), Rk, R)
    return np.ma.masked_less(R, 0.2)


PRODUCTS = {
 "ppi": dict(field="reflectivity", lv=np.arange(5,80,5.0), title="Reflectivity",
             lab="dBZ", elev=1.3),
 "vel": dict(field="velocity", lv=np.arange(-16,16.1,1.0), title="Radial velocity",
             lab="m/s", elev=1.3),
 "qpe": dict(field="__rain", lv=np.array([0.2,0.5,1,2,3,5,8,12,18,25,35,50,70,100,150]),
             title="Rain rate", lab="mm/h", elev=1.3),
}


def render(radar, dstr, product, stride=3, dpi=105, keep=None, hh0=None, hh1=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    import matplotlib.patheffects as pe
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    import pyart
    P=PRODUCTS[product]
    # Standard NWS reflectivity scale: discrete 5 dBZ bins with the colours
    # anchored to real dBZ values, not to arbitrary positions along a ramp.
    # Getting this wrong (fractional stops on a 5-70 range) pushes yellow up
    # to ~48 dBZ and makes ordinary convection look like drizzle.
    NWS_BOUNDS=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
    NWS_COLORS=["#04e9e7","#019ff4","#0300f4","#02fd02","#01c501","#008e00",
                "#fdf802","#e5bc00","#fd9500","#fd0000","#d40000","#bc0000",
                "#f800fd","#9854c6"]
    from matplotlib.colors import ListedColormap
    NWS=ListedColormap(NWS_COLORS)
    cmap = plt.get_cmap("RdBu_r") if product=="vel" else NWS
    if product=="ppi":
        P=dict(P); P["lv"]=np.array(NWS_BOUNDS,float)
    # One volume is split across four files on a ~5 min cycle and only one
    # carries the 1.3 deg sweep. File SIZE separates the four tasks, but the
    # absolute sizes drift with echo content, so a fixed threshold fails on
    # other days. Instead: cluster the day's sizes into the four families,
    # then download ONE file per family and read its actual elevation. Four
    # downloads identifies the task; the rest are selected by membership.
    ks=s3_keys(f"l2_data/{dstr[:4]}/{dstr[4:6]}/{dstr[6:8]}/{radar}/", with_size=True)
    if not ks: return 0
    ks.sort(key=lambda t: t[0])
    sz=np.array([s for _, s in ks], float)
    # Equal quartiles are NOT safe: the four tasks do not produce equal file
    # counts, so a quartile boundary can cut through a family and the 1.3 deg
    # task disappears entirely (seen on 2026-08-18). Use finer bins and probe
    # each, then take every bin that reports the elevation we want.
    NB=12
    order=np.argsort(sz); fam=np.empty(len(ks),int)
    for i,idx in enumerate(np.array_split(order,NB)): fam[idx]=i
    cache=Path.home()/"colombia_hydro"/"raw"/"radar"/dstr/radar
    cache.mkdir(parents=True, exist_ok=True)
    import pyart as _pa
    # Do not hardcode an elevation: Barrancabermeja's lowest sweep is 1.3 deg,
    # Munchique's is 1.5. Probe every bin, then take whichever bins carry the
    # LOWEST elevation present — that is the surveillance scan on any radar.
    elev_of={}
    for f in range(NB):
        member=[i for i in range(len(ks)) if fam[i]==f]
        if not member: continue
        k=ks[member[len(member)//2]][0]; probe=cache/("probe_"+k.split("/")[-1])
        try:
            urllib.request.urlretrieve(f"{B}/{k}", probe)
            e=float(_pa.io.read_sigmet(str(probe)).fixed_angle["data"][0])
        except Exception:
            e=None
        finally:
            probe.unlink(missing_ok=True)
        if e is not None: elev_of[f]=e
        print(f"    bin {f:2d}: {np.median(sz[fam==f])/1e6:5.2f} MB -> "
              f"{('%.1f deg'%e) if e is not None else 'unreadable'}", flush=True)
    if not elev_of:
        print("    no bin could be read — skipping"); return 0
    lowest=min(elev_of.values())
    good={f for f,e in elev_of.items() if abs(e-lowest)<0.15}
    P=dict(P); P["elev"]=lowest
    print(f"    lowest sweep = {lowest:.1f} deg ({len(good)} of {NB} bins)", flush=True)
    target=good
    keys=[ks[i][0] for i in range(len(ks)) if fam[i] in target]
    print(f"    {len(ks)} objects -> {len(keys)} scans at {P['elev']} deg", flush=True)
    if hh0 is not None:                                       # UTC hour window
        keys=[k for k in keys if hh0 <= int(k.split("/")[-1][9:11]) <= hh1]
    out=ARCH/radar/dstr/product; out.mkdir(parents=True,exist_ok=True)
    for f in out.glob("*.webp"): f.unlink()
    meta=[]; lat0,lon0=SITES.get(radar,(None,None))
    n_try=0
    for k in keys:
        if len(meta)>0 and n_try%stride!=0 and False: pass
        n_try+=1
        name=k.split("/")[-1]; p=cache/name
        if not p.exists():
            try: urllib.request.urlretrieve(f"{B}/{k}",p)
            except Exception: continue
        try: r=pyart.io.read_sigmet(str(p))
        except Exception: p.unlink(missing_ok=True); continue
        # ONE scan task only: the low surveillance sweep
        if abs(float(r.fixed_angle["data"][0]) - P["elev"]) > 0.15:
            p.unlink(missing_ok=True); continue   # belt-and-braces after size filter
        if len(meta)%stride:                      # subsample AFTER task selection
            meta.append(None); p.unlink(missing_ok=True); continue
        sw=r.extract_sweeps([0])
        lat0=float(sw.latitude["data"][0]); lon0=float(sw.longitude["data"][0])
        d = rain_rate(sw) if P["field"]=="__rain" else qc(sw,P["field"])
        az=np.deg2rad(sw.azimuth["data"]); rng=sw.range["data"]
        R,A=np.meshgrid(rng,az); x=R*np.sin(A); y=R*np.cos(A)
        la=lat0+y/110540.0; lo=lon0+x/(111320.0*np.cos(np.deg2rad(lat0)))
        ts=str(sw.time["units"]).split("since")[-1].strip()[:19]
        fig=plt.figure(figsize=(11.0,10.2),dpi=dpi,facecolor="#0f1626")
        ax=fig.add_axes([.045,.05,.895,.865],projection=ccrs.PlateCarree())
        ax.set_extent([lon0-3.05,lon0+3.05,lat0-2.95,lat0+2.95],ccrs.PlateCarree())
        ax.set_facecolor("#131c30")
        ax.pcolormesh(lo,la,np.ma.masked_invalid(d),cmap=cmap,
                      norm=BoundaryNorm(P["lv"],cmap.N),shading="nearest",zorder=2)
        for rr in (100,200,300):
            th=np.linspace(0,2*np.pi,361)
            ax.plot(lon0+rr*1000*np.sin(th)/(111320*np.cos(np.deg2rad(lat0))),
                    lat0+rr*1000*np.cos(th)/110540,color="#8296ad",lw=.45,alpha=.45,zorder=4)
        gj=Path(os.environ.get("RADAR_BASINS") or
                Path.home()/"wrf"/"data"/"colombia_hydro_regions.geojson")
        if not gj.exists(): _no_basins(gj)
        else:
            for ft in json.loads(gj.read_text())["features"]:
                g=ft["geometry"]
                for poly in (g["coordinates"] if g["type"]=="MultiPolygon" else [g["coordinates"]]):
                    rr2=np.array(poly[0]); ax.plot(rr2[:,0],rr2[:,1],color="#ffd166",lw=1.0,alpha=.85,zorder=5)
        ax.add_feature(cfeature.NaturalEarthFeature("physical","coastline","10m"),
                       edgecolor="#e9eff8",facecolor="none",lw=.75,zorder=6)
        ax.add_feature(cfeature.NaturalEarthFeature("cultural","admin_1_states_provinces_lines","10m"),
                       edgecolor="#5d6b85",facecolor="none",lw=.4,linestyle=(0,(4,3)),zorder=6)
        ax.plot(lon0,lat0,marker="^",ms=8,color="#5eead4",mec="#0f1626",mew=1.1,zorder=8)
        gl=ax.gridlines(draw_labels=True,lw=.28,color="#5d6b85",alpha=.35,linestyle=":")
        gl.top_labels=gl.right_labels=False
        gl.xlabel_style=gl.ylabel_style={"size":7.5,"color":"#8296ad"}
        sm=plt.cm.ScalarMappable(cmap=cmap,norm=BoundaryNorm(P["lv"],cmap.N))
        cb=fig.colorbar(sm,ax=ax,fraction=.029,pad=.011)
        cb.set_label(P["lab"],fontsize=9.5,color="#e9eff8"); cb.ax.tick_params(labelsize=8,colors="#8296ad")
        fig.text(.045,.955,f"{radar.replace('_',' ').title()} — {P['title']}",
                 fontsize=15,fontweight="bold",color="#e9eff8")
        fig.text(.045,.925,f"{ts} UTC · {P['elev']}° sweep · dual-pol QC · rings 100/200/300 km",
                 fontsize=9.5,color="#8296ad")
        png=out/f"{len([m for m in meta if m]):03d}.webp"
        fig.savefig(png,facecolor=fig.get_facecolor()); plt.close(fig)
        meta.append({"idx":len([m for m in meta if m]),"file":png.name,"date":ts,
                     "label":ts[11:16]+"Z"})
        p.unlink(missing_ok=True)
        if keep and len([m for m in meta if m])>=keep: break
    meta=[m for m in meta if m]
    (ARCH/radar/dstr/f"{product}_manifest.json").write_text(json.dumps(
        {"radar":radar,"date":dstr,"product":product,"site":[lat0,lon0],
         "n_frames":len(meta),"frames":meta}))
    return len(meta)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--radar")
    ap.add_argument("--products",nargs="+",default=["ppi"])
    ap.add_argument("--recent",type=int)
    ap.add_argument("--stride",type=int,default=3)
    ap.add_argument("--keep",type=int,default=96)
    ap.add_argument("--hours",nargs=2,type=int,metavar=("H0","H1"),
                    help="restrict to a UTC hour window, e.g. --hours 16 23")
    a=ap.parse_args()
    days=[a.date] if a.date else [
        (date.today()-timedelta(days=i)).strftime("%Y%m%d") for i in range(1,(a.recent or 2)+1)]
    for d in days:
        try: avail=sites_on(d)
        except Exception as e:
            print(f"{d}: listing failed ({e})"); continue
        if not avail: print(f"{d}: no data in the bucket yet"); continue
        for rad in ([a.radar] if a.radar else avail):
            if rad not in avail: continue
            for prod in a.products:
                n=render(rad,d,prod,stride=a.stride,keep=a.keep,
                         hh0=(a.hours[0] if a.hours else None),
                         hh1=(a.hours[1] if a.hours else None))
                print(f"  {d} {rad:18s} {prod:4s}: {n} frames",flush=True)

if __name__=="__main__": main()
