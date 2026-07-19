"""Julia fast-path for synoptic frame rendering.

Python stays in charge of everything meteorological and cartographic —
GRIB decode, region slicing, map projection, colormap/norm construction,
Natural Earth feature projection — and serializes flat arrays + style
dicts per frame. Julia (scripts/julia/synoptic_render.jl) is a pure
rasterizer: curvilinear quad mesh + polylines + colorbar + title.

Frames that need special overlays (plant markers, wind arrows, point
values) stay on the matplotlib path; anything staged here that Julia
fails to render is re-rendered by matplotlib from the same staged arrays,
so the fallback needs no re-fetch.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
STAGING = HERE / ".julia_staging"
OVERLAY_CACHE = HERE / ".julia_overlays"
JULIA_SCRIPT = HERE.parent / "julia" / "synoptic_render.jl"

_PC = None


def _pc():
    global _PC
    if _PC is None:
        import cartopy.crs as ccrs
        _PC = ccrs.PlateCarree()
    return _PC


def available() -> bool:
    import shutil
    return (os.environ.get("SYNOPTIC_RENDERER", "") == "julia"
            and shutil.which("julia") is not None)


def serialize_cmap(cmap, norm, vmin, vmax) -> dict:
    """ListedColormap+BoundaryNorm -> explicit colors+bounds; anything else
    -> 256-sample continuous ramp with (vmin, vmax)."""
    import matplotlib.colors as mcolors
    to_hex = mcolors.to_hex
    d: dict = {}
    if isinstance(norm, mcolors.BoundaryNorm):
        bounds = [float(b) for b in norm.boundaries]
        cols = [to_hex(cmap(norm(0.5 * (a + b)))) for a, b in zip(bounds, bounds[1:])]
        d.update(kind="banded", bounds=bounds, colors=cols)
    else:
        lo = float(vmin if vmin is not None else (norm.vmin if norm else 0.0))
        hi = float(vmax if vmax is not None else (norm.vmax if norm else 1.0))
        cols = [to_hex(cmap(x)) for x in np.linspace(0, 1, 256)]
        d.update(kind="continuous", vmin=lo, vmax=hi, colors=cols)
    for attr, key in (("_rgba_under", "under"), ("_rgba_over", "over")):
        rgba = getattr(cmap, attr, None)
        if rgba is not None:
            d[key] = to_hex(rgba)
    return d


def ensure_overlays(region_id: str, proj, scale: str, coast_scale: str = "10m") -> str:
    """Project Natural Earth line features once per region; cache npz of
    NaN-separated polylines."""
    OVERLAY_CACHE.mkdir(exist_ok=True)
    out = OVERLAY_CACHE / f"{region_id}.npz"
    if out.exists():
        return str(out)
    import cartopy.feature as cfeature
    groups = {}
    specs = [("coast", cfeature.NaturalEarthFeature("physical", "coastline", coast_scale)),
             ("states", cfeature.NaturalEarthFeature("cultural",
              "admin_1_states_provinces_lines", scale)),
             ("borders", cfeature.NaturalEarthFeature("cultural",
              "admin_0_boundary_lines_land", scale))]
    for name, feat in specs:
        xs, ys = [], []
        for geom in feat.geometries():
            lines = getattr(geom, "geoms", [geom])
            for line in lines:
                arr = np.asarray(line.coords)
                p = proj.transform_points(_pc(), arr[:, 0], arr[:, 1])
                xs.append(p[:, 0]); xs.append([np.nan])
                ys.append(p[:, 1]); ys.append([np.nan])
        groups[f"{name}_x"] = np.concatenate(xs).astype(np.float64)
        groups[f"{name}_y"] = np.concatenate(ys).astype(np.float64)
    np.savez_compressed(out, **groups)
    return str(out)


_regrid_cache: dict = {}


def _regrid_index(key: str, lats, lons, proj, extent_xy, shape=(880, 1560)):
    """Nearest-neighbor index mapping a regular projected raster onto the
    curvilinear grid. Computed once per (grid, region) and cached on disk —
    turns every subsequent frame regrid into one fancy-index op."""
    if key in _regrid_cache:
        return _regrid_cache[key]
    OVERLAY_CACHE.mkdir(exist_ok=True)
    f = OVERLAY_CACHE / f"regrid_{key}.npz"
    if f.exists():
        with np.load(f) as z:
            _regrid_cache[key] = (z["idx"], z["mask"])
        return _regrid_cache[key]
    from scipy.spatial import cKDTree
    p = proj.transform_points(_pc(), lons, lats)
    px, py = p[:, :, 0].ravel(), p[:, :, 1].ravel()
    tree = cKDTree(np.column_stack([px, py]))
    x0, x1, y0, y1 = extent_xy
    gx, gy = np.meshgrid(np.linspace(x0, x1, shape[1]),
                         np.linspace(y0, y1, shape[0]))
    dx_cell = np.hypot(px[1] - px[0], py[1] - py[0])
    d, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]),
                        distance_upper_bound=max(4 * dx_cell, 12000.0))
    mask = np.isfinite(d)
    idx = np.where(mask, idx, 0).astype(np.int64)
    np.savez_compressed(f, idx=idx, mask=mask)
    _regrid_cache[key] = (idx, mask)
    return _regrid_cache[key]


def stage_frame(*, frame_id: str, values, lats, lons, proj, extent_xy,
                cmap_spec: dict, title: str, cbar_label: str,
                cbar_ticks, figsize, out_path: str, overlays_npz: str,
                grid_key: str = "default") -> None:
    """Regrid to a regular projected raster (cached NN index) and stage."""
    STAGING.mkdir(exist_ok=True)
    shape = (880, 1560)
    idx, mask = _regrid_index(grid_key, lats, lons, proj, extent_xy, shape)
    flat = np.asarray(values, dtype=np.float32).ravel()
    reg = np.where(mask, flat[idx], np.nan).reshape(shape).astype(np.float32)
    np.savez_compressed(STAGING / f"{frame_id}.npz", values=reg)
    meta = dict(frame_id=frame_id, cmap=cmap_spec, title=title,
                cbar_label=cbar_label,
                cbar_ticks=[float(t) for t in cbar_ticks] if cbar_ticks else None,
                figsize=list(figsize), out_path=str(out_path),
                extent=[float(v) for v in extent_xy], overlays=overlays_npz)
    (STAGING / f"{frame_id}.json").write_text(json.dumps(meta))


def render_staged() -> tuple[list[str], list[str]]:
    """Run Julia over everything staged; convert PNG->WebP. Returns
    (done_ids, failed_ids); staged files for failures are left in place."""
    metas = sorted(p for p in STAGING.glob("*.json") if p.name != "spec.json")
    if not metas:
        return [], []
    spec = {"frames": [json.loads(m.read_text()) for m in metas],
            "staging": str(STAGING)}
    spec_path = STAGING / "spec.json"
    spec_path.write_text(json.dumps(spec))
    try:
        r = subprocess.run(["julia", "--project=" + str(JULIA_SCRIPT.parent), str(JULIA_SCRIPT), str(spec_path)],
                           capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0 and "JULIA RENDER DONE" in r.stdout
        if not ok:
            print(f"  synoptic julia failed:\n{r.stderr[-400:]}")
    except Exception as e:                                     # noqa: BLE001
        print(f"  synoptic julia failed ({str(e)[:80]})")
        ok = False
    done, failed = [], []
    from PIL import Image
    for m in metas:
        meta = json.loads(m.read_text())
        png = STAGING / f"{meta['frame_id']}.png"
        if ok and png.exists():
            Image.open(png).save(meta["out_path"], quality=84, method=6)
            png.unlink()
            (STAGING / f"{meta['frame_id']}.npz").unlink(missing_ok=True)
            m.unlink()
            done.append(meta["frame_id"])
        else:
            failed.append(meta["frame_id"])
    if done:
        print(f"  synoptic julia: rendered {len(done)} frames"
              + (f", {len(failed)} fell back" if failed else ""))
    return done, failed


def load_failed(frame_id: str):
    """Arrays + meta for a matplotlib fallback re-render."""
    meta = json.loads((STAGING / f"{frame_id}.json").read_text())
    with np.load(STAGING / f"{frame_id}.npz") as z:
        vals = z["values"]
    return meta, vals


def clear_frame(frame_id: str) -> None:
    (STAGING / f"{frame_id}.npz").unlink(missing_ok=True)
    (STAGING / f"{frame_id}.json").unlink(missing_ok=True)
