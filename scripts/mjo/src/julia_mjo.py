"""Julia fast-path for the MJO cross-section animators (walker / MMSF).

Python stays in charge of the science — spherical-harmonic streamfunctions,
climatology evaluation, indices — and serializes flat arrays + a per-frame
style spec; Julia (scripts/julia/mjo_render.jl) is a pure rasterizer
(filled contours + line contours + arrows + texts on one axis). Frames it
fails to render are re-rendered by the existing matplotlib path from the
same in-memory fields, so the fallback costs nothing.

Gate: MJO_RENDERER=julia + a julia binary on PATH (same opt-in shape as
synoptic's SYNOPTIC_RENDERER).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent / "data" / ".julia_staging"
JULIA_SCRIPT = HERE.parent.parent / "julia" / "mjo_render.jl"

_frames: list[dict] = []


def available() -> bool:
    return (os.environ.get("MJO_RENDERER", "") == "julia"
            and shutil.which("julia") is not None)


def cmap_hex(name: str, n: int) -> list[str]:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap(name)
    return [mcolors.to_hex(cmap((k + 0.5) / n)) for k in range(n)]


def reset() -> None:
    _frames.clear()
    STAGING.mkdir(parents=True, exist_ok=True)
    for p in STAGING.glob("*"):
        p.unlink()


def stage(frame_id: str, out_path: Path | str, arrays: dict, meta: dict) -> None:
    """arrays: name -> ndarray (saved to <frame_id>.npz); meta: the frame spec
    (fill/contours/arrows/texts/axis config per mjo_render.jl's docstring)."""
    np.savez_compressed(STAGING / f"{frame_id}.npz",
                        **{k: np.asarray(v, np.float64) for k, v in arrays.items()})
    fr = dict(meta)
    fr["frame_id"] = frame_id
    fr["_out"] = str(out_path)
    _frames.append(fr)


def render() -> dict[str, bool]:
    """Render everything staged; convert PNG -> WebP at each frame's out path.
    Returns {frame_id: succeeded} — callers matplotlib-re-render the False ones."""
    if not _frames:
        return {}
    spec = dict(staging=str(STAGING),
                frames=[{k: v for k, v in f.items() if k != "_out"} for f in _frames])
    (STAGING / "spec.json").write_text(json.dumps(spec))
    ok = False
    try:
        r = subprocess.run(
            ["julia", "--project=" + str(JULIA_SCRIPT.parent),
             str(JULIA_SCRIPT), str(STAGING / "spec.json")],
            capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0 and "JULIA RENDER DONE" in r.stdout
        if ok:
            print("  " + r.stdout.strip().splitlines()[-2], flush=True)
        else:
            print(f"  mjo julia failed:\n{(r.stderr or r.stdout)[-400:]}", flush=True)
    except Exception as e:                                     # noqa: BLE001
        print(f"  mjo julia failed ({str(e)[:80]})", flush=True)
    from PIL import Image
    status: dict[str, bool] = {}
    for f in _frames:
        png = STAGING / (f["frame_id"] + ".png")
        good = ok and png.exists()
        if good:
            out = Path(f["_out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.open(png).save(out, quality=84, method=6)
            png.unlink()
        status[f["frame_id"]] = good
    return status
