#!/usr/bin/env python3
"""One map style for the outlook pages (user 2026-09-07: "redesign the maps to be a similar style
for consistency"): the SEAS5 single-map look — plate carrée, most of the world, bold title top-left,
one-line subtitle, 50 m coastlines, light gridlines with labels, a horizontal colour bar below.
seas5_build._global_map draws exactly this; sfs_maps / sfs_daily import it.

    fig, ax, H, pc = open_map(kind="atm")            # global, cut at 40 °E (sst: 20 °E)
    fig, ax, H, pc = open_map(extent=[-180, 180, 20, 90], central=-140)
    ... draw with transform=pc ...
    features(ax, land_only=False); heading(fig, H, title, sub); colorbar(fig, H, mappable, label)
    save(fig, out)
"""
from __future__ import annotations
import numpy as np

CENTRAL = {"sst": -160.0, "atm": -140.0}
LAT = {"sst": (-70, 70), "atm": (-60, 85)}
WIDTH = 14.0


def open_map(kind: str = "atm", extent=None, central: float | None = None, width: float = WIDTH, top: float = 0.95, bot: float = 0.85):
    """→ (fig, ax, H, PlateCarree). `extent` = [W, E, S, N] in degrees (None → global by kind)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    c = CENTRAL.get(kind, CENTRAL["atm"]) if central is None else central
    proj = ccrs.PlateCarree(central_longitude=c)
    if extent is None:
        lat0, lat1 = LAT.get(kind, LAT["atm"]); lon_span = 360.0
    else:
        lat0, lat1 = extent[2], extent[3]; lon_span = extent[1] - extent[0]
    map_h = width * (lat1 - lat0) / lon_span; H = map_h + top + bot
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0.03, bot / H, 0.94, map_h / H], projection=proj)
    if extent is None or (extent[1] - extent[0]) >= 360:
        ax.set_extent([-180, 180, lat0, lat1], crs=proj)
    else:
        ax.set_extent([extent[0], extent[1], lat0, lat1], crs=pc)
    return fig, ax, H, pc


def features(ax, land_only: bool = False, states: bool = True, gridlines: bool = True):
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.LAND, facecolor="#f1f0eb", zorder=0)
    if land_only:
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
    if states:
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#6f6b64", zorder=3)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#6f6b64", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(-60, 91, 30), zorder=4)
        gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#555"}


def heading(fig, H: float, title: str, sub: str, title_size: float = 13.5, sub_size: float = 8.6, wrap: int | None = None):
    """Bold title top-left, subtitle under it; `wrap` = characters per subtitle line for narrow figures."""
    if wrap:
        import textwrap
        sub = "\n".join(textwrap.wrap(sub, wrap))
    fig.text(0.03, 1 - 0.14 / H, title, fontsize=title_size, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / H, sub, fontsize=sub_size, color="#444", va="top", linespacing=1.3)


def colorbar(fig, H: float, mappable, label: str, levels=None, extend: str = "both"):
    cax = fig.add_axes([0.25, 0.42 / H, 0.50, 0.14 / H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend, spacing="uniform")
    cb.set_label(label, fontsize=8.5); cb.ax.tick_params(labelsize=7)
    if levels is not None and len(levels) > 16:
        cb.set_ticks([x for x in levels if abs(x - round(x)) < 1e-6])
    return cb


def symmetric_levels(vmax: float, n: int = 6, gap: float | None = None) -> list[float]:
    """n bins each side of zero with a white band ±gap (default vmax/n/2) in the middle."""
    step = vmax / n; gap = step / 2 if gap is None else gap
    pos = [gap] + [step * k for k in range(1, n + 1)]
    return [-x for x in pos[::-1]] + pos


def save(fig, out, dpi: int = 125, quality: int = 86, lossless: bool = False):
    import matplotlib.pyplot as plt
    pk = {"lossless": True, "quality": 100, "method": 6} if lossless else {"quality": quality, "method": 6}
    fig.savefig(out, dpi=dpi, pil_kwargs=pk); plt.close(fig)


# ── SST: one anomaly palette and one absolute palette for every SST figure on the site ─────────────────────────────
# (user 2026-10-08: "improve the quality of ALL SST anomaly and absolute SST charts"). Discrete classes on round,
# symmetric levels, so a colour means the same number on every product (the daily OISST loops, MUR, the Walker SST
# panels, SEAS5, SFS) and the colour bar can be read. Values, baselines and methods stay the callers' own.
#
# Anomaly: a white neutral band +-0.2 °C, 0.2 °C classes to +-3 (the SEAS5 "highly detailed" ladder the user asked
# for on 2026-09-07), 0.5 °C to +-5, then 6 and 7 for the daily tail (the 2026 Niño-1+2 core passed +6 on OISST; the
# old asinh scale ran to +-7 for that reason). Colours: the site's own SST ramp (pale peach -> orange -> red ->
# maroon; pale blue -> blue -> navy), lightness falling away from white on both sides; the beyond-5 tail turns
# violet so a saturated core still reads as a different class.
SST_ANOM_POS = [0.2] + [round(0.2 * k, 1) for k in range(2, 16)] + [3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
SST_ANOM_LEVELS = [-x for x in SST_ANOM_POS[::-1]] + SST_ANOM_POS
_WARM = [(0.00, "#fdeee3"), (0.12, "#fbd6bb"), (0.32, "#f5a274"), (0.52, "#e3613c"), (0.70, "#bf2a22"),
         (0.84, "#87101c"), (0.92, "#5e0a24"), (1.00, "#3b0a3a")]
_COOL = [(0.00, "#e6f1f9"), (0.12, "#cde3f3"), (0.32, "#94c3e8"), (0.52, "#4d8fd6"), (0.70, "#2a5fb8"),
         (0.84, "#18388a"), (0.92, "#14245e"), (1.00, "#1b1240")]
SST_NEUTRAL = "#ffffff"
SST_LAND = "#d6d3cb"          # mid-light warm grey: distinct from the white neutral band (the outlook land #f1f0eb is not)
SST_COAST = "#3a3a3a"


def _ramp(stops, n):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("r", stops)
    return [cm(float(x)) for x in (np.arange(n) + 0.5) / n]


def _anom_colours():
    """One colour per SST_ANOM_LEVELS class (len(levels) - 1), plus the beyond-the-ladder under/over colours."""
    n = len(SST_ANOM_POS) - 1                                  # classes per side outside the neutral band
    warm, cool = _ramp(_WARM, n + 1), _ramp(_COOL, n + 1)      # the last ramp step is the extension colour
    return cool[:n][::-1] + [SST_NEUTRAL] + warm[:n], cool[n], warm[n]


def _cut_style(full, cls, under, over, lv, name):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    i0, i1 = full.index(lv[0]), full.index(lv[-1])
    use = cls[i0:i1]
    cmap = ListedColormap(use, name=name)
    cmap.set_under(cls[i0 - 1] if i0 > 0 else under)
    cmap.set_over(cls[i1] if i1 < len(cls) else over)
    cmap.set_bad(alpha=0.0)
    return cmap, BoundaryNorm(lv, len(use))


def sst_anom_style(vmax: float = 7.0, levels=None):
    """-> dict(levels, cmap, norm, ticks) for an SST anomaly field, cut to +-vmax (a value on the ladder). Every
    product takes its colours from the full ladder: +1.4 °C is the same red on a +-3 map as on a +-7 one. `levels`
    (a contiguous run of SST_ANOM_LEVELS) overrides the cut."""
    full = SST_ANOM_LEVELS
    cls, under, over = _anom_colours()
    lv = [x for x in full if abs(x) <= vmax + 1e-9] if levels is None else list(levels)
    cmap, norm = _cut_style(full, cls, under, over, lv, "sst_anom")
    big = max(abs(lv[0]), abs(lv[-1]))
    step = 1.0 if big > 3.01 else 0.5 if big > 1.51 else 0.2          # whole degrees on a +-7 bar, halves on +-3
    ticks = [0.0] + [x for x in lv if abs(x / step - round(x / step)) < 1e-6]
    return dict(levels=lv, cmap=cmap, norm=norm, ticks=sorted(set(ticks)))


# Absolute SST: 4 °C classes to 16 °C, 2 °C to 24, then 0.5 °C classes 24-31 °C so the 26.5 / 27 / 27.5 / 28 °C steps
# that matter for deep convection and the warm pool each get their own colour (on the old turbo -2..32 scale the whole
# tropical Pacific was one red). Sequential, lightness rising with temperature (navy -> blue -> teal -> mauve -> rose ->
# coral -> orange -> amber -> pale yellow); the 0.5 °C warm-range classes take most of the ramp.
SST_ABS_LEVELS = [-2.0, 0.0, 4.0, 8.0, 12.0, 16.0, 18.0, 20.0, 22.0, 24.0] + [24.5 + 0.5 * k for k in range(14)] + [32.0]
# stops by class position: 9 cold classes (to 24 °C) take the first 37.5 % (navy -> blue -> teal), the 15 warm ones the
# rest (mauve -> rose -> coral -> orange -> amber -> pale yellow); CIE L* rises monotonically from ~17 to ~87
_THERMAL = [(0.00, "#0c1a3a"), (0.13, "#1b3a78"), (0.26, "#2a5f9e"), (0.33, "#2d7095"), (0.38, "#5a7aa0"), (0.44, "#8a72aa"),
            (0.52, "#b8669a"), (0.63, "#dc6f72"), (0.75, "#f08c4c"), (0.87, "#f8b53a"), (1.00, "#f7ef8e")]
SST_ABS_ISOTHERMS = [26.5, 28.0, 30.0]


def sst_abs_style(lo: float | None = None, hi: float | None = None):
    """-> dict(levels, cmap, norm, ticks) for absolute SST, optionally cut to [lo, hi] (values on the ladder). Each
    class takes its colour from the full ladder, so 27.5 °C is the same colour on every absolute-SST figure."""
    full = SST_ABS_LEVELS
    cols = _ramp(_THERMAL, len(full) + 1)                      # under, the classes, over
    under, cls, over = cols[0], cols[1:-1], cols[-1]
    lv = [x for x in full if (lo is None or x >= lo - 1e-9) and (hi is None or x <= hi + 1e-9)]
    cmap, norm = _cut_style(full, cls, under, over, lv, "sst_abs")
    if lv[0] < 20:            # whole bar: every 4 °C in the cold range, every 2 to 24, every degree in the warm range
        ticks = [x for x in lv if (x <= 20 and x % 4 == 0) or (20 < x <= 24 and x % 2 == 0) or (x > 24 and x % 1 == 0)]
    else:
        ticks = [x for x in lv if x % 1 == 0]
    return dict(levels=lv, cmap=cmap, norm=norm, ticks=ticks)


def sst_tick_labels(ticks):
    """Colour-bar labels with a real minus sign and no trailing zeros."""
    return [("−" if t < 0 else "") + f"{abs(t):g}" for t in ticks]


def sst_colorbar(cb, style, label: str, size: float = 8.5, tick_size: float = 7.5):
    """Shared ticks / labels / outline for a colour bar drawn from sst_anom_style or sst_abs_style. The bar is
    uniform-spaced (one equal step per class), so ticks are thinned to the room the drawn bar actually has: outward
    from 0, a tick closer than ~1.3 label heights to the last kept one is dropped, and the end values always stay."""
    lv = style["levels"]
    bb = cb.ax.get_position()
    W, H = cb.ax.figure.get_size_inches()
    length_pt = (bb.width * W if cb.orientation == "horizontal" else bb.height * H) * 72.0
    per = length_pt / max(len(lv) - 1, 1)
    sep = tick_size * (1.3 if cb.orientation == "vertical" else 2.6)       # horizontal labels are wider than tall
    pos = {t: float(np.interp(t, lv, np.arange(len(lv)))) * per for t in style["ticks"] if lv[0] <= t <= lv[-1]}
    ticks = sorted(pos)
    keep = []
    for side in (sorted([t for t in ticks if t >= 0]), sorted([t for t in ticks if t < 0], reverse=True)):
        k = []
        for t in side:
            if not k and keep and t < 0 and abs(pos[t] - pos[keep[0]]) < sep:
                continue                                            # too close to 0 on the negative side
            if k and abs(pos[t] - pos[k[-1]]) < sep:
                if t == side[-1]:
                    k[-1] = t                                       # the end value wins over its inner neighbour
                continue
            k.append(t)
        keep = k if not keep else keep + k
    ticks = sorted(keep)
    cb.set_ticks(ticks)
    lab = sst_tick_labels(ticks)
    if cb.orientation == "horizontal":
        cb.ax.set_xticklabels(lab)
    else:
        cb.ax.set_yticklabels(lab)
    cb.ax.tick_params(labelsize=tick_size, length=2.5, width=0.5)
    cb.minorticks_off()
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, fontsize=size)
    return cb


def sst_features(ax, scale: str = "50m", coast_lw: float = 0.45, land: bool = True):
    """Land and coastline for an SST map: land drawn over the field (it hides the grid's land-mask stair steps) and
    the axes background in the land colour, so a coastal no-data cell reads as land, never as a white 'zero'."""
    import cartopy.feature as cfeature
    ax.set_facecolor(SST_LAND)
    if land:
        ax.add_feature(cfeature.LAND.with_scale(scale), facecolor=SST_LAND, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), edgecolor=SST_COAST, linewidth=coast_lw, zorder=3)


def save_webp_palette(fig, out, dpi: int, n: int = 256, facecolor: str = "white", **kw):
    """Save a flat-class map as LOSSLESS WebP on a palette of its `n` most frequent colours.

    A discrete-class SST map is ~97 % pixels in a few dozen exact class/land/white colours; the rest are anti-aliased
    edges of text, coastlines and box outlines. Keeping the most frequent colours EXACT (every class colour is among
    them, so no two adjacent 0.2 °C classes are ever merged - which a generic quantizer can do) and snapping only the
    rare edge pixels to their nearest palette colour gives a lossless-looking frame ~20 % smaller than plain lossless
    WebP, with none of the 4:2:0 chroma smearing a lossy encode puts on class edges and coastlines. Encode study on an
    OISST anomaly frame (1785 x 1373 px), 2026-10-08: PNG 630 KB, plain lossless WebP 302 KB, this 242 KB; lossy q70
    177 KB / q86 296 KB with 8 % / 3.5 % of pixels off by more than 12 levels in some channel, this 0 %."""
    import io
    import matplotlib.pyplot as plt
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=facecolor, **kw)
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    a = np.asarray(im)
    key = (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2].astype(np.int32)
    u, inv, cnt = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    if len(u) > n:
        keep = np.argsort(-cnt)[:n]
        pal = np.stack([(u[keep] >> 16) & 255, (u[keep] >> 8) & 255, u[keep] & 255], axis=1).astype(np.int32)
        allc = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], axis=1).astype(np.int32)
        lut = np.empty(len(u), dtype=np.int64)
        for s in range(0, len(u), 4096):                       # nearest palette colour for every unique colour
            d = ((allc[s:s + 4096, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
            lut[s:s + 4096] = d.argmin(axis=1)
        lut[keep] = np.arange(len(keep))
        a = pal[lut[inv]].reshape(a.shape).astype(np.uint8)
        im = Image.fromarray(a, "RGB")
    im.save(out, "WEBP", lossless=True, quality=100, method=4)     # method 6 is 20x slower for <1 % fewer bytes


def webp_kwargs(lossless: bool = False, quality: int = 86) -> dict:
    """pil_kwargs for fig.savefig(... .webp): lossless suits flat-class maps (crisp text and class edges, and usually
    smaller than a lossy encode of the same picture); lossy at `quality` otherwise."""
    return {"lossless": True, "quality": 100, "method": 6} if lossless else {"quality": quality, "method": 6}
