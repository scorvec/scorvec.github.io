"""One AAM-budget reducer for every model in the conservation-and-drag intercomparison (scripts/amdrag/).

Every model's fields go through `Reducer.torque_step` (the 6-hourly surface fields) and `Reducer.level_step` (the
24-hourly pressure-level winds), on the model's own published grid, so GEM, GFS/GEFS, IFS/AIFS and ERA5 are reduced
by exactly the same arithmetic. Outputs are small: 5-degree latitude-band sums, hemisphere-splittable, plus the
torque study's regions and the site's six mountain-range boxes.

Conventions (as aifs-aam-budget.html / scripts/mjo/src/aam.py; torque ON the atmosphere, positive = adds westerly
AAM; 1 Hadley = 1e18 N m):
  relative AAM   M_r = (a^3/g) sum u cos^2(phi) dlam dphi dp   layers between level mid-points, top edge 0 hPa,
                                                                bottom edge 1100 hPa, every layer clipped at p_s
  mass AAM       M_O = (Omega a^4/g) sum p_s cos^3(phi) dlam dphi
  mountain       T_m = a^2 sum h dp_s/dlam cos(phi) dlam dphi   (== -sum p_s dh/dlam dA; centred, periodic)
  friction       T_f = -a^3 sum tau_east cos^2(phi) dlam dphi   (tau = stress ON the surface, same sign as the wind)
  bulk friction  tau = C_d rho |V10| u10, rho = p_s/(R 288 K), C_d(month, 5-degree cell) = drag_cd_field.nc, fitted
                 per cell against ERA5 turbulent stress 1991-2020 (the site torque product's drag) - ONE drag law for
                 every model, so its differences measure each model's near-surface wind, not its drag scheme.
Band sums use exact row overlaps: a grid row stands for [lat - dlat/2, lat + dlat/2] and is shared between bands in
proportion to the overlap (the equator row half/half), so the 36 bands add up to the global integral exactly.

Mountain torque is computed on the native (published) grid and again after area-averaging p_s and h together onto a
common 0.5-degree and 1.5-degree grid (the resolution sensitivity of Brown 2004, QJRMS 130, 3029-3046: averaging both
fields removes the sub-cell covariance of p_s and dh/dlam, i.e. the torque the coarse grid cannot resolve).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

A = 6.371e6
G = 9.80665
OMEGA = 7.292115e-5
RD = 287.05
TREF = 288.0
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / "scripts" / "mjo" / "data" / "reference"
DATA = HERE / "data"
MASKS = DATA / "region_masks_0p25.npz"

BAND_EDGES = np.arange(-90.0, 90.1, 5.0)                       # 36 bands of 5 degrees, south to north
NB = len(BAND_EDGES) - 1
# the site's mountain-range boxes (torque_map_anim.RANGES), lon0, lon1, lat0, lat1 on 0..360E
RANGES = {"HIM": (70, 105, 25, 45), "ROC": (232, 258, 30, 62), "AND": (282, 296, -56, 12),
          "GRL": (300, 345, 58, 84), "ANT": (0, 360, -90, -66), "ALP": (0, 45, 36, 50)}
RKEYS = list(RANGES)
# the torque study's regions (torque_increments_maps.py), split by hemisphere: ORO = terrain > 1000 m or ERA5 sdor >
# 100 m plus 500 km downstream; OCX = ocean 30-65 deg outside ORO; OCO = other ocean; LND = other land
REGIONS = ["ORO", "OCX", "OCO", "LND"]
RHKEYS = [f"{r}_{h}" for r in REGIONS for h in ("N", "S")]
# level sets. C14 = the 14 levels every physics model here publishes the u AND v of except GEPS/ECMWF gaps:
#   ECMWF open data: 1000 925 850 700 600 500 400 300 250 200 150 100 50 (+10 where published)
#   GEPS raw:        1000 925 850 700     500 400 300 250 200     100 50 10
# C11 = the levels EVERY model publishes (no 600/150/10), C12 = GEPS's set (C11 + 10 hPa), C13 = ECMWF's 13.
C14 = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
C11 = [50, 100, 200, 250, 300, 400, 500, 700, 850, 925, 1000]
C12 = [10] + C11
C13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEVEL_SETS = {"C11": C11, "C12": C12, "C13": C13, "C14": C14}
LOWLEV = (925, 850)
# 5-degree map grid (display only)
MAP_LAT = -87.5 + 5.0 * np.arange(36)
MAP_LON = 2.5 + 5.0 * np.arange(72)


def layer_weights(p_pa: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Layer thickness (Pa) per level (p ascending), mid-point edges, top 0, bottom 1100 hPa, clipped to p_s."""
    edges = np.empty(len(p_pa) + 1)
    edges[1:-1] = 0.5 * (p_pa[1:] + p_pa[:-1])
    edges[0] = 0.0
    edges[-1] = 110000.0
    lo = np.minimum(edges[:-1][:, None, None], sp[None])
    hi = np.minimum(edges[1:][:, None, None], sp[None])
    return np.clip(hi - lo, 0.0, None)


def band_rows(lat: np.ndarray) -> np.ndarray:
    """(NB, nlat) fraction of each row's cell inside each 5-degree band (rows sum to 1)."""
    lat = np.asarray(lat, float)
    d = abs(lat[1] - lat[0])
    lo = np.clip(lat - d / 2, -90, 90)
    hi = np.clip(lat + d / 2, -90, 90)
    W = np.zeros((NB, len(lat)))
    for b in range(NB):
        e0, e1 = BAND_EDGES[b], BAND_EDGES[b + 1]
        ov = np.clip(np.minimum(hi, e1) - np.maximum(lo, e0), 0, None)
        W[b] = ov / np.where(hi - lo > 0, hi - lo, 1.0)
    return W


def hemi_rows(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row weights for NH and SH (the equator row half/half)."""
    lat = np.asarray(lat, float)
    eq = np.isclose(lat, 0.0)
    return (lat > 0) + 0.5 * eq, (lat < 0) + 0.5 * eq


class Grid:
    """A regular global lat-lon grid, latitude ASCENDING (callers flip), any longitude origin."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)
        assert self.lat[1] > self.lat[0], "latitude must ascend"
        dl = self.lon[1] - self.lon[0]
        assert abs(abs(dl) * len(self.lon) - 360.0) < 1e-3, f"not a global longitude circle ({len(self.lon)} x {dl})"
        self.lonsign = np.sign(dl)
        self.dlam = np.deg2rad(abs(dl))
        self.dphi = np.deg2rad(self.lat[1] - self.lat[0])
        self.cos = np.cos(np.deg2rad(self.lat))
        self.cos[np.abs(self.lat) >= 90.0 - 1e-9] = 0.0
        self.WB = band_rows(self.lat)                         # (NB, nlat)
        self.wN, self.wS = hemi_rows(self.lat)
        self.ieq = int(np.argmin(np.abs(self.lat)))
        self.has_eq = abs(self.lat[self.ieq]) < 1e-6
        self.area = (A ** 2) * self.cos * self.dlam * self.dphi   # (nlat,) cell area per row
        lon360 = self.lon % 360.0
        self.box = {}
        for k, (x0, x1, y0, y1) in RANGES.items():
            inlon = (lon360 >= x0) & (lon360 < x1) if x1 < 360 else np.ones_like(lon360, bool)
            inlat = (self.lat >= y0) & (self.lat <= y1)
            self.box[k] = inlat[:, None] & inlon[None, :]
        self._masks = None
        self._cd = {}
        # 5-degree map binning
        self.mi = np.clip(np.floor((self.lat + 90.0) / 5.0).astype(int), 0, 35)
        self.mj = np.floor(lon360 / 5.0).astype(int) % 72

    # ---- static helpers ---------------------------------------------------------------------------------------
    def masks(self):
        """Region masks and the ERA5 land-sea mask, nearest-neighbour from the 0.25-degree reference."""
        if self._masks is None:
            z = np.load(MASKS)
            lat0, lon0 = z["lat"], z["lon"]                      # lat ascending -90..90, lon 0..359.75
            iy = np.clip(np.rint((self.lat - lat0[0]) / 0.25).astype(int), 0, len(lat0) - 1)
            ix = np.rint((self.lon % 360.0) / 0.25).astype(int) % len(lon0)
            bits = z["bits"][iy][:, ix]
            m = {}
            for i, r in enumerate(REGIONS):
                rb = (bits >> i) & 1 == 1
                m[f"{r}_N"] = rb & (self.lat[:, None] > 0)
                m[f"{r}_S"] = rb & (self.lat[:, None] < 0)
                eq = np.isclose(self.lat, 0.0)
                if eq.any():                                       # equator row: count it in the NH only (regions
                    m[f"{r}_N"] |= rb & eq[:, None]               # are area means, so no half weights needed)
            m["land"] = z["lsm"][iy][:, ix] >= 0.5
            self._masks = m
        return self._masks

    def cd(self, month: int) -> np.ndarray:
        if month not in self._cd:
            import xarray as xr
            c = xr.open_dataset(REF / "drag_cd_field.nc")["cd"].sel(month=month).values   # (36,72) lat asc, lon 0..
            iy = np.clip(np.floor((self.lat + 90.0) / 5.0).astype(int), 0, 35)
            ix = np.floor((self.lon % 360.0) / 5.0).astype(int) % 72
            out = c[iy][:, ix]
            self._cd[month] = np.where(np.isfinite(out), out, np.nanmedian(c)).astype(np.float64)
        return self._cd[month]

    # ---- reductions -------------------------------------------------------------------------------------------
    def bands(self, rowsum: np.ndarray) -> np.ndarray:
        """(nlat,) row integrals -> (NB,) band integrals."""
        return self.WB @ rowsum

    def region_sum(self, f: np.ndarray) -> np.ndarray:
        m = self.masks()
        return np.array([float((f * m[k]).sum()) for k in RHKEYS])

    def region_mean(self, f: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """Area-mean of f over each region x hemisphere (cells where `valid`)."""
        m = self.masks()
        w = np.broadcast_to(self.area[:, None], f.shape)
        if valid is not None:
            w = w * valid
        out = []
        for k in RHKEYS:
            ww = w * m[k]
            s = ww.sum()
            out.append(float((f * ww).sum() / s) if s > 0 else np.nan)
        return np.array(out)

    def range_sum(self, f: np.ndarray) -> np.ndarray:
        return np.array([float(f[self.box[k]].sum()) for k in RKEYS])

    def map5(self, f: np.ndarray, how: str = "mean", valid: np.ndarray | None = None) -> np.ndarray:
        """5-degree map (36, 72): area-mean (how='mean') or sum (how='sum')."""
        idx = (self.mi[:, None] * 72 + self.mj[None, :]).ravel()
        if how == "sum":
            return np.bincount(idx, weights=f.ravel(), minlength=36 * 72).reshape(36, 72)
        w = np.broadcast_to(self.area[:, None], f.shape)
        if valid is not None:
            w = w * valid
        num = np.bincount(idx, weights=(f * w).ravel(), minlength=36 * 72)
        den = np.bincount(idx, weights=np.asarray(w).ravel(), minlength=36 * 72)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (num / den).reshape(36, 72)

    # ---- budget terms (per-cell densities so they can be summed any way) --------------------------------------
    def mass_cells(self, sp):
        return sp * (self.area / G)[:, None]                      # kg

    def mass_aam_cells(self, sp):
        return sp * (OMEGA * A ** 4 / G * self.cos ** 3 * self.dlam * self.dphi)[:, None]

    def mountain_cells(self, sp, h):
        dpd = (np.roll(sp, -1, axis=1) - np.roll(sp, 1, axis=1)) / (2 * self.dlam) * self.lonsign
        return h * dpd * (A ** 2 * self.cos * self.dlam * self.dphi)[:, None]

    def friction_cells(self, tau):
        return -tau * (A ** 3 * self.cos ** 2 * self.dlam * self.dphi)[:, None]

    def bulk_tau(self, u10, v10, sp, month):
        return self.cd(month) * (sp / (RD * TREF)) * np.hypot(u10, v10) * u10

    def rel_aam_lb(self, u, p_pa, sp):
        """u (lev, lat, lon) on ascending p -> relative AAM per level and band (lev, NB), kg m2 s-1. Level by level
        (a 0.15-degree, 33-level 3-D weight array would need GBs)."""
        edges = np.empty(len(p_pa) + 1)
        edges[1:-1] = 0.5 * (p_pa[1:] + p_pa[:-1])
        edges[0], edges[-1] = 0.0, 110000.0
        fac = A ** 3 / G * self.cos ** 2 * self.dlam * self.dphi
        out = np.empty((len(p_pa), NB))
        for k in range(len(p_pa)):
            w = np.clip(np.minimum(edges[k + 1], sp) - np.minimum(edges[k], sp), 0.0, None)
            out[k] = self.bands((u[k] * w).sum(1, dtype=np.float64) * fac)
        return out

    def eq_flux(self, u, v, p_pa, sp):
        """Relative-AAM flux into the NH across the equator, (a^2/g) sum_lam int u v dp dlam (N m = kg m2 s-2)."""
        if not self.has_eq:
            return np.nan
        i = self.ieq
        w = layer_weights(p_pa, sp[i:i + 1, :])[:, 0, :]
        return float((A ** 2 / G) * (u[:, i, :] * v[:, i, :] * w).sum() * self.dlam)


def coarse_grid(res: float):
    lat = np.arange(-90.0, 90.0 + 1e-9, res)
    lon = np.arange(0.0, 360.0 - 1e-9, res)
    return lat, lon


class Coarsener:
    """Area-weighted bin average of fine fields onto a coarse lat-lon grid (bins centred on the coarse points)."""

    def __init__(self, g: Grid, res: float):
        self.res = res
        lat, lon = coarse_grid(res)
        self.grid = Grid(lat, lon)
        self.iy = np.clip(np.floor((g.lat + 90.0 + res / 2) / res).astype(int), 0, len(lat) - 1)
        self.ix = np.floor(((g.lon % 360.0) + res / 2) / res).astype(int) % len(lon)
        self.idx = (self.iy[:, None] * len(lon) + self.ix[None, :]).ravel()
        w = np.broadcast_to(np.maximum(g.cos, 1e-6)[:, None], (len(g.lat), len(g.lon)))
        self.w = w.ravel()
        self.n = len(lat) * len(lon)
        self.den = np.bincount(self.idx, weights=self.w, minlength=self.n)
        self.shape = (len(lat), len(lon))

    def __call__(self, f):
        num = np.bincount(self.idx, weights=(f.ravel() * self.w), minlength=self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / self.den
        return out.reshape(self.shape)


# steps at which the per-level x band AAM arrays and the 5-degree maps are kept (the rest: totals only)
LB_STEPS = (0, 24, 72, 120, 168, 240, 360)
MAP_STEPS = (0, 120, 240, 360)
TORQUE_PREFIX = ("Tm", "Tf", "Tg")          # time-averaged into daily means; everything else is a state


class Reducer:
    """The per-step reductions for one model grid and orography (stateless: callers collect the step dicts)."""

    def __init__(self, g: Grid, h: np.ndarray):
        self.g = g
        self.h = h.astype(np.float64)
        self.coarse = {r: Coarsener(g, r) for r in (0.5, 1.5)}
        self.hc = {r: c(self.h) for r, c in self.coarse.items()}

    def torque_step(self, month: int, sp, u10, v10, tau_turb=None, tau_gwd=None, tcwv=None, maps=False):
        g = self.g
        out = {}
        out["mass_b"] = g.bands(g.mass_cells(sp).sum(1))
        out["MO_b"] = g.bands(g.mass_aam_cells(sp).sum(1))
        tm = g.mountain_cells(sp, self.h)
        out["Tm_b"] = g.bands(tm.sum(1))
        out["Tm_r"] = g.range_sum(tm)
        out["Tm_rh"] = g.region_sum(tm)
        for r, c in self.coarse.items():
            cg = c.grid
            tmc = cg.mountain_cells(c(sp), self.hc[r])
            tag = "05" if r == 0.5 else "15"
            out[f"Tm{tag}_b"] = cg.bands(tmc.sum(1))
            out[f"Tm{tag}_r"] = cg.range_sum(tmc)
        tb = g.friction_cells(g.bulk_tau(u10, v10, sp, month))
        out["Tfb_b"] = g.bands(tb.sum(1))
        out["Tfb_r"] = g.range_sum(tb)
        out["Tfb_rh"] = g.region_sum(tb)
        land = g.masks()["land"]
        out["Tfb_land_b"] = g.bands((tb * land).sum(1))
        out["u10_b"] = g.bands((u10 * g.area[:, None]).sum(1))          # area-weighted sums; / area_b later
        out["u10_rh"] = g.region_mean(u10)
        out["spd10_rh"] = g.region_mean(np.hypot(u10, v10))
        if tau_turb is not None:
            tt = g.friction_cells(tau_turb)
            out["Tft_b"] = g.bands(tt.sum(1))
            out["Tft_r"] = g.range_sum(tt)
            out["Tft_rh"] = g.region_sum(tt)
            out["Tft_land_b"] = g.bands((tt * land).sum(1))
        if tau_gwd is not None:
            tg = g.friction_cells(tau_gwd)
            out["Tg_b"] = g.bands(tg.sum(1))
            out["Tg_r"] = g.range_sum(tg)
            out["Tg_rh"] = g.region_sum(tg)
            out["Tg_land_b"] = g.bands((tg * land).sum(1))
        if tcwv is not None:
            out["wv_b"] = g.bands((tcwv * g.area[:, None]).sum(1))      # kg of water vapour per band
        out["xmap_Tm"] = g.map5(tm, "sum")                               # forecast-mean maps (packed as means)
        out["xmap_Tfb"] = g.map5(tb, "sum")
        if maps:
            out["map_u10"] = g.map5(u10)
        return out

    def level_step(self, levels, u: np.ndarray, v: dict | None, sp, lb=False, maps=False):
        """u: (nlev, lat, lon) on `levels` (any order); v: {level: 2-D} on (at least) the level sets' levels."""
        g = self.g
        levels = [int(l) for l in levels]
        order = np.argsort(levels)
        lev = np.array(levels)[order]
        u = u[order]
        out = {}
        full = g.rel_aam_lb(u, lev * 100.0, sp)                           # (lev, NB) on the full published set
        out["Mr_b"] = full.sum(0)
        if lb:
            out["Mr_lb"] = full
        pos = {int(l): i for i, l in enumerate(lev)}
        for nm, ls in LEVEL_SETS.items():
            if not all(l in pos for l in ls):
                continue
            ii = [pos[l] for l in ls]
            lbs = g.rel_aam_lb(u[ii], np.array(ls) * 100.0, sp)
            out[f"Mr{nm}_b"] = lbs.sum(0)
            if nm == "C11" and lb:
                out["MrC11_lb"] = lbs
            if v is not None and all(l in v for l in ls):
                vv = np.stack([v[l] for l in ls])
                out[f"F{nm}"] = g.eq_flux(u[ii], vv, np.array(ls) * 100.0, sp)
        for L in LOWLEV:
            if L in pos:
                ok = sp > L * 100.0
                out[f"u{L}_rh"] = g.region_mean(u[pos[L]], ok)
                out[f"u{L}_b"] = g.bands((u[pos[L]] * ok * g.area[:, None]).sum(1))
                out[f"a{L}_b"] = g.bands((ok * g.area[:, None]).sum(1))
                if maps and L == 850:
                    out["map_u850"] = g.map5(u[pos[L]], valid=ok)
        return out


def pack_run(prefix: str, T: dict, L: dict, dt: int = 6) -> dict:
    """One run's step dicts -> compact arrays {prefix|name}:
      D|<torque>  daily means over day d = [24d, 24d+24] h, trapezoid of the dt-hourly samples (NaN if one is missing)
      A|<torque>  the step-0 (analysis) value;  H|<x>  every dt-hourly step, (SH, NH) sums of the band arrays
      S|<state>   instantaneous at the 24-h steps (mass, mass AAM, 10 m wind ...);  L|<x>  at the 24-h level steps
      M|<map>     5-degree maps at MAP_STEPS (float16);  X|<map>  mean over days 1-10 of the torque maps
    Band arrays are float32 (36 bands; relative precision 6e-8 is far below every signal here)."""
    res = {}
    tsteps = sorted(T)
    if tsteps:
        last = max(tsteps)
        nd = last // 24
        days = np.arange(nd + 1) * 24
        keys = sorted({k for d in T.values() for k in d})
        res[f"{prefix}|Ssteps"] = days
        for k in keys:
            ex = np.asarray(next(d[k] for d in T.values() if k in d), dtype=np.float64)
            if k.startswith("xmap_"):
                sel = [s for s in tsteps if 24 <= s <= 240 and k in T[s]]
                if sel:
                    res[f"{prefix}|X|{k[5:]}"] = np.mean([T[s][k] for s in sel], 0).astype(np.float32)
                continue
            if k.startswith("map_"):
                arr = np.full((len(MAP_STEPS),) + ex.shape, np.nan)
                for i, s in enumerate(MAP_STEPS):
                    if s in T and k in T[s]:
                        arr[i] = T[s][k]
                res[f"{prefix}|M|{k[4:]}"] = arr.astype(np.float16)
                continue
            def at(s):
                return np.asarray(T[s][k], dtype=np.float64) if (s in T and k in T[s]) else np.full(ex.shape, np.nan)
            if k.startswith(TORQUE_PREFIX):
                w = np.array([0.5] + [1.0] * (24 // dt - 1) + [0.5]) / (24 // dt)
                D = np.full((nd,) + ex.shape, np.nan)
                for d in range(nd):
                    ss = [24 * d + j * dt for j in range(24 // dt + 1)]
                    D[d] = sum(wi * at(s) for wi, s in zip(w, ss))
                res[f"{prefix}|D|{k}"] = D.astype(np.float32)
                res[f"{prefix}|A|{k}"] = at(0).astype(np.float32)
            else:
                res[f"{prefix}|S|{k}"] = np.stack([at(s) for s in days]).astype(np.float32)
            if k.endswith("_b") and ex.shape == (NB,):
                H = np.stack([at(s) for s in range(0, last + 1, dt)])
                res[f"{prefix}|H|{k}"] = np.stack([H[:, :NB // 2].sum(1), H[:, NB // 2:].sum(1)], 1)
    lsteps = sorted(L)
    if lsteps:
        res[f"{prefix}|Lsteps"] = np.array(lsteps)
        keys = sorted({k for d in L.values() for k in d})
        for k in keys:
            ex = np.asarray(next(d[k] for d in L.values() if k in d), dtype=np.float64)
            if k.startswith("map_"):
                arr = np.full((len(MAP_STEPS),) + ex.shape, np.nan)
                for i, s in enumerate(MAP_STEPS):
                    if s in L and k in L[s]:
                        arr[i] = L[s][k]
                res[f"{prefix}|M|{k[4:]}"] = arr.astype(np.float16)
                continue
            if k.endswith("_lb"):
                arr = np.full((len(LB_STEPS),) + ex.shape, np.nan)
                for i, s in enumerate(LB_STEPS):
                    if s in L and k in L[s]:
                        arr[i] = L[s][k]
                res[f"{prefix}|L|{k}"] = arr.astype(np.float32)
                continue
            arr = np.full((len(lsteps),) + ex.shape, np.nan)
            for i, s in enumerate(lsteps):
                if k in L[s]:
                    arr[i] = L[s][k]
            res[f"{prefix}|L|{k}"] = arr if ex.ndim == 0 else arr.astype(np.float32)
    return res
