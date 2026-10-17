"""The month-ahead outlook's statistical method: five empirical components and their cross-validated consolidation.

Targets: 500 hPa height, 2 m temperature and precipitation, monthly means over 0-90N on the archive's 3° grid.

Nothing here reads a dynamical forecast. The outlook for target month T of year Y is issued on the 15th of month
T-1, and its predictors are the ERA5 means of the ~30-day window ending on the 7th of T-1 (mo_grid: block 1 of T-2
plus block 0 of T-1). The predictand is the calendar-month anomaly against 1991-2020.

Components (each a map of the target month's anomaly):
    TR   trend: per grid point, the linear trend of the 3-month (T-1..T+1) anomaly over the 30 years before Y,
         extrapolated to Y, after removing the part explained by relative Niño-3.4 (so La Niña or El Niño runs at
         the end of the window are not extrapolated as climate change)
    IX   index regression: the detrended target anomaly regressed on four indices of the window - relative
         Niño-3.4 (the RONI quantity) split into its El Niño and La Niña halves, equatorial u50 (QBO) and 57-63N u50
         (the lower polar vortex)
    CA   constructed analogue on the tropics (Van den Dool 1994): the window's detrended SST and rainfall anomalies
         20S-20N (each scaled to unit total variance) are written as a ridge-regularised combination of the library
         years', and the same weights combine those years' detrended target-month anomalies. Rainfall is the
         heating that actually forces the teleconnections; extratropical SST is left out on purpose, being mostly
         the footprint of the atmospheric circulation rather than a forcing of it
    PE   damped local persistence: per grid point, the detrended target anomaly regressed on the window's anomaly
         of the same field (for 2 m temperature also on the window's sea ice, snow depth and topsoil moisture, for
         precipitation on topsoil moisture)
    AN   analogue years: the eight library years closest to the window's state (tropical SST pattern, ENSO, QBO,
         vortex and the hemispheric 500 hPa pattern); their mean detrended target anomaly

IX and PE train on the target month pooled with its two neighbouring calendar months (each with its own window),
about 200 cases instead of 65; CA and AN use the target month alone, which scored better for them.

Consolidation: per grid point a non-negative least-squares regression of the observed anomaly on the five
components' cross-validated hindcasts, with the normal equations smoothed over ~2500 km so the weights vary only on
large scales. Hindcast years leave out the target year and its neighbours (Y-1, Y, Y+1) from every training step;
the consolidation weights for a hindcast year leave that year out as well, so the skill maps are fully
cross-validated except for the slow detrending (a centred local-linear fit, which sees the target year's own value
and its neighbours; see the methods page).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from itertools import combinations

import numpy as np

import mo_grid as G

CLIM = (1991, 2020)
COMPONENTS = ("TR", "IX", "CA", "PE", "AN")
NAMES = {
    "TR": "Trend",
    "IX": "RONI (El Niño, La Niña) / QBO / vortex regression",
    "CA": "Tropical SST and rainfall constructed analogue",
    "PE": "Local persistence",
    "AN": "Analogue years",
}
TARGETS = ("z500", "t2m", "prcp")
TREND_YEARS = 30
TREND_ENSO_ADJUST = True
POOL = (-1, 0, 1)  # calendar-month offsets pooled into the training library
POOLED = ("IX", "PE")  # the components trained on the pooled library (the rest use the target month alone)
PE_EXTRA = {"t2m": ("sic", "snow", "swvl"), "prcp": ("swvl",)}  # local surface predictors beside a field's own window
PE_FLOOR = {"win_sic": 0.01, "win_snow": 0.003, "win_swvl": 0.003}  # rms anomaly below which a predictor is dropped
SST_LAT = (20, 20)  # the constructed analogue's SST domain: 20S-20N
CA_VARS = ("sst", "prcp_t")  # the constructed analogue's state: tropical SST and tropical rainfall (the heating)
WIN_FIELDS = ("z500", "t2m", "sic", "sst", "snow", "swvl", "prcp")
ENSO_SPLIT = True
LOESS_HALF = 20  # half-width (years) of the tricube local-linear detrending
N_ANALOGS = 8
CA_RIDGE = 0.2  # λ as a fraction of the mean diagonal of the SST Gram matrix
SMOOTH_KM = 2500.0
CONS_RIDGE = 0.1
EARTH_KM = 6371.0


# ---------------------------------------------------------------------------------------------- archive assembly
@dataclass
class Archive:
    years: np.ndarray  # (nY,)
    cal: dict  # var -> (nY*12, ...) calendar-month means, NaN where a block is incomplete
    win: dict  # var -> (nY*12, ...) predictor window ending on the 7th of that month
    n: np.ndarray  # (nY*12, 2) samples per block

    def t(self, y: int, m: int) -> int:
        """Flat index of (year, month 1-12)."""
        return (y - int(self.years[0])) * 12 + (m - 1)


def load_archive(y0: int = 1959, y1: int | None = None) -> Archive:
    ys = sorted(int(p.stem.split("_")[1]) for p in G.DATA.glob("era5_[0-9]*.npz"))
    ys = [y for y in ys if y >= y0 and (y1 is None or y <= y1)]
    years = np.arange(ys[0], ys[-1] + 1)
    per = [G.load_year(int(y)) for y in years]
    keys = list(G.FIELDS) + list(G.INDICES)
    blk = {k: np.concatenate([p[k] for p in per], 0) for k in keys}  # (nY*12, 2, ...)
    n = np.concatenate([p["n"] for p in per], 0)
    full = np.array(
        [[n[i, h] >= G.expected_samples(int(years[i // 12]), i % 12 + 1, h) for h in (0, 1)] for i in range(len(n))]
    )
    cal, win = {}, {}
    for k in keys:
        b = blk[k]
        ext = (1,) * (b.ndim - 2)
        w = n.reshape(n.shape + ext).astype(np.float32)
        c = (b[:, 0] * w[:, 0] + b[:, 1] * w[:, 1]) / np.maximum(w[:, 0] + w[:, 1], 1)
        c[~full.all(1)] = np.nan
        cal[k] = c
        wv = np.full_like(c, np.nan)
        # window of month i = block 1 of month i-1 + block 0 of month i
        ok = full[:-1, 1] & full[1:, 0]
        ww = (b[:-1, 1] * w[:-1, 1] + b[1:, 0] * w[1:, 0]) / np.maximum(w[:-1, 1] + w[1:, 0], 1)
        ww[~ok] = np.nan
        wv[1:] = ww
        win[k] = wv
    return Archive(years=years, cal=cal, win=win, n=n)


def month_series(arr: np.ndarray, years: np.ndarray, m: int, y0: int) -> np.ndarray:
    """(nY_out, ...) values of calendar month m (1-12; may be 0 or 13 to step into the neighbouring year) for the
    years `years`, from a flat monthly array whose first year is y0. Out of range -> NaN."""
    out = np.full((len(years), *arr.shape[1:]), np.nan, np.float32)
    for i, y in enumerate(years):
        t = (int(y) - y0) * 12 + (m - 1)
        if 0 <= t < len(arr):
            out[i] = arr[t]
    return out


def clim_anom(x: np.ndarray, years: np.ndarray) -> np.ndarray:
    sel = (years >= CLIM[0]) & (years <= CLIM[1])
    return x - np.nanmean(x[sel], 0)


# -------------------------------------------------------------------------------------------------- statistics
def loess(x: np.ndarray, years: np.ndarray, half: float = LOESS_HALF) -> np.ndarray:
    """Tricube local-linear fit of x (nY, P) along years, evaluated at every year (NaNs ignored)."""
    yy = years.astype(float)
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    out = np.full(x.shape, np.nan, np.float64)
    for i, y in enumerate(yy):
        d = np.abs(yy - y) / half
        w = np.where(d < 1, (1 - d**3) ** 3, 0.0)[:, None] * ok
        dy = (yy - y)[:, None]
        s0, s1, s2 = w.sum(0), (w * dy).sum(0), (w * dy * dy).sum(0)
        t0, t1 = (w * x0).sum(0), (w * dy * x0).sum(0)
        det = s0 * s2 - s1 * s1
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(det > 1e-9, (s2 * t0 - s1 * t1) / det, t0 / np.maximum(s0, 1e-9))
    return out.astype(np.float32)


def linfit_at(x: np.ndarray, years: np.ndarray, at: float) -> np.ndarray:
    """OLS line through x (nY, P) along years (NaNs ignored), evaluated at `at`."""
    ok = np.isfinite(x)
    yy = np.where(ok, years[:, None].astype(float), 0.0)
    n = ok.sum(0)
    x0 = np.where(ok, x, 0.0)
    my = yy.sum(0) / np.maximum(n, 1)
    mx = x0.sum(0) / np.maximum(n, 1)
    dy = np.where(ok, yy - my, 0.0)
    b = (dy * (x0 - mx)).sum(0) / np.maximum((dy * dy).sum(0), 1e-9)
    return np.where(n >= 15, mx + b * (at - my), np.nan)


def regress(Xtr: np.ndarray, ytr: np.ndarray, Xnew: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Multiple regression through the origin of ytr (n, P) on predictors Xtr (n, k) shared by all points; returns
    (P,) at Xnew (k,). Predictors and predictands are detrended anomalies (zero mean by construction), and fitting
    no intercept avoids the cross-validation degeneracy of Barnston and van den Dool (1993): a training mean that
    leaves the target year out is shifted against that year, so a no-skill regression scores negative."""
    A = Xtr.T @ Xtr
    A = A + ridge * np.trace(A) / len(A) * np.eye(len(A))
    B = np.linalg.solve(A, Xtr.T @ ytr)  # (k, P)
    return Xnew @ B


def local_regress(xtr: np.ndarray, ytr: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    """Per-point regression through the origin (see regress): xtr (n, k, P), ytr (n, P), xnew (k, P)."""
    sd = np.maximum(np.sqrt((xtr**2).mean(0)), 1e-6)  # standardised, so the ridge treats sea ice and temperature alike
    xc = xtr / sd
    A = np.einsum("nip,njp->pij", xc, xc)
    k = A.shape[1]
    A = A + 0.05 * np.einsum("pii->p", A)[:, None, None] / k * np.eye(k)
    b = np.einsum("nip,np->pi", xc, ytr)
    beta = np.linalg.solve(A, b[..., None])[..., 0]  # (P, k)
    return np.einsum("pk,kp->p", beta, xnew / sd)


# ----------------------------------------------------------------------------------------------- the predictors
def nh_rows(v: str) -> slice:
    """Rows of a field's archive grid that make up the 0-90N predictand grid (precipitation is stored 30S-90N)."""
    return slice(int(np.searchsorted(G.LAT_ALL, 0.0)), None) if G.FIELDS.get(v) is G.LAT_ALL else slice(None)


class MonthBlock:
    """One target calendar month's library: per year, the predictor window's state and the target month's anomalies."""

    def __init__(self, A: Archive, month: int):
        self.month = month
        self.years = A.years
        nY = len(self.years)
        y0 = int(A.years[0])
        im = month - 1 if month > 1 else 12  # the issue month
        yi = self.years + (0 if month > 1 else -1)
        # predictands: anomalies vs 1991-2020, flattened (nY, P), 0-90N
        self.obs, self.det, self.seas, self.dseas = {}, {}, {}, {}
        for v in TARGETS:
            r = nh_rows(v)
            self.obs[v] = clim_anom(month_series(A.cal[v], self.years, month, y0)[:, r].reshape(nY, -1), self.years)
            s3 = np.stack(
                [
                    clim_anom(month_series(A.cal[v], self.years, month + d, y0)[:, r].reshape(nY, -1), self.years)
                    for d in (-1, 0, 1)
                ]
            )
            self.seas[v] = np.nanmean(s3, 0)
            # detrended anomaly: minus the slow (LOESS) part of the 3-month anomaly
            slow = loess(self.seas[v], self.years)
            self.det[v] = self.obs[v] - slow
            self.dseas[v] = self.seas[v] - slow
        # predictor window (issued in month im of year yi), detrended like the predictands; precipitation also as
        # its tropical band (prcp_t, 20S-20N) for the constructed analogue
        self.win = {}
        for v in (*WIN_FIELDS, "u50eq", "u50v"):
            x = month_series(A.win[v], yi, im, y0)
            if v == "prcp":
                lat = G.LAT_ALL
                xt = x[:, (lat >= -20) & (lat <= 20)].reshape(nY, -1)
                a = clim_anom(xt, self.years)
                self.win["prcp_t"] = a - loess(a, self.years)
                x = x[:, nh_rows(v)]
            a = clim_anom(x.reshape(nY, -1), self.years)
            self.win[v] = a - loess(a, self.years) if v not in ("u50eq", "u50v") else a
        sst = clim_anom(month_series(A.win["sst"], yi, im, y0), self.years)  # (nY, lat, lon), not detrended
        lat_s = G.LAT_SST
        ilat = (lat_s >= -5) & (lat_s <= 5)
        ilon = (G.LON >= 190) & (G.LON <= 240)
        n34 = np.nanmean(sst[:, ilat][:, :, ilon], axis=(1, 2))
        itr = (lat_s >= -20) & (lat_s <= 20)
        wt = np.cos(np.deg2rad(lat_s[itr]))[:, None] * np.ones(len(G.LON))
        t = sst[:, itr]
        trop = np.nansum(np.nan_to_num(t) * wt * np.isfinite(t), (1, 2)) / np.nansum(wt * np.isfinite(t), (1, 2))
        self.n34 = n34
        self.n34r = n34 - trop  # relative Niño-3.4 (van Oldenborgh et al. 2021), unscaled
        # RONI-style value for display: rescaled so its 1991-2020 standard deviation matches Niño-3.4's (the scale
        # does not change the regression)
        c = (self.years >= CLIM[0]) & (self.years <= CLIM[1])
        self.roni = self.n34r * np.nanstd(n34[c]) / np.nanstd(self.n34r[c])
        # El Niño and La Niña enter as separate predictors (their teleconnections are not mirror images)
        enso = [np.maximum(self.n34r, 0), np.minimum(self.n34r, 0)] if ENSO_SPLIT else [self.n34r, 0 * self.n34r]
        self.idx = np.stack([*enso, self.win["u50eq"][:, 0], self.win["u50v"][:, 0]], 1)
        self.valid = np.isfinite(self.idx).all(1) & np.isfinite(self.win["z500"]).all(1)
        for v in TARGETS:
            self.valid &= np.isfinite(self.obs[v]).all(1)
        self.ready = np.isfinite(self.idx).all(1) & np.isfinite(self.win["z500"]).all(1)  # predictors only
        self.time = self.years + (month - 1) / 12.0


class Problem:
    """The target month's library, pooled with the two neighbouring calendar months (each with its own predictor
    window and anomalies), which triples the sample behind every regression; the trend and the analogue years use
    the target month alone."""

    def __init__(self, A: Archive, month: int):
        self.POOL = POOL
        self.A = A
        self.month = month
        self.years = A.years
        self.blocks = [MonthBlock(A, (month - 1 + d) % 12 + 1) for d in self.POOL]
        self.c = self.blocks[self.POOL.index(0)]  # the target month
        cat = lambda get: np.concatenate([get(b) for b in self.blocks], 0)  # noqa: E731
        self.L = {
            "time": cat(lambda b: b.time),
            "valid": cat(lambda b: b.valid),
            "idx": cat(lambda b: b.idx),
            **{f"det_{v}": cat(lambda b, v=v: b.det[v]) for v in TARGETS},
            **{f"win_{v}": cat(lambda b, v=v: b.win[v]) for v in (*WIN_FIELDS, "prcp_t")},
        }
        self.n34r, self.n34, self.roni = self.c.n34r, self.c.n34, self.c.roni
        self.obs, self.idx = self.c.obs, self.c.idx
        # tropical SST (20S-20N) for the constructed analogue and the analogue distance: there SST anomalies force
        # the atmosphere; poleward of that they are mostly its footprint, so they would only re-encode persistence
        latg = np.repeat(G.LAT_SST, len(G.LON))
        s = self.L["win_sst"]
        self.sst_ok = (np.isfinite(s).mean(0) >= 0.9) & (latg >= -SST_LAT[0]) & (latg <= SST_LAT[1])
        self.sst_w = np.sqrt(np.cos(np.deg2rad(latg)))
        self.w = (np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))).ravel()
        self.off = len(self.years) * self.POOL.index(0)  # library index of the target month's first year

    def available(self, y: int, need_target: bool = True) -> bool:
        i = int(y - self.years[0])
        if not (0 <= i < len(self.years)):
            return False
        return bool(self.c.valid[i] if need_target else self.c.ready[i])

    # ------------------------------------------------------------------- one forecast, given the training years
    def forecast(self, y: int, exclude_years: float, target_year: int) -> dict:
        """Component forecasts {comp: {var: (P,)}} for year y. The library leaves out every sample within
        `exclude_years` of the target (in time) and every sample of `target_year` (the outlook being issued)."""
        i = int(y - self.years[0])
        ti = y + (self.month - 1) / 12.0
        L = self.L
        tr = L["valid"] & (np.abs(L["time"] - ti) > exclude_years) & (np.floor(L["time"]) != target_year)
        center = np.zeros_like(tr)
        center[self.off : self.off + len(self.years)] = True
        lib = {k: (tr if k in POOLED else tr & center) for k in COMPONENTS}
        out = {c: {} for c in COMPONENTS}
        # TR: past-only linear trend of the target month's 3-month anomaly, fitted after removing the part linearly
        # explained by relative Niño-3.4, so a run of La Niña (or El Niño) years near the end of the window is not
        # extrapolated as climate change (the slope on the index is fitted on the training years, detrended)
        c = self.c
        trc = tr[self.off : self.off + len(self.years)]
        past = (self.years >= y - TREND_YEARS) & (self.years < y)
        x = c.n34r
        for v in TARGETS:
            src = c.seas[v]
            if TREND_ENSO_ADJUST:
                ok = trc & np.isfinite(c.dseas[v]).all(1)
                b = (x[ok, None] * c.dseas[v][ok]).sum(0) / (x[ok] ** 2).sum()
                src = src - x[:, None] * b
            s = np.where(past[:, None], src, np.nan)
            out["TR"][v] = np.nan_to_num(linfit_at(s, self.years.astype(float), float(y)))
        # IX: index regression (El Niño, La Niña, QBO, vortex), pooled library
        m = lib["IX"]
        X = L["idx"][m]
        sd = np.maximum(np.sqrt((X**2).mean(0)), 1e-9)
        for v in TARGETS:
            out["IX"][v] = regress(X / sd, L[f"det_{v}"][m], c.idx[i] / sd, ridge=0.05)
        # CA: constructed analogue on tropical SST, pooled library
        S = self.ca_state()
        m = lib["CA"]
        Str, s0 = S[m], S[self.off + i]
        Gm = Str @ Str.T
        lam = CA_RIDGE * np.trace(Gm) / len(Gm)
        alpha = np.linalg.solve(Gm + lam * np.eye(len(Gm)), Str @ s0)
        for v in TARGETS:
            out["CA"][v] = alpha @ L[f"det_{v}"][m]
        self.last_alpha = (L["time"][m], alpha)
        # PE: local persistence; 2 m temperature also on the window's sea ice, snow depth and topsoil moisture,
        # precipitation on topsoil moisture (a surface predictor whose anomalies are negligible at a point, e.g. snow
        # in the subtropics, is left out there)
        m = lib["PE"]
        for v in TARGETS:
            keys = [f"win_{v}"] + [f"win_{k}" for k in PE_EXTRA.get(v, ())]
            xtr = np.stack([np.nan_to_num(L[k][m]) for k in keys], 1)
            xnew = np.stack([np.nan_to_num(L[k][self.off + i]) for k in keys], 0)
            live = np.sqrt((xtr**2).mean(0)) > np.array([PE_FLOOR.get(k, 0.0) for k in keys])[:, None]
            out["PE"][v] = local_regress(xtr * live, L[f"det_{v}"][m], xnew * live)
        # AN: analogue years of the target month by state distance
        d = self.state_distance(i, trc)
        order = np.argsort(d)
        pick = np.where(trc)[0][order[:N_ANALOGS]]
        self.last_analogs = [(int(self.years[j]), float(d[k])) for k, j in zip(order[:N_ANALOGS], pick)]
        for v in TARGETS:
            out["AN"][v] = c.det[v][pick].mean(0)
        return out

    def ca_state(self) -> np.ndarray:
        """The constructed analogue's state vector per library sample: tropical SST (sqrt-cos weighted ocean cells),
        and with "prcp_t" in CA_VARS also tropical rainfall, each block scaled to unit total variance."""
        parts = []
        for v in CA_VARS:
            if v == "sst":
                x = np.nan_to_num(self.L["win_sst"][:, self.sst_ok] * self.sst_w[self.sst_ok])
            else:
                lat = np.repeat(G.LAT_ALL[(G.LAT_ALL >= -20) & (G.LAT_ALL <= 20)], len(G.LON))
                x = np.nan_to_num(self.L["win_prcp_t"] * np.sqrt(np.cos(np.deg2rad(lat))))
            parts.append(x / np.sqrt((x**2).sum(1).mean() + 1e-12))
        return np.concatenate(parts, 1)

    def state_distance(self, i: int, tr: np.ndarray) -> np.ndarray:
        """Distance of year i's window state from each training year's (target month only): the standardised
        indices (weight 1 each), the leading 6 tropical SST PCs (total weight 3) and the leading 4 hemispheric
        Z500 PCs (total weight 2)."""
        c = self.c
        X = np.stack([c.n34r, c.win["u50eq"][:, 0], c.win["u50v"][:, 0]], 1)
        parts = [(X[tr] - X[i]) / X[tr].std(0)]
        for key, k, wt in (("sst", 6, 3.0), ("z500", 4, 2.0)):
            if key == "sst":
                F = np.nan_to_num(c.win["sst"][:, self.sst_ok] * self.sst_w[self.sst_ok])
            else:
                F = c.win["z500"] * np.sqrt(self.w)
            Ft = F[tr] - F[tr].mean(0)
            _, _, vt = np.linalg.svd(Ft, full_matrices=False)
            pcs_tr = Ft @ vt[:k].T
            pc_i = (F[i] - F[tr].mean(0)) @ vt[:k].T
            scale = np.sqrt((pcs_tr**2).sum(1).mean())  # rms PC-vector length
            parts.append((pcs_tr - pc_i) / scale * np.sqrt(wt * X.shape[1] / 3.0))
        D = np.concatenate(parts, 1)
        return np.sqrt((D**2).sum(1))


# ----------------------------------------------------------------------------------------------- consolidation
def sphere_kernel(L_km: float | None = None) -> np.ndarray:
    L_km = L_km or SMOOTH_KM
    lat = np.deg2rad(np.repeat(G.LAT_NH, len(G.LON)))
    lon = np.deg2rad(np.tile(G.LON, len(G.LAT_NH)))
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    c = np.clip(xyz @ xyz.T, -1, 1)
    d = EARTH_KM * np.arccos(c)
    K = np.exp(-((d / L_km) ** 2)) * np.cos(lat)[None, :]
    return (K / K.sum(1, keepdims=True)).astype(np.float32)


def nnls_batched(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Non-negative solution of A β = b for every point: A (P, k, k) normal matrix, b (P, k). Exhaustive over the
    active sets (k = 5: 31 subsets), choosing the feasible subset of least quadratic loss ½βᵀAβ − bᵀβ."""
    P, k = b.shape
    best = np.zeros((P, k))
    best_loss = np.zeros(P)  # β = 0 has loss 0
    for s in (s for r in range(1, k + 1) for s in combinations(range(k), r)):
        idx = list(s)
        As = A[:, idx][:, :, idx]
        bs = b[:, idx]
        try:
            x = np.linalg.solve(As, bs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            continue
        feas = (x >= 0).all(1)
        loss = 0.5 * np.einsum("pi,pij,pj->p", x, As, x) - (bs * x).sum(1)
        better = feas & (loss < best_loss - 1e-12)
        if better.any():
            best[better] = 0
            best[np.ix_(better, idx)] = x[better]
            best_loss[better] = loss[better]
    return best


def consolidation_weights(F: np.ndarray, Y: np.ndarray, K: np.ndarray, ridge: float | None = None) -> np.ndarray:
    """F (n, k, P) component hindcasts, Y (n, P) observed anomalies -> β (P, k) ≥ 0 from the spatially smoothed
    normal equations, ridge-damped toward zero (λ = ridge × mean diagonal)."""
    A = np.einsum("nip,njp->pij", F, F)
    b = np.einsum("nip,np->pi", F, Y)
    A = np.einsum("qp,pij->qij", K, A)
    b = K @ b
    k = A.shape[1]
    A = A + (CONS_RIDGE if ridge is None else ridge) * np.einsum("pii->p", A)[:, None, None] / k * np.eye(k)
    return nnls_batched(A, b)


def target_label(y: int, m: int) -> str:
    return f"{calendar.month_name[m]} {y}"
