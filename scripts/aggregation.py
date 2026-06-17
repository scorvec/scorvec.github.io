"""
Aggregation from per-turbine power to BA / ISO totals.

Mapping path:
    USWTDB.eia_id  →  EIA-860 plant_code  →  balancing_authority_code  →  ISO

EIA-860 (annual electric generator survey, https://www.eia.gov/electricity/data/eia860/)
provides `plant_code → balancing_authority_code` in the "Plant" sheet.
The set of BA codes is enumerated by EIA; we map the seven major US ISOs
(plus a "non-ISO" bucket) using the table below.

For operational use you will want to refresh EIA-860 yearly. The mapping
is *plant-level*, not turbine-level — turbines at the same `eia_id` go
to the same BA.

If a turbine's `eia_id` is missing from the EIA-860 mapping (very small
plants, recent commissions, repowers between EIA-860 vintages), we fall
back to a state-level default BA. The state-level fallback is
deliberately coarse and flagged so you can audit it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# EIA Balancing Authority code → ISO label.
# Reference: EIA-860 codebook / EIA-930 BA list.
BA_TO_ISO = {
    "ERCO": "ERCOT",
    "CISO": "CAISO",
    "MISO": "MISO",
    "PJM":  "PJM",
    "SWPP": "SPP",
    "ISNE": "ISO-NE",
    "NYIS": "NYISO",
    # Non-ISO BAs in wind-heavy regions
    "BPAT": "BPA",
    "PSCO": "PSCo",
    "PNM":  "PNM",
    "AZPS": "APS",
    "WACM": "WAPA-RM",
    "WAUW": "WAPA-UM",
    "PACE": "PacifiCorp-East",
    "PACW": "PacifiCorp-West",
    "NWMT": "NorthWestern",
    "AVA":  "Avista",
    "IPCO": "Idaho Power",
    "PGE":  "Portland GE",
    "PSEI": "Puget Sound",
    "TVA":  "TVA",
    "SOCO": "Southern Co",
    "DUK":  "Duke",
    "FPL":  "FPL",
    "SC":   "South Carolina E&G",
    "SCEG": "Dominion SC",
    "SEC":  "Seminole",
    "SPA":  "Southwestern Power Admin",
    "AECI": "Associated Electric",
    "EPE":  "El Paso Electric",
}


# Coarse state → likely-BA fallback. Only used when EIA-860 mapping is
# unavailable; flagged in the output so you know it's a fallback.
STATE_DEFAULT_BA = {
    "TX": "ERCO",   # most TX wind is in ERCOT (Panhandle is in SPP, see notes)
    "OK": "SWPP",
    "KS": "SWPP",
    "NE": "SWPP",
    "ND": "MISO",
    "SD": "SWPP",
    "IA": "MISO",
    "IL": "PJM",    # most IL wind is MISO; flagged for audit
    "IN": "MISO",
    "MN": "MISO",
    "WI": "MISO",
    "MI": "MISO",
    "MO": "MISO",
    "OH": "PJM",
    "PA": "PJM",
    "WV": "PJM",
    "VA": "PJM",
    "NJ": "PJM",
    "MD": "PJM",
    "NY": "NYIS",
    "MA": "ISNE",
    "ME": "ISNE",
    "NH": "ISNE",
    "VT": "ISNE",
    "CT": "ISNE",
    "RI": "ISNE",
    "CA": "CISO",
    "NV": "CISO",
    "AZ": "AZPS",
    "NM": "PNM",
    "CO": "PSCO",
    "WY": "PACE",
    "MT": "NWMT",
    "ID": "IPCO",
    "WA": "BPAT",
    "OR": "BPAT",
    "UT": "PACE",
}


def load_eia860_ba_map(path: str | Path) -> pd.DataFrame:
    """Load the EIA-860 'Plant' sheet and produce a plant_code → BA map.

    The annual EIA-860 ZIP unpacks to several Excel workbooks. The
    relevant one is `2___Plant_Y<year>.xlsx`, sheet `Plant`. Columns of
    interest: `Plant Code` and `Balancing Authority Code`.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name="Plant", header=1)
    else:
        df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    pc_col = next(c for c in df.columns if c.lower().startswith("plant code"))
    ba_col = next(c for c in df.columns
                  if c.lower().startswith("balancing authority code"))

    out = df[[pc_col, ba_col]].rename(columns={
        pc_col: "eia_id",
        ba_col: "ba_code",
    })
    out["eia_id"] = pd.to_numeric(out["eia_id"], errors="coerce").astype("Int64")
    out["ba_code"] = out["ba_code"].astype(str).str.strip().str.upper()
    out["iso"] = out["ba_code"].map(BA_TO_ISO).fillna("NON-ISO")
    return out.dropna(subset=["eia_id"])


def attach_ba_iso(inventory: pd.DataFrame,
                  eia860_map: pd.DataFrame | None,
                  overrides_csv: str | Path | None = "plant_overrides.csv"
                  ) -> pd.DataFrame:
    """Add ba_code, iso, ba_source columns to the inventory.

    Resolution order per turbine:
      1. plant_overrides.csv (manual eia_id → ba_code mapping, audited)
      2. EIA-860 plant_code → balancing_authority_code
      3. State-level default BA fallback

    Each row carries a `ba_source` provenance label so misroutes are
    easy to find later.
    """
    out = inventory.copy()

    if eia860_map is not None and not eia860_map.empty:
        m = eia860_map.set_index("eia_id")
        out["ba_code"] = out["eia_id"].map(m["ba_code"])
        out["iso"] = out["ba_code"].map(BA_TO_ISO).fillna("NON-ISO")
        out["ba_source"] = np.where(out["ba_code"].notna(),
                                    "eia860", None)
    else:
        out["ba_code"] = pd.Series([pd.NA] * len(out), dtype="object")
        out["iso"] = pd.Series([pd.NA] * len(out), dtype="object")
        out["ba_source"] = pd.Series([pd.NA] * len(out), dtype="object")

    # Fallback for unmapped turbines
    miss = out["ba_code"].isna()
    out.loc[miss, "ba_code"] = out.loc[miss, "t_state"].map(STATE_DEFAULT_BA)
    out.loc[miss & out["ba_code"].notna(), "ba_source"] = "state_fallback"

    # Apply manual overrides last so they win over both EIA-860 and state
    if overrides_csv is not None:
        ov_path = Path(overrides_csv)
        if ov_path.exists():
            try:
                ov = pd.read_csv(ov_path, comment="#")
            except Exception as e:
                log.warning("Could not read overrides %s: %s", ov_path, e)
                ov = pd.DataFrame()
            if not ov.empty and {"eia_id", "ba_code"}.issubset(ov.columns):
                ov_map = dict(zip(ov["eia_id"].astype("Int64"),
                                  ov["ba_code"].astype(str)))
                override_mask = out["eia_id"].isin(ov_map.keys())
                if override_mask.any():
                    out.loc[override_mask, "ba_code"] = out.loc[override_mask,
                                                                "eia_id"].map(ov_map)
                    out.loc[override_mask, "ba_source"] = "override"
                    log.info("Applied %d plant override(s) covering %d turbines",
                             len(ov_map), int(override_mask.sum()))

    out["iso"] = out["ba_code"].map(BA_TO_ISO).fillna("NON-ISO")

    return out


# Project-level breakouts. Plants in this map appear as standalone regions
# in the `iso` aggregation in addition to being summed into their parent ISO
# total. This is for headline projects that are interesting to track on
# their own (e.g. SunZia at 1+ GW exported from NM into CAISO).
#
# Maps eia_id -> display name shown in the dashboard dropdown.
PROJECT_BREAKOUTS: dict[int, str] = {
    99001: "SunZia (CAISO)",
}


def aggregate(forecast_df: pd.DataFrame,
              level: str = "ba",
              power_col: str = "power_net_kW") -> pd.DataFrame:
    """Sum per-turbine power to a chosen level.

    Parameters
    ----------
    forecast_df : DataFrame
        Output of `forecast_fleet`, with `ba_code`, `iso`, `valid_time`.
    level : {"plant", "ba", "iso", "state", "national"}
    power_col : str
        Column to sum (default net power after losses).

    For `level="iso"`, also emits one row per project listed in
    PROJECT_BREAKOUTS — these are treated as their own region in the
    output (so the dashboard dropdown lists them), but their generation
    is still included in the parent ISO total. This is useful for
    isolated visibility into headline projects without breaking the
    ISO-level apples-to-apples comparison against gridstatus actuals.

    Returns
    -------
    DataFrame with the group-key column(s), `valid_time`, and `MW`.
    """
    keys = {
        "plant":    ["eia_id", "p_name"],
        "ba":       ["ba_code"],
        "iso":      ["iso"],
        "state":    ["t_state"],
        "national": [],
    }[level]

    group_cols = keys + ["valid_time"]
    agg = (forecast_df.groupby(group_cols)[power_col].sum() / 1000.0)
    agg.name = "MW"
    out = agg.reset_index()

    # Project breakouts at iso level only
    if level == "iso" and PROJECT_BREAKOUTS:
        for eia_id, label in PROJECT_BREAKOUTS.items():
            mask = forecast_df["eia_id"] == eia_id
            if not mask.any():
                continue
            proj = (forecast_df.loc[mask].groupby("valid_time")[power_col]
                    .sum() / 1000.0).reset_index()
            proj["iso"] = label
            proj = proj.rename(columns={power_col: "MW"})[["iso", "valid_time", "MW"]]
            out = pd.concat([out, proj], ignore_index=True)

    return out
