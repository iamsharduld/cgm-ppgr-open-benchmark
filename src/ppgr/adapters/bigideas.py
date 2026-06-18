"""BIG IDEAs Lab adapter (docs/bigideas-schema.md).

Reads Dexcom_<ID>.csv (CGM) and Food_Log_<ID>.csv (meals) for subjects 001-016
and emits a `Cohort` with the shared tidy interface.

Verified data quirks handled here:
  * Dexcom CSV: first ~12 rows are metadata; glucose rows are `Event Type==EGV`,
    value in `Glucose Value (mg/dL)`, time in `Timestamp (YYYY-MM-DDThh:mm:ss)`.
  * Food_Log: meal anchor = `time_begin`. Multiple food rows can share one
    `time_begin` (components of one eating event) -> aggregated (sum macros).
  * Subject 003's Food_Log has NO header and only 11 columns (missing
    `time_end`, `protein`, `total_fat`) -> handled explicitly; protein/fat = NaN.
"""
from __future__ import annotations

import os

import pandas as pd

from ppgr.loader import MEAL_COLUMNS, Cohort

STD_FOODLOG_COLS = [
    "date", "time", "time_begin", "time_end", "logged_food", "amount", "unit",
    "searched_food", "calorie", "total_carb", "dietary_fiber", "sugar",
    "protein", "total_fat",
]
# subject 003: headerless, 11 columns (no time_end / protein / total_fat)
ALT_FOODLOG_COLS = [
    "date", "time", "time_begin", "logged_food", "amount", "unit",
    "searched_food", "calorie", "total_carb", "dietary_fiber", "sugar",
]

# macro fields aggregated within a single eating event (same time_begin)
_SUM_FIELDS = ["calorie", "total_carb", "dietary_fiber", "protein", "total_fat"]


def _read_cgm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    egv = df[df["Event Type"] == "EGV"].copy()
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(egv["Timestamp (YYYY-MM-DDThh:mm:ss)"]),
            "glucose": pd.to_numeric(egv["Glucose Value (mg/dL)"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "glucose"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def _read_foodlog(path: str) -> pd.DataFrame:
    """Read a Food_Log file, handling the standard and the headerless-003 layout."""
    head = pd.read_csv(path, nrows=1)
    if "time_begin" in head.columns:
        df = pd.read_csv(path)
        for c in ("protein", "total_fat", "dietary_fiber"):
            if c not in df.columns:
                df[c] = pd.NA
    else:
        # headerless variant (subject 003): infer by column count
        ncol = head.shape[1]
        names = ALT_FOODLOG_COLS if ncol == len(ALT_FOODLOG_COLS) else STD_FOODLOG_COLS
        df = pd.read_csv(path, header=None, names=names)
        for c in ("protein", "total_fat"):
            if c not in df.columns:
                df[c] = pd.NA
    return df


def load(root: str, subjects: list[str] | None = None) -> Cohort:
    """Load the BIG IDEAs cohort from the PhysioNet 1.1.2 directory tree.

    `root` is .../big-ideas-glycemic-wearable/1.1.2
    """
    if subjects is None:
        subjects = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )

    meal_rows = []
    cgm: dict[str, pd.DataFrame] = {}

    for sid in subjects:
        sdir = os.path.join(root, sid)
        cgm[sid] = _read_cgm(os.path.join(sdir, f"Dexcom_{sid}.csv"))

        fl = _read_foodlog(os.path.join(sdir, f"Food_Log_{sid}.csv"))
        fl = fl[fl["time_begin"].notna()].copy()
        fl["time_begin"] = pd.to_datetime(fl["time_begin"], errors="coerce")
        fl = fl[fl["time_begin"].notna()]
        for c in _SUM_FIELDS:
            fl[c] = pd.to_numeric(fl[c], errors="coerce")

        # aggregate food rows that share a time_begin into one eating event.
        # carbs/cal/fiber/protein/fat sum across components; a component with a
        # missing macro contributes 0 to that sum unless ALL components miss it
        # (then the aggregate is NaN -> excluded later if that field is required).
        grp = fl.groupby("time_begin", sort=True)
        for tb, g in grp:
            def agg(col: str) -> float:
                vals = g[col]
                if vals.notna().any():
                    return float(vals.sum(min_count=1))
                return float("nan")

            meal_rows.append(
                {
                    "subject_id": sid,
                    "meal_time": tb,
                    "carbs": agg("total_carb"),
                    "protein": agg("protein"),
                    "fat": agg("total_fat"),
                    "fiber": agg("dietary_fiber"),
                    "calorie": agg("calorie"),
                }
            )

    meals = pd.DataFrame(meal_rows, columns=MEAL_COLUMNS)
    meals = meals.sort_values(["subject_id", "meal_time"]).reset_index(drop=True)
    return Cohort(name="bigideas", meals=meals, cgm=cgm)
