"""Shared cohort interface (prereg.md; docs/*-schema.md).

A `Cohort` exposes a tidy per-meal table and the per-subject CGM series so that
the iAUC/inclusion/feature/eval modules are dataset-agnostic. The BIG IDEAs
adapter is implemented here; a CGMacros adapter can be added with the same
interface (it just needs to populate the same MEAL_COLUMNS + CGM frames).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# tidy per-meal schema (the shared interface)
MEAL_COLUMNS = [
    "subject_id",
    "meal_time",
    "carbs",      # = total_carb (CGMacros: Carbs)
    "protein",    # = protein     (CGMacros: Protein)
    "fat",        # = total_fat   (CGMacros: Fat)
    "fiber",      # = dietary_fiber (CGMacros: Fiber)
    "calorie",    # = calorie     (CGMacros: Calories)
]


@dataclass
class Cohort:
    """A loaded cohort with a shared interface."""

    name: str
    meals: pd.DataFrame                      # columns == MEAL_COLUMNS (+ extras ok)
    cgm: dict[str, pd.DataFrame]             # subject_id -> DataFrame[timestamp, glucose]

    def cgm_window(
        self, subject_id: str, meal_time: pd.Timestamp, lo_min: float, hi_min: float
    ) -> pd.DataFrame:
        """Return CGM rows in [meal_time+lo_min, meal_time+hi_min]."""
        s = self.cgm[subject_id]
        lo = meal_time + pd.Timedelta(minutes=lo_min)
        hi = meal_time + pd.Timedelta(minutes=hi_min)
        m = (s["timestamp"] >= lo) & (s["timestamp"] <= hi)
        return s.loc[m, ["timestamp", "glucose"]]
