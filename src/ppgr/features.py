"""Feature builder (prereg.md §5.3).

Nested feature tiers for the ablation (prereg §5.3, GitHub #24, H4):

1. **macros** — carbs, protein, fat, fiber, calorie (baseline tier).
2. **+context** — meal hour-of-day, meal index within the day.
3. **+anthro/labs** — subject-level anthropometrics + clinical labs from
   `bio.csv` (Age, Gender→0/1, BMI, A1c, Fasting GLU, Insulin, Triglycerides,
   Cholesterol, HDL, LDL). Constant within subject; attached by the adapter.
4. **+prior-response-history** — LEAKAGE-SAFE: for each meal, summaries of the
   subject's OWN earlier meals only (running mean iAUC of prior included meals,
   running mean carbs, count of prior meals). Strictly time-ordered, no future,
   within-subject => valid under LOSO (a test subject's history uses only that
   subject's own past, never any other subject and never the future).
5. **+microbiome** — the 22 ordinal Viome gut-health scores (NOT the 1979 binary
   taxa: p>>n at n=31). Subject-level; attached by the adapter.

Tiers 1-2 are derived per-meal here; tiers 3 & 5 are subject-level columns the
adapter attaches; tier 4 is built here from the per-meal `iauc_pos` outcome and
is therefore computed AFTER inclusion (the outcome must exist).

`add_features`/`feature_sets` keep their original tier-1/2 behaviour so the prior
`run_cgmacros.py` pipeline and tests are unaffected; the ablation-specific tier
helpers are additive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_FEATURES = ["carbs", "protein", "fat", "fiber", "calorie"]
CONTEXT_FEATURES = ["meal_hour", "meal_index_in_day"]

# Tier 3 (anthropometrics + labs): subject-level columns attached by the
# adapter (constant within subject). `gender_bin` is Gender mapped F->0, M->1.
ANTHRO_FEATURES = [
    "age", "gender_bin", "bmi", "a1c", "fasting_glu", "insulin",
    "triglycerides", "cholesterol", "hdl", "ldl",
]

# Tier 4 (prior-response history): leakage-safe within-subject running summaries
# of the subject's OWN earlier meals (see build_history_features).
HISTORY_FEATURES = ["prior_mean_iauc", "prior_mean_carbs", "prior_meal_count"]

# Tier 5 (microbiome): the 22 ordinal Viome gut-health scores. The adapter
# attaches them as `gh_0`..`gh_21` (subject-level, constant within subject).
N_GUT_SCORES = 22
MICROBIOME_FEATURES = [f"gh_{i}" for i in range(N_GUT_SCORES)]


def add_features(meals: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with context features added (macros already present)."""
    df = meals.copy()
    mt = pd.to_datetime(df["meal_time"])
    df["meal_hour"] = mt.dt.hour + mt.dt.minute / 60.0
    df["meal_date"] = mt.dt.date
    df["meal_index_in_day"] = (
        df.sort_values(["subject_id", "meal_time"])
        .groupby(["subject_id", "meal_date"])
        .cumcount()
    )
    # restore original row order
    df = df.loc[meals.index]
    return df.drop(columns=["meal_date"])


def build_history_features(
    meals: pd.DataFrame, target: str = "iauc_pos"
) -> pd.DataFrame:
    """Add leakage-safe prior-response-history features (prereg §5.3 tier 4).

    For each meal, using ONLY that subject's meals strictly earlier in time:
      * `prior_mean_iauc`  — running mean of prior included meals' `target`.
      * `prior_mean_carbs` — running mean of prior meals' carbs.
      * `prior_meal_count` — number of the subject's prior included meals.

    A subject's FIRST meal has no history => prior means are NaN and count 0.
    This is leakage-safe under LOSO: the features depend only on the test
    subject's OWN earlier meals, never on other subjects and never on the
    future. (The running mean of a subject's prior outcomes is the same
    construction the leakage-safe per-person-mean baseline uses, §6.1.2.)
    """
    df = meals.copy()
    order = df.sort_values(["subject_id", "meal_time"]).index
    df = df.loc[order]

    g = df.groupby("subject_id", sort=False)
    # running mean of PRIOR rows only: cumulative sum/count shifted by one.
    prior_n = g.cumcount()  # number of strictly-earlier meals for this subject
    cum_iauc = g[target].cumsum() - df[target]
    cum_carbs = g["carbs"].cumsum() - df["carbs"]
    with np.errstate(invalid="ignore"):
        df["prior_mean_iauc"] = np.where(prior_n > 0, cum_iauc / prior_n, np.nan)
        df["prior_mean_carbs"] = np.where(prior_n > 0, cum_carbs / prior_n, np.nan)
    df["prior_meal_count"] = prior_n.astype(float)

    return df.loc[meals.index]


def feature_sets() -> dict[str, list[str]]:
    """Named feature sets used by the models (prereg §5.3 / §6.3).

    Unchanged from the original tier-1/2 pipeline (used by run_cgmacros.py).
    """
    return {
        "macros": MACRO_FEATURES,
        "macros+context": MACRO_FEATURES + CONTEXT_FEATURES,
    }


# ---- ablation tier definitions (prereg §5.3; GitHub #24) ----

def ablation_tiers() -> dict[str, list[str]]:
    """The five nested feature tiers, in add-in order (prereg §5.3)."""
    return {
        "macros": MACRO_FEATURES,
        "context": CONTEXT_FEATURES,
        "anthro_labs": ANTHRO_FEATURES,
        "history": HISTORY_FEATURES,
        "microbiome": MICROBIOME_FEATURES,
    }


def nested_addin_sets() -> dict[str, list[str]]:
    """Nested add-in feature sets: macros, +context, +anthro/labs, +history,
    +microbiome (each cumulatively adds the next tier; prereg §5.3 / §6.3)."""
    tiers = ablation_tiers()
    names = list(tiers)
    out: dict[str, list[str]] = {}
    cols: list[str] = []
    labels: list[str] = []
    for name in names:
        cols = cols + tiers[name]
        labels.append(name)
        out["+".join(labels)] = list(cols)
    return out


def leave_one_tier_out_sets() -> dict[str, list[str]]:
    """Leave-one-tier-out feature sets: all five tiers MINUS one (prereg §5.3)."""
    tiers = ablation_tiers()
    names = list(tiers)
    out: dict[str, list[str]] = {}
    for drop in names:
        cols: list[str] = []
        for name in names:
            if name == drop:
                continue
            cols += tiers[name]
        out[f"all_minus_{drop}"] = cols
    return out
