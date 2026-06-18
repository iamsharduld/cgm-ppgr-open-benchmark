"""CGMacros adapter (docs/cgmacros-schema.md, prereg.md).

Reads the per-participant time series `CGMacros-0XX/CGMacros-0XX.csv` and the
subject-level `bio.csv`, and emits a `Cohort` with the shared tidy interface.

Verified data quirks handled here (see docs/cgmacros-schema.md "Verified against
the extracted data"):
  * Leading index column `Unnamed: 0` (dropped on read).
  * `Amount Consumed ` carries a TRAILING SPACE in the header (not used here).
  * Photo column is `Image path` (lower-case `p`; not used here).
  * `Timestamp` is `YYYY-MM-DD HH:MM:SS` (date-shifted), per-minute rows.
  * CGM (primary) stream = rows with non-null `Dexcom GL` (prereg §4.3).
  * Meals = rows with non-null `Meal Type` ∈ {Breakfast, Lunch, Dinner, Snacks};
    snacks ARE meals per prereg §3.3. A meal row's `Timestamp` coincides with a
    CGM reading (per-minute rows) => exact meal↔CGM alignment.
  * Diabetes group is DERIVED from `bio.csv` `A1c PDL (Lab)` (%) via ADA
    thresholds (no explicit group column exists): healthy <5.7, pre-DM
    5.7–<6.5, T2D >=6.5. This reproduces the published 15/16/14 split exactly.
  * `bio.csv` `subject` is an int (1..49 with gaps); maps to the zero-padded
    folder id `CGMacros-0{subject:03d}`.

Subject-level feature tiers (prereg §5.3, for the GitHub #24 ablation) are
attached to the per-meal table as constant-within-subject columns when the
files are present in `root`:
  * Tier 3 (anthro/labs) from `bio.csv`: age, gender_bin (F->0/M->1), bmi, a1c,
    fasting_glu, insulin, triglycerides, cholesterol, hdl, ldl.
  * Tier 5 (microbiome) from `gut_health_test.csv`: the 22 ordinal Viome
    gut-health scores as `gh_0`..`gh_21` (NOT the 1979 binary taxa; p>>n at
    n=31). Subjects with no Viome sample get all-NaN scores (XGBoost handles
    NaN natively => no imputation, leakage-safe).
These columns are ADDITIVE: the prior tier-1/2 pipeline ignores them, so its
behaviour is unchanged.
"""
from __future__ import annotations

import os

import pandas as pd

from ppgr.loader import MEAL_COLUMNS, Cohort

# ADA HbA1c (%) thresholds (prereg §3.1; docs/cgmacros-schema.md verified section)
A1C_HEALTHY_MAX = 5.7   # healthy: A1c < 5.7
A1C_PREDM_MAX = 6.5     # pre-DM: 5.7 <= A1c < 6.5 ; T2D: A1c >= 6.5

MEAL_TYPES = {"Breakfast", "Lunch", "Dinner", "Snacks"}

# Tier 3 (anthro/labs): bio.csv column -> tidy feature name (prereg §5.3).
# Trailing spaces match the verified raw headers (docs/cgmacros-schema.md).
BIO_FEATURE_MAP = {
    "Age": "age",
    "BMI": "bmi",
    "A1c PDL (Lab)": "a1c",
    "Fasting GLU - PDL (Lab)": "fasting_glu",
    "Insulin ": "insulin",
    "Triglycerides": "triglycerides",
    "Cholesterol": "cholesterol",
    "HDL": "hdl",
    "LDL (Cal)": "ldl",
}
N_GUT_SCORES = 22  # 22 ordinal Viome gut-health scores (tier 5)


def _subject_to_folder(subject: int) -> str:
    """bio.csv int subject -> zero-padded participant folder/file id."""
    return f"CGMacros-{int(subject):03d}"


def _a1c_group(a1c: float) -> str:
    """ADA-threshold diabetes group from HbA1c (%) (prereg §3.1)."""
    if pd.isna(a1c):
        return "unknown"
    if a1c < A1C_HEALTHY_MAX:
        return "healthy"
    if a1c < A1C_PREDM_MAX:
        return "pre-DM"
    return "T2D"


def _read_participant(
    path: str, glucose_col: str = "Dexcom GL"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cgm_df[timestamp, glucose], meal_df[MEAL_COLUMNS w/o subject_id]).

    CGM = rows with non-null `glucose_col`; meals = rows with non-null `Meal Type`.
    `glucose_col` defaults to `Dexcom GL` (prereg §4.3 primary stream); pass
    `Libre GL` for the CGM-brand sensitivity (prereg §4.3 / issue #25).
    """
    df = pd.read_csv(path, low_memory=False)
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")

    # --- CGM stream: glucose_col (default Dexcom GL, prereg §4.3) ---
    dex = pd.to_numeric(df[glucose_col], errors="coerce")
    cgm = pd.DataFrame({"timestamp": ts, "glucose": dex})
    cgm = cgm.dropna(subset=["timestamp", "glucose"]).sort_values("timestamp")
    cgm = cgm.reset_index(drop=True)

    # --- meals: rows with a non-null Meal Type (Breakfast/Lunch/Dinner/Snacks) ---
    mt = df["Meal Type"].astype("string").str.strip()
    is_meal = mt.notna() & (mt != "") & (mt != "nan")
    m = df.loc[is_meal].copy()
    meals = pd.DataFrame(
        {
            "meal_time": pd.to_datetime(m["Timestamp"], errors="coerce"),
            "meal_type": mt.loc[is_meal].to_numpy(),
            "carbs": pd.to_numeric(m["Carbs"], errors="coerce"),
            "protein": pd.to_numeric(m["Protein"], errors="coerce"),
            "fat": pd.to_numeric(m["Fat"], errors="coerce"),
            "fiber": pd.to_numeric(m["Fiber"], errors="coerce"),
            "calorie": pd.to_numeric(m["Calories"], errors="coerce"),
        }
    )
    meals = meals.dropna(subset=["meal_time"]).reset_index(drop=True)
    return cgm, meals


def _subject_features(root: str, bio: pd.DataFrame) -> dict[str, dict]:
    """Build per-folder subject-level feature dicts (prereg §5.3 tiers 3 & 5).

    Tier 3 (anthro/labs) from `bio.csv`; tier 5 (microbiome) = the 22 ordinal
    Viome gut-health scores from `gut_health_test.csv` (gh_0..gh_21). Returns
    {folder_id: {feature: value}}. Missing files/subjects yield NaN features
    (XGBoost handles NaN natively).
    """
    feats: dict[str, dict] = {}

    # tier 3: anthropometrics + labs (gender_bin: F->0, M->1)
    gender = bio["Gender"].astype("string").str.strip().str.upper()
    bio = bio.assign(gender_bin=gender.map({"F": 0.0, "M": 1.0}))
    for _, row in bio.iterrows():
        sid = row["folder"]
        d: dict[str, float] = {"gender_bin": row["gender_bin"]}
        for src, dst in BIO_FEATURE_MAP.items():
            d[dst] = pd.to_numeric(pd.Series([row.get(src)]), errors="coerce").iloc[0]
        feats[sid] = d

    # tier 5: 22 ordinal Viome gut-health scores -> gh_0..gh_21
    gh_path = os.path.join(root, "gut_health_test.csv")
    if os.path.exists(gh_path):
        gh = pd.read_csv(gh_path)
        score_cols = [c for c in gh.columns if c != "subject"][:N_GUT_SCORES]
        for _, row in gh.iterrows():
            sid = _subject_to_folder(row["subject"])
            d = feats.setdefault(sid, {})
            for i, c in enumerate(score_cols):
                d[f"gh_{i}"] = pd.to_numeric(
                    pd.Series([row[c]]), errors="coerce"
                ).iloc[0]
    # ensure every subject has all gh_ columns (NaN if no Viome sample)
    for sid, d in feats.items():
        for i in range(N_GUT_SCORES):
            d.setdefault(f"gh_{i}", float("nan"))

    return feats


def load(
    root: str,
    subjects: list[str] | None = None,
    glucose_col: str = "Dexcom GL",
) -> Cohort:
    """Load the CGMacros cohort.

    `root` is the directory holding `bio.csv` and the `CGMacros-0XX/` folders
    (i.e. .../data/cgmacros/extracted/CGMacros).

    `subjects`: optional list of folder ids (e.g. ["CGMacros-001", ...]) to
    restrict to. Defaults to every participant present in `bio.csv` that also
    has a per-participant CSV on disk.

    `glucose_col`: CGM stream column. Defaults to `Dexcom GL` (prereg §4.3
    primary stream); pass `Libre GL` for the CGM-brand sensitivity (issue #25).

    Each subject gets a per-subject `group` (healthy/pre-DM/T2D) and `a1c`
    attached as columns on the meal table (constant within subject).
    """
    bio = pd.read_csv(os.path.join(root, "bio.csv"))
    bio["a1c"] = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    bio["group"] = bio["a1c"].map(_a1c_group)
    bio["folder"] = bio["subject"].map(_subject_to_folder)

    folder_meta = bio.set_index("folder")[["a1c", "group"]].to_dict("index")

    # --- subject-level feature tiers (prereg §5.3): anthro/labs + gut-health ---
    subj_features = _subject_features(root, bio)

    if subjects is None:
        subjects = [
            f for f in bio["folder"].tolist()
            if os.path.isdir(os.path.join(root, f))
            and os.path.exists(os.path.join(root, f, f"{f}.csv"))
        ]

    meal_frames = []
    cgm: dict[str, pd.DataFrame] = {}

    for sid in subjects:
        csv_path = os.path.join(root, sid, f"{sid}.csv")
        sub_cgm, sub_meals = _read_participant(csv_path, glucose_col=glucose_col)
        cgm[sid] = sub_cgm

        sub_meals.insert(0, "subject_id", sid)
        meta = folder_meta.get(sid, {"a1c": float("nan"), "group": "unknown"})
        sub_meals["a1c"] = meta["a1c"]
        sub_meals["group"] = meta["group"]

        # attach subject-level tiers (constant within subject; prereg §5.3)
        feat = subj_features.get(sid, {})
        for col, val in feat.items():
            sub_meals[col] = val
        meal_frames.append(sub_meals)

    meals = pd.concat(meal_frames, ignore_index=True)
    # ensure the shared MEAL_COLUMNS exist and lead; keep extras (meal_type,
    # a1c, group) for sanity-checks / population restriction.
    ordered = MEAL_COLUMNS + [c for c in meals.columns if c not in MEAL_COLUMNS]
    meals = meals[ordered]
    meals = meals.sort_values(["subject_id", "meal_time"]).reset_index(drop=True)
    return Cohort(name="cgmacros", meals=meals, cgm=cgm)
