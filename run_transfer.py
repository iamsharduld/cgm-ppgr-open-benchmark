"""Cross-cohort TRANSFER experiment (prereg.md §8 generalization / H3; GitHub #23).

The NOVEL core test: train a meal-response model on ALL included meals of one
cohort and predict ALL included meals of the OTHER cohort. The two cohorts are
disjoint by construction (different studies, no shared subjects), so the entire
target cohort is a clean held-out set -- no LOSO and no personal-calibration
split is needed (those guard within-cohort leakage, which cannot occur here).

Cohorts (loaded via the reviewed adapters, UNCHANGED):
  * CGMacros -- PRIMARY non-diabetic population = healthy + pre-DM (31 subjects,
    A1c-derived ADA groups per the adapter; prereg §3.1).
  * BIG IDEAs -- all evaluable subjects (those that yield >=1 included meal;
    4/16 are unevaluable due to verified CGM<->food-log date misalignment, so
    12/16 evaluate -- handled entirely inside inclusion, not here).

TRANSFERRED models (macro-feature only -- comparable features across cohorts):
  1. carb_only        -- LinearRegression(iAUC ~ carbs)              (prereg §6.1.3)
  2. elasticnet_macros -- ElasticNetCV on MACRO_FEATURES             (prereg §6.2.1)
  3. xgboost_macros   -- XGBoost (grouped nested grid) on MACRO_FEATURES (§6.2.2)

EXCLUDED from transfer (explicitly, per the task spec):
  * per_person_mean / population_mean -- need TARGET-cohort calibration (a test
    subject's own level is unknowable from the source cohort); not transferable.
  * *_macros+context -- context features (meal hour / meal-index-in-day) are
    cohort-specific in distribution and not part of the harmonized macro set;
    excluded so the transfer compares like-for-like macro signal only.

Reuses src/ppgr UNCHANGED: adapters, inclusion.apply_inclusion, features.add_features
(for parity of the included-meal frame), and evaluate.py's model constructors,
metric helper (_metrics), Prediction dataclass, and subject-level bootstrap
(_subject_bootstrap_ci) so the CI methodology is identical to the within-cohort runs.

Usage:  PYTHONPATH=src python3 run_transfer.py
Writes: results/transfer_results.csv  and  results/transfer_results.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ppgr.adapters import bigideas, cgmacros
from ppgr.evaluate import (
    Prediction,
    _fit_xgb,
    _grouped_inner_cv,
    _make_elasticnet,
    _metrics,
    _subject_bootstrap_ci,
)
from ppgr.features import MACRO_FEATURES, add_features
from ppgr.inclusion import apply_inclusion

CGMACROS_ROOT = "data/cgmacros/extracted/CGMacros"
BIGIDEAS_ROOT = "data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2"
OUT_DIR = "results"
TARGET = "iauc_pos"
NON_DIABETIC = {"healthy", "pre-DM"}

# transferred models only (macro-comparable); see module docstring for exclusions
TRANSFER_MODELS = ["carb_only", "elasticnet_macros", "xgboost_macros"]

# within-cohort Pearson R for the SAME models, read from the reviewed results CSVs
WITHIN_CSV = {
    "cgmacros": os.path.join(OUT_DIR, "cgmacros_results.csv"),
    "bigideas": os.path.join(OUT_DIR, "bigideas_results.csv"),
}


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.3f}"


def load_included(name: str) -> pd.DataFrame:
    """Load a cohort, restrict population if needed, apply inclusion, add features.

    Returns the included-meal table (one row per included meal) with the iAUC
    target and the shared MACRO_FEATURES columns. Identical feature columns
    across cohorts by construction (both adapters emit MEAL_COLUMNS).
    """
    if name == "cgmacros":
        cohort = cgmacros.load(CGMACROS_ROOT)
        nd = sorted(
            sid for sid, g in
            cohort.meals.groupby("subject_id")["group"].first().items()
            if g in NON_DIABETIC
        )
        cohort.meals = cohort.meals[cohort.meals["subject_id"].isin(nd)].copy()
        cohort.cgm = {s: cohort.cgm[s] for s in nd}
    elif name == "bigideas":
        cohort = bigideas.load(BIGIDEAS_ROOT)
    else:
        raise ValueError(name)

    included, _att = apply_inclusion(cohort)
    included = add_features(included)
    return included


def fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit `model` on the TRAIN cohort, predict the TEST cohort.

    Mirrors evaluate.py's model construction EXACTLY (same helpers, same nested
    inner CV / grid search on the training cohort's subjects, same imputation),
    so the only change vs the within-cohort run is the train/test partition =
    whole-cohort transfer instead of LOSO folds.
    """
    ytr = train[TARGET].to_numpy()

    if model == "carb_only":
        lr = LinearRegression().fit(train[["carbs"]], ytr)
        return lr.predict(test[["carbs"]])

    if model == "elasticnet_macros":
        cols = MACRO_FEATURES
        med = train[cols].median()
        Xtr = train[cols].fillna(med)
        Xte = test[cols].fillna(med)  # impute test with TRAIN medians (no leakage)
        groups_tr = train["subject_id"].to_numpy()
        inner_cv = _grouped_inner_cv(groups_tr)
        en = _make_elasticnet(inner_cv).fit(Xtr, ytr)
        return en.predict(Xte)

    if model == "xgboost_macros":
        cols = MACRO_FEATURES
        groups_tr = train["subject_id"].to_numpy()
        xgb = _fit_xgb(train[cols], ytr, groups_tr)  # native NaN handling -> raw
        return xgb.predict(test[cols])

    raise ValueError(model)


def run_direction(train_name: str, test_name: str,
                  train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Run all transferred models for one TRAIN->TEST direction.

    Returns {model: (Prediction, point_R, R_lo, R_hi, rmse, rmse_lo, rmse_hi,
                      spearman, mae)} computed on the held-out TEST cohort with
    subject-level bootstrap CIs (resampling TEST-cohort subjects, §7.3).
    """
    sid_test = test_df["subject_id"].to_numpy()
    yte = test_df[TARGET].to_numpy()
    out = {}
    for model in TRANSFER_MODELS:
        yhat = np.asarray(fit_predict(model, train_df, test_df), dtype=float)
        pred = Prediction(method=model, subject_id=sid_test,
                          y_true=yte, y_pred=yhat)
        r, r_lo, r_hi = _subject_bootstrap_ci(
            pred, lambda yt, yp: _metrics(yt, yp)["pearson_r"])
        rmse, rmse_lo, rmse_hi = _subject_bootstrap_ci(
            pred, lambda yt, yp: _metrics(yt, yp)["rmse"])
        m = _metrics(yte, yhat)
        out[model] = {
            "pred": pred,
            "pearson_r": r, "pearson_r_lo": r_lo, "pearson_r_hi": r_hi,
            "spearman_r": m["spearman_r"],
            "rmse": rmse, "rmse_lo": rmse_lo, "rmse_hi": rmse_hi,
            "mae": m["mae"],
            "n_meals": len(yte),
            "n_subjects": int(len(np.unique(sid_test))),
        }
    return out


def within_r() -> dict:
    """Read within-cohort Pearson R (+CI) for the transferred models per cohort."""
    w = {}
    for name, path in WITHIN_CSV.items():
        df = pd.read_csv(path).set_index("method")
        w[name] = {
            m: {
                "pearson_r": float(df.loc[m, "pearson_r"]),
                "pearson_r_lo": float(df.loc[m, "pearson_r_lo"]),
                "pearson_r_hi": float(df.loc[m, "pearson_r_hi"]),
                "rmse": float(df.loc[m, "rmse"]),
            }
            for m in TRANSFER_MODELS
        }
    return w


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading cohorts + applying inclusion (reviewed modules) ...")
    cg = load_included("cgmacros")
    bi = load_included("bigideas")
    print(f"  CGMacros (non-diabetic) included meals: {len(cg)} "
          f"from {cg['subject_id'].nunique()} subjects")
    print(f"  BIG IDEAs included meals:               {len(bi)} "
          f"from {bi['subject_id'].nunique()} subjects")

    # feature-harmonization sanity: identical macro columns, same target, ranges
    print("\n===== FEATURE-HARMONIZATION SANITY =====")
    for nm, df in [("cgmacros", cg), ("bigideas", bi)]:
        miss = [c for c in MACRO_FEATURES + [TARGET] if c not in df.columns]
        print(f"  {nm}: macro+target cols present: {not miss} "
              f"(missing={miss})")
        desc = df[MACRO_FEATURES].agg(["mean", "min", "max"]).round(1)
        print(f"    carbs mean/min/max: {desc.loc['mean','carbs']}/"
              f"{desc.loc['min','carbs']}/{desc.loc['max','carbs']} g  | "
              f"iAUC mean: {df[TARGET].mean():.0f} mg/dL·min")
        nan_frac = df[MACRO_FEATURES].isna().mean().round(3).to_dict()
        print(f"    macro NaN fraction: {nan_frac}")

    within = within_r()

    directions = [
        ("cgmacros", "bigideas", cg, bi),  # CGMacros -> BIG IDEAs
        ("bigideas", "cgmacros", bi, cg),  # BIG IDEAs -> CGMacros
    ]

    rows = []
    results = {}
    for tr_name, te_name, tr_df, te_df in directions:
        print(f"\n===== TRANSFER: train {tr_name} -> test {te_name} =====")
        res = run_direction(tr_name, te_name, tr_df, te_df)
        results[(tr_name, te_name)] = res
        for model in TRANSFER_MODELS:
            r = res[model]
            wr = within[te_name][model]  # within-cohort R for the TEST cohort
            degr = wr["pearson_r"] - r["pearson_r"]
            rows.append({
                "direction": f"{tr_name}->{te_name}",
                "train_cohort": tr_name,
                "test_cohort": te_name,
                "model": model,
                "n_test_meals": r["n_meals"],
                "n_test_subjects": r["n_subjects"],
                "within_R": wr["pearson_r"],
                "within_R_lo": wr["pearson_r_lo"],
                "within_R_hi": wr["pearson_r_hi"],
                "transfer_R": r["pearson_r"],
                "transfer_R_lo": r["pearson_r_lo"],
                "transfer_R_hi": r["pearson_r_hi"],
                "delta_R_degradation": degr,
                "transfer_spearman": r["spearman_r"],
                "transfer_rmse": r["rmse"],
                "transfer_rmse_lo": r["rmse_lo"],
                "transfer_rmse_hi": r["rmse_hi"],
                "transfer_mae": r["mae"],
                "within_rmse": wr["rmse"],
            })
            print(f"  {model:20s} transfer R={fmt(r['pearson_r'])} "
                  f"[{fmt(r['pearson_r_lo'])},{fmt(r['pearson_r_hi'])}]  "
                  f"within R={fmt(wr['pearson_r'])}  "
                  f"ΔR(degr)={fmt(degr)}  "
                  f"RMSE={fmt(r['rmse'])} [{fmt(r['rmse_lo'])},{fmt(r['rmse_hi'])}]")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "transfer_results.csv"), index=False)

    _write_md(table, cg, bi)
    print("\nWrote results/transfer_results.csv and results/transfer_results.md")


def _write_md(table: pd.DataFrame, cg: pd.DataFrame, bi: pd.DataFrame) -> None:
    lines = []
    lines.append("# Cross-cohort TRANSFER — results (prereg §8 / H3; GitHub #23)\n")
    lines.append(
        "**The novel core test.** Train a meal-response model on ALL included "
        "meals of one cohort, predict ALL included meals of the OTHER cohort. "
        "The two cohorts are disjoint by construction (different studies, no "
        "shared subjects), so the whole target cohort is a clean held-out set — "
        "no LOSO and no personal-calibration split are needed. Primary metric: "
        "Pearson R; subject-level bootstrap 95% CIs resample the **test** "
        "cohort's subjects (prereg §7.3). Pipeline reuses `src/ppgr` unchanged "
        "(adapters, `apply_inclusion`, model constructors, metric + bootstrap "
        "helpers in `evaluate.py`).\n"
    )
    lines.append("## Cohorts (included meals)\n")
    lines.append(
        f"- **CGMacros** — non-diabetic (healthy + pre-DM, A1c-derived per "
        f"prereg §3.1): **{len(cg)} included meals** from "
        f"**{cg['subject_id'].nunique()} subjects**.")
    lines.append(
        f"- **BIG IDEAs** — all evaluable subjects: **{len(bi)} included meals** "
        f"from **{bi['subject_id'].nunique()} subjects** (12/16; 4 unevaluable "
        "due to verified CGM↔food-log date misalignment, handled in inclusion).\n")

    lines.append("## Transferred models (macro-feature only)\n")
    lines.append(
        "Only models whose features are **comparable across cohorts** are "
        "transferred: `carb_only` (linear on carbs), `elasticnet_macros`, "
        "`xgboost_macros` (both on the harmonized macros carbs/protein/fat/"
        "fiber/calorie).\n")
    lines.append("**Excluded from transfer (explicit):**")
    lines.append(
        "- `per_person_mean` / `population_mean` — require **target-cohort "
        "calibration** (a test subject's own level is unknowable from the source "
        "cohort); not a transferable function of meal features.")
    lines.append(
        "- `*_macros+context` — **context features** (meal hour, meal-index-in-"
        "day) are cohort-specific and outside the harmonized macro set; excluded "
        "so transfer compares like-for-like macro signal.\n")

    lines.append("## Transfer results: [direction × model]\n")
    lines.append(
        "| direction | model | n_test meals (subj) | within-R [95% CI] | "
        "transfer-R [95% CI] | ΔR (degradation = within − transfer) | "
        "transfer RMSE [95% CI] | transfer Spearman ρ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        lines.append(
            f"| {r['direction']} | {r['model']} | "
            f"{int(r['n_test_meals'])} ({int(r['n_test_subjects'])}) | "
            f"{fmt(r['within_R'])} [{fmt(r['within_R_lo'])}, {fmt(r['within_R_hi'])}] | "
            f"{fmt(r['transfer_R'])} [{fmt(r['transfer_R_lo'])}, {fmt(r['transfer_R_hi'])}] | "
            f"{fmt(r['delta_R_degradation'])} | "
            f"{fmt(r['transfer_rmse'])} [{fmt(r['transfer_rmse_lo'])}, {fmt(r['transfer_rmse_hi'])}] | "
            f"{fmt(r['transfer_spearman'])} |")
    lines.append("")
    lines.append(
        "- *within-R* is the within-cohort LOSO Pearson R for the **same model "
        "on the test cohort**, read from `results/{cgmacros,bigideas}_results.csv` "
        "(the reviewed within-cohort runs).")
    lines.append(
        "- *ΔR (degradation)* > 0 means transfer is **worse** than within-cohort; "
        "≈0 or <0 means transfer holds up.\n")

    lines.append("## Interpretation (H3)\n")
    lines.append(
        "H3 predicts cross-cohort R drops substantially vs within-cohort, and "
        "that meal-only (macro) models transfer relatively better than "
        "personalized ones. Personalized models are not transferable here (see "
        "exclusions), so this run tests the **macro-model transfer** leg of H3 "
        "directly. Read the ΔR column and the overlap of transfer-R vs within-R "
        "CIs; with 12 evaluable BIG IDEAs subjects (and 31 CGMacros) the CIs are "
        "wide — verdict is reported with explicit small-n caveats. Numbers are "
        "as printed by the run; no value is hand-edited.\n")

    lines.append("## Data / feature-harmonization notes\n")
    lines.append(
        "- **Identical feature columns + target** across cohorts by construction "
        "(both adapters emit `MEAL_COLUMNS`; iAUC via the same `apply_inclusion`).")
    lines.append(
        "- **Imputation:** ElasticNet uses **train-cohort** macro medians to fill "
        "any missing macros in both train and test (no target-cohort leakage); "
        "XGBoost uses native NaN handling on raw macros — exactly as in the "
        "within-cohort pipeline.")
    lines.append(
        "- **BIG IDEAs subject 003** has a headerless food log (no protein/fat) → "
        "those macros are NaN and imputed; **4/16 BIG IDEAs subjects** are "
        "unevaluable (CGM↔food-log date misalignment) so transfer onto BIG IDEAs "
        "tests 12 subjects.")
    lines.append("")

    with open(os.path.join(OUT_DIR, "transfer_results.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
