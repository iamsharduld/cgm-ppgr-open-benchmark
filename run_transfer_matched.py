"""MATCHED within-vs-transfer comparison (prereg §8 / H3; issue #25).

The reviewed transfer report (`results/transfer_results.md`) compares
transfer-R against a within-cohort LOSO R that was computed by the PRIMARY
pipeline (`run_loso`), which reserves each test subject's earliest 3 meals as a
personal-calibration set and evaluates only the REMAINING meals (§6.1.2). The
transfer run instead scores the FULL target cohort (no reserve — there is no
within-cohort leakage to guard across disjoint cohorts). So the ΔR
(within − transfer) mixed two different evaluation sets.

This script recomputes a MATCHED within-cohort R for the three macro models
  carb_only, elasticnet_macros, xgboost_macros
under LOSO on the FULL cohort — NO calibration reserve — so the evaluation set
is apples-to-apples with the transfer run (the whole cohort is scored). The
macro models don't need the calibration reserve (it exists only for the
per-person-mean baseline, which is not transferred). Everything else (model
constructors, nested inner-CV / grid search, imputation, the subject-level
bootstrap CI, the included-meal frames) is IDENTICAL to the reviewed pipeline
via the same `evaluate.py` helpers and the same `load_included` used by
run_transfer.py.

Then ΔR_matched = within_R_matched(test_cohort) − transfer_R, using the
transfer_R already in `results/transfer_results.csv` (unchanged).

Usage:  PYTHONPATH=src python3 run_transfer_matched.py
Writes: results/transfer_matched.csv and results/transfer_matched.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ppgr.evaluate import (
    Prediction,
    _fit_xgb,
    _grouped_inner_cv,
    _make_elasticnet,
    _metrics,
    _subject_bootstrap_ci,
)
from ppgr.features import MACRO_FEATURES

# reuse run_transfer's cohort loader EXACTLY (same population restriction,
# inclusion, feature build) so the included-meal frames are identical.
from run_transfer import load_included

OUT_DIR = "results"
TARGET = "iauc_pos"
MACRO_MODELS = ["carb_only", "elasticnet_macros", "xgboost_macros"]
TRANSFER_CSV = os.path.join(OUT_DIR, "transfer_results.csv")


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.3f}"


def fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Identical model construction to run_transfer.fit_predict (same helpers,
    same train-median imputation / native NaN, same nested inner CV)."""
    ytr = train[TARGET].to_numpy()
    if model == "carb_only":
        lr = LinearRegression().fit(train[["carbs"]], ytr)
        return lr.predict(test[["carbs"]])
    if model == "elasticnet_macros":
        cols = MACRO_FEATURES
        med = train[cols].median()
        Xtr = train[cols].fillna(med)
        Xte = test[cols].fillna(med)
        inner_cv = _grouped_inner_cv(train["subject_id"].to_numpy())
        en = _make_elasticnet(inner_cv).fit(Xtr, ytr)
        return en.predict(Xte)
    if model == "xgboost_macros":
        cols = MACRO_FEATURES
        xgb = _fit_xgb(train[cols], ytr, train["subject_id"].to_numpy())
        return xgb.predict(test[cols])
    raise ValueError(model)


def matched_within_loso(included: pd.DataFrame) -> dict[str, Prediction]:
    """LOSO over the FULL cohort (NO calibration reserve) for the macro models.

    For each subject: train on all OTHER subjects, predict ALL of this subject's
    meals (the whole cohort is scored, exactly as the transfer run scores the
    whole target cohort). Returns pooled held-out Predictions per model.
    """
    subjects = sorted(included["subject_id"].unique())
    acc = {m: {"sid": [], "yt": [], "yp": []} for m in MACRO_MODELS}
    for test_sid in subjects:
        train = included[included["subject_id"] != test_sid]
        test = included[included["subject_id"] == test_sid]
        if test.empty:
            continue
        yte = test[TARGET].to_numpy()
        for model in MACRO_MODELS:
            yhat = np.asarray(fit_predict(model, train, test), dtype=float)
            acc[model]["sid"].extend([test_sid] * len(test))
            acc[model]["yt"].extend(yte.tolist())
            acc[model]["yp"].extend(yhat.tolist())
    return {
        m: Prediction(method=m, subject_id=np.array(d["sid"]),
                      y_true=np.array(d["yt"], dtype=float),
                      y_pred=np.array(d["yp"], dtype=float))
        for m, d in acc.items()
    }


def within_metrics(preds: dict[str, Prediction]) -> dict[str, dict]:
    out = {}
    for m, p in preds.items():
        r, r_lo, r_hi = _subject_bootstrap_ci(
            p, lambda yt, yp: _metrics(yt, yp)["pearson_r"])
        out[m] = {
            "pearson_r": r, "pearson_r_lo": r_lo, "pearson_r_hi": r_hi,
            "rmse": _metrics(p.y_true, p.y_pred)["rmse"],
            "n_meals": len(p.y_true),
            "n_subjects": int(len(np.unique(p.subject_id))),
        }
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading cohorts (run_transfer.load_included, identical frames) ...")
    cg = load_included("cgmacros")
    bi = load_included("bigideas")
    print(f"  CGMacros included meals: {len(cg)} from {cg['subject_id'].nunique()} subj")
    print(f"  BIG IDEAs included meals: {len(bi)} from {bi['subject_id'].nunique()} subj")

    print("Matched within-cohort LOSO (FULL cohort, NO calibration reserve) ...")
    cg_w = within_metrics(matched_within_loso(cg))
    bi_w = within_metrics(matched_within_loso(bi))
    matched = {"cgmacros": cg_w, "bigideas": bi_w}

    # full-cohort meal counts for sanity vs the reserved within-run
    n_full = {"cgmacros": len(cg), "bigideas": len(bi)}

    transfer = pd.read_csv(TRANSFER_CSV)

    rows = []
    for _, t in transfer.iterrows():
        model = t["model"]
        if model not in MACRO_MODELS:
            continue
        test_cohort = t["test_cohort"]
        w = matched[test_cohort][model]
        tr_R = float(t["transfer_R"])
        matched_within_R = w["pearson_r"]
        delta_matched = matched_within_R - tr_R
        rows.append({
            "direction": t["direction"],
            "train_cohort": t["train_cohort"],
            "test_cohort": test_cohort,
            "model": model,
            "n_test_meals": int(t["n_test_meals"]),
            "n_full_within_meals": n_full[test_cohort],
            "n_test_subjects": int(t["n_test_subjects"]),
            # original (reserved within-LOSO) numbers, for the record
            "within_R_reserved": float(t["within_R"]),
            "delta_R_reserved": float(t["delta_R_degradation"]),
            # matched (full-cohort, no reserve) within-LOSO numbers
            "within_R_matched": matched_within_R,
            "within_R_matched_lo": w["pearson_r_lo"],
            "within_R_matched_hi": w["pearson_r_hi"],
            "transfer_R": tr_R,
            "transfer_R_lo": float(t["transfer_R_lo"]),
            "transfer_R_hi": float(t["transfer_R_hi"]),
            "delta_R_matched": delta_matched,
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "transfer_matched.csv"), index=False)

    print("\n===== MATCHED within (full cohort, no reserve) =====")
    for ck, mv in matched.items():
        for m in MACRO_MODELS:
            print(f"  {ck:9s} {m:18s} within_R(matched)={fmt(mv[m]['pearson_r'])} "
                  f"[{fmt(mv[m]['pearson_r_lo'])},{fmt(mv[m]['pearson_r_hi'])}]  "
                  f"n={mv[m]['n_meals']}")
    print("\n===== CORRECTED ΔR (matched within − transfer) =====")
    for _, r in table.iterrows():
        print(f"  {r['direction']:20s} {r['model']:18s} "
              f"within(matched)={fmt(r['within_R_matched'])}  "
              f"transfer={fmt(r['transfer_R'])}  "
              f"ΔR_matched={fmt(r['delta_R_matched'])}  "
              f"(was reserved ΔR={fmt(r['delta_R_reserved'])})")

    _write_md(table, matched, n_full)
    print("\nWrote results/transfer_matched.csv and results/transfer_matched.md")


def _write_md(table: pd.DataFrame, matched: dict, n_full: dict) -> None:
    lines = ["# MATCHED within-vs-transfer comparison (prereg §8 / H3; #25)\n"]
    lines.append(
        "**Why this exists.** The reviewed transfer report computed ΔR as "
        "`within-LOSO R − transfer R`, but the within-LOSO R came from the "
        "PRIMARY pipeline (`run_loso`), which reserves each test subject's "
        "earliest 3 meals as a personal-calibration set and scores only the "
        "**remaining** meals (prereg §6.1.2). The transfer run scores the "
        "**FULL** target cohort. The two R's were therefore on different "
        "evaluation sets, so the ΔR mixed apples and oranges.\n")
    lines.append(
        "**The fix.** For the three macro models (`carb_only`, "
        "`elasticnet_macros`, `xgboost_macros`) — which do NOT use the "
        "calibration reserve (it exists only for the per-person-mean baseline, "
        "and that baseline is not transferable) — recompute a within-cohort "
        "LOSO R on the **FULL** cohort with **NO calibration reserve**: train on "
        "all other subjects, predict ALL of the held-out subject's meals. This "
        "matches exactly how transfer scores the whole target cohort. Model "
        "construction, nested inner-CV / grid search, imputation, and the "
        "subject-level bootstrap CI are identical to the reviewed pipeline "
        "(same `evaluate.py` helpers; same included-meal frames via "
        "`run_transfer.load_included`). `transfer_R` is unchanged from "
        "`results/transfer_results.csv`.\n")

    lines.append("## Matched within-cohort R (full cohort, no calibration reserve)\n")
    lines.append("| cohort | model | within-R (matched) [95% CI] | n meals (full) |")
    lines.append("|---|---|---|---|")
    for ck in ["cgmacros", "bigideas"]:
        for m in MACRO_MODELS:
            mv = matched[ck][m]
            lines.append(
                f"| {ck} | {m} | {fmt(mv['pearson_r'])} "
                f"[{fmt(mv['pearson_r_lo'])}, {fmt(mv['pearson_r_hi'])}] | "
                f"{mv['n_meals']} |")
    lines.append("")

    lines.append("## Corrected ΔR (matched within − transfer)\n")
    lines.append(
        "ΔR > 0 ⇒ transfer is **worse** than within-cohort (degradation, as H3 "
        "predicts); ΔR ≈ 0 or < 0 ⇒ transfer holds up (or even exceeds "
        "within-cohort).\n")
    lines.append(
        "| direction | model | within-R (matched) [95% CI] | transfer-R [95% CI] "
        "| **ΔR matched** | ΔR (old, reserved) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        lines.append(
            f"| {r['direction']} | {r['model']} | "
            f"{fmt(r['within_R_matched'])} [{fmt(r['within_R_matched_lo'])}, "
            f"{fmt(r['within_R_matched_hi'])}] | "
            f"{fmt(r['transfer_R'])} [{fmt(r['transfer_R_lo'])}, "
            f"{fmt(r['transfer_R_hi'])}] | "
            f"**{fmt(r['delta_R_matched'])}** | {fmt(r['delta_R_reserved'])} |")
    lines.append("")

    lines.append("## Interpretation (corrected H3 read)\n")
    lines.append(
        "- The matched within-R replaces the reserved-meal within-R that "
        "inflated/deflated the old ΔR (different evaluation set). The signs and "
        "magnitudes in the **ΔR matched** column are the apples-to-apples "
        "degradations.")
    lines.append(
        "- Read each ΔR with its CI overlap: transfer-R and within-R CIs are wide "
        "(12 evaluable BIG IDEAs subjects; 31 CGMacros), so most single-cell "
        "degradations are not individually significant; the column shows the "
        "direction and size of the corrected gap.")
    lines.append(
        "- This corrects ONLY the within-vs-transfer ΔR bookkeeping; the transfer "
        "R's themselves and the primary within-cohort headline "
        "(`cgmacros_results.md`, reserved-meal LOSO) are unchanged.")
    lines.append("")

    with open(os.path.join(OUT_DIR, "transfer_matched.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
