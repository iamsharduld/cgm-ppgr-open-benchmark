"""CGMacros non-diabetic headline SENSITIVITY analyses (prereg §3.3, §4.3, §5.1).

Three pre-registered robustness checks on the CGMacros non-diabetic (healthy +
pre-DM, A1c-derived; prereg §3.1) headline. Each re-runs the SAME reviewed
pipeline (adapters, apply_inclusion, add_features, run_loso, summarize,
paired_delta_ci) with exactly ONE thing changed, so deltas vs the primary
headline are clean. NOTHING here mutates the primary results files.

  (A) Snack-excluded (prereg §3.3): drop rows whose normalized `Meal Type` is a
      snack BEFORE inclusion; primary outcome iAUC_pos, Dexcom stream.
        -> results/cgmacros_snack_excluded.{md,csv}
  (B) CGM-brand / Libre GL (prereg §4.3, issue #25): load the cohort with the
      `Libre GL` stream instead of `Dexcom GL`; everything else identical.
        -> results/cgmacros_libre.{md,csv}
  (C) Peak-rise outcome (prereg §5.1, issue #25): run LOSO with target=peak_rise
      instead of iauc_pos; Dexcom stream, snacks included (primary inclusion).
        -> results/cgmacros_peakrise.{md,csv}

Usage:  PYTHONPATH=src python3 run_cgmacros_sensitivity.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ppgr.adapters import cgmacros
from ppgr.evaluate import paired_delta_ci, run_loso, summarize
from ppgr.features import add_features
from ppgr.inclusion import apply_inclusion

DATA_ROOT = "data/cgmacros/extracted/CGMacros"
OUT_DIR = "results"
NON_DIABETIC = {"healthy", "pre-DM"}

ORDER = [
    "population_mean", "per_person_mean", "carb_only", "carb_calorie",
    "elasticnet_macros", "elasticnet_macros+context",
    "mlp_macros", "mlp_macros+context",
    "xgboost_macros", "xgboost_macros+context",
]
XGB_METHODS = ["xgboost_macros", "xgboost_macros+context"]


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.3f}"


def _normalize_meal_type(s: pd.Series) -> pd.Series:
    """Collapse `Meal Type` casing/spelling variants to 4 canonical classes
    (identical to run_cgmacros.py)."""
    n = s.astype("string").str.strip().str.lower()
    n = n.str.replace(r"^snack.*", "snack", regex=True)
    n = n.replace({"snacks": "snack"})
    return n


def restrict_non_diabetic(cohort):
    nd = sorted(
        sid for sid, g in cohort.meals.groupby("subject_id")["group"].first().items()
        if g in NON_DIABETIC
    )
    cohort.meals = cohort.meals[cohort.meals["subject_id"].isin(nd)].copy()
    cohort.cgm = {s: cohort.cgm[s] for s in nd}
    return cohort, nd


def order_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["__o"] = summary["method"].map({m: i for i, m in enumerate(ORDER)})
    return summary.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def deltas_h2(preds, summary):
    best_xgb = max(
        XGB_METHODS, key=lambda m: summary.set_index("method").loc[m, "pearson_r"]
    )
    d = {}
    for comp in ["per_person_mean", "carb_only"]:
        d[comp] = paired_delta_ci(preds[best_xgb], preds[comp], "pearson_r")
    return best_xgb, d


def metrics_table(summary: pd.DataFrame) -> list[str]:
    lines = ["| method | n_meals | Pearson R [95% CI] | Spearman ρ | RMSE [95% CI] | MAE |",
             "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {int(r['n_meals'])} | "
            f"{fmt(r['pearson_r'])} [{fmt(r['pearson_r_lo'])}, {fmt(r['pearson_r_hi'])}] | "
            f"{fmt(r['spearman_r'])} | "
            f"{fmt(r['rmse'])} [{fmt(r['rmse_lo'])}, {fmt(r['rmse_hi'])}] | "
            f"{fmt(r['mae'])} |")
    return lines


def h2_table(best_xgb, deltas) -> list[str]:
    lines = [f"Best XGBoost variant by Pearson R: **{best_xgb}**\n",
             "| comparison | ΔR (XGB − baseline) | 95% CI | CI excludes 0? | ΔR ≥ 0.05? | beats baseline? |",
             "|---|---|---|---|---|---|"]
    for comp, (pt, lo, hi) in deltas.items():
        excl = (lo > 0 or hi < 0)
        marg = pt >= 0.05
        beats = "yes" if (excl and marg) else "no"
        lines.append(
            f"| {best_xgb} − {comp} | {fmt(pt)} | [{fmt(lo)}, {fmt(hi)}] | "
            f"{'yes' if excl else 'no'} | {'yes' if marg else 'no'} | {beats} |")
    return lines


def load_primary_summary() -> pd.DataFrame:
    """The primary CGMacros headline (Dexcom, iAUC_pos, snacks included)."""
    return pd.read_csv(os.path.join(OUT_DIR, "cgmacros_results.csv")).set_index("method")


# ---------------------------------------------------------------------------

def run_snack_excluded() -> dict:
    print("\n##### (A) SNACK-EXCLUDED SENSITIVITY (prereg §3.3) #####")
    cohort = cgmacros.load(DATA_ROOT)                 # Dexcom stream (primary)
    cohort, nd = restrict_non_diabetic(cohort)

    norm = _normalize_meal_type(cohort.meals["meal_type"])
    n_before = len(cohort.meals)
    n_snack = int(norm.eq("snack").sum())
    cohort.meals = cohort.meals[norm.ne("snack")].copy()
    print(f"  non-diabetic subjects: {len(nd)}; raw rows {n_before} - snacks {n_snack}"
          f" -> {len(cohort.meals)} kept (snack-excluded)")

    included, att = apply_inclusion(cohort)
    included = add_features(included)
    print(f"  included meals (snack-excluded): {len(included)}")

    preds = run_loso(included)
    summary = order_summary(summarize(preds))
    summary.to_csv(os.path.join(OUT_DIR, "cgmacros_snack_excluded.csv"), index=False)
    best_xgb, deltas = deltas_h2(preds, summary)

    prim = load_primary_summary()
    lines = ["# CGMacros — SNACK-EXCLUDED sensitivity (prereg §3.3)\n"]
    lines.append(
        "Re-runs the CGMacros **non-diabetic** (healthy + pre-DM) headline with "
        "snack rows dropped **before** inclusion (normalized `Meal Type == snack`). "
        "Everything else is identical to the primary run (Dexcom GL stream, "
        "iAUC_pos outcome, LOSO, same reviewed pipeline). The primary analysis "
        "INCLUDES snacks per prereg §3.3; this is the pre-registered "
        "snack-excluded robustness check.\n")
    lines.append(
        f"- Non-diabetic raw meal rows: **{n_before}**; snack rows removed: "
        f"**{n_snack}**; non-snack rows fed to inclusion: **{len(cohort.meals)}**.")
    lines.append(
        f"- Included meals after inclusion: **{len(included)}** (primary, "
        "snacks-included: **851**).\n")
    lines.append("## Per-method metrics (snack-excluded)\n")
    lines += metrics_table(summary)
    lines.append("")
    lines.append("## Δ vs primary (snack-included) headline — Pearson R\n")
    lines.append("| method | R (snack-excluded) | R (primary, snack-incl) | ΔR (excl − incl) |")
    lines.append("|---|---|---|---|")
    for _, r in summary.iterrows():
        m = r["method"]
        rp = float(prim.loc[m, "pearson_r"]) if m in prim.index else float("nan")
        lines.append(f"| {m} | {fmt(r['pearson_r'])} | {fmt(rp)} | {fmt(r['pearson_r'] - rp)} |")
    lines.append("")
    lines.append("## H2 head-to-head (snack-excluded)\n")
    lines += h2_table(best_xgb, deltas)
    lines.append("")
    with open(os.path.join(OUT_DIR, "cgmacros_snack_excluded.md"), "w") as f:
        f.write("\n".join(lines))

    # console
    for _, r in summary.iterrows():
        m = r["method"]
        rp = float(prim.loc[m, "pearson_r"]) if m in prim.index else float("nan")
        print(f"    {m:28s} R={fmt(r['pearson_r'])}  primary={fmt(rp)}  "
              f"ΔR={fmt(r['pearson_r'] - rp)}")
    return {"n_included": len(included), "summary": summary, "primary": prim,
            "best_xgb": best_xgb, "deltas": deltas}


def run_libre() -> dict:
    print("\n##### (B) CGM-BRAND SENSITIVITY: Libre GL (prereg §4.3, #25) #####")
    cohort = cgmacros.load(DATA_ROOT, glucose_col="Libre GL")
    cohort, nd = restrict_non_diabetic(cohort)
    print(f"  non-diabetic subjects: {len(nd)}; raw meal rows: {len(cohort.meals)}")

    included, att = apply_inclusion(cohort)
    included = add_features(included)
    n_incl_subj = included["subject_id"].nunique() if not included.empty else 0
    print(f"  included meals (Libre): {len(included)} from {n_incl_subj} subjects"
          f"  (Dexcom primary: 851 from 31)")

    preds = run_loso(included)
    summary = order_summary(summarize(preds))
    summary.to_csv(os.path.join(OUT_DIR, "cgmacros_libre.csv"), index=False)
    best_xgb, deltas = deltas_h2(preds, summary)

    prim = load_primary_summary()
    lines = ["# CGMacros — CGM-BRAND sensitivity: Libre GL stream (prereg §4.3 / #25)\n"]
    lines.append(
        "Re-runs the CGMacros **non-diabetic** headline using the **Abbott "
        "FreeStyle Libre Pro** stream (`Libre GL`, 15-min native cadence) instead "
        "of the primary **Dexcom G6 Pro** stream (`Dexcom GL`, 5-min). Outcome, "
        "inclusion rules, features, LOSO, models all identical. Motivated by the "
        "verified device-disagreement literature (Hengist & Hall 2024/25; Selvin "
        "2023). The 30-min max-gap inclusion rule (§3.2.1) is stricter relative to "
        "Libre's coarser 15-min sampling, so the included-meal count differs.\n")
    lines.append(
        f"- Non-diabetic subjects: **{len(nd)}**; raw meal rows: **{len(cohort.meals)}**.")
    lines.append(
        f"- Included meals (Libre stream): **{len(included)}** from "
        f"**{n_incl_subj}** subjects (Dexcom primary: **851** from **31**).\n")
    lines.append("## Per-method metrics (Libre GL stream)\n")
    lines += metrics_table(summary)
    lines.append("")
    lines.append("## Δ vs primary (Dexcom GL) headline — Pearson R\n")
    lines.append(
        "> Note: meal sets differ between streams (different coverage under the "
        "max-gap rule), so this Δ is a device-level comparison of the headline "
        "conclusions, not a paired per-meal delta.\n")
    lines.append("| method | R (Libre) | R (primary, Dexcom) | ΔR (Libre − Dexcom) |")
    lines.append("|---|---|---|---|")
    for _, r in summary.iterrows():
        m = r["method"]
        rp = float(prim.loc[m, "pearson_r"]) if m in prim.index else float("nan")
        lines.append(f"| {m} | {fmt(r['pearson_r'])} | {fmt(rp)} | {fmt(r['pearson_r'] - rp)} |")
    lines.append("")
    lines.append("## H2 head-to-head (Libre GL)\n")
    lines += h2_table(best_xgb, deltas)
    lines.append(
        "\n> **H2 verdict across devices:** compare this table to the primary "
        "(Dexcom) H2 table in `cgmacros_results.md`. Does XGBoost still beat "
        "carb_only (CI excludes 0 AND ΔR≥0.05) and still NOT clear the bar vs "
        "per_person_mean? See the run console / report for the verdict.\n")
    with open(os.path.join(OUT_DIR, "cgmacros_libre.md"), "w") as f:
        f.write("\n".join(lines))

    for _, r in summary.iterrows():
        m = r["method"]
        rp = float(prim.loc[m, "pearson_r"]) if m in prim.index else float("nan")
        print(f"    {m:28s} R={fmt(r['pearson_r'])}  Dexcom={fmt(rp)}  "
              f"ΔR={fmt(r['pearson_r'] - rp)}")
    print("    H2 (Libre):")
    for comp, (pt, lo, hi) in deltas.items():
        excl = (lo > 0 or hi < 0)
        print(f"      {best_xgb} - {comp}: dR={fmt(pt)} CI=[{fmt(lo)},{fmt(hi)}]  "
              f"excludes0={excl} dR>=0.05={pt>=0.05}")
    return {"n_included": len(included), "n_subj": n_incl_subj,
            "summary": summary, "primary": prim, "best_xgb": best_xgb,
            "deltas": deltas}


def run_peakrise() -> dict:
    print("\n##### (C) PEAK-RISE OUTCOME SENSITIVITY (prereg §5.1, #25) #####")
    cohort = cgmacros.load(DATA_ROOT)                 # Dexcom stream (primary)
    cohort, nd = restrict_non_diabetic(cohort)
    included, att = apply_inclusion(cohort)           # same primary inclusion
    included = add_features(included)
    print(f"  non-diabetic subjects: {len(nd)}; included meals: {len(included)} "
          "(same set as primary)")
    print(f"  peak_rise (mg/dL): mean {included['peak_rise'].mean():.1f}, "
          f"median {included['peak_rise'].median():.1f}")

    preds = run_loso(included, target="peak_rise")    # secondary outcome
    summary = order_summary(summarize(preds))
    summary.to_csv(os.path.join(OUT_DIR, "cgmacros_peakrise.csv"), index=False)
    best_xgb, deltas = deltas_h2(preds, summary)

    prim = load_primary_summary()
    lines = ["# CGMacros — PEAK-RISE outcome sensitivity (prereg §5.1 / #25)\n"]
    lines.append(
        "Re-runs the CGMacros **non-diabetic** headline with the **secondary "
        "outcome `peak_rise`** = max(glucose over 0–120) − baseline (mg/dL; "
        "prereg §5.1) as the target, instead of the primary `iauc_pos`. Same "
        "Dexcom GL stream, same inclusion (snacks included), same included-meal "
        "set, LOSO, models. Tests whether the headline conclusions are robust to "
        "the choice of PPGR target. (RMSE/MAE are in mg/dL here, NOT mg/dL·min — "
        "not comparable to the iAUC tables.)\n")
    lines.append(
        f"- Non-diabetic subjects: **{len(nd)}**; included meals: **{len(included)}** "
        "(identical set to the primary iAUC run).")
    lines.append(
        f"- peak_rise distribution (mg/dL): mean **{included['peak_rise'].mean():.1f}**, "
        f"median **{included['peak_rise'].median():.1f}**.\n")
    lines.append("## Per-method metrics (target = peak_rise)\n")
    lines += metrics_table(summary)
    lines.append("")
    lines.append("## Pearson R: peak_rise vs primary iAUC_pos (same models)\n")
    lines.append(
        "> R is unitless so the columns are comparable; RMSE/MAE are not "
        "(different units).\n")
    lines.append("| method | R (peak_rise) | R (primary iAUC_pos) | ΔR (peak − iAUC) |")
    lines.append("|---|---|---|---|")
    for _, r in summary.iterrows():
        m = r["method"]
        rp = float(prim.loc[m, "pearson_r"]) if m in prim.index else float("nan")
        lines.append(f"| {m} | {fmt(r['pearson_r'])} | {fmt(rp)} | {fmt(r['pearson_r'] - rp)} |")
    lines.append("")
    lines.append("## H2 head-to-head (target = peak_rise)\n")
    lines += h2_table(best_xgb, deltas)
    lines.append("")
    with open(os.path.join(OUT_DIR, "cgmacros_peakrise.md"), "w") as f:
        f.write("\n".join(lines))

    for _, r in summary.iterrows():
        m = r["method"]
        rp = float(prim.loc[m, "pearson_r"]) if m in prim.index else float("nan")
        print(f"    {m:28s} R={fmt(r['pearson_r'])}  iAUC={fmt(rp)}  "
              f"ΔR={fmt(r['pearson_r'] - rp)}")
    print("    H2 (peak_rise):")
    for comp, (pt, lo, hi) in deltas.items():
        excl = (lo > 0 or hi < 0)
        print(f"      {best_xgb} - {comp}: dR={fmt(pt)} CI=[{fmt(lo)},{fmt(hi)}]  "
              f"excludes0={excl} dR>=0.05={pt>=0.05}")
    return {"n_included": len(included), "summary": summary, "primary": prim,
            "best_xgb": best_xgb, "deltas": deltas}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    run_snack_excluded()
    run_libre()
    run_peakrise()
    print("\nWrote results/cgmacros_{snack_excluded,libre,peakrise}.{md,csv}")


if __name__ == "__main__":
    main()
