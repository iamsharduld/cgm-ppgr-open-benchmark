"""Run the CGMacros PPGR benchmark end-to-end and write results.

Usage:  PYTHONPATH=src python3 run_cgmacros.py

Pipeline per `prereg.md`: load CGMacros -> restrict to the PRIMARY non-diabetic
population (healthy + pre-DM = 31 subjects, derived from A1c via ADA thresholds,
§3.1) -> apply per-meal inclusion (§3.2) -> LOSO CV baselines + models (§6-§7,
LOSO is the pre-specified small-n fallback §7.1) -> summarize.

Writes:
  results/cgmacros_results.csv  -- per-method metrics + CIs
  results/cgmacros_results.md   -- sanity check, attrition funnel, metrics, H2
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


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.3f}"


def _normalize_meal_type(s: pd.Series) -> pd.Series:
    """Collapse the real `Meal Type` casing/spelling variants to 4 canonical
    classes. Verified raw values across the 45 files:
      Breakfast/breakfast, Lunch/lunch, Dinner/dinner,
      Snacks/snack/Snack/'snack 1'  -> snack.
    """
    n = s.astype("string").str.strip().str.lower()
    n = n.str.replace(r"^snack.*", "snack", regex=True)
    n = n.replace({"snacks": "snack"})
    return n


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CGMacros cohort ...")
    cohort = cgmacros.load(DATA_ROOT)
    sanity = _sanity_check(cohort)

    # ---- restrict to PRIMARY non-diabetic population (healthy + pre-DM) ----
    nd_subjects = sorted(
        sid for sid, g in cohort.meals.groupby("subject_id")["group"].first().items()
        if g in NON_DIABETIC
    )
    print(f"Restricting to non-diabetic (healthy+pre-DM): {len(nd_subjects)} subjects")
    cohort.meals = cohort.meals[cohort.meals["subject_id"].isin(nd_subjects)].copy()
    cohort.cgm = {s: cohort.cgm[s] for s in nd_subjects}

    # snack-excluded sensitivity COUNT (secondary; primary keeps snacks, §3.3)
    norm_mt = _normalize_meal_type(cohort.meals["meal_type"])
    n_snacks_nd = int(norm_mt.eq("snack").sum())
    n_meals_nd = len(cohort.meals)
    print(f"  non-diabetic raw meal rows: {n_meals_nd}  (of which snacks: {n_snacks_nd})")

    print("Applying inclusion filters (prereg §3.2) ...")
    included, att = apply_inclusion(cohort)
    print(f"  included meals: {len(included)}")

    included = add_features(included)

    print("Running LOSO CV: baselines + models (prereg §6-§7) ...")
    preds = run_loso(included)
    summary = summarize(preds)

    order = [
        "population_mean", "per_person_mean", "carb_only", "carb_calorie",
        "elasticnet_macros", "elasticnet_macros+context",
        "mlp_macros", "mlp_macros+context",
        "xgboost_macros", "xgboost_macros+context",
    ]
    summary["__o"] = summary["method"].map({m: i for i, m in enumerate(order)})
    summary = summary.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    summary.to_csv(os.path.join(OUT_DIR, "cgmacros_results.csv"), index=False)

    # head-to-head: best XGBoost vs per-person-mean and carb-only (H2)
    xgb_methods = ["xgboost_macros", "xgboost_macros+context"]
    best_xgb = max(
        xgb_methods, key=lambda m: summary.set_index("method").loc[m, "pearson_r"]
    )
    deltas = {}
    for comp in ["per_person_mean", "carb_only"]:
        deltas[comp] = paired_delta_ci(preds[best_xgb], preds[comp], "pearson_r")

    _write_md(sanity, n_meals_nd, n_snacks_nd, len(nd_subjects),
              included, att, summary, best_xgb, deltas)
    _print_console(sanity, n_meals_nd, n_snacks_nd, included, att, summary,
                   best_xgb, deltas)


def _sanity_check(cohort) -> dict:
    """Print + return the pre-trust sanity checks (whole 45-subject cohort)."""
    gmins, gmaxs, ntot = [], [], 0
    for df in cohort.cgm.values():
        if len(df):
            gmins.append(df["glucose"].min())
            gmaxs.append(df["glucose"].max())
            ntot += len(df)
    raw_counts = cohort.meals["meal_type"].value_counts().to_dict()
    norm_counts = _normalize_meal_type(cohort.meals["meal_type"]).value_counts().to_dict()
    sub_grp = cohort.meals.groupby("subject_id")["group"].first().value_counts().to_dict()
    n_snacks = int(_normalize_meal_type(cohort.meals["meal_type"]).eq("snack").sum())

    s = {
        "n_subjects": len(cohort.cgm),
        "glucose_min": float(min(gmins)),
        "glucose_max": float(max(gmaxs)),
        "n_cgm_readings": ntot,
        "n_meals_total": len(cohort.meals),
        "meal_raw_counts": raw_counts,
        "meal_norm_counts": norm_counts,
        "n_snacks": n_snacks,
        "group_counts": sub_grp,
    }
    print("\n===== SANITY CHECK (all 45 subjects) =====")
    print(f"  subjects: {s['n_subjects']}")
    print(f"  Dexcom GL global min/max: {s['glucose_min']} / {s['glucose_max']}  (expect 40-400)")
    print(f"  total Dexcom CGM readings: {s['n_cgm_readings']}")
    print(f"  total meal rows: {s['n_meals_total']}")
    print(f"  meal rows by Meal Type (normalized): {s['meal_norm_counts']}")
    print(f"  #snacks (any variant): {s['n_snacks']}")
    print(f"  A1c-derived group counts: {s['group_counts']}  (expect healthy=15, pre-DM=16, T2D=14)")
    ok = s["group_counts"].get("healthy") == 15 and \
        s["group_counts"].get("pre-DM") == 16 and s["group_counts"].get("T2D") == 14
    print(f"  group counts == 15/16/14 ? {'YES' if ok else 'NO'}")
    return s


def _write_md(sanity, n_meals_nd, n_snacks_nd, n_nd_subj,
              included, att, summary, best_xgb, deltas) -> None:
    lines = []
    lines.append("# CGMacros PPGR benchmark — results\n")
    n_incl_subj = included["subject_id"].nunique() if not included.empty else 0
    lines.append(
        "Pipeline per `prereg.md` (§3 inclusion, §4 iAUC, §5 features, §6 "
        "baselines/models, §7 evaluation). Primary outcome: `iAUC_pos` (0–120 "
        "min, trapezoidal area-above-baseline; primary stream = **Dexcom GL**, "
        "§4.3). Disjoint-subject CV = leave-one-subject-out (the pre-specified "
        "small-n fallback, §7.1). Primary population = **non-diabetic** "
        "(healthy + pre-DM), derived from `bio.csv` `A1c PDL (Lab)` via ADA "
        f"thresholds (§3.1): **{n_nd_subj} subjects**. Of these, **{n_incl_subj}** "
        "contribute >=1 included meal.\n"
    )

    # sanity
    lines.append("## Sanity checks (whole 45-subject cohort, before population restriction)\n")
    lines.append(f"- Subjects (with per-participant CSV): **{sanity['n_subjects']}**")
    lines.append(
        f"- Dexcom GL global min/max: **{sanity['glucose_min']} / {sanity['glucose_max']}** "
        f"mg/dL (expected 40–400); total Dexcom readings: {sanity['n_cgm_readings']:,}"
    )
    lines.append(f"- Total meal rows (non-null `Meal Type`): **{sanity['n_meals_total']}**")
    lines.append(f"- Meal rows by type (normalized): `{sanity['meal_norm_counts']}`")
    lines.append(f"- Raw `Meal Type` values (casing/spelling variants present): `{sanity['meal_raw_counts']}`")
    lines.append(f"- Snacks (any variant): **{sanity['n_snacks']}**")
    g = sanity["group_counts"]
    ok = g.get("healthy") == 15 and g.get("pre-DM") == 16 and g.get("T2D") == 14
    lines.append(
        f"- A1c-derived group counts: healthy=**{g.get('healthy')}**, "
        f"pre-DM=**{g.get('pre-DM')}**, T2D=**{g.get('T2D')}** — "
        f"matches published 15/16/14? **{'YES' if ok else 'NO'}**\n"
    )

    # population
    lines.append("## Primary population (non-diabetic) & snack sensitivity\n")
    lines.append(f"- Non-diabetic subjects (healthy + pre-DM): **{n_nd_subj}**")
    lines.append(f"- Raw meal rows in non-diabetic subjects: **{n_meals_nd}**")
    lines.append(
        f"- Of these, snacks: **{n_snacks_nd}** (secondary: a snack-EXCLUDED "
        "sensitivity would drop these; primary analysis INCLUDES snacks per "
        "prereg §3.3).\n"
    )

    # attrition
    lines.append("## Meal-inclusion attrition (CONSORT-style funnel, non-diabetic pop.)\n")
    lines.append(f"- Total logged meal anchors: **{att.total}**")
    lines.append(f"- Passed (2) known carbs: **{att.pass_carb}**")
    lines.append(f"- Passed (3a) no overlapping meal in (0,120]: **{att.pass_no_overlap}**")
    lines.append(f"- Passed (3b) prior-meal washout (preceding 120 min): **{att.pass_washout}**")
    lines.append(f"- Passed (1) CGM coverage of t=0 and t=120: **{att.pass_cgm_coverage}**")
    lines.append(f"- Passed (1) no interpolation gap >30 min => **INCLUDED: {att.pass_gap}**")
    lines.append(f"- Evaluable subjects (>=1 included meal): **{n_incl_subj}** / {n_nd_subj}\n")

    lines.append("### Per-subject attrition\n")
    lines.append("| subject | group | total | +carbs | +no-overlap | +washout | +cgm-cov | included |")
    lines.append("|---|---|---|---|---|---|---|---|")
    grp_by_sid = (
        included.groupby("subject_id")["group"].first().to_dict()
        if "group" in included.columns and not included.empty else {}
    )
    for sid in sorted(att.per_subject):
        p = att.per_subject[sid]
        grp = grp_by_sid.get(sid, "")
        lines.append(
            f"| {sid} | {grp} | {p['total']} | {p['pass_carb']} | {p['pass_no_overlap']} | "
            f"{p['pass_washout']} | {p['pass_cgm_coverage']} | {p['included']} |"
        )
    lines.append("")

    # outcome distribution
    if not included.empty:
        lines.append("## Outcome distribution (included meals)\n")
        lines.append(
            f"- iAUC_pos (mg/dL·min): mean {included['iauc_pos'].mean():.1f}, "
            f"median {included['iauc_pos'].median():.1f}, sd {included['iauc_pos'].std():.1f}, "
            f"min {included['iauc_pos'].min():.1f}, max {included['iauc_pos'].max():.1f}"
        )
        lines.append(
            f"- peak-rise (mg/dL): mean {included['peak_rise'].mean():.1f}, "
            f"median {included['peak_rise'].median():.1f}"
        )
        gs = included.groupby("subject_id").size()
        lines.append(
            f"- meals/subject: min {gs.min()}, median {int(gs.median())}, max {gs.max()}\n"
        )

    # metrics table
    lines.append("## Per-method metrics (pooled held-out meals; subject-level bootstrap 95% CIs)\n")
    lines.append("| method | n_meals | Pearson R [95% CI] | Spearman ρ | RMSE [95% CI] | MAE |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {int(r['n_meals'])} | "
            f"{fmt(r['pearson_r'])} [{fmt(r['pearson_r_lo'])}, {fmt(r['pearson_r_hi'])}] | "
            f"{fmt(r['spearman_r'])} | "
            f"{fmt(r['rmse'])} [{fmt(r['rmse_lo'])}, {fmt(r['rmse_hi'])}] | "
            f"{fmt(r['mae'])} |"
        )
    lines.append("")

    # H2 head-to-head
    lines.append("## Head-to-head: best XGBoost vs key baselines (ΔPearson R, paired bootstrap) — H2\n")
    lines.append(f"Best XGBoost variant by Pearson R: **{best_xgb}**\n")
    lines.append("| comparison | ΔR (XGB − baseline) | 95% CI | CI excludes 0? | ΔR ≥ 0.05? | beats baseline? |")
    lines.append("|---|---|---|---|---|---|")
    for comp, (pt, lo, hi) in deltas.items():
        excl = (lo > 0 or hi < 0)
        marg = pt >= 0.05
        beats = "yes" if (excl and marg) else "no"
        lines.append(
            f"| {best_xgb} − {comp} | {fmt(pt)} | [{fmt(lo)}, {fmt(hi)}] | "
            f"{'yes' if excl else 'no'} | {'yes' if marg else 'no'} | {beats} |"
        )
    lines.append("")
    lines.append(
        "> H2 pre-specified bar (prereg §7.2): XGBoost \"meaningfully beats\" a "
        "baseline only if the ΔR 95% CI excludes 0 AND the point ΔR ≥ 0.05.\n"
    )

    lines.append("## Notes\n")
    lines.append(
        "- **`Meal Type` casing/spelling variants.** The extracted files carry "
        "`Breakfast/breakfast`, `Lunch/lunch`, `Dinner/dinner`, and "
        "`Snacks/snack/Snack/'snack 1'` — more than the 4 the dictionary lists. "
        "The adapter anchors meals on any non-null `Meal Type`, so all are kept; "
        "the snack count normalizes these variants."
    )
    lines.append(
        "- **Snacks included** in the primary analysis per prereg §3.3 (the "
        "no-overlap / washout rules govern); a snack-excluded sensitivity is a "
        "secondary count, reported above."
    )
    lines.append(
        "- **`population_mean` Pearson R** is degenerate for a (near-)constant "
        "predictor (it varies only across LOSO folds); RMSE/MAE are the honest "
        "metrics for that baseline. Reported as printed for transparency."
    )
    lines.append("")

    with open(os.path.join(OUT_DIR, "cgmacros_results.md"), "w") as f:
        f.write("\n".join(lines))


def _print_console(sanity, n_meals_nd, n_snacks_nd, included, att, summary,
                   best_xgb, deltas) -> None:
    print("\n===== ATTRITION (non-diabetic pop.) =====")
    print(f"total={att.total} carbs={att.pass_carb} no_overlap={att.pass_no_overlap} "
          f"washout={att.pass_washout} cgm_cov={att.pass_cgm_coverage} INCLUDED={att.pass_gap}")
    n_incl_subj = included["subject_id"].nunique() if not included.empty else 0
    print(f"evaluable subjects: {n_incl_subj}")
    print("\n===== PER-METHOD METRICS (Pearson R [CI], RMSE [CI]) =====")
    for _, r in summary.iterrows():
        print(f"  {r['method']:30s}  R={fmt(r['pearson_r'])} "
              f"[{fmt(r['pearson_r_lo'])},{fmt(r['pearson_r_hi'])}]  "
              f"RMSE={fmt(r['rmse'])} [{fmt(r['rmse_lo'])},{fmt(r['rmse_hi'])}]  "
              f"MAE={fmt(r['mae'])}  n={int(r['n_meals'])}")
    print(f"\n===== H2 (best XGBoost = {best_xgb}) =====")
    for comp, (pt, lo, hi) in deltas.items():
        excl = (lo > 0 or hi < 0)
        marg = pt >= 0.05
        print(f"  XGB - {comp}: dR={fmt(pt)} CI=[{fmt(lo)},{fmt(hi)}]  "
              f"excludes0={excl} dR>=0.05={marg} -> beats={'YES' if (excl and marg) else 'NO'}")
    print("\nWrote results/cgmacros_results.csv and results/cgmacros_results.md")


if __name__ == "__main__":
    main()
