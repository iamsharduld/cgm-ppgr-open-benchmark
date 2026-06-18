"""Feature-group ablation (GitHub #24; prereg §5.3 tiers + hypothesis H4).

Usage:  PYTHONPATH=src python3 run_ablation.py

Runs the nested feature-tier ablation on the CGMacros NON-DIABETIC population
(healthy + pre-DM = 31 subjects, A1c-derived, prereg §3.1), with the SAME model
(XGBoost — the field workhorse, §6.2) under the SAME LOSO + subject-grouped
nested CV and leakage-safe per-subject calibration as `evaluate.py`.

Feature tiers (prereg §5.3, nested add-in):
  1. macros        — carbs, protein, fat, fiber, calorie (baseline)
  2. +context      — meal hour-of-day, meal index in day
  3. +anthro/labs  — subject-level Age, Gender, BMI, A1c, Fasting GLU, Insulin,
                     Triglycerides, Cholesterol, HDL, LDL (from bio.csv)
  4. +history      — LEAKAGE-SAFE prior-response history: running mean iAUC /
                     carbs of the subject's OWN earlier included meals + count
  5. +microbiome   — the 22 ordinal Viome gut-health scores (NOT the 1979
                     binary taxa: p>>n at n=31)

Reports, for each nested tier AND each leave-one-tier-out variant: Pearson R
(primary, prereg §7.2) with subject-level bootstrap 95% CIs (§7.3), plus the
marginal ΔR (paired bootstrap CI) of the microbiome tier and the prior-history
tier specifically (H4).

Writes:
  results/cgmacros_ablation.csv -- per-tier metrics + CIs
  results/cgmacros_ablation.md  -- tiers, leave-one-out, marginal deltas, H4
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ppgr.adapters import cgmacros
from ppgr.evaluate import (
    N_CALIB,
    SEED,
    TARGET,
    Prediction,
    _fit_xgb,
    _split_calibration,
    paired_delta_ci,
    summarize,
)
from ppgr.features import (
    ablation_tiers,
    add_features,
    build_history_features,
    leave_one_tier_out_sets,
    nested_addin_sets,
)
from ppgr.inclusion import apply_inclusion

DATA_ROOT = "data/cgmacros/extracted/CGMacros"
OUT_DIR = "results"
NON_DIABETIC = {"healthy", "pre-DM"}


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.3f}"


def run_xgb_loso(meals: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict[str, Prediction]:
    """XGBoost under LOSO for each named feature set (same protocol as
    evaluate.run_loso: leakage-safe per-subject calibration so all feature sets
    are evaluated on the SAME held-out meals; subject-grouped nested grid search
    for hyperparameters; native NaN handling => raw features, no imputation)."""
    df = _split_calibration(meals)
    subjects = sorted(df["subject_id"].unique())
    acc = {name: {"sid": [], "yt": [], "yp": []} for name in feature_sets}

    for test_sid in subjects:
        train = df[df["subject_id"] != test_sid]
        test_all = df[df["subject_id"] == test_sid]
        test = test_all[~test_all["_is_calib"]]
        if test.empty:
            continue
        ytr = train[TARGET].to_numpy()
        yte = test[TARGET].to_numpy()
        groups_tr = train["subject_id"].to_numpy()

        for name, cols in feature_sets.items():
            # XGBoost native missing handling -> feed raw (no imputation), same
            # as evaluate.run_loso (prereg §6.2/§6.3).
            xgb = _fit_xgb(train[cols], ytr, groups_tr)
            yp = xgb.predict(test[cols])
            acc[name]["sid"].extend([test_sid] * len(test))
            acc[name]["yt"].extend(yte.tolist())
            acc[name]["yp"].extend(np.asarray(yp, dtype=float).tolist())

    return {
        name: Prediction(
            method=name,
            subject_id=np.array(d["sid"]),
            y_true=np.array(d["yt"], dtype=float),
            y_pred=np.array(d["yp"], dtype=float),
        )
        for name, d in acc.items()
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CGMacros cohort ...")
    cohort = cgmacros.load(DATA_ROOT)

    nd_subjects = sorted(
        sid for sid, g in cohort.meals.groupby("subject_id")["group"].first().items()
        if g in NON_DIABETIC
    )
    print(f"Non-diabetic (healthy+pre-DM): {len(nd_subjects)} subjects")
    cohort.meals = cohort.meals[cohort.meals["subject_id"].isin(nd_subjects)].copy()
    cohort.cgm = {s: cohort.cgm[s] for s in nd_subjects}

    print("Applying inclusion filters (prereg §3.2) ...")
    included, _att = apply_inclusion(cohort)
    print(f"  included meals: {len(included)} over "
          f"{included['subject_id'].nunique()} subjects")

    # tier 2 (context) + tier 4 (history; needs iauc_pos => after inclusion)
    included = add_features(included)
    included = build_history_features(included, target=TARGET)

    # how many subjects have ANY valid Viome gut-health score (tier 5)
    gh_cols = [c for c in included.columns if c.startswith("gh_")]
    has_viome = (
        included.groupby("subject_id")[gh_cols].first().notna().any(axis=1).sum()
    )
    print(f"  subjects with >=1 Viome gut-health score: {has_viome} "
          f"/ {included['subject_id'].nunique()}")

    nested = nested_addin_sets()
    loto = leave_one_tier_out_sets()

    cache = os.path.join(OUT_DIR, ".ablation_preds.pkl")
    if os.environ.get("ABLATION_USE_CACHE") == "1" and os.path.exists(cache):
        import pickle
        print("Loading cached LOSO predictions ...")
        with open(cache, "rb") as f:
            nested_preds, loto_preds = pickle.load(f)
    else:
        print("Running XGBoost LOSO for each nested tier ...")
        nested_preds = run_xgb_loso(included, nested)
        print("Running XGBoost LOSO for each leave-one-tier-out set ...")
        loto_preds = run_xgb_loso(included, loto)
        import pickle
        with open(cache, "wb") as f:
            pickle.dump((nested_preds, loto_preds), f)

    nested_summary = summarize(nested_preds)
    loto_summary = summarize(loto_preds)

    # keep add-in order / leave-one-out order
    nested_summary["__o"] = nested_summary["method"].map(
        {m: i for i, m in enumerate(nested)}
    )
    nested_summary = nested_summary.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    loto_summary["__o"] = loto_summary["method"].map({m: i for i, m in enumerate(loto)})
    loto_summary = loto_summary.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    # --- marginal deltas (paired subject-level bootstrap CI on Pearson R) ---
    tier_names = list(ablation_tiers())  # ["macros","context","anthro_labs",...]
    nested_names = list(nested)          # cumulative add-in set names (add-in order)
    # add-in marginal of EACH tier (i>=1) = nested[i] - nested[i-1], keyed by TIER
    addin_deltas = {}
    for i in range(1, len(nested_names)):
        cur, prev = nested_names[i], nested_names[i - 1]
        addin_deltas[tier_names[i]] = paired_delta_ci(
            nested_preds[cur], nested_preds[prev], "pearson_r"
        )
    # leave-one-out marginal of a tier = full - all_minus_tier, keyed by TIER
    full_name = nested_names[-1]  # the all-tier set
    loto_deltas = {}
    for drop in list(loto):
        tier = drop.replace("all_minus_", "")
        loto_deltas[tier] = paired_delta_ci(
            nested_preds[full_name], loto_preds[drop], "pearson_r"
        )

    # combine + write
    combined = pd.concat([nested_summary, loto_summary], ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "cgmacros_ablation.csv"), index=False)

    _write_md(included, has_viome, nested_summary, loto_summary,
              addin_deltas, loto_deltas, tier_names)
    _print_console(nested_summary, loto_summary, addin_deltas, loto_deltas)


def _ci_row(r) -> str:
    return (f"| {r['method']} | {int(r['n_meals'])} | "
            f"{fmt(r['pearson_r'])} [{fmt(r['pearson_r_lo'])}, {fmt(r['pearson_r_hi'])}] | "
            f"{fmt(r['spearman_r'])} | "
            f"{fmt(r['rmse'])} [{fmt(r['rmse_lo'])}, {fmt(r['rmse_hi'])}] | "
            f"{fmt(r['mae'])} |")


def _write_md(included, has_viome, nested_summary, loto_summary,
              addin_deltas, loto_deltas, tier_names) -> None:
    n_subj = included["subject_id"].nunique()
    lines = []
    lines.append("# CGMacros feature-group ablation — results\n")
    lines.append(
        "GitHub **#24** (feature-group ablation + personalization-value); "
        "prereg **§5.3** feature tiers + hypothesis **H4**. Model = **XGBoost** "
        "(the field workhorse, §6.2) under the SAME LOSO + subject-grouped nested "
        "CV and leakage-safe per-subject calibration as `evaluate.py` (earliest "
        f"{N_CALIB} meals/subject reserved; all tiers evaluated on the SAME "
        "held-out meals). Primary outcome `iAUC_pos` (§4); primary metric "
        "**Pearson R** with subject-level bootstrap 95% CIs (§7.2-§7.3, "
        f"seed={SEED}). Population = **non-diabetic** (healthy + pre-DM), "
        f"A1c-derived (§3.1): **{n_subj}** subjects contribute >=1 included meal; "
        f"{len(included)} included meals.\n"
    )
    lines.append("## Feature tiers (prereg §5.3)\n")
    lines.append("1. **macros** — carbs, protein, fat, fiber, calorie (baseline)")
    lines.append("2. **+context** — meal hour-of-day, meal index in day")
    lines.append("3. **+anthro/labs** — subject-level Age, Gender(→0/1), BMI, A1c, "
                 "Fasting GLU, Insulin, Triglycerides, Cholesterol, HDL, LDL (bio.csv)")
    lines.append("4. **+history** — LEAKAGE-SAFE prior-response history: running "
                 "mean iAUC & carbs of the subject's OWN earlier included meals + "
                 "prior-meal count (strictly time-ordered, within-subject => valid "
                 "under LOSO)")
    lines.append("5. **+microbiome** — the **22 ordinal Viome gut-health scores** "
                 "(NOT the 1979 binary taxa: p>>n at n=31, so the binary tier is "
                 f"omitted by design). Subjects with no Viome sample "
                 f"({n_subj - has_viome} of {n_subj}) have all-NaN scores — XGBoost "
                 "routes NaN natively (no imputation, leakage-safe).\n")

    lines.append("## Nested add-in tiers (XGBoost, Pearson R [95% CI])\n")
    lines.append("| feature set | n_meals | Pearson R [95% CI] | Spearman ρ | RMSE [95% CI] | MAE |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in nested_summary.iterrows():
        lines.append(_ci_row(r))
    lines.append("")

    lines.append("## Leave-one-tier-out (all five tiers MINUS one)\n")
    lines.append("| feature set | n_meals | Pearson R [95% CI] | Spearman ρ | RMSE [95% CI] | MAE |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in loto_summary.iterrows():
        lines.append(_ci_row(r))
    lines.append("")

    lines.append("## Marginal ΔR per tier (paired subject-level bootstrap)\n")
    lines.append("**Add-in marginal** = R(with tier) − R(previous nested set). "
                 "**Leave-out marginal** = R(all tiers) − R(all tiers minus this one).\n")
    lines.append("| tier | add-in ΔR [95% CI] | CI excl. 0? | leave-out ΔR [95% CI] | CI excl. 0? |")
    lines.append("|---|---|---|---|---|")
    for i in range(1, len(tier_names)):
        tier = tier_names[i]
        ap, alo, ahi = addin_deltas[tier]
        lp, llo, lhi = loto_deltas[tier]
        aex = (alo > 0 or ahi < 0)
        lex = (llo > 0 or lhi < 0)
        lines.append(
            f"| {tier} | {fmt(ap)} [{fmt(alo)}, {fmt(ahi)}] | {'yes' if aex else 'no'} | "
            f"{fmt(lp)} [{fmt(llo)}, {fmt(lhi)}] | {'yes' if lex else 'no'} |"
        )
    # macros is the baseline tier (no add-in delta); show its leave-out only
    if "macros" in loto_deltas:
        lp, llo, lhi = loto_deltas["macros"]
        lex = (llo > 0 or lhi < 0)
        lines.append(
            f"| macros (baseline) | — | — | {fmt(lp)} [{fmt(llo)}, {fmt(lhi)}] | "
            f"{'yes' if lex else 'no'} |"
        )
    lines.append("")

    # H4 verdict
    mic_a = addin_deltas["microbiome"]
    mic_l = loto_deltas["microbiome"]
    his_a = addin_deltas["history"]
    his_l = loto_deltas["history"]
    mac_l = loto_deltas["macros"]
    lines.append("## H4 verdict (prereg §1, H4)\n")
    lines.append(
        "> **H4.** Meal macronutrients (+ a short history of the person's prior "
        "responses) dominate the predictive signal; microbiome (Viome) adds "
        "little for *glucose*. *Falsified if* the microbiome tier adds predictive "
        "value comparable to/exceeding the macro tier, **or** if prior-response "
        "history adds negligibly.\n"
    )
    lines.append(f"- **Microbiome marginal:** add-in ΔR = {fmt(mic_a[0])} "
                 f"[{fmt(mic_a[1])}, {fmt(mic_a[2])}]; leave-out ΔR = {fmt(mic_l[0])} "
                 f"[{fmt(mic_l[1])}, {fmt(mic_l[2])}].")
    lines.append(f"- **Prior-history marginal:** add-in ΔR = {fmt(his_a[0])} "
                 f"[{fmt(his_a[1])}, {fmt(his_a[2])}]; leave-out ΔR = {fmt(his_l[0])} "
                 f"[{fmt(his_l[1])}, {fmt(his_l[2])}].")
    lines.append(f"- **Macro tier (leave-out, for comparison):** ΔR = {fmt(mac_l[0])} "
                 f"[{fmt(mac_l[1])}, {fmt(mac_l[2])}].\n")
    lines.append(
        "_Interpretation rule (pre-specified, §1 H4):_ H4 is **supported** if "
        "microbiome's marginal ΔR is small and not larger than the macro tier's, "
        "and prior-history adds non-negligibly; **falsified** otherwise. See the "
        "console/summary verdict for the data-driven call.\n"
    )

    lines.append("## Notes\n")
    lines.append("- **Microbiome tier = 22 ordinal Viome gut-health scores**, NOT "
                 "the 1979 binary taxon-presence indicators. With n=31 subjects the "
                 "binary set is p>>n (1979≫31) and would dominate any honest "
                 "ablation by overfitting noise; the 22 ordinal scores are the "
                 "defensible subject-level microbiome representation. This choice is "
                 "pre-specified here.")
    lines.append("- **Leakage-safety:** anthro/labs and gut-health are "
                 "constant-within-subject and a held-out subject's values are never "
                 "in any training meal; history uses only the subject's own earlier "
                 "meals. Under LOSO no test-subject information leaks into training.")
    lines.append("- **All tiers paired** on the same held-out meals (per-subject "
                 "calibration scheme), so every ΔR is a paired comparison.")
    lines.append("")

    with open(os.path.join(OUT_DIR, "cgmacros_ablation.md"), "w") as f:
        f.write("\n".join(lines))


def _print_console(nested_summary, loto_summary, addin_deltas, loto_deltas) -> None:
    print("\n===== NESTED ADD-IN TIERS (XGBoost, Pearson R [CI]) =====")
    for _, r in nested_summary.iterrows():
        print(f"  {r['method']:45s} R={fmt(r['pearson_r'])} "
              f"[{fmt(r['pearson_r_lo'])},{fmt(r['pearson_r_hi'])}]  "
              f"RMSE={fmt(r['rmse'])}  n={int(r['n_meals'])}")
    print("\n===== LEAVE-ONE-TIER-OUT (Pearson R [CI]) =====")
    for _, r in loto_summary.iterrows():
        print(f"  {r['method']:45s} R={fmt(r['pearson_r'])} "
              f"[{fmt(r['pearson_r_lo'])},{fmt(r['pearson_r_hi'])}]  n={int(r['n_meals'])}")
    print("\n===== MARGINAL ΔR (paired bootstrap) =====")
    for tier, (p, lo, hi) in addin_deltas.items():
        print(f"  add-in   {tier:12s}: ΔR={fmt(p)} [{fmt(lo)},{fmt(hi)}]")
    for tier, (p, lo, hi) in loto_deltas.items():
        print(f"  leave-out{tier:12s}: ΔR={fmt(p)} [{fmt(lo)},{fmt(hi)}]")
    print("\nWrote results/cgmacros_ablation.csv and results/cgmacros_ablation.md")


if __name__ == "__main__":
    main()
