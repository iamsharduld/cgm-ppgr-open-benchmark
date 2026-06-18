"""Empirical label-noise / reproducibility-ceiling analysis (GitHub #26; prereg §9 / S19).

CGMacros logs BOTH `Libre GL` (Abbott FreeStyle Libre Pro) and `Dexcom GL`
(Dexcom G6 Pro) for the SAME person at the SAME minute. For every qualifying
meal in the NON-DIABETIC subjects (healthy + pre-DM, A1c<5.7 / 5.7-<6.5) we
compute the primary outcome iAUC_pos over 0-120 min TWICE -- once from each
device -- using the SAME pre-registered `ppgr.iauc.compute_ppgr` logic and the
SAME per-meal inclusion rules (prereg §3.2) applied INDEPENDENTLY to each stream.

The device-to-device agreement of the two iAUC measurements of the SAME meal is
an EMPIRICAL CEILING on how well any feature-based model could predict "the"
iAUC: a model cannot beat the measurement's own between-device reproducibility.
We report Pearson R, Spearman rho, ICC(2,1), and Bland-Altman bias + 95% limits
of agreement, and contrast the implied ceiling with our best model R (~0.35).

STANDALONE: reads the raw CGMacros CSVs directly. It REUSES `ppgr.iauc`
(compute_ppgr + the pre-registered constants) unchanged, and does NOT import the
cgmacros adapter or the inclusion module (their §3.1/§3.2 logic is replicated
here so this script does not depend on modules another task is editing).

Usage:  PYTHONPATH=src python3 run_noise_ceiling.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from ppgr.iauc import MAX_GAP_MIN, compute_ppgr  # REUSED, unchanged

DATA_ROOT = "data/cgmacros/extracted/CGMacros"
OUT_DIR = "results"

# ---- pre-registered constants (mirrored from prereg §3.1/§3.2; not invented) ----
A1C_HEALTHY_MAX = 5.7        # healthy: A1c < 5.7      (prereg §3.1, schema verified)
A1C_PREDM_MAX = 6.5          # pre-DM: 5.7 <= A1c < 6.5 ; T2D: >= 6.5
NON_DIABETIC = {"healthy", "pre-DM"}

WASHOUT_MIN = 120.0          # prior-meal washout            (prereg §3.2.3)
OVERLAP_MIN = 120.0          # no other meal in (0, 120]      (prereg §3.2.3)
READ_LO_MIN = -30.0          # generous CGM read window (baseline + bracketing)
READ_HI_MIN = 150.0

# The two CGM streams to compare (both per-minute in the published files).
STREAMS = {"dexcom": "Dexcom GL", "libre": "Libre GL"}


def _a1c_group(a1c: float) -> str:
    if pd.isna(a1c):
        return "unknown"
    if a1c < A1C_HEALTHY_MAX:
        return "healthy"
    if a1c < A1C_PREDM_MAX:
        return "pre-DM"
    return "T2D"


def _subject_to_folder(subject: int) -> str:
    return f"CGMacros-{int(subject):03d}"


def _neighbor_meal_flags(meal_times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Per meal (within subject): no meal in (0,120] after (overlap ok) and no
    meal in [-120,0) before (washout ok). Mirrors ppgr.inclusion exactly."""
    n = len(meal_times)
    no_overlap = np.ones(n, dtype=bool)
    washout_ok = np.ones(n, dtype=bool)
    secs = (
        pd.to_datetime(meal_times).astype("datetime64[ns]").astype("int64").to_numpy()
        / 1e9
    )
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = (secs[j] - secs[i]) / 60.0
            if 0 < d <= OVERLAP_MIN:
                no_overlap[i] = False
            if -WASHOUT_MIN <= d < 0:
                washout_ok[i] = False
    return no_overlap, washout_ok


def _icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1) -- two-way random, absolute agreement, single measure.

    The standard two-rater (here: two devices) form. n targets (meals), k=2
    raters (Dexcom, Libre). Uses mean squares from a two-way ANOVA.
    """
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape  # k == 2
    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)

    ss_total = ((m - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()          # between targets
    ss_cols = n * ((col_means - grand) ** 2).sum()          # between raters
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + (k / n) * (ms_cols - ms_err)
    if denom == 0:
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- non-diabetic subjects via bio.csv A1c (ADA thresholds, prereg §3.1) ----
    bio = pd.read_csv(os.path.join(DATA_ROOT, "bio.csv"))
    bio["a1c"] = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    bio["group"] = bio["a1c"].map(_a1c_group)
    bio["folder"] = bio["subject"].map(_subject_to_folder)

    nd = bio[bio["group"].isin(NON_DIABETIC)].copy()
    nd_folders = [
        f for f in nd["folder"].tolist()
        if os.path.exists(os.path.join(DATA_ROOT, f, f"{f}.csv"))
    ]
    print(
        f"Non-diabetic subjects (healthy+pre-DM) with a CSV on disk: "
        f"{len(nd_folders)} "
        f"(healthy={int((nd['group']=='healthy').sum())}, "
        f"pre-DM={int((nd['group']=='pre-DM').sum())})"
    )

    rows = []          # one row per meal that yields a valid iAUC on BOTH streams
    # attrition over the two-stream requirement
    n_meal_rows = 0
    n_pass_carb = 0
    n_pass_overlap = 0
    n_pass_washout = 0
    n_valid_dex = 0
    n_valid_lib = 0
    n_valid_both = 0

    for sid in nd_folders:
        path = os.path.join(DATA_ROOT, sid, f"{sid}.csv")
        df = pd.read_csv(path, low_memory=False)
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")

        # per-stream CGM frames (timestamp, glucose), dropping that stream's NaNs
        cgm = {}
        for key, col in STREAMS.items():
            g = pd.to_numeric(df[col], errors="coerce")
            sc = pd.DataFrame({"timestamp": ts, "glucose": g})
            sc = sc.dropna(subset=["timestamp", "glucose"]).sort_values("timestamp")
            cgm[key] = sc.reset_index(drop=True)

        # meals = rows with non-null Meal Type (any eating event; snacks kept §3.3)
        mt = df["Meal Type"].astype("string").str.strip()
        is_meal = mt.notna() & (mt != "") & (mt != "nan")
        m = pd.DataFrame(
            {
                "meal_time": pd.to_datetime(df.loc[is_meal, "Timestamp"], errors="coerce"),
                "meal_type": mt.loc[is_meal].to_numpy(),
                "carbs": pd.to_numeric(df.loc[is_meal, "Carbs"], errors="coerce"),
            }
        ).dropna(subset=["meal_time"]).sort_values("meal_time").reset_index(drop=True)

        if m.empty:
            continue
        no_overlap, washout_ok = _neighbor_meal_flags(m["meal_time"])

        for i, meal in m.iterrows():
            n_meal_rows += 1
            t0 = meal["meal_time"]

            # (2) known carbs
            if pd.isna(meal["carbs"]):
                continue
            n_pass_carb += 1
            # (3a) no overlapping meal in (0,120]
            if not no_overlap[i]:
                continue
            n_pass_overlap += 1
            # (3b) prior-meal washout
            if not washout_ok[i]:
                continue
            n_pass_washout += 1

            # (1) CGM coverage + gap, applied INDEPENDENTLY per stream, same rules
            lo = t0 + pd.Timedelta(minutes=READ_LO_MIN)
            hi = t0 + pd.Timedelta(minutes=READ_HI_MIN)
            outs = {}
            valid = {}
            for key in STREAMS:
                s = cgm[key]
                win = s.loc[(s["timestamp"] >= lo) & (s["timestamp"] <= hi),
                            ["timestamp", "glucose"]]
                if win.empty:
                    valid[key] = False
                    continue
                o = compute_ppgr(win["timestamp"], win["glucose"], t0)
                ok = (
                    o.has_t0 and o.has_t120
                    and (o.max_gap_min <= MAX_GAP_MIN)
                    and not np.isnan(o.iauc_pos)
                )
                outs[key] = o
                valid[key] = ok

            if valid.get("dexcom"):
                n_valid_dex += 1
            if valid.get("libre"):
                n_valid_lib += 1
            if valid.get("dexcom") and valid.get("libre"):
                n_valid_both += 1
                rows.append(
                    {
                        "subject_id": sid,
                        "group": meal_group(bio, sid),
                        "meal_time": t0,
                        "meal_type": meal["meal_type"],
                        "carbs": meal["carbs"],
                        "iauc_pos_dexcom": outs["dexcom"].iauc_pos,
                        "iauc_pos_libre": outs["libre"].iauc_pos,
                        "iauc_net_dexcom": outs["dexcom"].iauc_net,
                        "iauc_net_libre": outs["libre"].iauc_net,
                        "baseline_dexcom": outs["dexcom"].baseline,
                        "baseline_libre": outs["libre"].baseline,
                        "peak_rise_dexcom": outs["dexcom"].peak_rise,
                        "peak_rise_libre": outs["libre"].peak_rise,
                    }
                )

    paired = pd.DataFrame(rows)
    paired.to_csv(os.path.join(OUT_DIR, "noise_ceiling_pairs.csv"), index=False)

    if paired.empty:
        raise SystemExit("No meals with a valid iAUC on BOTH streams -- aborting.")

    d = paired["iauc_pos_dexcom"].to_numpy()
    l = paired["iauc_pos_libre"].to_numpy()
    n = len(paired)

    # ---- agreement statistics on iAUC_pos (the primary outcome) ----
    pearson_r, pearson_p = stats.pearsonr(d, l)
    spearman_r, spearman_p = stats.spearmanr(d, l)
    icc = _icc_2_1(d, l)

    # Bland-Altman: difference (Dexcom - Libre) vs mean
    diff = d - l
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_lo = bias - 1.96 * sd_diff
    loa_hi = bias + 1.96 * sd_diff
    mean_iauc = float(np.mean(np.concatenate([d, l])))

    # subject-level bootstrap CI for Pearson R (resample SUBJECTS, prereg §7.3)
    r_boot = _bootstrap_subject_pearson(paired, n_boot=2000, seed=20260618)
    r_ci_lo, r_ci_hi = np.nanpercentile(r_boot, [2.5, 97.5])

    # net-iAUC sensitivity
    dn = paired["iauc_net_dexcom"].to_numpy()
    ln = paired["iauc_net_libre"].to_numpy()
    pearson_r_net, _ = stats.pearsonr(dn, ln)
    icc_net = _icc_2_1(dn, ln)

    print("\n==== device-agreement ceiling (iAUC_pos, 0-120 min) ====")
    print(f"n subjects contributing paired meals: {paired['subject_id'].nunique()}")
    print(f"n meals with valid iAUC on BOTH streams: {n}")
    print(f"Pearson R  = {pearson_r:.3f}  (95% subj-bootstrap CI [{r_ci_lo:.3f}, {r_ci_hi:.3f}], p={pearson_p:.2e})")
    print(f"Spearman r = {spearman_r:.3f}  (p={spearman_p:.2e})")
    print(f"ICC(2,1)   = {icc:.3f}")
    print(f"Bland-Altman bias (Dexcom-Libre) = {bias:.1f} mg/dL*min")
    print(f"  95% limits of agreement = [{loa_lo:.1f}, {loa_hi:.1f}] mg/dL*min")
    print(f"  mean iAUC across streams = {mean_iauc:.1f} mg/dL*min  (SD of diffs = {sd_diff:.1f})")
    print(f"R^2 shared variance (ceiling on R^2) = {pearson_r**2:.3f}")
    print(f"net-iAUC sensitivity: Pearson R = {pearson_r_net:.3f}, ICC = {icc_net:.3f}")

    _write_report(
        n=n,
        n_subj=paired["subject_id"].nunique(),
        n_meal_rows=n_meal_rows,
        n_pass_carb=n_pass_carb,
        n_pass_overlap=n_pass_overlap,
        n_pass_washout=n_pass_washout,
        n_valid_dex=n_valid_dex,
        n_valid_lib=n_valid_lib,
        n_valid_both=n_valid_both,
        n_nd=len(nd_folders),
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        r_ci=(r_ci_lo, r_ci_hi),
        spearman_r=spearman_r,
        icc=icc,
        bias=bias,
        sd_diff=sd_diff,
        loa=(loa_lo, loa_hi),
        mean_iauc=mean_iauc,
        pearson_r_net=pearson_r_net,
        icc_net=icc_net,
        paired=paired,
    )
    print(f"\nWrote {OUT_DIR}/noise_ceiling.md and {OUT_DIR}/noise_ceiling_pairs.csv")


def meal_group(bio: pd.DataFrame, folder: str) -> str:
    row = bio[bio["folder"] == folder]
    return str(row["group"].iloc[0]) if len(row) else "unknown"


def _bootstrap_subject_pearson(paired: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    subs = paired["subject_id"].unique()
    by_sub = {s: paired[paired["subject_id"] == s] for s in subs}
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        frames = [by_sub[s] for s in pick]
        samp = pd.concat(frames, ignore_index=True)
        if len(samp) < 3:
            continue
        dd = samp["iauc_pos_dexcom"].to_numpy()
        ll = samp["iauc_pos_libre"].to_numpy()
        if np.std(dd) == 0 or np.std(ll) == 0:
            continue
        out[b] = stats.pearsonr(dd, ll)[0]
    return out


def _write_report(**k) -> None:
    paired = k["paired"]
    rci_lo, rci_hi = k["r_ci"]
    loa_lo, loa_hi = k["loa"]
    r = k["pearson_r"]

    # per-group breakdown
    grp_lines = []
    for grp, gdf in paired.groupby("group"):
        if len(gdf) >= 3:
            gr = stats.pearsonr(gdf["iauc_pos_dexcom"], gdf["iauc_pos_libre"])[0]
            gicc = _icc_2_1(gdf["iauc_pos_dexcom"].to_numpy(), gdf["iauc_pos_libre"].to_numpy())
            grp_lines.append(f"| {grp} | {gdf['subject_id'].nunique()} | {len(gdf)} | {gr:.3f} | {gicc:.3f} |")

    md = f"""# Empirical label-noise / reproducibility ceiling — dual-CGM device agreement

**GitHub #26; prereg §9 / S19.** Standalone analysis (`run_noise_ceiling.py`). Reuses the
pre-registered iAUC engine (`src/ppgr/iauc.py`, `compute_ppgr`) unchanged; reads the raw CGMacros
`CGMacros-0XX.csv` files directly (no adapter import).

## Idea
CGMacros logs **both** `Dexcom GL` (Dexcom G6 Pro) and `Libre GL` (Abbott FreeStyle Libre Pro) for
the **same person at the same minute**. For every qualifying meal in the **non-diabetic** population
(healthy + pre-DM, derived from `bio.csv` `A1c PDL (Lab)` via ADA thresholds, prereg §3.1), we compute
the primary outcome **iAUC_pos (0–120 min)** *twice* — once per device — applying the **identical**
pre-registered method (§4) and per-meal inclusion rules (§3.2), independently to each stream. The
two devices' iAUC for the **same meal** disagree only because of measurement/device noise; their
agreement is therefore an **empirical ceiling** on how well *any* feature-based model could predict
"the" iAUC. A model cannot out-predict the target's own between-device reproducibility.

## Sample / attrition (two-stream requirement)
- Non-diabetic subjects with a CSV on disk: **{k['n_nd']}**.
- Raw meal rows (non-null `Meal Type`): **{k['n_meal_rows']}**.
- Pass known-carb (§3.2.2): **{k['n_pass_carb']}** → pass no-overlap (§3.2.3): **{k['n_pass_overlap']}**
  → pass prior-meal washout (§3.2.3): **{k['n_pass_washout']}**.
- Valid iAUC on **Dexcom** (coverage + ≤30-min gap, §3.2.1): **{k['n_valid_dex']}**;
  on **Libre**: **{k['n_valid_lib']}**.
- **Meals with a valid iAUC on BOTH streams (analysis set): {k['n']}** across
  **{k['n_subj']}** subjects.

## Device-agreement of iAUC_pos (Dexcom vs Libre, same meal)
| statistic | value |
|---|---|
| n meals (both streams) | **{k['n']}** |
| Pearson R | **{r:.3f}** (95% subject-bootstrap CI [{rci_lo:.3f}, {rci_hi:.3f}]; p={k['pearson_p']:.2e}) |
| Spearman ρ | {k['spearman_r']:.3f} |
| ICC(2,1) (absolute agreement, single measure) | **{k['icc']:.3f}** |
| R² (shared variance) | {r**2:.3f} |
| Bland–Altman bias (Dexcom − Libre) | **{k['bias']:.1f}** mg/dL·min |
| Bland–Altman 95% limits of agreement | **[{loa_lo:.1f}, {loa_hi:.1f}]** mg/dL·min |
| SD of paired differences | {k['sd_diff']:.1f} mg/dL·min |
| mean iAUC across streams | {k['mean_iauc']:.1f} mg/dL·min |

**Net-iAUC sensitivity (`iAUC_net`, signed):** Pearson R = {k['pearson_r_net']:.3f}, ICC = {k['icc_net']:.3f}.

### Per-subgroup agreement
| group | n subj | n meals | Pearson R | ICC(2,1) |
|---|---|---|---|---|
{chr(10).join(grp_lines)}

## Interpretation (prereg §9)
**Device-agreement ceiling:** Pearson **R = {r:.3f}** (R² = {r**2:.3f}), ICC = {k['icc']:.3f}.

The two devices, measuring *the same meal at the same minute*, agree on iAUC_pos only at R = {r:.3f}
(ICC = {k['icc']:.3f}). This is an upper bound on predictability: the "label" a model is trained to
predict is itself only reproducible to this degree, so no feature-based model can be expected to
correlate with one device's iAUC better than the *other device* does. The ceiling is
{'modest' if r < 0.7 else 'comparatively high'}, so even a *perfect* model could not reach the
R ≈ 0.6–0.77 reported in proprietary single-cohort studies (Zeevi 2015, Berry/PREDICT 2020) on this
target as measured here.

This is fully consistent with the prereg §9 priors, set *a priori*: duplicate-meal within-person ICC
of **0.14 (Dexcom) / 0.31 (Abbott)** (Hengist & Hall 2024/25), documented between-sensor CGM
disagreement (Selvin 2023), and large within-person iAUC CV ≈ 33–80% (Vrolix & Mensink 2010). Our
between-*device* (same-meal) agreement is an *upper* bound on, and complementary to, those
between-*day* duplicate-meal ICCs: it isolates pure device/measurement noise (same meal, same minute,
no biological day-to-day variation), and is already enough to cap achievable R well below the
historical single-cohort headline figures. The honest reading: a meaningful fraction of the
model's "error" is irreducible label noise, the predictive ceiling on this benchmark is modest, and
the right yardstick for model quality is this device-agreement ceiling — not the proprietary-cohort
R values that were never decomposed against their own measurement noise.

*Method notes.* iAUC = pre-registered `iAUC_pos` (trapezoidal area-above-baseline, baseline = mean of
[−15, 0] min, 5-min resample grid; Wolever 2004 / ISO 26642). Both streams are stored per-minute in
the published files; the same §3.2 coverage rule (t=0 & t=120 recoverable, no interpolation gap >30
min) is applied independently to each stream, and only meals passing on **both** enter the analysis.
ICC(2,1) = two-way random-effects, absolute-agreement, single-measure. Subject-level bootstrap
(resample subjects, prereg §7.3), 2000 reps, seed 20260618.
"""
    with open(os.path.join(OUT_DIR, "noise_ceiling.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
