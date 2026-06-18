"""Disjoint-subject CV, baselines, models, metrics (prereg.md §6-§7).

Splitting (§7.1): leave-one-subject-out (LOSO) -- pre-specified fallback for
small n (n=16 here), matching the CGMacros paper's LOSO.

Personal-calibration scheme (§6.1.2): for EACH test subject reserve their
EARLIEST 3 meals as a personal-calibration set used only to estimate that
subject's level (per-person-mean baseline). ALL methods are then evaluated only
on that subject's REMAINING meals, so every comparison is paired on the same
held-out meals.

Baselines (§6.1): population-mean, per-person-mean (leakage-safe), carb-only
(linear), carb+calorie (linear). Models (§6.2): ElasticNet, XGBoost.

Metrics (§7.2): Pearson R (primary), Spearman, RMSE, MAE -- pooled over held-out
meals -- with subject-level bootstrap 95% CIs (resample subjects) (§7.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ppgr.features import MACRO_FEATURES, feature_sets

TARGET = "iauc_pos"
N_CALIB = 3            # earliest meals reserved per test subject (§6.1.2)
SEED = 12345           # fixed seed (§6.3 / §7.1)
N_BOOT = 2000          # bootstrap resamples for subject-level CIs (§7.3)


@dataclass
class Prediction:
    """Pooled held-out predictions for one method."""

    method: str
    subject_id: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


def _split_calibration(meals: pd.DataFrame) -> pd.DataFrame:
    """Tag each test subject's earliest N_CALIB meals as calibration (§6.1.2)."""
    df = meals.copy()
    df["_is_calib"] = False
    for _, g in df.groupby("subject_id"):
        earliest = g.sort_values("meal_time").index[:N_CALIB]
        df.loc[earliest, "_is_calib"] = True
    return df


def _grouped_inner_cv(groups: np.ndarray, n_splits: int = 5):
    """Subject-grouped inner-CV splits within the outer training fold (prereg §6.3).

    Returns a list of (train_idx, val_idx) with NO subject shared across the inner
    split, or a plain int k as a degenerate fallback if there are too few groups.
    """
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        return 2
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


def _make_elasticnet(cv) -> Pipeline:
    # inner CV for the regularization path = subject-grouped folds (§6.3)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    alphas=50,  # sklearn>=1.9: int => that many alphas on the path
                    cv=cv,
                    max_iter=20000,
                    random_state=SEED,
                ),
            ),
        ]
    )


# Small MLP (prereg §6.2.3): a deliberately small net for the small effective n.
# Fixed, sensible defaults (NO heavy search, per the small-n rationale):
#   * hidden_layer_sizes=(16,)   -- a single small hidden layer; with ~25-50
#     features and a few hundred meals, more capacity just overfits.
#   * early_stopping=True        -- 10% internal validation hold-out with
#     n_iter_no_change patience => regularization without a tuning loop.
#   * alpha=1e-2                 -- modest L2 weight decay (heavier than the
#     sklearn 1e-4 default, to curb overfitting at this n).
#   * StandardScaler + median-impute upstream (same preprocessing as ElasticNet).
# Mirrors ElasticNet's StandardScaler pipeline so the two linear/non-linear
# regularized models see identical, leakage-safe (train-median imputed) inputs.
_MLP_PARAMS = dict(
    hidden_layer_sizes=(16,),
    activation="relu",
    solver="adam",
    alpha=1e-2,
    early_stopping=True,
    n_iter_no_change=20,
    max_iter=2000,
    random_state=SEED,
)


def _make_mlp() -> Pipeline:
    """Small MLP with StandardScaler (prereg §6.2.3); median-impute upstream."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", MLPRegressor(**_MLP_PARAMS)),
        ]
    )


# XGBoost: fixed base + a SMALL subject-grouped nested grid search (§6.3)
_XGB_BASE = dict(
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=SEED, n_jobs=4,
)
_XGB_GRID = {"max_depth": [2, 3], "n_estimators": [200, 400]}


def _fit_xgb(X, y, groups: np.ndarray) -> XGBRegressor:
    """Subject-grouped nested grid search for XGBoost (§6.3); native NaN handling.

    Falls back to fixed sensible defaults if too few subjects for grouped CV.
    """
    n_groups = len(np.unique(groups))
    k = min(5, n_groups)
    base = XGBRegressor(**_XGB_BASE)
    if k < 2:
        return base.set_params(n_estimators=300, max_depth=3).fit(X, y)
    gs = GridSearchCV(
        base, _XGB_GRID, cv=GroupKFold(n_splits=k), scoring="r2", n_jobs=1,
    )
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_


def run_loso(meals: pd.DataFrame, target: str = TARGET) -> dict[str, Prediction]:
    """Run all baselines + models under LOSO; return pooled held-out preds.

    `target` defaults to the primary outcome `iauc_pos` (prereg §4); pass
    `peak_rise` for the secondary-outcome sensitivity (prereg §5.1 / issue #25).
    The personal-calibration split (§6.1.2) is independent of the target.
    """
    fs = feature_sets()
    df = _split_calibration(meals)
    subjects = sorted(df["subject_id"].unique())

    acc: dict[str, dict[str, list]] = {
        m: {"sid": [], "yt": [], "yp": []}
        for m in [
            "population_mean",
            "per_person_mean",
            "carb_only",
            "carb_calorie",
            "elasticnet_macros",
            "elasticnet_macros+context",
            "xgboost_macros",
            "xgboost_macros+context",
            "mlp_macros",
            "mlp_macros+context",
        ]
    }

    for test_sid in subjects:
        train = df[df["subject_id"] != test_sid]
        test_all = df[df["subject_id"] == test_sid]
        calib = test_all[test_all["_is_calib"]]
        test = test_all[~test_all["_is_calib"]]
        if test.empty:
            continue  # subject has <= N_CALIB meals: no evaluable meals

        ytr = train[target].to_numpy()
        yte = test[target].to_numpy()

        def record(method: str, preds: np.ndarray) -> None:
            acc[method]["sid"].extend([test_sid] * len(test))
            acc[method]["yt"].extend(yte.tolist())
            acc[method]["yp"].extend(np.asarray(preds, dtype=float).tolist())

        # --- baselines ---
        # population-mean: train-fold mean iAUC (§6.1.1)
        record("population_mean", np.full(len(test), float(ytr.mean())))

        # per-person-mean: leakage-safe, from this subject's calibration meals
        # (§6.1.2); if no calibration meals, fall back to population mean.
        ppm = float(calib[target].mean()) if not calib.empty else float(ytr.mean())
        record("per_person_mean", np.full(len(test), ppm))

        # carb-only linear (§6.1.3)
        lr = LinearRegression().fit(train[["carbs"]], ytr)
        record("carb_only", lr.predict(test[["carbs"]]))

        # carb+calorie linear (§6.1.4)
        cc_cols = ["carbs", "calorie"]
        tr_cc = train.dropna(subset=cc_cols)
        lr2 = LinearRegression().fit(tr_cc[cc_cols], tr_cc[target].to_numpy())
        record("carb_calorie", lr2.predict(test[cc_cols].fillna(tr_cc[cc_cols].median())))

        # --- models, per feature set (§6.2) with subject-grouped nested CV (§6.3) ---
        groups_tr = train["subject_id"].to_numpy()
        inner_cv = _grouped_inner_cv(groups_tr)
        for fs_name, cols in fs.items():
            tr_imp_med = train[cols].median()
            Xtr = train[cols].fillna(tr_imp_med)
            Xte = test[cols].fillna(tr_imp_med)

            en = _make_elasticnet(inner_cv).fit(Xtr, ytr)
            record(f"elasticnet_{fs_name}", en.predict(Xte))

            # Small MLP (§6.2.3): same StandardScaler + train-median-imputed
            # inputs as ElasticNet (Xtr/Xte), fixed small-net defaults.
            mlp = _make_mlp().fit(Xtr, ytr)
            record(f"mlp_{fs_name}", mlp.predict(Xte))

            # XGBoost: native missing handling -> feed raw (no imputation)
            xgb = _fit_xgb(train[cols], ytr, groups_tr)
            record(f"xgboost_{fs_name}", xgb.predict(test[cols]))

    out = {}
    for m, d in acc.items():
        out[m] = Prediction(
            method=m,
            subject_id=np.array(d["sid"]),
            y_true=np.array(d["yt"], dtype=float),
            y_pred=np.array(d["yp"], dtype=float),
        )
    _ = MACRO_FEATURES
    return out


# ---- metrics ----

def _metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    if len(yt) < 2 or np.std(yp) == 0:
        pear = float("nan")
        spear = float("nan")
    else:
        pear = float(stats.pearsonr(yt, yp)[0])
        spear = float(stats.spearmanr(yt, yp)[0])
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return {"pearson_r": pear, "spearman_r": spear, "rmse": rmse, "mae": mae}


def _subject_bootstrap_ci(
    pred: Prediction, metric_fn, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """Subject-level bootstrap 95% CI: resample SUBJECTS with replacement (§7.3)."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(pred.subject_id)
    by_subj = {s: np.where(pred.subject_id == s)[0] for s in subjects}
    point = metric_fn(pred.y_true, pred.y_pred)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subj[s] for s in chosen])
        val = metric_fn(pred.y_true[idx], pred.y_pred[idx])
        if not np.isnan(val):
            boots.append(val)
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize(preds: dict[str, Prediction]) -> pd.DataFrame:
    """Per-method metrics with subject-level bootstrap 95% CIs."""
    metric_names = ["pearson_r", "spearman_r", "rmse", "mae"]
    rows = []
    for method, pred in preds.items():
        row = {"method": method, "n_meals": len(pred.y_true),
               "n_subjects": len(np.unique(pred.subject_id))}
        for mn in metric_names:
            point, lo, hi = _subject_bootstrap_ci(
                pred, lambda yt, yp, _m=mn: _metrics(yt, yp)[_m]
            )
            row[mn] = point
            row[f"{mn}_lo"] = lo
            row[f"{mn}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_delta_ci(
    a: Prediction, b: Prediction, metric: str = "pearson_r",
    n_boot: int = N_BOOT, seed: int = SEED,
) -> tuple[float, float, float]:
    """Paired subject-level bootstrap CI for metric(a) - metric(b).

    `a` and `b` must be evaluated on the same held-out meals (they are, by the
    paired calibration scheme). Resamples the shared subjects.
    """
    assert np.array_equal(a.subject_id, b.subject_id)
    rng = np.random.default_rng(seed)
    subjects = np.unique(a.subject_id)
    by_subj = {s: np.where(a.subject_id == s)[0] for s in subjects}

    def delta(idx):
        return _metrics(a.y_true[idx], a.y_pred[idx])[metric] - \
               _metrics(b.y_true[idx], b.y_pred[idx])[metric]

    point = delta(np.arange(len(a.y_true)))
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subj[s] for s in chosen])
        d = delta(idx)
        if not np.isnan(d):
            boots.append(d)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)
