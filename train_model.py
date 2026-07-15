"""
train_model.py

Builds the "spatial fused" chronic-disease risk model end to end, replacing
every fabricated number the earlier dashboard review flagged with something
actually computed:

  - Individual BRFSS records are joined to real regional PLACES data via a
    real state key (src.data_hygiene.attach_state_abbr), not simulated per
    demographic selection.
  - Evaluation metrics (precision/recall curve, macro-F1 per target) are
    computed on a genuine held-out split, not synthesized.
  - Feature importance uses SHAP if it's installed; otherwise falls back to
    sklearn permutation importance -- and the output file records *which*
    method was used, so a fallback is never mistaken for real SHAP values.
  - The equity/fairness audit computes real FPR/FNR by neighborhood
    disadvantage quartile on the held-out set. No claim about fairness is
    written anywhere; the dashboard just displays these numbers and lets
    the reader judge.
  - LightGBM is used if installed (matching the "trained LightGBM model"
    comment in the original dashboard code); otherwise falls back to
    sklearn's HistGradientBoostingClassifier, and the backend actually used
    is recorded in eval_metrics.json so provenance is never ambiguous.

Outputs (all consumed directly by app.py):
    models/spatial_fused_production_model.joblib
    models/eval_metrics.json
    models/shap_importance.csv
    data/equity_audit.csv

Usage:
    python train_model.py --brfss LLCP2024.XPT --spatial data/aggregated_spatial_sdoh.csv
"""
import os
import sys
import json
import logging
import argparse

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

from config.logging_config import setup_logger
from src.data_ingestion import load_brfss_transport
from src.data_hygiene import clean_brfss_demographics_and_targets

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "age_code", "income_code", "educa_code", "race_code", "sex_code",
    "spatial_lack_insurance", "spatial_poor_mental_health",
    # Behavioral/clinical risk factors (see src/data_hygiene.py for cleaning
    # details). general_health_code (GENHLTH) is deliberately excluded: it's
    # a strong predictor partly because it's downstream of existing disease
    # status (someone already sick tends to rate their health worse), which
    # leaks the outcome back in as a "feature" rather than measuring an
    # independent risk factor. It's still cleaned in src/data_hygiene.py in
    # case a future version of this model wants to revisit it.
    "bmi", "smoker_code", "physical_activity_code",
    "heavy_drinking_code",
]
TARGET_COLUMNS = [
    "target_diabetes", "target_stroke", "target_cancer",
    "target_cardio", "target_chronic_respiratory",
]
TARGET_DISPLAY_NAMES = {
    "target_diabetes": "Diabetes Risk",
    "target_stroke": "Stroke Risk",
    "target_cancer": "Cancer Prevalence",
    "target_cardio": "Cardiovascular Event",
    "target_chronic_respiratory": "Chronic Respiratory Disease",
}
# Maps cleaned BRFSS covariate names to the feature names app.py builds its
# request from. Keeping this mapping in one place means the model's
# training-time schema and the dashboard's inference-time schema can't
# silently drift apart without an explicit change to this dict.
# NOTE: MSCODE (metro status) is a real per-respondent BRFSS field, but is
# deliberately NOT used as a model feature here -- in the real 2024 extract
# it was missing/suppressed for ~75% of respondents (see train_model.py's
# earlier attrition diagnostics), which makes whatever signal remains too
# unreliable to trust rather than a gap worth imputing around. It's still
# cleaned for validity in src/data_hygiene.py in case a future version of
# this model wants to revisit it (e.g. if a less-suppressed BRFSS release
# becomes available), but load_and_join below does not pull it in.
RAW_TO_FEATURE_RENAME = {
    "_AGEG5YR": "age_code",
    "INCOME3": "income_code",
    "EDUCA": "educa_code",
    "_RACEGR3": "race_code",
    "SEXVAR": "sex_code",
    "BMI": "bmi",
    "_SMOKER3": "smoker_code",
    "_TOTINDA": "physical_activity_code",
    "_RFDRHV9": "heavy_drinking_code",
}


def get_model_backend():
    try:
        from lightgbm import LGBMClassifier
        logger.info("Using LightGBM as the model backend.")
        return (lambda: LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1)), "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        logger.warning(
            "lightgbm is not installed in this environment. Falling back to "
            "sklearn.ensemble.HistGradientBoostingClassifier. Install lightgbm "
            "and re-run for parity with the originally intended model family "
            "(see the 'trained LightGBM model' comment in the original app.py)."
        )
        return (lambda: HistGradientBoostingClassifier(max_depth=6, random_state=42)), "HistGradientBoostingClassifier"


MIN_GROUP_N_FOR_MEDIAN = 30  # below this, a group's median is too noisy to trust; fall back to a coarser group


def impute_income_group_median(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing income_code with the median income among respondents
    who share the same (age_code, educa_code), falling back to a
    educa_code-only median, then the overall median, when a specific
    age x education cell has fewer than MIN_GROUP_N_FOR_MEDIAN real
    (non-missing-income) observations to compute a stable median from.

    income_code is an ordinal 1-8 scale (not continuous), so all imputed
    values are rounded to the nearest integer and clipped to [1, 8] --
    a fractional or out-of-range "income category" isn't meaningful.

    Every imputation tier's usage is logged so it's clear how much of the
    final feature is real respondent-reported data vs. a group median.
    """
    out = df.copy()
    missing_mask = out["income_code"].isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return out

    observed = out[~missing_mask]

    age_educ_median = observed.groupby(["age_code", "educa_code"])["income_code"].agg(["median", "count"])
    age_educ_lookup = age_educ_median[age_educ_median["count"] >= MIN_GROUP_N_FOR_MEDIAN]["median"]

    educ_median = observed.groupby("educa_code")["income_code"].agg(["median", "count"])
    educ_lookup = educ_median[educ_median["count"] >= MIN_GROUP_N_FOR_MEDIAN]["median"]

    overall_median = observed["income_code"].median()

    def resolve(row):
        key = (row["age_code"], row["educa_code"])
        if key in age_educ_lookup.index:
            return age_educ_lookup.loc[key], "age_x_education_group"
        if row["educa_code"] in educ_lookup.index:
            return educ_lookup.loc[row["educa_code"]], "education_group"
        return overall_median, "overall_median"

    tiers_used = {"age_x_education_group": 0, "education_group": 0, "overall_median": 0}
    imputed_values = []
    for _, row in out.loc[missing_mask].iterrows():
        value, tier = resolve(row)
        imputed_values.append(value)
        tiers_used[tier] += 1

    imputed_values = np.clip(np.round(imputed_values), 1, 8)
    out.loc[missing_mask, "income_code"] = imputed_values
    out["income_code_imputed"] = missing_mask  # audit column; not a model feature

    logger.info(
        "Imputed income_code for %d rows (%.1f%%) using group-adjusted medians: "
        "%d via age x education group, %d via education-only group (age x education "
        "cell too small), %d via overall median (education group too small).",
        n_missing, 100 * n_missing / len(out),
        tiers_used["age_x_education_group"], tiers_used["education_group"], tiers_used["overall_median"],
    )
    return out


def load_and_join(brfss_path: str, spatial_path: str) -> pd.DataFrame:
    if not os.path.exists(brfss_path):
        raise FileNotFoundError(f"BRFSS transport file not found: {brfss_path}")
    if not os.path.exists(spatial_path):
        raise FileNotFoundError(f"{spatial_path} not found. Run generate_spatial_features.py first.")

    raw = load_brfss_transport(brfss_path)
    cleaned = clean_brfss_demographics_and_targets(raw)

    if "stateabbr" not in cleaned.columns:
        raise KeyError(
            "clean_brfss_demographics_and_targets did not produce a 'stateabbr' "
            "column -- cannot join to spatial data. Ensure _STATE was present "
            "in the raw BRFSS extract requested from run_profile.py's target list."
        )

    cleaned = cleaned.rename(columns=RAW_TO_FEATURE_RENAME)

    spatial = pd.read_csv(spatial_path, dtype={"stateabbr": str})
    missing_spatial_cols = [c for c in ["stateabbr", "spatial_lack_insurance", "spatial_poor_mental_health"] if c not in spatial.columns]
    if missing_spatial_cols:
        raise KeyError(f"{spatial_path} is missing expected columns: {missing_spatial_cols}")

    state_avg = (
        spatial.groupby("stateabbr")[["spatial_lack_insurance", "spatial_poor_mental_health"]]
        .mean()
        .reset_index()
    )
    merged = cleaned.merge(state_avg, on="stateabbr", how="inner")
    logger.info("Joined BRFSS to spatial SDoH on stateabbr (state-level averages).")

    n_before, n_after = len(cleaned), len(merged)
    logger.info(
        "Joined BRFSS to spatial SDoH: %d -> %d rows (%d rows dropped, likely "
        "unmatched/territory states with no PLACES coverage in the current pull)",
        n_before, n_after, n_before - n_after,
    )

    required = FEATURE_COLUMNS + TARGET_COLUMNS
    missing_cols = [c for c in required if c not in merged.columns]
    if missing_cols:
        raise KeyError(f"Joined dataset is missing expected columns: {missing_cols}")

    # income_code (BRFSS INCOME3) has real item non-response (~18% missing
    # in the real 2024 extract) -- refusing to state income is itself a
    # meaningful minority behavior, not the near-total suppression MSCODE
    # had, so it's worth recovering via imputation rather than excluding
    # entirely OR lumping into a blunt "unknown" bucket. Income correlates
    # strongly with both age and education, so missing values are imputed
    # with the median income_code within each (age_code, educa_code) group
    # among respondents who did report income -- falling back to a coarser
    # educa_code-only group median, then the overall median, wherever a
    # specific age x education cell has too few real observations for a
    # stable estimate.
    merged = impute_income_group_median(merged)

    # Diagnose which specific column(s) are driving attrition BEFORE dropping
    # anything, so a large final loss is traceable to a specific field
    # instead of showing up only as a mystery row count. Report the top
    # offenders regardless of severity, and warn loudly if any single column
    # accounts for a large share of the loss on its own.
    null_counts = merged[required].isna().sum().sort_values(ascending=False)
    null_pct = (null_counts / len(merged) * 100).round(1)
    logger.info(
        "Per-column missingness before final dropna (n=%d):\n%s",
        len(merged),
        "\n".join(f"  {col}: {cnt} missing ({pct}%)" for col, cnt, pct in zip(null_counts.index, null_counts.values, null_pct.values) if cnt > 0),
    )
    worst_col, worst_pct = null_pct.index[0], null_pct.iloc[0]
    if worst_pct > 30:
        logger.warning(
            "'%s' alone is missing in %.1f%% of joined rows -- this single column "
            "is likely the dominant driver of any large row loss below, not an "
            "even spread across features. Investigate that field specifically "
            "before assuming general data quality issues.",
            worst_col, worst_pct,
        )

    model_df = merged.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS)
    n_after_dropna = len(model_df)
    pct_lost_final = 100 * (len(merged) - n_after_dropna) / len(merged) if len(merged) else 0
    logger.info(
        "Final modeling frame after dropping remaining NaNs in features/targets: "
        "%d -> %d rows (%.1f%% additional loss)",
        len(merged), n_after_dropna, pct_lost_final,
    )
    return model_df


def compute_disadvantage_quartile(df: pd.DataFrame) -> pd.Series:
    """Neighborhood disadvantage quartile used ONLY as a fairness-audit
    stratifier -- never as a model feature, and never conflated with
    urbanicity/MSCODE. Keeping these uses separate avoids the circularity
    bug the original MSCODE derivation had (defining 'rural' as 'highest
    uninsurance quartile' guarantees a fake correlation between the two).
    """
    return pd.qcut(
        df["spatial_lack_insurance"].rank(method="first"), 4,
        labels=["Q1 (Least Disadvantaged)", "Q2", "Q3", "Q4 (Most Disadvantaged)"],
    )


def main(brfss_path: str, spatial_path: str, test_size: float = 0.2):
    setup_logger()
    model_factory, backend_name = get_model_backend()

    model_df = load_and_join(brfss_path, spatial_path)
    disadvantage_quartile = compute_disadvantage_quartile(model_df)

    X = model_df[FEATURE_COLUMNS]
    Y = model_df[TARGET_COLUMNS]

    X_train, X_test, Y_train, Y_test, q_train, q_test = train_test_split(
        X, Y, disadvantage_quartile, test_size=test_size, random_state=42
    )
    logger.info("Training on %d rows, evaluating on %d held-out rows.", len(X_train), len(X_test))

    model = MultiOutputClassifier(model_factory())
    model.fit(X_train, Y_train)

    y_pred_all = model.predict(X_test)
    y_proba_all = model.predict_proba(X_test)  # list of (n_samples, 2) arrays, one per target

    f1_scores, pr_curves, pr_thresholds, equity_rows = {}, {}, {}, []
    roc_auc_scores, pr_auc_scores, prevalence_by_target = {}, {}, {}

    for i, target in enumerate(TARGET_COLUMNS):
        y_true = Y_test[target].values
        y_pred = y_pred_all[:, i]
        y_proba = y_proba_all[i][:, 1]

        f1_scores[target] = float(f1_score(y_true, y_pred, average="macro"))

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        baseline_rate = float(y_true.mean())
        pr_curves[target] = {
            "recall": recall.tolist(),
            "precision": precision.tolist(),
            "baseline": [baseline_rate] * len(recall),
        }
        pr_thresholds[target] = thresholds

        # F1 at the default 0.5 decision threshold is a poor summary for
        # targets this imbalanced (prevalence ranges ~4.6%-20.7% here) --
        # MultiOutputClassifier's .predict() rarely crosses 0.5 for the rare
        # classes, so F1 ends up hugging the "always predict negative"
        # baseline even when the model has real, threshold-independent
        # discriminative power. ROC-AUC and PR-AUC (average precision)
        # measure ranking quality across all thresholds, so they surface
        # that signal instead of hiding it behind one bad operating point.
        roc_auc_scores[target] = float(roc_auc_score(y_true, y_proba))
        pr_auc_scores[target] = float(average_precision_score(y_true, y_proba))
        prevalence_by_target[target] = baseline_rate

        for q in sorted(q_test.dropna().unique().tolist()):
            mask = (q_test == q).values
            if mask.sum() < 10:
                logger.warning("Skipping equity row for %s / %s: fewer than 10 held-out samples", target, q)
                continue
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) else float("nan")
            fnr = fn / (fn + tp) if (fn + tp) else float("nan")
            equity_rows.append({
                "target": TARGET_DISPLAY_NAMES[target],
                "group": q,
                "n": int(mask.sum()),
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4),
            })

    primary_target = "target_stroke" if "target_stroke" in pr_curves else TARGET_COLUMNS[0]

    # Operating point: the cutoff probability at which the model's predictions
    # would actually be used to call something "high risk." Chosen as the
    # point on the primary target's precision-recall curve with the highest
    # F1 score -- the best balance found anywhere on the curve between
    # catching real cases (recall) and avoiding false alarms (precision).
    # This is a real, computed choice from the held-out data, not a
    # subjective pick -- a different priority (e.g. "catch more cases even
    # at the cost of more false alarms") would justify a different cutoff.
    _p = np.array(pr_curves[primary_target]["precision"][:-1])
    _r = np.array(pr_curves[primary_target]["recall"][:-1])
    _t = pr_thresholds[primary_target]
    with np.errstate(divide="ignore", invalid="ignore"):
        _f1 = np.where((_p + _r) > 0, 2 * _p * _r / (_p + _r), 0)
    _best_idx = int(np.argmax(_f1))
    operating_point = {
        "target": TARGET_DISPLAY_NAMES[primary_target],
        "threshold": round(float(_t[_best_idx]), 4),
        "precision": round(float(_p[_best_idx]), 4),
        "recall": round(float(_r[_best_idx]), 4),
        "f1": round(float(_f1[_best_idx]), 4),
        "rationale": (
            "Chosen as the cutoff with the highest F1 score on the held-out precision-recall curve -- "
            "the best balance point found between catching real cases and avoiding false alarms. A "
            "program that weighs missed cases more heavily than false alarms (or vice versa) would "
            "reasonably pick a different cutoff."
        ),
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    joblib.dump(model, "models/spatial_fused_production_model.joblib")
    logger.info("Saved trained model to models/spatial_fused_production_model.joblib")

    # --- Feature importance: real SHAP if available, else permutation
    # importance, with the method explicitly recorded per-row. ---
    method = None
    importance_rows = []
    try:
        import shap
        method = "shap"
        for target, est in zip(TARGET_COLUMNS, model.estimators_):
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_test)
            vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            mean_abs = np.abs(vals).mean(axis=0)
            for feat, imp in zip(FEATURE_COLUMNS, mean_abs):
                importance_rows.append({"feature": feat, "target": target, "importance": float(imp), "method": method})
    except ImportError:
        method = "permutation"
        logger.warning(
            "shap is not installed; falling back to sklearn permutation importance. "
            "This is a rougher, differently-defined measure of feature influence than "
            "SHAP -- recorded as method='permutation' so it's never mistaken for SHAP."
        )
        for target, est in zip(TARGET_COLUMNS, model.estimators_):
            result = permutation_importance(est, X_test, Y_test[target], n_repeats=10, random_state=42)
            for feat, imp in zip(FEATURE_COLUMNS, result.importances_mean):
                importance_rows.append({"feature": feat, "target": target, "importance": float(imp), "method": method})

    shap_df = (
        pd.DataFrame(importance_rows)
        .groupby(["feature", "method"], as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    shap_df.to_csv("models/shap_importance.csv", index=False)
    logger.info("Saved feature importances (method=%s) to models/shap_importance.csv", method)

    equity_df = pd.DataFrame(equity_rows)
    equity_df.to_csv("data/equity_audit.csv", index=False)
    logger.info("Saved real subgroup fairness audit to data/equity_audit.csv")

    eval_metrics = {
        "model_backend": backend_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "macro_f1_by_target": {TARGET_DISPLAY_NAMES[k]: round(v, 4) for k, v in f1_scores.items()},
        "roc_auc_by_target": {TARGET_DISPLAY_NAMES[k]: round(v, 4) for k, v in roc_auc_scores.items()},
        "pr_auc_by_target": {TARGET_DISPLAY_NAMES[k]: round(v, 4) for k, v in pr_auc_scores.items()},
        "prevalence_by_target": {TARGET_DISPLAY_NAMES[k]: round(v, 4) for k, v in prevalence_by_target.items()},
        "precision_recall": pr_curves[primary_target],
        "precision_recall_target": TARGET_DISPLAY_NAMES[primary_target],
        "operating_point": operating_point,
        # NOTE: no "impact_summary" key is written here. Translating raw
        # precision/recall into operational claims like "6.2x faster outreach"
        # requires assumptions about a real deployment workflow (staffing,
        # current screening cost, etc.) that this script has no basis for.
        # app.py's Tab 3 will show "no data" rather than invent one. Add an
        # impact_summary block here yourself, ONLY once you have numbers
        # from an actual pilot or documented operational assumptions.
    }
    with open("models/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info("Saved real evaluation metrics to models/eval_metrics.json")

    print("Training complete.")
    print(f"Backend: {backend_name}")
    print(f"Feature importance method: {method}")
    print(f"Macro F1 by target: {eval_metrics['macro_f1_by_target']}")
    print(f"ROC-AUC by target: {eval_metrics['roc_auc_by_target']}")
    print(f"PR-AUC by target (vs. baseline prevalence): "
          + ", ".join(f"{k}: {v:.4f} (baseline {eval_metrics['prevalence_by_target'][k]:.4f})"
                       for k, v in eval_metrics['pr_auc_by_target'].items()))
    print("\nMean SHAP importance across all targets (overall ranking):")
    print(shap_df.to_string(index=False))
    print("\nSHAP importance by target:")
    per_target_shap = (
        pd.DataFrame(importance_rows)
        .pivot_table(index="feature", columns="target", values="importance")
        .reindex(FEATURE_COLUMNS)
    )
    print(per_target_shap.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brfss", default="LLCP2024.XPT")
    parser.add_argument("--spatial", default="data/aggregated_spatial_sdoh.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    main(args.brfss, args.spatial, args.test_size)
