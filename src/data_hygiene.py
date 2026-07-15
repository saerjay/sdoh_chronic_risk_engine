import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# BRFSS uses these codes consistently for "Don't know/Not sure" and "Refused"
# across most categorical items. EDUCA typically only uses 9.
_DK_REFUSED = [7.0, 9.0]

# Official ANSI/FIPS numeric state codes, as used in BRFSS's _STATE variable.
# This mapping is a stable federal standard, not derived/estimated data, so
# it's safe to hardcode -- unlike the ZCTA-to-state guess in
# generate_spatial_features.py, there's no ambiguity here.
_STATE_FIPS_TO_ABBR = {
    1.0: "AL", 2.0: "AK", 4.0: "AZ", 5.0: "AR", 6.0: "CA", 8.0: "CO", 9.0: "CT",
    10.0: "DE", 11.0: "DC", 12.0: "FL", 13.0: "GA", 15.0: "HI", 16.0: "ID",
    17.0: "IL", 18.0: "IN", 19.0: "IA", 20.0: "KS", 21.0: "KY", 22.0: "LA",
    23.0: "ME", 24.0: "MD", 25.0: "MA", 26.0: "MI", 27.0: "MN", 28.0: "MS",
    29.0: "MO", 30.0: "MT", 31.0: "NE", 32.0: "NV", 33.0: "NH", 34.0: "NJ",
    35.0: "NM", 36.0: "NY", 37.0: "NC", 38.0: "ND", 39.0: "OH", 40.0: "OK",
    41.0: "OR", 42.0: "PA", 44.0: "RI", 45.0: "SC", 46.0: "SD", 47.0: "TN",
    48.0: "TX", 49.0: "UT", 50.0: "VT", 51.0: "VA", 53.0: "WA", 54.0: "WV",
    55.0: "WI", 56.0: "WY", 66.0: "GU", 72.0: "PR", 78.0: "VI",
}


def attach_state_abbr(df: pd.DataFrame) -> pd.DataFrame:
    """Maps BRFSS's numeric _STATE (FIPS) code to a 'stateabbr' column, so
    individual BRFSS records can be joined to the state-level PLACES spatial
    data produced by generate_spatial_features.py. Without this, there is no
    real key linking the two data sources -- previously nothing in the
    pipeline did this at all, which is why the dashboard's map couldn't be
    honestly stratified by demographic group.
    """
    if "_STATE" not in df.columns:
        logger.warning("_STATE column not present in BRFSS extract; cannot attach stateabbr")
        return df
    out = df.copy()
    out["stateabbr"] = out["_STATE"].map(_STATE_FIPS_TO_ABBR)
    unmapped = out["stateabbr"].isna().sum()
    if unmapped:
        logger.warning(
            "%d rows had a _STATE code not in the FIPS mapping and were left "
            "unmapped (they will be dropped at the join step in train_model.py)",
            unmapped,
        )
    return out


def _mark_invalid_as_nan(series: pd.Series, invalid_codes: list) -> pd.Series:
    """Returns a copy of series with invalid/refused codes replaced by NaN,
    WITHOUT dropping any rows. Row-dropping is deferred to a single,
    explicit, logged step so we always know how much data we're losing
    and why.
    """
    return series.where(~series.isin(invalid_codes), np.nan)


def clean_brfss_demographics_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw BRFSS responses into binary disease targets and categorical
    demographic covariates.

    Design principle: compute every column first (setting invalid/refused
    responses to NaN), and only drop rows at the very end, in one explicit,
    logged step. This avoids two problems in the previous implementation:

    1. Composite (OR-based) targets no longer silently lose rows where
       one contributing question was skipped but the other still tells us
       the true label (e.g., COPD=Yes, Asthma=skipped -> target is still
       known to be 1, not dropped).
    2. We can report exactly how many respondents were lost to missingness
       at each stage, so downstream users can judge whether non-response
       bias is a concern.
    """
    logger.info("Starting BRFSS cleaning on %d raw rows", len(df))
    cleaned_df = df.copy()
    n_start = len(cleaned_df)
    cleaned_df = attach_state_abbr(cleaned_df)

    # --- DIABETE4 is NOT binary like the other single-question targets --
    # it has four valid answer codes (1=Yes, 2=Yes-during-pregnancy-only,
    # 3=No, 4=No/pre-diabetes), not just 1/2. An earlier version of this
    # function mapped it with the same {1.0: 1, 2.0: 0} used for the
    # genuinely-binary targets below, which silently turned every "No"
    # (code 3 -- the most common answer, ~85-90% of respondents) into NaN,
    # then dropped at listwise deletion. That one bug accounted for the
    # overwhelming majority of a reported 85% row loss. Gestational-only
    # diabetes (code 2) is coded as 0 here to match CDC's own "diagnosed
    # diabetes" prevalence definition (DIABETE4==1), which excludes it as
    # not indicating chronic/ongoing diabetes.
    if "DIABETE4" in cleaned_df.columns:
        valid_diabete_codes = [1.0, 2.0, 3.0, 4.0]
        valid = cleaned_df["DIABETE4"].where(cleaned_df["DIABETE4"].isin(valid_diabete_codes), np.nan)
        cleaned_df["target_diabetes"] = valid.map({1.0: 1, 2.0: 0, 3.0: 0, 4.0: 0})
    else:
        logger.warning("DIABETE4 not found in source data; target_diabetes will not be created")

    # --- Single-question targets that ARE genuinely binary (1=Yes, 2=No) ---
    single_question_targets = {
        "CVDSTRK3": "target_stroke",
        "CHCOCNC1": "target_cancer",
    }
    for raw_col, target_col in single_question_targets.items():
        if raw_col in cleaned_df.columns:
            valid = _mark_invalid_as_nan(cleaned_df[raw_col], _DK_REFUSED)
            cleaned_df[target_col] = valid.map({1.0: 1, 2.0: 0})
        else:
            logger.warning("%s not found in source data; %s will not be created", raw_col, target_col)

    # _MICHD is CDC's own pre-computed composite (1 = had MI/CHD, 2 = did not);
    # it does not use the standard 7/9 don't-know/refused scheme, and missing
    # is represented as NaN directly.
    if "_MICHD" in cleaned_df.columns:
        cleaned_df["target_cardio"] = cleaned_df["_MICHD"].map({1.0: 1, 2.0: 0})
    else:
        logger.warning("_MICHD not found; target_cardio will not be created")

    # --- Composite OR target: asthma OR COPD ---
    # A row should only be dropped if BOTH source questions are unusable.
    # If either question affirmatively indicates disease, that's enough to
    # label the row positive even if the other question was skipped.
    if "ASTHMA3" in cleaned_df.columns and "CHCCOPD3" in cleaned_df.columns:
        asthma = cleaned_df["ASTHMA3"]
        copd = cleaned_df["CHCCOPD3"]
        has_asthma = asthma == 1.0
        has_copd = copd == 1.0
        asthma_valid = ~asthma.isin(_DK_REFUSED) & asthma.notna()
        copd_valid = ~copd.isin(_DK_REFUSED) & copd.notna()

        cleaned_df["target_chronic_respiratory"] = np.select(
            condlist=[has_asthma | has_copd, asthma_valid & copd_valid],
            choicelist=[1, 0],
            default=np.nan,
        )
    else:
        logger.warning(
            "ASTHMA3 and/or CHCCOPD3 not found; target_chronic_respiratory will not be created"
        )

    # --- Demographic covariates: clean don't-know/refused codes to NaN,
    # but do NOT drop rows here -- missingness on a covariate the model
    # doesn't strictly need shouldn't cost us a labeled training example. ---
    if "INCOME3" in cleaned_df.columns:
        cleaned_df["INCOME3"] = _mark_invalid_as_nan(cleaned_df["INCOME3"], [77.0, 99.0])
    if "_AGEG5YR" in cleaned_df.columns:
        # 14 = "Don't know/Refused/Missing" in the 13-category _AGEG5YR scheme
        cleaned_df["_AGEG5YR"] = _mark_invalid_as_nan(cleaned_df["_AGEG5YR"], [14.0])
    if "_RACEGR3" in cleaned_df.columns:
        cleaned_df["_RACEGR3"] = _mark_invalid_as_nan(cleaned_df["_RACEGR3"], [9.0])
    if "EDUCA" in cleaned_df.columns:
        cleaned_df["EDUCA"] = _mark_invalid_as_nan(cleaned_df["EDUCA"], [9.0])
    if "SEXVAR" in cleaned_df.columns:
        # SEXVAR is binary in BRFSS with no legitimate missing code; leave as-is
        # but flag anything outside {1.0, 2.0} as a data quality problem rather
        # than silently accepting it.
        bad_sex = ~cleaned_df["SEXVAR"].isin([1.0, 2.0])
        if bad_sex.any():
            logger.warning("%d rows have unexpected SEXVAR values; setting to NaN", bad_sex.sum())
            cleaned_df.loc[bad_sex, "SEXVAR"] = np.nan

    if "MSCODE" in cleaned_df.columns:
        # Official BRFSS MSCODE categories: 1=center city of an MSA,
        # 2=outside center city but same county, 3=suburban county of the
        # MSA, 4=MSA with no center city (rare), 5=not in an MSA. Anything
        # outside {1,2,3,4,5} is not a valid code for this field (it has no
        # standard don't-know/refused code since it's derived from the
        # sampling frame, not a survey question) and is nulled rather than
        # passed through.
        valid_mscode = [1.0, 2.0, 3.0, 4.0, 5.0]
        bad_mscode = ~cleaned_df["MSCODE"].isin(valid_mscode) & cleaned_df["MSCODE"].notna()
        if bad_mscode.any():
            logger.warning(
                "%d rows have an MSCODE value outside {1,2,3,4,5}; setting to NaN",
                bad_mscode.sum(),
            )
            cleaned_df.loc[bad_mscode, "MSCODE"] = np.nan

    # --- Behavioral/clinical risk factors -----------------------------
    if "_BMI5" in cleaned_df.columns:
        # _BMI5 is BMI * 100 (e.g. 2731 = BMI 27.31); already blank/NaN for
        # respondents missing height or weight, no separate DK/refused code.
        # Implausible values (data entry errors CDC's own QC missed) are
        # nulled rather than silently kept -- valid adult BMI is roughly
        # 12-100; outside that range is almost certainly a data error, not
        # a real body mass index.
        cleaned_df["BMI"] = cleaned_df["_BMI5"] / 100.0
        implausible_bmi = (cleaned_df["BMI"] < 12) | (cleaned_df["BMI"] > 100)
        if implausible_bmi.any():
            logger.warning("%d rows have an implausible BMI outside [12, 100]; setting to NaN", implausible_bmi.sum())
            cleaned_df.loc[implausible_bmi, "BMI"] = np.nan
    else:
        logger.warning("_BMI5 not found in source data; BMI will not be created")

    if "_SMOKER3" in cleaned_df.columns:
        # 1=current smoker daily, 2=current smoker some days, 3=former smoker,
        # 4=never smoked, 9=don't know/refused/missing.
        cleaned_df["_SMOKER3"] = _mark_invalid_as_nan(cleaned_df["_SMOKER3"], [9.0])

    if "_TOTINDA" in cleaned_df.columns:
        # 1=had leisure-time physical activity in past 30 days, 2=none, 9=DK/refused/missing.
        cleaned_df["_TOTINDA"] = _mark_invalid_as_nan(cleaned_df["_TOTINDA"], [9.0])

    if "GENHLTH" in cleaned_df.columns:
        # 1=excellent ... 5=poor, 7=don't know, 9=refused.
        # NOTE: self-rated general health is a strong predictor partly
        # *because* it's somewhat downstream of existing disease status --
        # someone already living with diabetes or heart disease is more
        # likely to rate their health as fair/poor. It's included here
        # because that's standard practice in this literature, but it is
        # not a "pure" independent risk factor the way BMI or smoking are.
        # Remove "GENHLTH": "general_health_code" from train_model.py's
        # RAW_TO_FEATURE_RENAME (and FEATURE_COLUMNS) if you'd rather
        # exclude it on those grounds.
        cleaned_df["GENHLTH"] = _mark_invalid_as_nan(cleaned_df["GENHLTH"], _DK_REFUSED)

    # NOTE: sleep hours (SLEPTIM1) is not present in the 2024 BRFSS release
    # this project uses -- confirmed via list_brfss_columns.py, with no
    # similarly-named alternative found. Not cleaned or used as a feature.

    if "_RFDRHV9" in cleaned_df.columns:
        # 1=not a heavy drinker, 2=heavy drinker, 9=don't know/refused/missing.
        # (Suffix version confirmed for this release via list_brfss_columns.py;
        # was _RFDRHV8 in earlier BRFSS years.)
        cleaned_df["_RFDRHV9"] = _mark_invalid_as_nan(cleaned_df["_RFDRHV9"], [9.0])

    cols_to_drop = ["DIABETE4", "CVDSTRK3", "ASTHMA3", "CHCCOPD3", "CHCOCNC1", "_MICHD", "_STATE", "_BMI5"]
    existing_drops = [c for c in cols_to_drop if c in cleaned_df.columns]
    cleaned_df = cleaned_df.drop(columns=existing_drops)

    # --- Single, explicit, logged listwise deletion on TARGET columns only.
    # (Demographic covariate NaNs are left for the model/imputer to handle,
    # since dropping on those too would compound attrition unnecessarily.) ---
    target_cols = [c for c in cleaned_df.columns if c.startswith("target_")]
    if target_cols:
        before = len(cleaned_df)

        # Per-target breakdown BEFORE the intersection is taken, so if total
        # loss is large it's traceable to a specific target rather than an
        # even spread across all five -- the same principle used for feature
        # attrition in train_model.py. Any single target missing far more
        # than the others points at a coding bug in that target's mapping,
        # not generic survey non-response.
        per_target_null = cleaned_df[target_cols].isna().sum().sort_values(ascending=False)
        per_target_pct = (per_target_null / before * 100).round(1)
        logger.info(
            "Per-target missingness BEFORE listwise deletion (n=%d):\n%s",
            before,
            "\n".join(f"  {col}: {cnt} missing ({pct}%)" for col, cnt, pct in
                      zip(per_target_null.index, per_target_null.values, per_target_pct.values)),
        )
        worst_target, worst_pct = per_target_pct.index[0], per_target_pct.iloc[0]
        others_max_pct = per_target_pct.iloc[1] if len(per_target_pct) > 1 else 0
        if worst_pct > 25 and worst_pct > others_max_pct * 2:
            logger.warning(
                "'%s' is missing in %.1f%% of rows, far more than any other "
                "target (next-highest: %.1f%%). This pattern -- one target "
                "wildly out of line with the rest -- usually means that "
                "target's specific answer-code mapping has a bug (e.g. an "
                "answer category not being recognized as valid), not that "
                "respondents are actually skipping that one question at a "
                "dramatically higher rate than the others. Check that "
                "target's raw source column and mapping logic specifically "
                "before assuming this is expected survey non-response.",
                worst_target, worst_pct, others_max_pct,
            )

        cleaned_df = cleaned_df.dropna(subset=target_cols)
        after = len(cleaned_df)
        pct_lost = 100 * (before - after) / before if before else 0
        logger.info(
            "Listwise deletion on %s: %d -> %d rows (%.1f%% lost to missing/invalid targets)",
            target_cols, before, after, pct_lost,
        )
        if pct_lost > 15:
            logger.warning(
                "More than 15%% of respondents were dropped for missing target data. "
                "Consider whether non-response bias affects who remains in the sample."
            )

    logger.info("BRFSS cleaning complete: %d -> %d final rows", n_start, len(cleaned_df))
    return cleaned_df


def clean_and_pivot_places_api(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw stacked PLACES records into wide-form social indicators.

    Uses 'locationid' as the ZCTA join key (confirmed for the PLACES ZCTA
    2024 release, dataset fu4u-a9bh) -- NOT 'locationname', which is a
    display label in some PLACES releases and not guaranteed to be a clean
    5-digit ZIP.
    """
    working_df = df.copy()
    working_df.columns = working_df.columns.str.lower()

    if "locationid" not in working_df.columns:
        raise KeyError(
            "Expected a 'locationid' column from the PLACES API response; "
            f"got columns: {list(working_df.columns)}"
        )

    working_df = working_df.rename(columns={"locationid": "ZCTA"})
    working_df["ZCTA"] = working_df["ZCTA"].astype(str).str.strip()

    # Sanity check: ZCTAs should be 5-digit numeric strings. If most values
    # aren't, this is very likely the wrong geography level or column.
    looks_like_zip = working_df["ZCTA"].str.fullmatch(r"\d{5}").mean()
    if looks_like_zip < 0.9:
        logger.warning(
            "Only %.0f%% of 'locationid' values look like 5-digit ZCTAs. "
            "Double-check that this Socrata dataset is the PLACES ZCTA-level "
            "release (fu4u-a9bh) and that 'locationid' is the right field.",
            looks_like_zip * 100,
        )

    working_df["ZCTA"] = working_df["ZCTA"].str.zfill(5)
    working_df["data_value"] = pd.to_numeric(working_df["data_value"], errors="coerce")

    # Preserve stateabbr through the pivot if the API response included it
    # (see generate_spatial_features.resolve_state, which prefers this over
    # any ZIP-prefix guessing when it's available).
    keep_cols = ["ZCTA"] + (["stateabbr"] if "stateabbr" in working_df.columns else [])
    static_cols = working_df[keep_cols].drop_duplicates(subset=["ZCTA"])

    pivoted_df = working_df.pivot_table(
        index="ZCTA", columns="measureid", values="data_value", aggfunc="first"
    ).reset_index()
    pivoted_df.columns.name = None

    if "stateabbr" in static_cols.columns:
        pivoted_df = pivoted_df.merge(static_cols, on="ZCTA", how="left")

    acc_col = "ACCESS2" if "ACCESS2" in pivoted_df.columns else "access2"
    mhlth_col = "MHLTH" if "MHLTH" in pivoted_df.columns else "mhlth"

    missing = [c for c, present in [("ACCESS2/access2", acc_col in pivoted_df.columns),
                                     ("MHLTH/mhlth", mhlth_col in pivoted_df.columns)] if not present]
    if missing:
        raise KeyError(f"Expected PLACES measure columns not found after pivot: {missing}")

    pivoted_df = pivoted_df.rename(
        columns={acc_col: "spatial_lack_insurance", mhlth_col: "spatial_poor_mental_health"}
    )
    return pivoted_df
