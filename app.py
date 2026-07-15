# app.py
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import plotly.express as px
from streamlit_folium import st_folium
from src.data_ingestion import merge_spatial_vectors, verify_zip_code

st.set_page_config(
    page_title="Social Determinants of Health As Predictors of Individual Chronic Disease Risk",
    layout="wide"
)

# ---------------------------------------------------------------------------
# THEME
# Colors are defined as CSS custom properties with a prefers-color-scheme
# override, so the app stays readable on both light and dark computers
# rather than forcing light-mode colors regardless of the viewer's system
# setting. GROUP_* colors are used as small "eyebrow" labels throughout
# Tab 1 to visually tie each input section to the output it feeds (e.g.
# the location picker and the regional stats it produces share a color),
# using color paired with a text label rather than color alone.
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
        :root {
            --page-bg: #fafafa;
            --text-h1: #0f172a;
            --text-h2: #1e293b;
            --text-body: #334155;
            --accent: #0d9488;
            --footer-border: #e0e0e0;
            --footer-title: #333333;
            --footer-item: #666666;
            --demo-bg: #fff7ed;
            --demo-border: #fdba74;
            --demo-text: #7c2d12;
            --group-demo-bg: #eef2ff;
            --group-demo-text: #4338ca;
            --group-location-bg: #ecfeff;
            --group-location-text: #0e7490;
            --group-health-bg: #fffbeb;
            --group-health-text: #b45309;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --page-bg: #0e1117;
                --text-h1: #f1f5f9;
                --text-h2: #e2e8f0;
                --text-body: #cbd5e1;
                --accent: #2dd4bf;
                --footer-border: #30363d;
                --footer-title: #cbd5e1;
                --footer-item: #94a3b8;
                --demo-bg: #451a03;
                --demo-border: #b45309;
                --demo-text: #fed7aa;
                --group-demo-bg: rgba(129, 140, 248, 0.16);
                --group-demo-text: #a5b4fc;
                --group-location-bg: rgba(34, 211, 238, 0.16);
                --group-location-text: #67e8f9;
                --group-health-bg: rgba(251, 191, 36, 0.16);
                --group-health-text: #fcd34d;
            }
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 5% !important;
            padding-right: 5% !important;
            background-color: var(--page-bg);
        }
        @media (max-width: 768px) {
            .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
        }
        h1 {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
            font-weight: 700 !important;
            color: var(--text-h1) !important;
            letter-spacing: -0.05rem;
            margin-bottom: 2rem !important;
        }
        h2, h3 {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
            font-weight: 600 !important;
            color: var(--text-h2) !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        p, li { color: var(--text-body) !important; line-height: 1.7 !important; font-size: 1.05rem !important; }
        .section-arrow { text-align: center; font-size: 1.75rem; color: var(--accent); margin: 2rem 0; font-weight: bold; }
        .stSelectbox, .stRadio { margin-bottom: 1.5rem !important; }
        .source-footer { margin-top: 5rem !important; padding-top: 2rem !important; border-top: 1px solid var(--footer-border) !important; }
        .source-title { font-size: 0.95rem !important; font-weight: bold !important; color: var(--footer-title) !important; margin-bottom: 0.5rem !important; }
        .source-item { font-size: 0.85rem !important; color: var(--footer-item) !important; line-height: 1.6 !important; margin-bottom: 0.4rem !important; }
        .sr-only {
            position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
            overflow: hidden; clip: rect(0, 0, 0, 0); border: 0;
        }
        .demo-banner {
            background-color: var(--demo-bg); border: 1px solid var(--demo-border); border-radius: 8px;
            padding: 0.9rem 1.25rem; margin-bottom: 1.5rem; color: var(--demo-text) !important;
            font-size: 0.95rem !important;
        }
        .group-label {
            display: inline-block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.06em; padding: 0.2rem 0.65rem; border-radius: 999px; margin-bottom: 0.85rem;
        }
        .group-demographics { background-color: var(--group-demo-bg); color: var(--group-demo-text); }
        .group-location { background-color: var(--group-location-bg); color: var(--group-location-text); }
        .group-health { background-color: var(--group-health-bg); color: var(--group-health-text); }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA / MODEL LOADING
# Everything in this section is designed to fail loudly (via st.warning) if
# real artifacts are missing, rather than silently substituting fabricated
# numbers -- that substitution was the core problem with the previous version.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_production_pipeline():
    model_path = "models/spatial_fused_production_model.joblib"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Found {model_path} but failed to load it: {e}")
            return None
    return None


@st.cache_data
def load_regional_sdoh_matrix():
    path = "data/aggregated_spatial_sdoh.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"stateabbr": str})
    return df


@st.cache_data
def load_zcta_level_sdoh():
    """Pre-aggregation, ZCTA-level SDoH table, for a precise 'enter your
    ZIP' lookup via src.data_ingestion.merge_spatial_vectors. Separate from
    load_regional_sdoh_matrix(), which is the coarser state-level average
    used for the state dropdown and Tab 2's map.
    """
    path = "data/zcta_level_sdoh.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"ZCTA": str, "stateabbr": str})


@st.cache_data
def load_demographic_risk_predictions():
    """Per-respondent real model predictions (see
    generate_demographic_risk_predictions.py), used by Tab 2's
    demographic-filtered risk map. Different from load_regional_sdoh_matrix():
    that one is a raw CDC statistic with no demographic breakdown; this one
    is the trained model's own predictions for real BRFSS respondents, which
    do carry demographics, so it can answer "what does the model predict for
    people like X, state by state?"
    """
    path = "data/model_predicted_risk_by_respondent.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"stateabbr": str})


@st.cache_data
def load_eval_artifacts():
    """Loads real model-evaluation artifacts if they exist:
      - models/eval_metrics.json: {"precision_recall": [...], "operating_point": {...}, "impact_summary": {...}}
      - models/shap_importance.csv: columns feature, importance
      - data/equity_audit.csv: columns group, n, fpr, fnr
    Returns a dict of whichever pieces are actually present; missing pieces
    are simply absent from the dict (checked by callers) rather than filled
    with placeholder numbers.
    """
    artifacts = {}
    if os.path.exists("models/eval_metrics.json"):
        with open("models/eval_metrics.json") as f:
            artifacts["eval_metrics"] = json.load(f)
    if os.path.exists("models/shap_importance.csv"):
        artifacts["shap"] = pd.read_csv("models/shap_importance.csv")
    if os.path.exists("data/equity_audit.csv"):
        artifacts["equity"] = pd.read_csv("data/equity_audit.csv")
    return artifacts


def attempt_model_prediction(model, feature_row: dict):
    """Tries to get a real prediction out of the loaded pipeline. Returns
    None if the model isn't loaded or the feature schema doesn't line up,
    so the caller can fall back to a clearly-labeled illustrative estimate
    instead of pretending a real prediction happened.
    """
    if model is None:
        return None
    try:
        X = pd.DataFrame([feature_row])
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            missing = [c for c in expected if c not in X.columns]
            if missing:
                st.session_state["_model_schema_mismatch"] = missing
                return None
            X = X[expected]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Multi-output classifiers return a list of per-target arrays;
            # single-output return one array.
            if isinstance(proba, list):
                return [float(p[0][1]) for p in proba]
            return [float(p) for p in proba[0]]
        elif hasattr(model, "predict"):
            pred = model.predict(X)
            return [float(v) for v in np.ravel(pred)]
    except Exception as e:
        st.session_state["_model_error"] = str(e)
        return None
    return None


model_pipeline = load_production_pipeline()
spatial_df = load_regional_sdoh_matrix()
zcta_df = load_zcta_level_sdoh()
demo_risk_df = load_demographic_risk_predictions()
eval_artifacts = load_eval_artifacts()
IS_LIVE_MODEL = model_pipeline is not None
HAS_SPATIAL_DATA = spatial_df is not None
HAS_ZCTA_DATA = zcta_df is not None
HAS_DEMO_RISK_DATA = demo_risk_df is not None

st.title("Social Determinants of Health As Predictors of Individual Chronic Disease Risk")

if not IS_LIVE_MODEL:
    st.markdown(
        '<div class="demo-banner">⚠️ <b>Demo mode:</b> No trained model was found at '
        '<code>models/spatial_fused_production_model.joblib</code>. The risk scores below are just '
        'placeholder guesses, not real predictions, and are labeled that way.</div>',
        unsafe_allow_html=True
    )
if not HAS_SPATIAL_DATA:
    st.markdown(
        '<div class="demo-banner">⚠️ <b>Missing data:</b> '
        '<code>data/aggregated_spatial_sdoh.csv</code> was not found. Run '
        '<code>generate_spatial_features.py</code> first. Neighborhood numbers below will not show up '
        'until that file exists.</div>',
        unsafe_allow_html=True
    )

st.markdown("""
    This dashboard looks at how your background and your neighborhood relate to your risk for common
    health problems, like diabetes or stroke. It uses real answers from the **2024 CDC BRFSS survey**
    and real neighborhood data from the **2024 CDC PLACES program**. If real data or a real result is
    not available for something, that is stated that the result is not available.
""")

AGE_MAP = {
    "Ages 18 to 24": 1.0, "Ages 25 to 29": 2.0, "Ages 30 to 34": 3.0,
    "Ages 35 to 39": 4.0, "Ages 40 to 44": 5.0, "Ages 45 to 49": 6.0,
    "Ages 50 to 54": 7.0, "Ages 55 to 59": 8.0, "Ages 60 to 64": 9.0,
    "Ages 65 to 69": 10.0, "Ages 70 to 74": 11.0, "Ages 75 to 79": 12.0,
    "Ages 80 or older": 13.0
}
INCOME_MAP = {
    "Less than $10,000": 1.0, "Less than $15,000": 2.0, "Less than $20,000": 3.0,
    "Less than $25,000": 4.0, "Less than $35,000": 5.0, "Less than $50,000": 6.0,
    "Less than $75,000": 7.0, "$75,000 or more": 8.0
}
EDUCATION_MAP = {
    "Some Grade School": 1.0, "Elementary School": 2.0, "Some High School": 3.0,
    "High School Graduate or GED": 4.0, "Some College or Technical School": 5.0,
    "College Graduate": 6.0
}
RACE_MAP = {
    "White, Non-Hispanic": 1.0,
    "Black, Non-Hispanic": 2.0,
    "American Indian or Alaska Native, Non-Hispanic": 3.0,
    "Asian, Non-Hispanic": 4.0,
    "Native Hawaiian or Other Pacific Islander, Non-Hispanic": 5.0,
    "Hispanic or Latino (Any Race)": 7.0,
    "Other / Multiracial, Non-Hispanic": 6.0
}

# Official CDC national numbers, for comparing against the survey sample
# and against the personal estimate in Tab 1. These come from CDC's own
# BRFSS national results (dataset "BRFSS Prevalence Data, 2011-present"),
# using the same or nearly the same question used for each condition --
# so the comparison is as close to apples-to-apples as CDC's own published
# data allows. Chronic Respiratory Disease has no single official combined
# number (the target here is asthma OR COPD, and CDC doesn't publish that
# combination), so the two separately-published rates are added together
# and flagged as such -- the real combined rate is a bit lower, since some
# people have both conditions.
OFFICIAL_NATIONAL_PREVALENCE = {
    "Diabetes Risk": 0.120,
    "Stroke Risk": 0.034,
    "Cancer Prevalence": 0.087,
    "Cardiovascular Event": 0.070,
    "Chronic Respiratory Disease": 0.173,
}
OFFICIAL_NATIONAL_PREVALENCE_SOURCE = (
    "CDC BRFSS Prevalence Data (2011-present), 2024, median across all states and DC. "
    "https://data.cdc.gov/d/dttw-5yxu"
)

DISEASES = ["Diabetes Risk", "Stroke Risk", "Cancer Prevalence", "Cardiovascular Event", "Chronic Respiratory Disease"]
DISEASE_TO_PRED_COL = {
    "Diabetes Risk": "pred_target_diabetes",
    "Stroke Risk": "pred_target_stroke",
    "Cancer Prevalence": "pred_target_cancer",
    "Cardiovascular Event": "pred_target_cardio",
    "Chronic Respiratory Disease": "pred_target_chronic_respiratory",
}

FEATURE_DISPLAY_NAMES = {
    "age_code": "Age",
    "income_code": "Income",
    "educa_code": "Education",
    "race_code": "Race/Ethnicity",
    "sex_code": "Sex",
    "bmi": "BMI",
    "smoker_code": "Smoking Status",
    "physical_activity_code": "Physical Activity",
    "heavy_drinking_code": "Heavy Drinking",
    "spatial_lack_insurance": "Neighborhood: Lack of Insurance",
    "spatial_poor_mental_health": "Neighborhood: Poor Mental Health",
}

st.header("Your Traits")
st.markdown("Select traits below to update the risk profile and the map further down the page.")

st.markdown('<span class="group-label group-demographics">Demographics</span>', unsafe_allow_html=True)
with st.container(border=True):
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        age_label = st.selectbox("Age Group", options=list(AGE_MAP.keys()), index=5)
        age_code = AGE_MAP[age_label]
    with col_f2:
        income_label = st.selectbox("Income Bracket", options=list(INCOME_MAP.keys()), index=4)
        income_code = INCOME_MAP[income_label]
    with col_f3:
        educa_label = st.selectbox("Highest Level of Education", options=list(EDUCATION_MAP.keys()), index=5)
        educa_code = EDUCATION_MAP[educa_label]
    with col_f4:
        race_label = st.selectbox("Race and Ethnicity", options=list(RACE_MAP.keys()))
        race_code = RACE_MAP[race_label]

st.markdown('<div class="section-arrow">↓</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Individual Risk Profile",
    "2. Regional SDoH Map",
    "3. Public Health Impact",
    "4. Model Intelligence & Diagnostics"
])

# ---------------------------------------------------------------------------
# TAB 1: INDIVIDUAL RISK PROFILE
# Color grouping (preattentive cue, paired with a text label so color isn't
# the only signal): Demographics = indigo, Location/regional data = cyan,
# Health & Lifestyle = amber. The location picker and the regional stats it
# produces share the same cyan label, since those numbers come directly and
# only from that one selection. The risk readout deliberately does NOT
# reuse any of those three colors -- it's a function of every input
# together, so tying it to one group's color would overstate that group's
# influence.
# ---------------------------------------------------------------------------
with tab1:
    st.header("Your Risk Profile")
    st.markdown("See how your health risk compares to your neighborhood, your peers in the survey data, and the whole country.")

    st.markdown('<span class="group-label group-location">Location &amp; Sex</span>', unsafe_allow_html=True)
    with st.container(border=True):
        lookup_mode = st.radio(
            "Look up your local area by:",
            options=["State (regional average)", "ZIP code (your specific area)"],
            horizontal=True,
            disabled=not (HAS_SPATIAL_DATA or HAS_ZCTA_DATA),
        )

        col_geo1, col_geo2 = st.columns(2)
        state_select, zip_input = None, None
        with col_geo1:
            if lookup_mode == "ZIP code (your specific area)":
                zip_input = st.text_input(
                    "ZIP code", max_chars=5, placeholder="e.g. 21201",
                    disabled=not HAS_ZCTA_DATA,
                    help="This looks up your exact ZIP code area. It's more precise than picking a whole state."
                )
                if not HAS_ZCTA_DATA:
                    st.caption("ZIP-level data isn't available yet. Run `generate_spatial_features.py` to build it.")
            else:
                available_states = sorted(spatial_df["stateabbr"].dropna().unique().tolist()) if HAS_SPATIAL_DATA else []
                state_select = st.selectbox(
                    "Target Analysis State",
                    options=available_states or ["No data available"],
                    disabled=not HAS_SPATIAL_DATA
                )
        with col_geo2:
            sex_select = st.radio("Sex", options=[1, 2],
                                   format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)

    st.markdown('<span class="group-label group-health">Health &amp; Lifestyle</span>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("The model uses these answers directly. They show more about your personal risk than your demographics or your neighborhood do.")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            height_ft = st.number_input("Height (feet)", min_value=3, max_value=8, value=5, step=1)
            height_in = st.number_input("Height (inches)", min_value=0, max_value=11, value=6, step=1)
            weight_lbs = st.number_input("Weight (lbs)", min_value=50, max_value=700, value=160, step=1)
            total_inches = height_ft * 12 + height_in
            bmi_value = 703 * weight_lbs / (total_inches ** 2) if total_inches > 0 else None
            if bmi_value:
                st.caption(f"Calculated BMI: {bmi_value:.1f}")
        with col_h2:
            SMOKER_LABELS = {1.0: "Current smoker, every day", 2.0: "Current smoker, some days",
                              3.0: "Former smoker", 4.0: "Never smoked"}
            smoker_select = st.selectbox("Smoking status", options=list(SMOKER_LABELS.keys()), format_func=lambda x: SMOKER_LABELS[x], index=3)
            activity_select = st.radio("Any leisure-time physical activity in the past 30 days?", options=[1.0, 2.0],
                                        format_func=lambda x: "Yes" if x == 1.0 else "No", horizontal=True)
            drinking_select = st.radio("Heavy drinker? (14+ drinks/week men, 7+ women)", options=[1.0, 2.0],
                                        format_func=lambda x: "No" if x == 1.0 else "Yes", horizontal=True)

    st.markdown('<div class="section-arrow">↓</div>', unsafe_allow_html=True)
    st.markdown('<span class="group-label group-location">Regional Data, From Your Location</span>', unsafe_allow_html=True)
    st.subheader("Your Local Area")

    ins_val, mhlth_val, region_label = None, None, "N/A"

    if lookup_mode == "ZIP code (your specific area)":
        if HAS_ZCTA_DATA and zip_input:
            if not verify_zip_code(zip_input):
                st.error("Enter a valid 5-digit ZIP code (e.g. 21201).")
            else:
                try:
                    matched_row = merge_spatial_vectors(zcta_df, zip_input)
                    ins_val = matched_row["spatial_lack_insurance"]
                    mhlth_val = matched_row["spatial_poor_mental_health"]
                    region_label = f"ZIP {zip_input.strip()}"
                    if "stateabbr" in matched_row.index and pd.notna(matched_row["stateabbr"]):
                        region_label += f" ({matched_row['stateabbr']})"
                except ValueError as e:
                    st.error(f"Couldn't look up that ZIP: {e}")
        elif HAS_ZCTA_DATA and not zip_input:
            st.caption("Enter a ZIP code above to see what your local area looks like.")
    else:
        if HAS_SPATIAL_DATA and state_select in spatial_df["stateabbr"].values:
            region_rows = spatial_df[spatial_df["stateabbr"] == state_select]
            ins_val = region_rows["spatial_lack_insurance"].mean()
            mhlth_val = region_rows["spatial_poor_mental_health"].mean()
            region_label = state_select

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label=f"People Without Health Insurance ({region_label})",
            value=f"{ins_val:.2f}%" if ins_val is not None and not pd.isna(ins_val) else "No data"
        )
    with col_m2:
        st.metric(
            label=f"People with Frequent Poor Mental Health ({region_label})",
            value=f"{mhlth_val:.2f}%" if mhlth_val is not None and not pd.isna(mhlth_val) else "No data",
            help="Share of adults who said their mental health was not good on 14 or more of the past 30 days.",
        )

    diseases = DISEASES

    feature_row = {
        "age_code": age_code, "income_code": income_code, "educa_code": educa_code,
        "race_code": race_code, "sex_code": float(sex_select),
        "spatial_lack_insurance": ins_val if ins_val is not None else np.nan,
        "spatial_poor_mental_health": mhlth_val if mhlth_val is not None else np.nan,
        "bmi": bmi_value if bmi_value is not None else np.nan,
        "smoker_code": float(smoker_select),
        "physical_activity_code": float(activity_select),
        "heavy_drinking_code": float(drinking_select),
    }
    live_prediction = attempt_model_prediction(model_pipeline, feature_row) if IS_LIVE_MODEL else None

    if live_prediction is not None and len(live_prediction) == len(diseases):
        score_col_label = "Your Risk"
        individual_probabilities = live_prediction
    else:
        score_col_label = "Your Risk (Demo Estimate, Not the Real Model)"
        if "_model_schema_mismatch" in st.session_state:
            st.info(
                f"The trained model expects some inputs ({st.session_state['_model_schema_mismatch']}) "
                "that this page doesn't collect yet, so it can't run. Showing a simple placeholder "
                "estimate below instead."
            )
        safe_ins = ins_val if ins_val is not None and not pd.isna(ins_val) else 0.0
        safe_mhlth = mhlth_val if mhlth_val is not None and not pd.isna(mhlth_val) else 0.0
        individual_probabilities = [
            0.14 + (age_code * 0.02) - (income_code * 0.015),
            0.04 + (age_code * 0.01) + (safe_ins * 0.003),
            0.08 + (age_code * 0.015),
            0.09 + (age_code * 0.02) + (safe_ins * 0.002),
            0.11 + (safe_mhlth * 0.004),
        ]
        individual_probabilities = [min(max(p, 0.01), 0.99) for p in individual_probabilities]

    # Live sidebar readout: persists across every tab and updates on every
    # widget change above, so the risk numbers stay visible and "watchable"
    # while scrolling through the trait inputs instead of only appearing
    # after scrolling all the way down to the table below.
    with st.sidebar:
        st.markdown("### Your Estimated Risk")
        st.caption(score_col_label)
        for disease, prob in zip(diseases, individual_probabilities):
            nat_avg = OFFICIAL_NATIONAL_PREVALENCE.get(disease)
            delta = f"{prob - nat_avg:+.1%} vs. national avg" if nat_avg is not None else None
            st.metric(label=disease, value=f"{prob:.1%}", delta=delta, delta_color="inverse")
        st.caption("Updates as you change your traits on the Individual Risk Profile tab.")

    sample_avg_col = "Average in Survey Data"
    national_avg_col = "U.S. National Average (CDC)"
    eval_metrics_dict = eval_artifacts.get("eval_metrics", {})
    sample_prevalence = eval_metrics_dict.get("prevalence_by_target", {})
    n_test = eval_metrics_dict.get("n_test")

    st.markdown('<div class="section-arrow">↓</div>', unsafe_allow_html=True)
    st.subheader("Your Risk, Compared")

    prob_df = pd.DataFrame({
        "Health Condition": diseases,
        score_col_label: individual_probabilities,
        sample_avg_col: [sample_prevalence.get(d) for d in diseases],
        national_avg_col: [OFFICIAL_NATIONAL_PREVALENCE.get(d) for d in diseases],
    })
    st.dataframe(
        prob_df.style.format(
            {score_col_label: "{:.1%}", sample_avg_col: "{:.1%}", national_avg_col: "{:.1%}"},
            na_rep="No data",
        ),
        use_container_width=True,
        hide_index=True
    )

    sample_size_note = f"about {n_test:,} real people" if n_test else "real people"
    st.caption(
        f"**{sample_avg_col}** is the real share of {sample_size_note} in the survey data who have each "
        f"condition. **{national_avg_col}** is the CDC's own official number for the whole country. "
        f"{OFFICIAL_NATIONAL_PREVALENCE_SOURCE} The Chronic Respiratory Disease national number combines "
        "two separate CDC rates (asthma and COPD), so it's a rough estimate, not an exact one."
    )
    st.caption(
        "These are estimates, not a diagnosis. Don't use them to make medical decisions on their own. "
        "Talk to a doctor about your own health."
    )

# ---------------------------------------------------------------------------
# TAB 2: MAP
# Two map modes:
#   1. "Neighborhood Health Access" -- the two predictor variables (mental health and insurance access),
#       averaged by state. PLACES doesn't publish these split by
#      age/income/race, so this map is never filtered by demographics --
#      that's a real limit of the data source, not something hidden here.
#   2. "Model-Predicted Risk" -- built from generate_demographic_risk_predictions.py,
#      which runs the trained model over every real BRFSS respondent. Since
#      real respondents DO have demographics, the map can filter by one trait
#      at a time (age, income, education, or race) and average the model's
#      real predictions by state. Filtering happens by only ONE trait at a
#      time, not all four together, because slicing by all four leaves only
#      a couple of real people per state -- too few to trust.
# ---------------------------------------------------------------------------
def render_us_choropleth(state_avg_df: pd.DataFrame, value_col: str, legend_name: str, map_key: str):
    us_states_geojson_url = "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
    us_map = folium.Map(location=[37.8, -96.0], zoom_start=4, tiles="CartoDB positron")

    metric_min, metric_max = state_avg_df[value_col].min(), state_avg_df[value_col].max()
    if pd.notna(metric_min) and pd.notna(metric_max) and metric_min != metric_max:
        explicit_bins = list(np.linspace(metric_min, metric_max, 6))
        folium.Choropleth(
            geo_data=us_states_geojson_url, name="choropleth", data=state_avg_df,
            columns=["stateabbr", value_col], key_on="feature.id", fill_color="viridis",
            fill_opacity=0.75, line_opacity=0.4, line_color="#cbd5e1", line_weight=1,
            bins=explicit_bins, legend_name=legend_name, highlight=True
        ).add_to(us_map)

    with st.container(border=True):
        st_folium(us_map, use_container_width=True, height=550, key=map_key)


with tab2:
    st.header("Map")

    map_mode = st.radio(
        "What do you want to see on the map?",
        options=["Neighborhood Health Access (CDC data)", "Model-Predicted Risk (pick one trait)"],
        horizontal=True,
    )

    if map_mode == "Neighborhood Health Access (CDC data)":
        if not HAS_SPATIAL_DATA:
            st.warning("No map data yet. Run `generate_spatial_features.py` to build `data/aggregated_spatial_sdoh.csv`.")
        else:
            st.markdown(
                "This map shows two variables from the CDC: how many people lack health "
                "insurance, and how many report frequent poor mental health. Each state's color is a real "
                "average across its counties. The CDC does not publish these numbers broken down by age, "
                "income, or race, so this map always shows the whole population of each state. Switch to "
                "**Model-Predicted Risk** above for a map that changes based on age, income, education, or "
                "race. **Model-Predicted Risk** uses the trained model instead of raw CDC numbers."
            )

            available_metrics = [c for c in ["spatial_lack_insurance", "spatial_poor_mental_health"] if c in spatial_df.columns]
            metric_labels = {"spatial_lack_insurance": "Lack of Health Insurance (%)", "spatial_poor_mental_health": "Frequent Poor Mental Health, 14+ Days (%)"}

            col_map_sel, _ = st.columns([1, 1])
            with col_map_sel:
                map_metric = st.selectbox(
                    "Select Metric",
                    options=available_metrics,
                    format_func=lambda c: metric_labels.get(c, c)
                )

            state_avg = spatial_df.groupby("stateabbr")[map_metric].mean().reset_index()
            render_us_choropleth(state_avg, map_metric, metric_labels.get(map_metric, map_metric), "us_choropleth_map_real")

            n_states = state_avg["stateabbr"].nunique()
            st.caption(
                f"<b>Figure 1.</b> Real state averages of {metric_labels.get(map_metric, map_metric)}, "
                f"built from {n_states} states with data in the current CDC pull. States not shown had no "
                "matching records in the source data.",
                unsafe_allow_html=True
            )
    else:
        if not HAS_DEMO_RISK_DATA:
            st.warning(
                "No model-predicted risk data yet. Run `generate_demographic_risk_predictions.py` to build "
                "`data/model_predicted_risk_by_respondent.csv`."
            )
        elif not IS_LIVE_MODEL:
            st.warning("No trained model found, so this map can't be built. Run `train_model.py` first.")
        else:
            st.markdown(
                "This map is built from the trained model's real predictions for real BRFSS survey takers. "
                "Select a health condition and one personal trait below to see "
                "the model's average predicted risk, for individuals who match that trait in each state. "
                "Filtering happens by only one trait at a time (not age AND income "
                "AND education AND race all together), because this keeps the sample group "
                "of surveyers large enough to trust the predictions."
            )

            col_disease_sel, col_dim_sel = st.columns(2)
            with col_disease_sel:
                map_disease = st.selectbox("Health Condition", options=DISEASES, key="map_disease_select")
            demographic_dimensions = {
                "Age Group": ("age_code", age_code, age_label),
                "Income Bracket": ("income_code", income_code, income_label),
                "Education Level": ("educa_code", educa_code, educa_label),
                "Race and Ethnicity": ("race_code", race_code, race_label),
            }
            with col_dim_sel:
                map_dimension = st.selectbox("Filter By", options=list(demographic_dimensions.keys()))

            dim_col, dim_value, dim_label = demographic_dimensions[map_dimension]
            pred_col = DISEASE_TO_PRED_COL[map_disease]

            MIN_SAMPLE_SIZE = 30
            matching = demo_risk_df[demo_risk_df[dim_col] == dim_value]
            state_counts = matching.groupby("stateabbr").size()
            state_avg = (
                matching.groupby("stateabbr")[pred_col]
                .mean()
                .reset_index()
                .merge(state_counts.rename("n").reset_index(), on="stateabbr")
            )
            enough_data = state_avg[state_avg["n"] >= MIN_SAMPLE_SIZE].copy()
            too_few = state_avg[state_avg["n"] < MIN_SAMPLE_SIZE]

            if enough_data.empty:
                st.info(
                    f"Not enough real survey takers who are '{dim_label}' in any single state (need at "
                    f"least {MIN_SAMPLE_SIZE}) to draw this map. Try a different trait."
                )
            else:
                enough_data[pred_col] = enough_data[pred_col] * 100
                render_us_choropleth(
                    enough_data, pred_col, f"Predicted {map_disease} (%), {dim_label}", "us_choropleth_model_risk"
                )
                st.caption(
                    f"<b>Figure 1.</b> Real model predictions for {int(enough_data['n'].sum()):,} real survey "
                    f"takers who are '{dim_label}', averaged by state. Only states with at least "
                    f"{MIN_SAMPLE_SIZE} matching survey takers are shown"
                    + (f" ({len(too_few)} state(s) had too few and are left blank)." if len(too_few) else ".")
                    + " This is a model prediction, not an official CDC number.",
                    unsafe_allow_html=True
                )

# ---------------------------------------------------------------------------
# TAB 3: PUBLIC HEALTH IMPACT
# All numbers here now come from models/eval_metrics.json if present.
# Nothing is hardcoded. If the file doesn't exist, the tab says so plainly.
# ---------------------------------------------------------------------------
with tab3:
    st.header("How This Model Could Improve Healthcare Processes")
    st.markdown("Practical applications for using this model, supported by the evidence.")

    with st.container(border=True):
        st.subheader("How Much Better Than Guessing?")
        st.markdown(
            "If an entity were to pilot this model in their practice, these numbers would be replaced by real, "
            "data comparing outcomes both pre and post model implementation. Instead, the below findings come from "
            "the model's test results. It is **not** a pilot result, and it does not gaurentee these results in the "
            "real world if it were implemented. However, it does show how much better the model performs at "
            "sorting high-risk people from low-risk people when compared to pure guessing."
        )
        _metrics_t3 = eval_artifacts.get("eval_metrics", {})
        pr_auc_by_target = _metrics_t3.get("pr_auc_by_target")
        prevalence_by_target_t3 = _metrics_t3.get("prevalence_by_target")
        if pr_auc_by_target and prevalence_by_target_t3:
            lift_cols = st.columns(len(DISEASES))
            for col, disease in zip(lift_cols, DISEASES):
                pr_auc = pr_auc_by_target.get(disease)
                prevalence = prevalence_by_target_t3.get(disease)
                with col:
                    if pr_auc and prevalence:
                        lift = pr_auc / prevalence
                        st.metric(label=disease, value=f"{lift:.1f}x", help=f"{pr_auc:.1%} vs. a {prevalence:.1%} baseline rate")
                    else:
                        st.metric(label=disease, value="No data")
            st.caption(
                "\"2x\" means the model is about twice as good as random guessing at telling who has the "
                "condition from who doesn't, averaged across every possible cutoff. Numbers come from "
                "`models/eval_metrics.json`, using each condition's held-out test data. See the glossary in "
                "Tab 4 for more on PR-AUC and how this is calculated."
            )
        else:
            st.info("No performance data found yet. Run `train_model.py` to generate `models/eval_metrics.json`.")

    with st.container(border=True):
        st.subheader("How Could This Be Used?")
        st.markdown(
            "These are ideas for how a tool like this could help, based on what it measures today. None of "
            "this has been tested in a real program yet -- see the pilot results section below for that."
        )
        st.markdown("""
- **Health departments** could use the map in Tab 2 to spot neighborhoods with low insurance access or high mental health burden, and send mobile clinics or outreach workers there first.
- **Community health workers** could use the risk table in Tab 1 as one more piece of information when deciding who might benefit from a check-up reminder or a home visit.
- **Researchers and policymakers** could use the patterns here to study how income, education, and neighborhood conditions connect to health risks, and to make the case for funding in specific areas.
""")
        st.caption(
            "These are possible uses, not proven results. A tool like this should always be one input among "
            "many, reviewed by real people, not an automatic decision-maker."
        )

    impact = eval_artifacts.get("eval_metrics", {}).get("impact_summary") if "eval_metrics" in eval_artifacts else None

    if not impact:
        st.caption(
            "No real pilot has run yet, so no pilot results are shown here. Pilot numbers can only come "
            "from an actual program, not a calculation -- once one runs, `add_pilot_impact_summary.py` "
            "can add its real, measured numbers to this section."
        )
    else:
        st.markdown('<div class="section-arrow">↓</div>', unsafe_allow_html=True)
        st.subheader("Real Pilot Results")
        st.markdown("This part only shows numbers measured from an actual program in the real world -- never estimates.")
        with st.container(border=True):
            col_metric_left, col_metric_right = st.columns([1, 1])
            with col_metric_left:
                st.metric(label="Care Outreach Efficiency", value=impact.get("outreach_efficiency", "N/A"))
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric(label="Clinical False-Alarm Reduction", value=impact.get("false_alarm_reduction", "N/A"))
            with col_metric_right:
                st.metric(label="Equity Reach Multiplier", value=impact.get("equity_reach_multiplier", "N/A"))
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric(label="Recall at Operating Point", value=impact.get("recall_at_operating_point", "N/A"))
            if impact.get("source"):
                st.caption(f"Source: {impact['source']}")
        st.caption("<b>Figure 2.</b> Computed from a real holdout evaluation; see <code>models/eval_metrics.json</code>.", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 4: MODEL INTELLIGENCE & DIAGNOSTICS
# Same principle: real SHAP importances, real PR curve, and real equity
# audit are loaded from artifact files if present; otherwise the tab shows
# an explicit "not yet computed" state instead of synthetic charts/tables.
# ---------------------------------------------------------------------------
GLOSSARY = {
    "Held-out / test set": "Real people set aside and never shown to the model while it was learning. They are used afterward to check the model's work honestly, like a pop quiz with questions the student never saw in advance.",
    "Prevalence (held-out)": "How common a condition actually is among the held-out group -- for example, if 5 out of 100 people in the test set really had a stroke, prevalence is 5%.",
    "Baseline / guessing": "What you'd get by always predicting \"no\" (or by randomly guessing at the real rate). A model needs to beat this to be worth anything.",
    "Macro F1": "A single score that blends how often the model is right when it says \"yes\" (precision) with how many real \"yes\" cases it actually catches (recall). It's calculated using a 50% cutoff, which can look bad for rare conditions even if the model has real skill -- see ROC-AUC and PR-AUC below.",
    "Precision": "Of all the times the model said someone has a condition, what share actually did?",
    "Recall": "Of everyone who actually has the condition, what share did the model catch?",
    "ROC-AUC": "A score from 0 to 1 for how well the model ranks people from highest to lowest risk, checked across every possible cutoff, not just 50%. A score of 0.5 means no better than a coin flip; 1.0 means perfect ranking.",
    "PR-AUC": "Similar to ROC-AUC, but more informative when a condition is rare. It's the area under the precision-recall curve; higher is better, and it should always be compared to the condition's real prevalence (a model with zero skill scores about equal to the prevalence).",
    "SHAP": "A method for figuring out how much each answer (like age or smoking status) pushed a specific prediction up or down. It's the standard way to explain what a model is \"paying attention to.\"",
    "FPR (False Positive Rate)": "Of the people who do NOT have a condition, what share did the model wrongly flag as having it?",
    "FNR (False Negative Rate)": "Of the people who DO have a condition, what share did the model miss?",
    "Neighborhood disadvantage quartile": "Neighborhoods sorted into four equal-sized groups (quarters) by how many residents lack health insurance, from least to most disadvantaged. This allows checking whether the model makes more mistakes for people in poorer neighborhoods.",
    "Choropleth map": "A map where areas (like states) are shaded by color based on a number -- darker usually means higher.",
    "Operating point": "The specific cutoff probability a model actually uses to decide \"high risk\" vs. \"not high risk.\" A model can rank people well but still need a chosen cutoff before it can make a yes/no call -- see Figure 3.",
    "Imputation": "Filling in a missing answer with an estimate (such as a group median) instead of leaving it blank or dropping that person from the data entirely.",
    "Selection bias": "A distortion that happens when the people who end up in a survey are not a perfect random sample of everyone -- for example, a phone survey underrepresents people without phone access.",
}


with tab4:
    st.header("How the Model Works")
    st.markdown(
        "This page shows the model's real, measured performance. "
        "Unfamiliar terms are explained in the glossary at the bottom."
    )

    with st.container(border=True):
        st.subheader("Figure 1: How Well Does the Model Score, Condition by Condition?")
        st.markdown(
            "This table shows four different ways of grading the model for each health condition, using "
            "real people it never saw during training. **Prevalence** is how common the condition really "
            "is. **Macro F1** uses a simple 50-50 cutoff and can look weak for rare conditions like stroke "
            "even when the model is doing real work **ROC-AUC** and **PR-AUC** grade the model across "
            "every possible cutoff instead of just one, so they're a fairer read here."
        )
        _metrics = eval_artifacts.get("eval_metrics", {})
        if "roc_auc_by_target" in _metrics:
            perf_df = pd.DataFrame({
                "Condition": list(_metrics["macro_f1_by_target"].keys()),
                "Prevalence (held-out)": list(_metrics.get("prevalence_by_target", {}).values()),
                "Macro F1 (0.5 threshold)": list(_metrics["macro_f1_by_target"].values()),
                "ROC-AUC": list(_metrics["roc_auc_by_target"].values()),
                "PR-AUC": list(_metrics["pr_auc_by_target"].values()),
            })
            st.dataframe(
                perf_df.style.format({
                    "Prevalence (held-out)": "{:.2%}", "Macro F1 (0.5 threshold)": "{:.3f}",
                    "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "A weak looking Macro F1 for a condition like Stroke Risk is a side effect of the 50-50 cutoff "
                "on a rare condition, not proof the model has no signal. Instead, compare PR-AUC to Prevalence in the "
                "same row. The bigger the gap, the more the model is beating pure guessing."
            )
        else:
            st.info(
                "No performance numbers found yet. Run `train_model.py` to build `models/eval_metrics.json`."
            )

    col_graph1, col_graph2 = st.columns([1, 1])

    with col_graph1:
        with st.container(border=True):
            st.subheader("Figure 2: What Matters Most to the Model?")
            st.markdown(
                "This chart ranks the answers (age, income, smoking, and so on) by how much they typically "
                "move a prediction up or down, averaged across every person and every condition. Longer bars "
                "matter more. This comes from SHAP, a standard method for explaining model decisions and "
                "evaluating the marginal impact of each predictor variable. See the glossary below for more information."
            )
            if "shap" in eval_artifacts:
                shap_data = eval_artifacts["shap"].copy()
                shap_data["feature"] = shap_data["feature"].map(FEATURE_DISPLAY_NAMES).fillna(shap_data["feature"])
                shap_data = shap_data.sort_values(by="importance", ascending=True)
                fig_shap = px.bar(shap_data, x="importance", y="feature", orientation="h")
                fig_shap.update_traces(marker_color='#0d9488')
                fig_shap.update_layout(
                    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="sans-serif", size=12),
                    xaxis=dict(showgrid=True, gridcolor="#e2e8f0", title="SHAP Importance"), yaxis=dict(title=None),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info(
                    "No SHAP values found yet. Run `train_model.py` (with the `shap` package installed) to "
                    "build `models/shap_importance.csv`."
                )

    with col_graph2:
        with st.container(border=True):
            st.subheader("Figure 3: Trading Off Catching Cases vs. False Alarms")
            st.markdown(
                "As you loosen the model's cutoff, it catches more real cases (moving right, higher "
                "**recall**) but also raises more false alarms (dropping **precision**, the teal line). The "
                "gold line is what a model with no real skill would score at every point -- the higher above "
                "gold the teal line sits, the more useful the model is."
            )
            _eval_metrics_fig3 = eval_artifacts.get("eval_metrics", {})
            pr_data = _eval_metrics_fig3.get("precision_recall")
            operating_point = _eval_metrics_fig3.get("operating_point")
            if pr_data:
                pr_df = pd.DataFrame(pr_data)  # expects columns: recall, precision, baseline
                fig_pr = px.line(
                    pr_df, x="recall", y=["precision", "baseline"],
                    color_discrete_map={"precision": "#0d9488", "baseline": "#dea30e"}
                )
                if operating_point:
                    fig_pr.add_scatter(
                        x=[operating_point["recall"]], y=[operating_point["precision"]],
                        mode="markers", name="Operating Point",
                        marker=dict(size=13, color="#dc2626", symbol="star"),
                    )
                fig_pr.update_layout(
                    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="sans-serif", size=12),
                    xaxis=dict(showgrid=True, gridcolor="#e2e8f0", title="Recall"),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0", range=[0, 1.05], title="Precision"),
                    legend=dict(title=None, orientation="h", y=-0.25, x=0.05),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_pr, use_container_width=True)
                if operating_point:
                    st.caption(
                        f"The red star is the chosen operating point for {operating_point['target']}: a cutoff "
                        f"of {operating_point['threshold']:.2f}, giving {operating_point['precision']:.0%} "
                        f"precision and {operating_point['recall']:.0%} recall. {operating_point['rationale']}"
                    )
            else:
                st.info(
                    "No precision-recall curve found yet. Run `train_model.py` to build "
                    "`models/eval_metrics.json`."
                )

    st.markdown('<div class="section-arrow">↓</div>', unsafe_allow_html=True)
    st.subheader("Figure 4: Does the Model Treat Every Neighborhood Fairly?")
    st.markdown(
        "Neighborhoods were split into four equal-sized groups, from least to most disadvantaged (based on "
        "how many residents lack health insurance), and the model's error rates were checked in each group. "
        "**FPR** is how often the model wrongly flags someone as at-risk; **FNR** is how often it misses "
        "someone who really is at risk. If FNR climbs sharply in the most disadvantaged group, that means "
        "the model is missing more real cases exactly where help may be needed most."
    )

    if "equity" in eval_artifacts:
        equity_df = eval_artifacts["equity"]
        st.table(equity_df)
        st.caption(
            "Real error rates from the held-out test set, by neighborhood disadvantage group. "
            "See <code>data/equity_audit.csv</code>.",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="demo-banner">⚠️ No fairness numbers found yet. An earlier version of this '
            'dashboard showed a made-up table claiming the model \"doesn\'t penalize people based on '
            'where they live.\" That claim was removed fairness should only be discussed after real '
            'error rates are measured on a real test set.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-arrow">↓</div>', unsafe_allow_html=True)
    st.subheader("Limitations")
    st.markdown("""
- **Not every ZIP code is covered.** The ZIP-code lookup on Tab 1 comes from a CDC database that caps each pull at 500,000 rows. That covers about 13,500 of the roughly 33,000 ZIP code areas in the country -- a ZIP code outside that set will show "No data."
- **Some answers are filled in, not reported.** About 19% of survey takers did not report their income. Those values are estimated using the typical income of similar people (same age group and education level) rather than left blank, which would have dropped those people from the model entirely. This is called imputation, and it can smooth over real variation within a group.
- **BRFSS has known survey limits.** BRFSS is a phone survey of adults with a working phone, so people without phone access, people in institutions (such as nursing homes or prisons), and people who declined to take part are underrepresented -- a pattern called selection bias. Answers are also self-reported, not confirmed by a lab test or a doctor.
- **CDC PLACES pulls are also capped at 500,000 rows.** This is the same limit behind the ZIP code gap above. It does not affect the state-level averages used for the map and the model, since those come from a smaller county-level release that stays well under the cap.
""")

    st.markdown('<div class="section-arrow">↓</div>', unsafe_allow_html=True)
    st.subheader("Glossary")
    st.markdown("Short, plain-language definitions for the data terms used on this page.")
    for term, definition in GLOSSARY.items():
        with st.expander(term):
            st.markdown(definition)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="source-footer">
        <div class="source-title">References and Data Notes</div>
        <div class="source-item">
             <b>Behavioral Risk Factor Surveillance System (BRFSS) 2024 Annual Survey Data.</b> Centers for Disease Control and Prevention (CDC), National Center for Chronic Disease Prevention and Health Promotion. Matrix Data Stream (LLCP2024.XPT).
        </div>
        <div class="source-item">
             <b>PLACES: Local Data for Better Health, County Data, 2024 Release.</b> Centers for Disease Control and Prevention (CDC), Robert Wood Johnson Foundation. Socrata Dataset ID: fu4u-a9bh. Used for state-level regional averages (Regional SDoH Map, Individual Risk Profile state lookup).
        </div>
        <div class="source-item">
             <b>PLACES: Local Data for Better Health, ZCTA Data, 2024 Release.</b> Centers for Disease Control and Prevention (CDC), Robert Wood Johnson Foundation. Socrata Dataset ID: 4r2x-hcfq. Used for the ZIP-code lookup (Individual Risk Profile).
        </div>
        <div class="source-item">
             <b>CDC/ATSDR Social Vulnerability Index (SVI) Procedural Guidance Framework.</b> Agency for Toxic Substances and Disease Registry, Geospatial Research, Analysis, and Services Program (GRASP).
        </div>
    </div>
""", unsafe_allow_html=True)
