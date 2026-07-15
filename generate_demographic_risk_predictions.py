"""
generate_demographic_risk_predictions.py

Builds a per-respondent table of real model-predicted risk, so app.py's
Regional SDoH Map tab can offer a second map: predicted disease risk by
state, filtered to whichever demographic group the user picks (age bracket,
income bracket, education level, race/ethnicity).

This is different from the PLACES map (data/aggregated_spatial_sdoh.csv),
which is a raw CDC statistic with no demographic breakdown -- PLACES simply
doesn't publish estimates split by age/income/race at the state or ZCTA
level. This script instead uses the real BRFSS respondents (who do have
those demographics) and the already-trained model to answer: "among people
who match this demographic profile, what does the model predict for their
average risk, state by state?" Every number on that map traces back to a
real respondent and a real model prediction -- nothing here is simulated.

Predictions are generated for the FULL cleaned/joined dataset (not just the
held-out test split) since this is a descriptive map, not a performance
claim -- Tab 4 already reports honest held-out performance separately.

Usage:
    python generate_demographic_risk_predictions.py --brfss LLCP2024.XPT --spatial data/aggregated_spatial_sdoh.csv
"""
import os
import argparse
import logging

import joblib
import pandas as pd

from config.logging_config import setup_logger
from train_model import load_and_join, FEATURE_COLUMNS, TARGET_COLUMNS

logger = logging.getLogger(__name__)

DEMOGRAPHIC_COLUMNS = ["stateabbr", "age_code", "income_code", "educa_code", "race_code"]


def main(brfss_path: str, spatial_path: str, model_path: str = "models/spatial_fused_production_model.joblib"):
    setup_logger()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run train_model.py first.")

    model = joblib.load(model_path)
    model_df = load_and_join(brfss_path, spatial_path)

    X = model_df[FEATURE_COLUMNS]
    proba_by_target = model.predict_proba(X)  # list of (n_samples, 2) arrays, one per target

    out = model_df[DEMOGRAPHIC_COLUMNS].copy()
    for i, target in enumerate(TARGET_COLUMNS):
        out[f"pred_{target}"] = proba_by_target[i][:, 1]

    os.makedirs("data", exist_ok=True)
    output_path = "data/model_predicted_risk_by_respondent.csv"
    out.to_csv(output_path, index=False)
    logger.info("Saved %d rows of per-respondent model predictions to %s", len(out), output_path)
    print(f"Saved {len(out)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brfss", default="LLCP2024.XPT")
    parser.add_argument("--spatial", default="data/aggregated_spatial_sdoh.csv")
    parser.add_argument("--model", default="models/spatial_fused_production_model.joblib")
    args = parser.parse_args()
    main(args.brfss, args.spatial, args.model)
