# SDoH Chronic Disease Risk Dashboard

**Live app:** [Streamlit App](https://chronic-disease-risk-engine.streamlit.app)

<img width="950" height="431" alt="Screenshot 2026-07-15 143758" src="https://github.com/user-attachments/assets/d1a0365d-ef5d-4c09-aaf4-9c151a58bb38" />


## What this is

A dashboard that estimates individual risk for five chronic conditions
(diabetes, stroke, cancer, cardiovascular event, chronic respiratory
disease), using real 2024 CDC BRFSS survey answers combined with real
neighborhood data from CDC PLACES.

I built this mainly to learn: prompt engineering, how to structure a real
model pipeline end to end, and where AI coding tools actually break down.
Along the way it touched on model architecture, data validation and
cleaning at scale, statistical evaluation (why a threshold-dependent
metric like F1 can be misleading on imbalanced data), working with public
APIs (CDC Socrata endpoints), git workflows, and deploying a real app.

Worth being upfront about: partway through this project, an AI assistant
I was using rewrote parts of the app on its own to fabricate model
predictions, a fake fairness audit, and made-up "impact" numbers. I
didn't catch it immediately, and it took a fair amount of time combing
through the code line by line to find and rip out the fabricated pieces
and rebuild them with real computations. That experience shaped how I
approached the rest of the build: verify everything against the actual
data, and don't trust a plausible-looking number just because an AI
wrote it.

## What it does

- **Individual risk profile.** Plug in your demographics, health/lifestyle
  answers, and either your state or your ZIP code. Get model predictions,
  plus how that compares to real survey averages and the CDC's own
  national numbers for each condition.
- **Two maps.** One of raw CDC PLACES neighborhood stats (insurance
  access, mental health burden), and one of the model's own predicted risk,
  filterable by one demographic trait at a time (filtering by all of them
  at once leaves too few real people per state to trust).
- **Model performance.** ROC-AUC/PR-AUC by condition, a computed
  operating point (not an arbitrary cutoff), SHAP feature importances, and
  a subgroup fairness audit by neighborhood disadvantage.
- **A limitations section.** ZIP coverage gaps from a 500K-row API cap,
  income imputation, BRFSS survey bias.

## Data

- **CDC BRFSS 2024** (`LLCP2024.XPT`, 457,670 respondents): individual
  demographics, health/lifestyle answers, and the 5 target conditions.
- **CDC PLACES 2024**: county-level release for state averages
  (`fu4u-a9bh`), ZCTA-level release for the ZIP lookup (`4r2x-hcfq`).

## Model

A LightGBM classifier (falls back to scikit-learn's
`HistGradientBoostingClassifier` if LightGBM isn't installed), wrapped in
a `MultiOutputClassifier` so one model predicts all 5 conditions at once
from the same inputs: age, income, education, race, sex, BMI, smoking,
physical activity, drinking, and two neighborhood-level PLACES stats.

It's trained on about 290K BRFSS respondents and evaluated on a held-out
~72K it never sees during training. ROC-AUC (how well it ranks
higher-risk people above lower-risk people) lands between 0.66 and 0.79
depending on the condition: real signal, but modest. This is not a
diagnostic tool and shouldn't be treated like one.

## Running it locally

```bash
pip install -r requirements.txt
python generate_spatial_features.py          # pulls CDC PLACES data
python train_model.py --brfss LLCP2024.XPT    # needs the raw BRFSS file, not in this repo
python generate_demographic_risk_predictions.py
streamlit run app.py
```

The raw `LLCP2024.XPT` file (~1GB) isn't in this repo. Get it from
[CDC BRFSS](https://www.cdc.gov/brfss/annual_data/annual_2024.html).
Everything `app.py` needs to run (`models/`, `data/`) is already committed,
so you don't need the raw file just to run the dashboard.

## Repo layout

```
app.py                                  # the dashboard
train_model.py                          # trains the model, writes models/ + data/equity_audit.csv
generate_spatial_features.py            # pulls CDC PLACES data
generate_demographic_risk_predictions.py # per-respondent predictions for the demographic map
add_pilot_impact_summary.py             # attaches real pilot numbers, once a real pilot exists
src/                                    # BRFSS ingestion + cleaning
config/                                 # logging config
data/, models/                          # generated artifacts, committed so the app runs out of the box
```

## Disclaimer

This dashboard is solely for exploratory research only. It should not be
used to guide medical decisions, diagnose, or as a medical reference.
Talk to a doctor if you have any concerns about your, or your loved one's
health.
