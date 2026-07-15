"""
add_pilot_impact_summary.py

Attaches real, pilot-measured operational impact numbers to
models/eval_metrics.json, for display in app.py's Tab 3 ("Public Health
Impact").

This is intentionally a separate script from train_model.py: model training
and evaluation (precision, recall, F1, SHAP, fairness) can be computed from
data alone. Operational impact numbers -- "outreach is now this much
faster," "false alarms dropped by this much" -- cannot. They only mean
something if they come from an actual pilot deployment (a real screening
program, a real comparison against the prior process), not from this
script or from the model's own metrics. This script will not accept
default or estimated values; every field is required, and a --source
description is mandatory so the provenance of each number is always
visible to anyone reading the dashboard later.

Usage example (replace with your real pilot's measured values):

    python add_pilot_impact_summary.py \\
        --source "Q3 2026 pilot, County Health Dept X, n=412 screened over 8 weeks" \\
        --outreach-efficiency "3.1x Faster" \\
        --false-alarm-reduction "42% Lower" \\
        --equity-reach-multiplier "2.4x Higher" \\
        --recall-at-operating-point "58% Caught"
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Attach real pilot-measured impact numbers to models/eval_metrics.json. "
                    "Every value must come from an actual pilot -- this script has no defaults."
    )
    parser.add_argument("--eval-metrics-path", default="models/eval_metrics.json")
    parser.add_argument("--source", required=True,
                         help="Where these numbers came from, e.g. 'Q3 2026 pilot, County X, n=412'. "
                              "Required so provenance is always visible alongside the numbers.")
    parser.add_argument("--outreach-efficiency", required=True,
                         help="Measured, e.g. '3.1x Faster' -- how much faster real outreach was "
                              "using the model's ranked list vs. the prior process, in the pilot.")
    parser.add_argument("--false-alarm-reduction", required=True,
                         help="Measured, e.g. '42% Lower' -- observed reduction in false-positive "
                              "screening referrals vs. the prior process, in the pilot.")
    parser.add_argument("--equity-reach-multiplier", required=True,
                         help="Measured, e.g. '2.4x Higher' -- observed increase in resources reaching "
                              "high-disadvantage-quartile individuals vs. the prior process, in the pilot.")
    parser.add_argument("--recall-at-operating-point", required=True,
                         help="Measured, e.g. '58% Caught' -- share of true cases the pilot's chosen "
                              "operating threshold actually caught, observed in the pilot.")
    args = parser.parse_args()

    if not os.path.exists(args.eval_metrics_path):
        print(f"ERROR: {args.eval_metrics_path} not found. Run train_model.py first.", file=sys.stderr)
        sys.exit(1)

    with open(args.eval_metrics_path) as f:
        eval_metrics = json.load(f)

    eval_metrics["impact_summary"] = {
        "source": args.source,
        "outreach_efficiency": args.outreach_efficiency,
        "false_alarm_reduction": args.false_alarm_reduction,
        "equity_reach_multiplier": args.equity_reach_multiplier,
        "recall_at_operating_point": args.recall_at_operating_point,
    }

    with open(args.eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Added impact_summary (source: {args.source}) to {args.eval_metrics_path}")


if __name__ == "__main__":
    main()
