"""
list_brfss_columns.py

Quick check: confirms whether the new risk-factor columns
(src/data_ingestion.py's target_cols) actually exist in your real .XPT
file, and if any expected name is wrong, searches for likely alternatives
by keyword so you can fix the name before running a full training pass.

BRFSS variable names carry year-specific suffixes (e.g. _RFDRHV8,
_SMOKER3) that sometimes change between annual releases. This can't be
verified without your actual file, so run this first -- it only reads
metadata (fast, no need to load all 457k rows) rather than the full
dataset.

Usage:
    python list_brfss_columns.py LLCP2024.XPT
"""
import sys
import pyreadstat

EXPECTED_NEW_COLUMNS = ["_BMI5", "_SMOKER3", "_TOTINDA", "GENHLTH", "_RFDRHV9"]
# SLEPTIM1 removed: confirmed not present in this project's 2024 BRFSS
# release, with no similarly-named alternative found either.

# If an expected name is missing, search for anything containing these
# keywords so you can spot the likely correct suffix-version yourself.
KEYWORD_HINTS = {
    "_BMI5": ["BMI"],
    "_SMOKER3": ["SMOK"],
    "_TOTINDA": ["TOTINDA", "PA1", "EXERANY"],
    "GENHLTH": ["GENHLTH"],
    "_RFDRHV9": ["RFDRHV", "DRNK"],
}


def main(file_path: str):
    print(f"Reading metadata only from {file_path} (fast -- not loading full data)...")
    _, meta = pyreadstat.read_xport(file_path, metadataonly=True, encoding="ISO-8859-1")
    all_columns = set(meta.column_names)
    print(f"File has {len(all_columns)} total columns.\n")

    all_found = True
    for expected in EXPECTED_NEW_COLUMNS:
        if expected in all_columns:
            print(f"  OK   {expected}")
        else:
            all_found = False
            print(f"  MISSING   {expected}")
            candidates = sorted(
                c for c in all_columns
                if any(kw in c.upper() for kw in KEYWORD_HINTS.get(expected, []))
            )
            if candidates:
                print(f"           Possible alternatives found in this file: {candidates}")
            else:
                print("           No similarly-named columns found -- this risk factor may not be in this release.")

    print()
    if all_found:
        print("All expected columns found. Safe to run train_model.py as-is.")
    else:
        print(
            "Some columns were missing. Update the matching name(s) in both "
            "src/data_ingestion.py's target_cols and src/data_hygiene.py's "
            "cleaning logic before running train_model.py, or that risk "
            "factor will simply be silently excluded (data_ingestion.py "
            "logs a warning for this, but it's easy to miss)."
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_brfss_columns.py <path_to_LLCP2024.XPT>")
        sys.exit(1)
    main(sys.argv[1])
