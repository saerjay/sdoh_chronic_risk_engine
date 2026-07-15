import re
import logging
import pandas as pd
import pyreadstat
from typing import Any

logger = logging.getLogger(__name__)


def verify_zip_code(zip_code: Any) -> bool:
    """Validates if a user-provided input is a structurally sound 5-digit US ZIP code."""
    if not isinstance(zip_code, str):
        return False
    return bool(re.match(r"^\d{5}$", zip_code.strip()))


def merge_spatial_vectors(places_df: pd.DataFrame, zip_code: str) -> pd.Series:
    """Locates and extracts SDOH features from CDC PLACES data using a ZCTA key.

    NOTE: this looks up a single ZIP against ZCTA-level PLACES data. It has
    no relationship to BRFSS individual records -- load_brfss_transport()
    below does not (and cannot, from the public national extract) return an
    individual's ZIP/ZCTA, only their state (_STATE) and metro status
    (MSCODE). This function is for a separate use case: looking up a
    user-supplied ZIP code directly against the ZCTA-level PLACES table,
    not for joining to BRFSS training data.
    """
    if not verify_zip_code(zip_code):
        raise ValueError(f"Malformed 5-digit ZIP code structure: {zip_code}")
    places_df = places_df.copy()
    places_df["ZCTA"] = places_df["ZCTA"].astype(str).str.strip()
    matched_zone = places_df[places_df["ZCTA"] == zip_code.strip()]
    if matched_zone.empty:
        raise ValueError("Spatial boundary missing from master PLACES index")
    return matched_zone.iloc[0]


def load_brfss_transport(file_path: str) -> pd.DataFrame:
    """
    Ingests individual-level BRFSS transport (.xpt) files.
    Ensures all 13 columns for targets, demographics, and spatial keys are explicitly extracted.
    """
    logger.info(f"Initiating stream ingestion for BRFSS file: {file_path}")
    try:
        target_cols = [
            "DIABETE4", "CVDSTRK3", "ASTHMA3", "CHCCOPD3", "CHCOCNC1", "_MICHD",
            "_AGEG5YR", "SEXVAR", "INCOME3", "_RACEGR3", "EDUCA",
            "_STATE", "MSCODE",
            # Behavioral/clinical risk factors, added to give the model real
            # signal beyond demographics + coarse state-level SDoH averages.
            "_BMI5",       # computed BMI, value = actual BMI * 100
            "_SMOKER3",    # 1=current smoker daily, 2=some days, 3=former, 4=never
            "_TOTINDA",    # 1=had leisure-time physical activity, 2=none
            "GENHLTH",     # 1=excellent ... 5=poor, self-rated general health
            # SLEPTIM1 (sleep hours) is not present in this BRFSS release --
            # confirmed via list_brfss_columns.py, no similarly-named column
            # found either. Not included.
            "_RFDRHV9",    # 1=not a heavy drinker, 2=heavy drinker (suffix version
                           # confirmed via list_brfss_columns.py for this release;
                           # was _RFDRHV8 in earlier years)
        ]

        df, meta = pyreadstat.read_xport(file_path, encoding="ISO-8859-1")

        # Enforce uppercase format for consistency across versions
        df.columns = [c.upper() if c.upper() == "MSCODE" else c for c in df.columns]

        existing_cols = [c for c in target_cols if c in df.columns]
        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            logger.warning(
                "BRFSS extract is missing expected columns: %s. Downstream "
                "cleaning/joining steps that depend on them will be skipped "
                "or will raise explicitly, rather than silently substituting "
                "defaults.", missing_cols,
            )
        df_optimized = df[existing_cols].copy()

        logger.info(f"Successfully loaded BRFSS matrix. Shape: {df_optimized.shape}")
        return df_optimized
    except Exception as e:
        logger.error(f"Error ingesting BRFSS transport file: {str(e)}")
        raise IOError(f"Failed to process BRFSS data stream: {e}")
