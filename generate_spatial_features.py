import os
import sys
import logging
import pandas as pd
from sodapy import Socrata
from config.logging_config import setup_logger
from src.data_hygiene import clean_and_pivot_places_api

logger = logging.getLogger(__name__)

# CDC PLACES: Local Data for Better Health, County Data, 2024 release.
# Used for the *state-level* aggregate (data/aggregated_spatial_sdoh.csv)
# that train_model.py joins to BRFSS and Tab 2's map reads. County-level
# rows carry a real, authoritative 'stateabbr' field, so state averages
# built from them don't depend on any ZIP-prefix approximation.
PLACES_COUNTY_DATASET_ID = "fu4u-a9bh"

# CDC PLACES: Local Data for Better Health, ZCTA Data, 2024 release.
# Used ONLY for the ZIP-code lookup table (data/zcta_level_sdoh.csv).
# IMPORTANT: an earlier version of this script used fu4u-a9bh (the COUNTY
# release above) for this table too, on the mistaken assumption it was the
# ZCTA release -- both are 5-digit numeric IDs, so a wrong-but-plausible
# lookup went undetected until manual spot-checking of ZIP 21201
# (Baltimore, MD) returned Kentucky's regional stats (21201 also happens to
# be a valid Kentucky county FIPS code). Verified against CDC's own dataset
# metadata (https://data.cdc.gov/api/views/<id>.json) before fixing. Unlike
# the county release, this one does NOT include a 'stateabbr' field, so
# resolve_state() below falls back to the approximate ZIP-prefix table for
# this pull specifically.
PLACES_ZCTA_DATASET_ID = "4r2x-hcfq"

# 2-digit ZIP-prefix -> state fallback. THIS IS AN APPROXIMATION and is only
# used if the PLACES response doesn't already include a usable state field
# and no verified crosswalk file is supplied (see load_zcta_state_crosswalk).
# Known limitations: some prefixes span multiple states/territories at the
# 3-digit level, and this table does not attempt to resolve those splits.
_ZIP2_STATE_FALLBACK = {
    '00': 'PR',  # NOTE: 00 also covers parts of NY (Holtsville), Guam, VI, AS -
                 # genuinely ambiguous at 2-digit resolution; flagged in logs.
    '01': 'MA', '02': 'MA', '03': 'NH', '04': 'ME', '05': 'VT', '06': 'CT', '07': 'NJ', '08': 'NJ', '09': 'NJ',
    '10': 'NY', '11': 'NY', '12': 'NY', '13': 'NY', '14': 'NY', '15': 'PA', '16': 'PA', '17': 'PA', '18': 'PA', '19': 'PA',
    '20': 'MD', '21': 'MD', '22': 'VA', '23': 'VA', '24': 'WV', '25': 'WV', '26': 'WV', '27': 'NC', '28': 'NC', '29': 'NC',
    '30': 'GA', '31': 'GA', '32': 'FL', '33': 'FL', '34': 'FL', '35': 'AL', '36': 'AL', '37': 'TN', '38': 'TN', '39': 'MS',
    '40': 'KY', '41': 'KY', '42': 'KY', '43': 'OH', '44': 'OH', '45': 'OH', '46': 'IN', '47': 'IN', '48': 'MI', '49': 'MI',
    '50': 'IA', '51': 'IA', '52': 'IA', '53': 'WI', '54': 'WI', '55': 'MN', '56': 'MN', '57': 'SD', '58': 'ND', '59': 'MT',
    '60': 'IL', '61': 'IL', '62': 'IL', '63': 'MO', '64': 'MO', '65': 'MO', '66': 'KS', '67': 'KS', '68': 'NE', '69': 'NE',
    '70': 'LA', '71': 'LA', '72': 'AR', '73': 'OK', '74': 'OK', '75': 'TX', '76': 'TX', '77': 'TX', '78': 'TX', '79': 'TX',
    '80': 'CO', '81': 'CO', '82': 'WY', '83': 'ID', '84': 'UT', '85': 'AZ', '86': 'AZ', '87': 'NM', '88': 'NM', '89': 'NV',
    '90': 'CA', '91': 'CA', '92': 'CA', '93': 'CA', '94': 'CA', '95': 'CA', '96': 'HI', '97': 'OR', '98': 'WA', '99': 'AK',
}


def load_zcta_state_crosswalk(path: str = "data/zcta_state_crosswalk.csv") -> pd.DataFrame | None:
    """Loads a verified ZCTA->state crosswalk if present. Expected columns:
    zcta (5-digit string), stateabbr. Returns None if not found.
    """
    if os.path.exists(path):
        cw = pd.read_csv(path, dtype={"zcta": str})
        cw["zcta"] = cw["zcta"].str.zfill(5)
        logger.info("Loaded verified ZCTA-state crosswalk with %d entries from %s", len(cw), path)
        return cw
    return None


def resolve_state(pivoted_places: pd.DataFrame) -> pd.DataFrame:
    """Attaches a 'stateabbr' column using the best available source, in
    order of trust:
      1. A 'stateabbr' field already present in the raw PLACES response
         (some Socrata releases include this even at ZCTA level -- check
         first rather than assume it's absent).
      2. A verified external crosswalk file (data/zcta_state_crosswalk.csv).
      3. The approximate 2-digit ZIP-prefix fallback table, which leaves
         unresolved ZCTAs as NaN rather than mislabeling them.
    """
    if "stateabbr" in pivoted_places.columns and pivoted_places["stateabbr"].notna().any():
        logger.info("PLACES response already includes a 'stateabbr' field; using it directly.")
        return pivoted_places

    crosswalk = load_zcta_state_crosswalk()
    if crosswalk is not None:
        merged = pivoted_places.merge(crosswalk.rename(columns={"zcta": "ZCTA"}), on="ZCTA", how="left")
        unmatched = merged["stateabbr"].isna().sum()
        if unmatched:
            logger.warning("%d ZCTAs had no state match in the crosswalk", unmatched)
        return merged

    logger.warning(
        "No 'stateabbr' field in the PLACES response and no verified crosswalk "
        "found at data/zcta_state_crosswalk.csv. Falling back to an approximate "
        "2-digit ZIP-prefix mapping. This WILL misassign some ZCTAs, especially "
        "near state borders and in PR/GU/VI/AS. Recommend building a real "
        "crosswalk (e.g. from the Census Bureau's ZCTA relationship files) "
        "before treating outputs as authoritative."
    )

    def zip_to_state_prefix(zip_str):
        z = str(zip_str).zfill(5)[:2]
        return _ZIP2_STATE_FALLBACK.get(z, None)

    out = pivoted_places.copy()
    out["stateabbr"] = out["ZCTA"].apply(zip_to_state_prefix)
    unresolved = out["stateabbr"].isna().sum()
    if unresolved:
        logger.warning(
            "%d ZCTAs (%.1f%%) could not be resolved to a state via the "
            "fallback prefix table and were left as NaN.",
            unresolved, 100 * unresolved / len(out),
        )
    return out


def fetch_places(dataset_id: str, select_query: str) -> pd.DataFrame:
    client = Socrata("data.cdc.gov", None)
    results = client.get(dataset_id, select=select_query, limit=500000)
    df = pd.DataFrame.from_records(results)
    if df.empty:
        raise ValueError(f"API payload for dataset {dataset_id} returned an empty dataframe structure.")
    df.columns = df.columns.str.lower()
    return df


def main():
    setup_logger()
    target_features = ["spatial_lack_insurance", "spatial_poor_mental_health"]

    # --- State-level aggregate: county-level PLACES release, which has a
    # real 'stateabbr' per row, joined to BRFSS by train_model.py and
    # rendered on Tab 2's map. ---
    logger.info("Connecting to 2024 CDC PLACES County endpoint [%s]...", PLACES_COUNTY_DATASET_ID)
    try:
        county_df = fetch_places(PLACES_COUNTY_DATASET_ID, "locationid, stateabbr, measureid, data_value")
        logger.info("County-level stream ingested. Raw records processed: %d", len(county_df))
        logger.info("Raw columns returned by Socrata: %s", list(county_df.columns))

        pivoted_county = clean_and_pivot_places_api(county_df)
        pivoted_county = resolve_state(pivoted_county)

        # This county release includes a "US" row (a nationwide rollup, not
        # a real state) alongside the 50 states/DC/territories. Letting it
        # through would put "US" in app.py's state dropdown as if it were a
        # selectable state -- drop it here rather than downstream.
        is_national_rollup = pivoted_county["stateabbr"] == "US"
        if is_national_rollup.any():
            logger.info("Dropping %d national rollup row(s) (stateabbr='US') from the state-level aggregate.", is_national_rollup.sum())
            pivoted_county = pivoted_county[~is_national_rollup]

        spatial_sdoh_matrix = (
            pivoted_county.dropna(subset=["stateabbr"])
            .groupby("stateabbr")[target_features]
            .mean()
            .reset_index()
        )

        # NOTE: no MSCODE / urbanicity column is produced here. PLACES has
        # no concept of metropolitan status, and BRFSS carries its own
        # per-respondent MSCODE field -- reconstructing an urbanicity proxy
        # from county-level insurance/mental-health data (as an earlier
        # version of this script did) would define "rural" as "highest
        # uninsurance quartile," which manufactures the exact correlation it
        # would then appear to discover. Note that train_model.py does NOT
        # actually use BRFSS's MSCODE as a model feature either -- it's ~75%
        # missing in the real 2024 extract, too suppressed to trust (see
        # train_model.py's RAW_TO_FEATURE_RENAME comment).

        os.makedirs("data", exist_ok=True)
        output_path = "data/aggregated_spatial_sdoh.csv"
        spatial_sdoh_matrix.to_csv(output_path, index=False)

        logger.info("Nationwide regional features saved! Matrix dimensions: %s", spatial_sdoh_matrix.shape)
        print(f"Unique States Extracted: {spatial_sdoh_matrix['stateabbr'].dropna().unique().tolist()}")
    except Exception as e:
        logger.error("Failed to compile state-level spatial matrix: %s", str(e))

    # --- ZCTA-level table: the actual ZCTA release, used ONLY for app.py's
    # ZIP-code lookup feature (src.data_ingestion.merge_spatial_vectors).
    # Not used by train_model.py -- BRFSS has no individual ZIP to join
    # against. This release has no 'stateabbr' field, so resolve_state()
    # falls back to the approximate ZIP-prefix table here. ---
    logger.info("Connecting to 2024 CDC PLACES ZCTA endpoint [%s]...", PLACES_ZCTA_DATASET_ID)
    try:
        zcta_df = fetch_places(PLACES_ZCTA_DATASET_ID, "locationid, measureid, data_value")
        logger.info("ZCTA-level stream ingested. Raw records processed: %d", len(zcta_df))
        logger.info("Raw columns returned by Socrata: %s", list(zcta_df.columns))

        pivoted_zcta = clean_and_pivot_places_api(zcta_df)
        pivoted_zcta = resolve_state(pivoted_zcta)

        zcta_output_path = "data/zcta_level_sdoh.csv"
        pivoted_zcta[["ZCTA", "stateabbr"] + target_features].to_csv(zcta_output_path, index=False)
        logger.info("Saved ZCTA-level table (for ZIP lookup) to %s", zcta_output_path)
    except Exception as e:
        logger.error("Failed to compile ZCTA-level spatial matrix: %s", str(e))


if __name__ == "__main__":
    main()
