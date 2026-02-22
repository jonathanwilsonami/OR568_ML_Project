

# ------------------------------------------------------------
# Aircraft Registry Lookup (Single or Multiple ICAO24)
# ------------------------------------------------------------
# Usage examples:
#
# lookup_aircraft_registry(
#     icao24_codes="a4bb8a",
#     registry_csv="faa_registry.csv",
#     output_csv="lookup.csv"
# )
#
# lookup_aircraft_registry(
#     icao24_codes=["a4bb8a","ab12cd","a3ff90"],
#     registry_csv="faa_registry.csv",
#     output_csv="lookup_list.csv"
# )
# ------------------------------------------------------------

import pandas as pd
from typing import Union, List


# -----------------------------
# Fields to return
# -----------------------------
RETURN_COLUMNS = [
    "icao24",
    "n_number",
    "aircraft_code",
    "aircraft_manufacturer",
    "aircraft_model",
    "aircraft_type",
    "aircraft_category",
    "num_engines",
    "num_seats",
    "cruising_speed_mph",
    "manufacturing_year",
    "registrant_type",
    "registrant_name",
    "registrant_city",
    "registrant_state",
    "registrant_country",
]


# -----------------------------
# Main lookup function
# -----------------------------
def lookup_aircraft_registry(
    icao24_codes: Union[str, List[str]],
    registry_csv: str,
    output_csv: str = "aircraft_lookup_results.csv",
) -> str:
    """
    Lookup aircraft registry info using one or multiple ICAO24 codes.

    Parameters
    ----------
    icao24_codes : str OR list[str]
        Single ICAO24 or list of ICAO24 values.
    registry_csv : str
        Path to FAA registry CSV.
    output_csv : str
        Output CSV file.

    Returns
    -------
    str : path to output CSV
    """

    # Normalize input to list
    if isinstance(icao24_codes, str):
        icao24_codes = [icao24_codes]

    # Normalize codes
    codes = (
        pd.Series(icao24_codes)
        .astype(str)
        .str.lower()
        .str.strip()
        .dropna()
        .unique()
    )

    print(f"Looking up {len(codes)} ICAO24 codes...")

    # Read registry
    df = pd.read_csv(registry_csv)

    # Normalize registry column
    df["icao24"] = (
        df["icao24"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Filter
    result = df[df["icao24"].isin(codes)]

    # Keep only requested columns (if present)
    cols_existing = [c for c in RETURN_COLUMNS if c in result.columns]
    result = result[cols_existing]

    # Save output
    result.to_csv(output_csv, index=False)

    print(f"Matches found: {len(result)}")
    print(f"Saved → {output_csv}")

    return output_csv


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    ICAO_LIST = [
        "abbe71",
        "abd19e",
        "a97d63",
        "ac508b",
        "a4bb8a"
    ]
    
    lookup_aircraft_registry(
        icao24_codes=ICAO_LIST,
        registry_csv="/home/jon/Documents/grad_school/OR568/project/OR568_ML_Project/data_pipeline/raw_data/faa_aircraft_registry.csv",
        output_csv="lookup_results.csv",
    )
