from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from config import CONFIG
from pipeline_main import FullNetworkPipeline


def make_single_year_config(year: int):
    cfg = deepcopy(CONFIG)

    # Run only one year at a time
    cfg.years = [year]
    cfg.months_by_year = {
        year: list(range(1, 13)),
    }

    # Keep the same full-network behavior
    cfg.route_filter.airports = None
    cfg.route_filter.airport_pairs = None
    cfg.route_filter.origin_filter = None
    cfg.route_filter.dest_filter = None

    # Only build the flights canonical output
    cfg.features.enabled = True
    cfg.features.write_flights_canonical = True

    cfg.features.build_aircraft_rotation_features = False
    cfg.features.build_airport_time_features = False
    cfg.features.build_route_time_features = False
    cfg.features.build_propagation_chain_features = False

    cfg.features.write_aircraft_rotation = False
    cfg.features.write_airport_time = False
    cfg.features.write_route_time = False
    cfg.features.write_propagation_chains = False

    # Do not write extra postprocess products
    cfg.postprocess.enabled = False
    cfg.postprocess.write_filtered_monthly = False
    cfg.postprocess.write_filtered_yearly = False
    cfg.postprocess.write_filtered_all_years = False
    cfg.postprocess.write_all_years_full = False
    cfg.postprocess.write_all_years_filtered = False

    # Keep BTS monthly rebuild on unless you already cached them
    cfg.run_bts_stage = True
    cfg.use_cached_bts_months = True

    # Reuse 2019 reference mapping so canonical structure stays consistent
    # These paths are already 2019-based in your config
    cfg.run_reference_stage = False

    # Weather:
    # If cached weather for a year exists, it will be used.
    # If not, the pipeline will build it.
    cfg.use_cached_weather = True
    cfg.run_weather_stage = True

    # Joined outputs:
    # Keep monthly false to save disk.
    # Keep yearly true because the feature builder currently consumes the full joined DF in memory anyway.
    cfg.use_cached_monthly_joined = False
    cfg.write_monthly_joined = False
    cfg.write_yearly_joined = False

    # Ensure directories exist as expected
    cfg.final_out_dir = Path("data/final")
    cfg.feature_out_dir = Path("data/features")
    cfg.market_out_dir = Path("data/markets")

    return cfg


def run_year(year: int):
    print(f"\n================ RUNNING YEAR {year} ================\n")
    cfg = make_single_year_config(year)
    pipeline = FullNetworkPipeline(cfg)
    pipeline.run()

    out_path = cfg.feature_out_dir / f"flights_canonical_{year}.parquet"
    if out_path.exists():
        print(f"\nSUCCESS: wrote {out_path}\n")
    else:
        print(f"\nWARNING: expected output not found: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Run canonical flights pipeline one year at a time.")
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to run, e.g. 2015",
    )
    parser.add_argument(
        "--all-missing",
        action="store_true",
        help="Run all years from 2015 to 2025 except years whose flights_canonical_yyyy.parquet already exist.",
    )
    args = parser.parse_args()

    if args.year is not None:
        run_year(args.year)
        return

    if args.all_missing:
        target_years = list(range(2015, 2026))
        feature_dir = Path("data/features")

        for year in target_years:
            out_path = feature_dir / f"flights_canonical_{year}.parquet"

            if year == 2019 and out_path.exists():
                print(f"Skipping {year}: canonical already exists -> {out_path}")
                continue

            if out_path.exists():
                print(f"Skipping {year}: canonical already exists -> {out_path}")
                continue

            run_year(year)
        return

    raise SystemExit("Provide either --year YYYY or --all-missing")


if __name__ == "__main__":
    main()