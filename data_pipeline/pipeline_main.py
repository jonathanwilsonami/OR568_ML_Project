from __future__ import annotations

import json

import polars as pl

from config import CONFIG, PipelineConfig
from bts_source import process_bts_month, write_bts_parquet, get_months_for_year
from reference_builder import (
    extract_unique_airports_from_bts,
    build_reference_dimensions,
)
from joins import add_station_keys_to_bts, join_weather_to_bts, add_weather_utc_timestamps
from weather_source import (
    pull_weather_for_period,
    pull_weather_for_year_chunked,
    build_year_date_window,
)
from canonical_features import build_all_canonical_feature_tables
from postprocess import maybe_write_filtered
from utils import ensure_dir


class FullNetworkPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        ensure_dir(self.cfg.bts.out_dir)
        ensure_dir(self.cfg.weather.out_dir)
        ensure_dir(self.cfg.reference.out_dir)
        ensure_dir(self.cfg.final_out_dir)
        ensure_dir(self.cfg.feature_out_dir)
        ensure_dir(self.cfg.market_out_dir)

    def run(self) -> None:
        for year in self.cfg.years:
            print(f"\n========== YEAR {year} ==========")
            months = get_months_for_year(year, self.cfg.months_by_year)

            yearly_bts_df = self._build_or_load_yearly_bts(year, months)
            airport_dim, airport_to_station, station_to_timezone = self._build_or_load_reference(year, yearly_bts_df)
            weather_df = self._get_weather_for_year(year, airport_to_station)
            weather_df = add_weather_utc_timestamps(weather_df, station_to_timezone)

            yearly_joined_df = self._join_months_for_year(year, months, airport_to_station, weather_df)

            if self.cfg.postprocess.enabled and self.cfg.postprocess.write_filtered_yearly:
                yearly_filtered_out = self.cfg.final_out_dir / f"bts_weather_filtered_{year}.parquet"
                maybe_write_filtered(
                    yearly_joined_df,
                    self.cfg.postprocess,
                    yearly_filtered_out,
                    dataset_name=f"yearly_filtered_{year}",
                )

            if self.cfg.run_feature_stage and self.cfg.features.enabled:
                print(f"\nBuilding canonical feature tables for {year}...")
                build_all_canonical_feature_tables(
                    flights_joined_df=yearly_joined_df,
                    airport_dim=airport_dim,
                    joins=self.cfg.joins,
                    cfg=self.cfg.features,
                    out_dir=self.cfg.feature_out_dir,
                    year_label=str(year),
                )

    def _build_or_load_yearly_bts(self, year: int, months: list[int]) -> pl.DataFrame:
        yearly_bts_path = self.cfg.final_out_dir / f"bts_only_{year}.parquet"
        monthly_dfs: list[pl.DataFrame] = []

        for month in months:
            monthly_bts_path = self.cfg.bts.out_dir / f"bts_filtered_{year}_{month:02d}.parquet"

            if self.cfg.use_cached_bts_months and monthly_bts_path.exists():
                print(f"Loading cached BTS month -> {monthly_bts_path}")
                bts_df = pl.read_parquet(monthly_bts_path)
            else:
                if not self.cfg.run_bts_stage:
                    raise RuntimeError(f"BTS stage disabled, but cached month not found: {monthly_bts_path}")

                bts_df, _ = process_bts_month(
                    year=year,
                    month=month,
                    bts_cfg=self.cfg.bts,
                    route_cfg=self.cfg.route_filter,
                    joins=self.cfg.joins,
                )
                write_bts_parquet(bts_df, monthly_bts_path)

            monthly_dfs.append(bts_df)

        yearly_bts_df = pl.concat(monthly_dfs, how="vertical_relaxed")
        yearly_bts_df.write_parquet(yearly_bts_path)
        print(f"Wrote yearly BTS-only -> {yearly_bts_path}")
        print(f"Year {year} BTS-only rows: {yearly_bts_df.height:,}")
        return yearly_bts_df

    def _build_or_load_reference(
        self,
        year: int,
        yearly_bts_df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, dict[str, str], dict[str, str]]:
        airport_dim_path = self.cfg.reference.airport_dim_path
        station_dim_path = self.cfg.reference.station_dim_path
        airport_station_json = self.cfg.reference.airport_station_json_path

        if airport_dim_path.exists() and station_dim_path.exists() and airport_station_json.exists():
            print(f"Loading cached airport_dim -> {airport_dim_path}")
            airport_dim = pl.read_parquet(airport_dim_path)
            station_dim = pl.read_parquet(station_dim_path)

            with open(airport_station_json, "r", encoding="utf-8") as f:
                airport_to_station = json.load(f)

            station_to_timezone = {
                row["station"]: row["station_timezone"]
                for row in station_dim.filter(pl.col("station_timezone").is_not_null())
                .select(["station", "station_timezone"]).iter_rows(named=True)
            }

            return airport_dim, airport_to_station, station_to_timezone

        if not self.cfg.run_reference_stage:
            raise RuntimeError("Reference stage disabled, but reference outputs not found.")

        unique_airports_df = extract_unique_airports_from_bts(
            yearly_bts_df,
            self.cfg.reference.unique_airports_path,
        )

        airport_dim, station_dim, _bridge, airport_to_station, _airport_to_timezone, station_to_timezone = build_reference_dimensions(
            unique_airports_df=unique_airports_df,
            cfg=self.cfg.reference,
        )

        return airport_dim, airport_to_station, station_to_timezone

    def _get_weather_for_year(
        self,
        year: int,
        airport_to_station: dict[str, str],
    ) -> pl.DataFrame:
        clean_path = self.cfg.weather.out_dir / f"weather_clean_{year}.parquet"
        raw_path = (
            self.cfg.weather.out_dir / f"weather_raw_{year}.parquet"
            if self.cfg.weather.raw_parquet
            else None
        )

        if self.cfg.use_cached_weather and clean_path.exists():
            print(f"Loading cached weather -> {clean_path}")
            weather_df = pl.read_parquet(clean_path)
            print(f"Cached weather rows for {year}: {weather_df.height:,}")
            return weather_df

        if not self.cfg.run_weather_stage:
            raise RuntimeError(
                f"Weather stage disabled, but cached weather not found for year {year}: {clean_path}"
            )

        stations = sorted(set(airport_to_station.values()))
        print(f"Pulling weather for year={year}, mapped stations={len(stations):,}")

        if self.cfg.weather.chunk_by_month:
            weather_df = pull_weather_for_year_chunked(
                stations=stations,
                year=year,
                cfg=self.cfg.weather,
                raw_output_path=raw_path,
                clean_output_path=clean_path,
            )
        else:
            start, end = build_year_date_window(year)
            weather_df = pull_weather_for_period(
                stations=stations,
                start=start,
                end=end,
                cfg=self.cfg.weather,
                raw_output_path=raw_path,
                clean_output_path=clean_path,
            )

        return weather_df

    def _join_months_for_year(
        self,
        year: int,
        months: list[int],
        airport_to_station: dict[str, str],
        weather_df: pl.DataFrame,
    ) -> pl.DataFrame:
        from canonical_features import join_airport_reference, add_utc_timestamps

        monthly_joined: list[pl.DataFrame] = []

        # Load airport_dim once so we can attach timezones before joining weather
        airport_dim = pl.read_parquet(self.cfg.reference.airport_dim_path)

        for month in months:
            print(f"\n----- Joining {year}-{month:02d} -----")

            monthly_bts_path = self.cfg.bts.out_dir / f"bts_filtered_{year}_{month:02d}.parquet"
            monthly_joined_path = self.cfg.final_out_dir / f"bts_weather_{year}_{month:02d}.parquet"
            monthly_filtered_path = self.cfg.final_out_dir / f"bts_weather_filtered_{year}_{month:02d}.parquet"

            if self.cfg.use_cached_monthly_joined and monthly_joined_path.exists():
                print(f"Loading cached monthly joined -> {monthly_joined_path}")
                joined_df = pl.read_parquet(monthly_joined_path)
                monthly_joined.append(joined_df)
                continue

            bts_df = pl.read_parquet(monthly_bts_path)

            # Attach airport timezone metadata
            bts_df = join_airport_reference(
                flights_df=bts_df,
                airport_dim=airport_dim,
                joins=self.cfg.joins,
            )

            # Build UTC timestamps from local BTS timestamps
            bts_df = add_utc_timestamps(
                df=bts_df,
                joins=self.cfg.joins,
            )

            # Add station keys for weather join
            bts_df = add_station_keys_to_bts(
                bts_df,
                airport_to_station=airport_to_station,
                joins=self.cfg.joins,
            )

            # Sanity check
            for c in [
                self.cfg.joins.dep_ts_col,
                self.cfg.joins.arr_ts_col,
                self.cfg.joins.dep_station_col,
                self.cfg.joins.arr_station_col,
            ]:
                if c not in bts_df.columns:
                    raise KeyError(f"Required join column missing after UTC/station prep: {c}")

                non_null = bts_df.select(pl.col(c).is_not_null().sum()).item()
                print(f"Non-null {c}: {non_null:,}")

            joined_df = join_weather_to_bts(
                bts_df=bts_df,
                weather_df=weather_df,
                joins=self.cfg.joins,
            )

            print(f"Rows after weather join: {joined_df.height:,}")

            if self.cfg.write_monthly_joined:
                joined_df.write_parquet(monthly_joined_path)
                print(f"Wrote monthly joined -> {monthly_joined_path}")

            if self.cfg.postprocess.enabled and self.cfg.postprocess.write_filtered_monthly:
                maybe_write_filtered(
                    joined_df,
                    self.cfg.postprocess,
                    monthly_filtered_path,
                    dataset_name=f"monthly_filtered_{year}_{month:02d}",
                )

            monthly_joined.append(joined_df)

        yearly_joined_df = pl.concat(monthly_joined, how="vertical_relaxed")
        yearly_out = self.cfg.final_out_dir / f"bts_weather_{year}.parquet"
        yearly_joined_df.write_parquet(yearly_out)

        print(f"\nWrote yearly joined -> {yearly_out}")
        print(f"Year {year} final rows: {yearly_joined_df.height:,}")

        return yearly_joined_df


if __name__ == "__main__":
    pipeline = FullNetworkPipeline(CONFIG)
    pipeline.run()