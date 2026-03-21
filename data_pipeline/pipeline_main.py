from __future__ import annotations

from pathlib import Path
import json

import polars as pl

from config import CONFIG, PipelineConfig
from bts_source import process_bts_month, write_bts_parquet, get_months_for_year
from airport_station_mapping import (
    extract_unique_airports_from_bts,
    build_airport_to_station_mapping,
)
from weather_source import (
    pull_weather_for_period,
    pull_weather_for_year_chunked,
    build_year_date_window,
)
from joins import add_station_keys_to_bts, join_weather_to_bts
from postprocess import maybe_write_filtered
from feature_engineering import build_all_feature_tables
from utils import ensure_dir, cleanup_path


class BTSWeatherNetworkPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        ensure_dir(self.cfg.bts.out_dir)
        ensure_dir(self.cfg.weather.out_dir)
        ensure_dir(self.cfg.mapping.out_dir)
        ensure_dir(self.cfg.final_out_dir)
        ensure_dir(self.cfg.feature_out_dir)

    def run(self) -> None:
        for year in self.cfg.years:
            print(f"\n========== YEAR {year} ==========")

            months = get_months_for_year(year, self.cfg.months_by_year)

            # Pass 1: BTS monthly extraction
            yearly_bts_df = self._build_or_load_yearly_bts(year, months)

            # Pass 2: airport discovery + mapping
            airport_to_station = self._build_or_load_mapping(year, yearly_bts_df)

            # Pass 3: weather pull using generated mapping
            weather_df = self._get_weather_for_year(year, airport_to_station)

            # Pass 4: join weather to monthly BTS and write joined files
            yearly_joined_df = self._join_months_for_year(
                year=year,
                months=months,
                airport_to_station=airport_to_station,
                weather_df=weather_df,
            )

            # Optional filtered output
            if self.cfg.postprocess.enabled and self.cfg.postprocess.write_filtered_yearly:
                yearly_filtered_out = self.cfg.final_out_dir / f"bts_weather_filtered_{year}.parquet"
                maybe_write_filtered(
                    yearly_joined_df,
                    self.cfg.postprocess,
                    yearly_filtered_out,
                    dataset_name=f"yearly_filtered_{year}",
                )

            # Pass 5: build feature tables
            if self.cfg.run_feature_stage and self.cfg.features.enabled:
                print(f"\nBuilding derived feature/network tables for {year}...")
                build_all_feature_tables(
                    flights_df=yearly_joined_df,
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
                    raise RuntimeError(
                        f"BTS stage disabled, but cached month not found: {monthly_bts_path}"
                    )

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

    def _build_or_load_mapping(
        self,
        year: int,
        yearly_bts_df: pl.DataFrame,
    ) -> dict[str, str]:
        mapping_json = self.cfg.mapping.airport_station_json_path

        if mapping_json.exists():
            print(f"Loading cached airport mapping -> {mapping_json}")
            with open(mapping_json, "r", encoding="utf-8") as f:
                return json.load(f)

        if not self.cfg.run_mapping_stage:
            raise RuntimeError(
                f"Mapping stage disabled, but cached mapping not found: {mapping_json}"
            )

        unique_airports_df = extract_unique_airports_from_bts(
            yearly_bts_df,
            self.cfg.mapping.unique_airports_path,
        )

        airport_to_station, _ = build_airport_to_station_mapping(
            unique_airports_df=unique_airports_df,
            cfg=self.cfg.mapping,
        )

        return airport_to_station

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

        print(f"Weather rows for {year}: {weather_df.height:,}")
        return weather_df

    def _join_months_for_year(
        self,
        year: int,
        months: list[int],
        airport_to_station: dict[str, str],
        weather_df: pl.DataFrame,
    ) -> pl.DataFrame:
        monthly_joined: list[pl.DataFrame] = []

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

            bts_df = add_station_keys_to_bts(
                bts_df,
                airport_to_station=airport_to_station,
                joins=self.cfg.joins,
            )

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

        if self.cfg.postprocess.write_all_years_full:
            all_years_full_path = self.cfg.final_out_dir / "bts_weather_all_years.parquet"
            yearly_joined_df.write_parquet(all_years_full_path)

        return yearly_joined_df


if __name__ == "__main__":
    pipeline = BTSWeatherNetworkPipeline(CONFIG)
    pipeline.run()