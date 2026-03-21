from __future__ import annotations

from pathlib import Path
import polars as pl

from config import CONFIG, PipelineConfig
from bts_source import process_bts_month, write_bts_parquet, get_months_for_year
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
        ensure_dir(self.cfg.final_out_dir)
        ensure_dir(self.cfg.feature_out_dir)

    def run(self) -> None:
        all_years_full: list[pl.DataFrame] = []

        for year in self.cfg.years:
            print(f"\n========== YEAR {year} ==========")
            months = get_months_for_year(year, self.cfg.months_by_year)

            weather_df = self._get_weather_for_year(year)

            monthly_joined: list[pl.DataFrame] = []
            download_records: list[tuple[Path, Path]] = []

            for month in months:
                print(f"\n----- Processing {year}-{month:02d} -----")

                monthly_joined_path = self.cfg.final_out_dir / f"bts_weather_{year}_{month:02d}.parquet"
                monthly_filtered_path = self.cfg.final_out_dir / f"bts_weather_filtered_{year}_{month:02d}.parquet"
                monthly_bts_path = self.cfg.bts.out_dir / f"bts_filtered_{year}_{month:02d}.parquet"

                if self.cfg.use_cached_monthly_joined and monthly_joined_path.exists():
                    print(f"Loading cached monthly joined -> {monthly_joined_path}")
                    joined_df = pl.read_parquet(monthly_joined_path)
                    print(f"Cached monthly joined rows: {joined_df.height:,}")

                    if self.cfg.postprocess.enabled and self.cfg.postprocess.write_filtered_monthly:
                        maybe_write_filtered(
                            joined_df,
                            self.cfg.postprocess,
                            monthly_filtered_path,
                            dataset_name=f"monthly_filtered_{year}_{month:02d}",
                        )

                    monthly_joined.append(joined_df)
                    continue

                bts_df, download_record = self._get_bts_for_month(year, month, monthly_bts_path)
                if download_record is not None:
                    download_records.append(download_record)

                print(f"BTS rows after extract/filter/timestamp build: {bts_df.height:,}")

                if bts_df.height == 0:
                    print(f"No BTS rows for {year}-{month:02d} after route filter/timestamp build.")
                    continue

                if not self.cfg.run_join_stage:
                    print("Skipping join stage per config.")
                    continue

                bts_df = add_station_keys_to_bts(
                    bts_df,
                    airport_to_station=self.cfg.airport_to_station,
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

            if monthly_joined and self.cfg.write_yearly_joined:
                df_year = pl.concat(monthly_joined, how="vertical_relaxed")
                yearly_out = self.cfg.final_out_dir / f"bts_weather_{year}.parquet"
                df_year.write_parquet(yearly_out)

                print(f"\nWrote yearly joined -> {yearly_out}")
                print(f"Year {year} final rows: {df_year.height:,}")

                all_years_full.append(df_year)

                if self.cfg.postprocess.enabled and self.cfg.postprocess.write_filtered_yearly:
                    yearly_filtered_out = self.cfg.final_out_dir / f"bts_weather_filtered_{year}.parquet"
                    maybe_write_filtered(
                        df_year,
                        self.cfg.postprocess,
                        yearly_filtered_out,
                        dataset_name=f"yearly_filtered_{year}",
                    )

                if self.cfg.run_feature_stage and self.cfg.features.enabled:
                    print(f"\nBuilding derived feature/network tables for {year}...")
                    build_all_feature_tables(
                        flights_df=df_year,
                        joins=self.cfg.joins,
                        cfg=self.cfg.features,
                        out_dir=self.cfg.feature_out_dir,
                        year_label=str(year),
                    )

                if (
                    self.cfg.bts.cleanup_downloads_if_final_has_data
                    and df_year.height > 0
                ):
                    for zip_path, extract_dir in download_records:
                        cleanup_path(zip_path)
                        cleanup_path(extract_dir)
                        print(f"Cleaned -> {zip_path} and {extract_dir}")

        self._write_all_years_outputs(all_years_full)

    def _write_all_years_outputs(self, yearly_dfs: list[pl.DataFrame]) -> None:
        if not yearly_dfs:
            print("No yearly datasets available to combine.")
            return

        all_years_df = pl.concat(yearly_dfs, how="vertical_relaxed")
        print(f"\nCombined all-years rows: {all_years_df.height:,}")
        print(f"Combined all-years cols: {len(all_years_df.columns)}")

        if self.cfg.postprocess.write_all_years_full:
            all_years_full_path = self.cfg.final_out_dir / "bts_weather_all_years.parquet"
            all_years_df.write_parquet(all_years_full_path)
            print(f"Wrote all-years full -> {all_years_full_path}")

        if self.cfg.postprocess.enabled and self.cfg.postprocess.write_all_years_filtered:
            all_years_filtered_path = self.cfg.final_out_dir / "bts_weather_filtered_all_years.parquet"
            maybe_write_filtered(
                all_years_df,
                self.cfg.postprocess,
                all_years_filtered_path,
                dataset_name="all_years_filtered",
            )

    def _get_weather_for_year(self, year: int) -> pl.DataFrame:
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

        stations = sorted(set(self.cfg.airport_to_station.values()))
        print(f"Pulling weather for year={year}, stations={stations}")

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

    def _get_bts_for_month(
        self,
        year: int,
        month: int,
        monthly_bts_path: Path,
    ) -> tuple[pl.DataFrame, tuple[Path, Path] | None]:
        if self.cfg.use_cached_bts_months and monthly_bts_path.exists():
            print(f"Loading cached BTS month -> {monthly_bts_path}")
            bts_df = pl.read_parquet(monthly_bts_path)
            return bts_df, None

        if not self.cfg.run_bts_stage:
            raise RuntimeError(
                f"BTS stage disabled, but cached BTS month not found: {monthly_bts_path}"
            )

        bts_df, download_record = process_bts_month(
            year=year,
            month=month,
            bts_cfg=self.cfg.bts,
            route_cfg=self.cfg.route_filter,
            joins=self.cfg.joins,
        )

        write_bts_parquet(bts_df, monthly_bts_path)
        return bts_df, download_record


if __name__ == "__main__":
    pipeline = BTSWeatherNetworkPipeline(CONFIG)
    pipeline.run()