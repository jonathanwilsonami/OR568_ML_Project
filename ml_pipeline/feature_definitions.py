from __future__ import annotations

TARGET_CLASS = "arr_del15"
TARGET_REG = "arr_delay"
TIME_COL = "dep_ts_actual_utc"
YEAR_COL = "model_year"


# ============================================================
# AIRCRAFT_FEATURES
# ============================================================

AIRCRAFT_FEATURES = [
    "aircraft_year_mfr",
    "aircraft_no_eng",
    "aircraft_no_seats",
    "aircraft_age",
    # Uncomment if you want categorical ones (need encoding first):
    # "aircraft_type_acft",
    # "aircraft_ac_weight",
    # "aircraft_model",
    # "aircraft_mfr",
]

# XGBoost uses flat features
XGB_FEATURE_SETS = {
    # ============================================================
    # Schedule-only baseline
    # ============================================================
    "xgb_schedule": [
        "distance",
        "dep_hour_local",
        "dep_weekday_local",
        "dep_month_local",
        "dep_time_bucket",
        "is_weekend",
        "is_holiday",
        "days_to_nearest_holiday",
        "crs_elapsed_time",
    ],

    # ============================================================
    # Schedule + weather + traffic
    # ============================================================
    "xgb_context": [
        "distance",
        "dep_hour_local",
        "dep_weekday_local",
        "dep_month_local",
        "dep_time_bucket",
        "is_weekend",
        "is_holiday",
        "days_to_nearest_holiday",
        "crs_elapsed_time",

        "dep_temp_c",
        "dep_wind_speed_m_s",
        "dep_wind_dir_deg",
        "dep_ceiling_height_m",
        "arr_temp_c",
        "arr_wind_speed_m_s",
        "arr_wind_dir_deg",
        "arr_ceiling_height_m",

        "route_frequency",
        "origin_flight_volume",
        "dest_flight_volume",
    ],

    # ============================================================
    # Conservative propagation model
    # ============================================================
    "xgb_2hop_propagation": [
        "distance",
        "dep_hour_local",
        "dep_weekday_local",
        "dep_month_local",
        "dep_time_bucket",
        "is_weekend",
        "is_holiday",
        "days_to_nearest_holiday",
        "crs_elapsed_time",

        "dep_temp_c",
        "dep_wind_speed_m_s",
        "dep_wind_dir_deg",
        "dep_ceiling_height_m",
        "arr_temp_c",
        "arr_wind_speed_m_s",
        "arr_wind_dir_deg",
        "arr_ceiling_height_m",

        "route_frequency",
        "origin_flight_volume",
        "dest_flight_volume",

        "prev1_arr_delay",
        "prev1_dep_delay",
        "prev1_arr_del15",
        "prev1_dep_del15",
        "prev2_arr_delay",
        "prev2_dep_delay",
        "prev2_arr_del15",
        "prev2_dep_del15",
        "prev1_turnaround_minutes",
        "time_since_prev2_arrival_minutes",
    ],

    # ============================================================
    # 2-hop delay-only propagation
    # ============================================================
    "xgb_2hop_delay_only": [
        "distance",
        "dep_hour_local",
        "dep_weekday_local",
        "dep_month_local",
        "dep_time_bucket",
        "is_weekend",
        "is_holiday",
        "days_to_nearest_holiday",
        "crs_elapsed_time",

        "prev1_arr_delay",
        "prev1_dep_delay",
        "prev1_arr_del15",
        "prev1_dep_del15",
        "prev2_arr_delay",
        "prev2_dep_delay",
        "prev2_arr_del15",
        "prev2_dep_del15",
        "prev1_turnaround_minutes",
        "time_since_prev2_arrival_minutes",
    ],

    # ============================================================
    # Aircraft state / operational context
    # ============================================================
    "xgb_aircraft_state": [
        "distance",
        "dep_hour_local",
        "dep_weekday_local",
        "dep_month_local",
        "dep_time_bucket",
        "is_weekend",
        "is_holiday",
        "days_to_nearest_holiday",
        "crs_elapsed_time",

        "route_frequency",
        "origin_flight_volume",
        "dest_flight_volume",

        "aircraft_leg_number_day",
        "relative_leg_position",
        "cum_dep_delay_aircraft_day",
        "cum_arr_delay_aircraft_day",
        "rotation_continuity_flag",
        "turnaround_minutes",
        "tight_turnaround_flag",
    ],

    # ============================================================
    # Full operations model
    # ============================================================
    "xgb_ops_full": [
        "distance",
        "dep_hour_local",
        "dep_weekday_local",
        "dep_month_local",
        "dep_time_bucket",
        "is_weekend",
        "is_holiday",
        "days_to_nearest_holiday",
        "crs_elapsed_time",

        "dep_temp_c",
        "dep_wind_speed_m_s",
        "dep_wind_dir_deg",
        "dep_ceiling_height_m",
        "arr_temp_c",
        "arr_wind_speed_m_s",
        "arr_wind_dir_deg",
        "arr_ceiling_height_m",

        "route_frequency",
        "origin_flight_volume",
        "dest_flight_volume",

        "prev_arr_delay",
        "prev_dep_delay",
        "prev_arr_del15",
        "prev_dep_del15",
        "prev_arr_delayed_flag",
        "prev_total_delay",

        "turnaround_minutes",
        "tight_turnaround_flag",
        "rotation_continuity_flag",
        "aircraft_leg_number_day",
        "relative_leg_position",
        "cum_dep_delay_aircraft_day",
        "cum_arr_delay_aircraft_day",

        "prev1_arr_delay",
        "prev1_dep_delay",
        "prev1_arr_del15",
        "prev1_dep_del15",
        "prev2_arr_delay",
        "prev2_dep_delay",
        "prev2_arr_del15",
        "prev2_dep_del15",
        "prev1_turnaround_minutes",
        "time_since_prev2_arrival_minutes",
    ],

    # ============================================================
    # Backward-compatible alias
    # ============================================================
    "xgb_full": [
        # -------------------------
        # Schedule / Calendar
        # -------------------------
        "distance",
        "dep_hour_local",
        "dep_weekday_local",
        "dep_month_local",
        "dep_time_bucket",
        "is_weekend",
        "is_holiday",
        "days_to_nearest_holiday",
        "crs_elapsed_time",

        # -------------------------
        # Weather / Context
        # -------------------------
        "dep_temp_c",
        "dep_wind_speed_m_s",
        "dep_wind_dir_deg",
        "dep_ceiling_height_m",

        "arr_temp_c",
        "arr_wind_speed_m_s",
        "arr_wind_dir_deg",
        "arr_ceiling_height_m",

        # -------------------------
        # Traffic / Network Demand
        # -------------------------
        "route_frequency",
        "origin_flight_volume",
        "dest_flight_volume",

        # -------------------------
        # Aircraft State / Ops
        # -------------------------
        "aircraft_leg_number_day",
        "relative_leg_position",
        "rotation_continuity_flag",

        # -------------------------
        # Conservative 2-Hop Propagation
        # -------------------------
        "prev1_arr_delay",
        "prev1_dep_delay",
        "prev1_arr_del15",
        "prev1_dep_del15",

        "prev2_arr_delay",
        "prev2_dep_delay",
        "prev2_arr_del15",
        "prev2_dep_del15",

        "prev1_turnaround_minutes",
        "time_since_prev2_arrival_minutes",

        # ============================================================
        # LOW LIVE-USABILITY FEATURES
        # ============================================================

        # Immediate prior realized delay often too close to departure
        # "prev_arr_delay",
        # "prev_dep_delay",
        # "prev_arr_del15",
        # "prev_dep_del15",
        # "prev_arr_delayed_flag",
        # "prev_total_delay",

        # Exact realized turnaround often unavailable hours before flight
        # "turnaround_minutes",
        # "tight_turnaround_flag",

        # Dangerous if not strictly computed only from completed legs
        # "cum_dep_delay_aircraft_day",
        # "cum_arr_delay_aircraft_day",
    ],
}

# XGB_FEATURE_SETS["xgb_full_aircraft"] = AIRCRAFT_FEATURES + XGB_FEATURE_SETS["xgb_schedule"]
XGB_FEATURE_SETS["xgb_full_aircraft"] = AIRCRAFT_FEATURES + XGB_FEATURE_SETS["xgb_full"]

LSTM_CONTEXT_CURRENT = [
    "distance",
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",
    "dep_time_bucket",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
    "crs_elapsed_time",
    "dep_temp_c",
    "dep_wind_speed_m_s",
    "dep_wind_dir_deg",
    "dep_ceiling_height_m",
    "arr_temp_c",
    "arr_wind_speed_m_s",
    "arr_wind_dir_deg",
    "arr_ceiling_height_m",
    "route_frequency",
    "origin_flight_volume",
    "dest_flight_volume",
    "tight_turnaround_flag",
    "relative_leg_position",
    "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",
]

LSTM_DELAY_ONLY_CURRENT = [
    "distance",
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",
    "dep_time_bucket",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
    "crs_elapsed_time",
]

STEP_FEATURES = [
    "arr_delay_prev",
    "dep_delay_prev",
    "arr_del15_prev",
    "dep_del15_prev",
    "gap_minutes",
    "distance",
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",
    "dep_time_bucket",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
    "crs_elapsed_time",
    "dep_temp_c",
    "dep_wind_speed_m_s",
    "dep_wind_dir_deg",
    "dep_ceiling_height_m",
    "arr_temp_c",
    "arr_wind_speed_m_s",
    "arr_wind_dir_deg",
    "arr_ceiling_height_m",
    "route_frequency",
    "origin_flight_volume",
    "dest_flight_volume",
    "tight_turnaround_flag",
    "relative_leg_position",
    "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",
]