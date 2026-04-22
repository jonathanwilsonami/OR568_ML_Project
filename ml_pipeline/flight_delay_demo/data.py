"""
data.py — AeroFlux
Uses Polars for loading and feature engineering (memory efficient),
converts to slim pandas DataFrame only at the end.
"""
from __future__ import annotations
import re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

_HOLIDAY_STRS = [
    "2019-01-01","2019-01-21","2019-02-18","2019-05-27","2019-07-04","2019-09-02","2019-10-14","2019-11-11","2019-11-28","2019-12-25",
    "2020-01-01","2020-01-20","2020-02-17","2020-05-25","2020-07-03","2020-09-07","2020-10-12","2020-11-11","2020-11-26","2020-12-25",
    "2021-01-01","2021-01-18","2021-02-15","2021-05-31","2021-07-05","2021-09-06","2021-10-11","2021-11-11","2021-11-25","2021-12-24",
    "2022-01-01","2022-01-17","2022-02-21","2022-05-30","2022-07-04","2022-09-05","2022-10-10","2022-11-11","2022-11-24","2022-12-25",
    "2023-01-01","2023-01-16","2023-02-20","2023-05-29","2023-07-04","2023-09-04","2023-10-09","2023-11-10","2023-11-23","2023-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-05-27","2024-07-04","2024-09-02","2024-10-14","2024-11-11","2024-11-28","2024-12-25",
    "2025-01-01","2025-01-20","2025-02-17","2025-05-26","2025-07-04","2025-09-01","2025-10-13","2025-11-11","2025-11-27","2025-12-25",
]
_HOLIDAY_SET   = frozenset(_HOLIDAY_STRS)
_HOLIDAY_DATES = [pl.Series([s]).str.strptime(pl.Date,"%Y-%m-%d")[0] for s in _HOLIDAY_STRS]

XGB_FULL_FEATURES = [
    "distance","dep_hour_local","dep_weekday_local","dep_month_local","dep_time_bucket","is_weekend","is_holiday","days_to_nearest_holiday","crs_elapsed_time",
    "dep_temp_c","dep_wind_speed_m_s","dep_wind_dir_deg","dep_ceiling_height_m","arr_temp_c","arr_wind_speed_m_s","arr_wind_dir_deg","arr_ceiling_height_m",
    "route_frequency","origin_flight_volume","dest_flight_volume",
    "prev_arr_delay","prev_dep_delay","prev_arr_del15","prev_dep_del15","prev_arr_delayed_flag","prev_total_delay",
    "turnaround_minutes","tight_turnaround_flag","rotation_continuity_flag","aircraft_leg_number_day","relative_leg_position",
    "cum_dep_delay_aircraft_day","cum_arr_delay_aircraft_day",
    "prev1_arr_delay","prev1_dep_delay","prev1_arr_del15","prev1_dep_del15",
    "prev2_arr_delay","prev2_dep_delay","prev2_arr_del15","prev2_dep_del15",
    "prev1_turnaround_minutes","time_since_prev2_arrival_minutes",
]
STEP_FEATURES = [
    "arr_delay_prev","dep_delay_prev","arr_del15_prev","dep_del15_prev","gap_minutes","distance","dep_hour_local","dep_weekday_local","dep_month_local",
    "dep_time_bucket","is_weekend","is_holiday","days_to_nearest_holiday","crs_elapsed_time","dep_temp_c","dep_wind_speed_m_s","dep_wind_dir_deg",
    "dep_ceiling_height_m","arr_temp_c","arr_wind_speed_m_s","arr_wind_dir_deg","arr_ceiling_height_m","route_frequency","origin_flight_volume",
    "dest_flight_volume","tight_turnaround_flag","relative_leg_position","cum_dep_delay_aircraft_day","cum_arr_delay_aircraft_day",
]
LSTM_CONTEXT_CURRENT = [
    "distance","dep_hour_local","dep_weekday_local","dep_month_local","dep_time_bucket","is_weekend","is_holiday","days_to_nearest_holiday","crs_elapsed_time",
    "dep_temp_c","dep_wind_speed_m_s","dep_wind_dir_deg","dep_ceiling_height_m","arr_temp_c","arr_wind_speed_m_s","arr_wind_dir_deg","arr_ceiling_height_m",
    "route_frequency","origin_flight_volume","dest_flight_volume","tight_turnaround_flag","relative_leg_position","cum_dep_delay_aircraft_day","cum_arr_delay_aircraft_day",
]
TARGET_CLASS = "arr_del15"
TARGET_REG   = "arr_delay"

_IDENTITY_COLS = [
    "tail_number","flight_id","flight_number_reporting_airline","origin","dest",
    "dep_ts_actual_utc","arr_ts_actual_utc","flight_date","arr_delay","dep_delay",
    "arr_del15","dep_del15","op_carrier","reporting_airline",
    "prev_origin","prev_dest","prev_flight_id_same_tail",
    "distance","crs_elapsed_time","dep_hour_local","dep_weekday_local","dep_month_local",
    "dep_temp_c","dep_wind_speed_m_s","dep_ceiling_height_m",
]

def _clean_col(col):
    col = re.sub(r"[^0-9a-zA-Z]+","_",col)
    col = re.sub(r"([A-Z]+)([A-Z][a-z])",r"\1_\2",col)
    col = re.sub(r"([a-z0-9])([A-Z])",r"\1_\2",col)
    col = re.sub(r"_+","_",col)
    return col.strip("_").lower()

def _parse_date(s):
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%m-%d-%Y","%d/%m/%Y"):
        try: return pd.to_datetime(s,format=fmt).date()
        except: pass
    try: return pd.to_datetime(s).date()
    except: return None

def _engineer_polars(df):
    print(f"  [eng] {df.height:,} rows — Polars engineering …")
    cols = set(df.columns)
    # Calendar
    cal=[]
    if "dep_ts_actual_utc" in cols:
        if "dep_hour_local"    not in cols: cal.append(pl.col("dep_ts_actual_utc").dt.hour().cast(pl.Int8).alias("dep_hour_local"))
        if "dep_weekday_local" not in cols: cal.append((pl.col("dep_ts_actual_utc").dt.weekday()+1).cast(pl.Int8).alias("dep_weekday_local"))
        if "dep_month_local"   not in cols: cal.append(pl.col("dep_ts_actual_utc").dt.month().cast(pl.Int8).alias("dep_month_local"))
        if "flight_date"       not in cols: cal.append(pl.col("dep_ts_actual_utc").dt.date().alias("flight_date"))
    if cal: df=df.with_columns(cal); cols=set(df.columns)
    bkt=[]
    if "dep_hour_local" in cols:
        bkt.append(pl.when(pl.col("dep_hour_local")<6).then(1).when(pl.col("dep_hour_local")<11).then(2).when(pl.col("dep_hour_local")<14).then(3).when(pl.col("dep_hour_local")<18).then(4).when(pl.col("dep_hour_local")<21).then(5).otherwise(6).cast(pl.Int8).alias("dep_time_bucket"))
    if "dep_weekday_local" in cols: bkt.append(pl.col("dep_weekday_local").is_in([6,7]).cast(pl.Int8).alias("is_weekend"))
    if "flight_date" in cols:
        print("  [eng] holiday features …")
        # Ensure flight_date is pl.Date regardless of source type
        if df["flight_date"].dtype != pl.Date:
            df = df.with_columns(pl.col("flight_date").cast(pl.Date))
            cols = set(df.columns)
        bkt.append(pl.col("flight_date").cast(pl.Utf8).is_in(list(_HOLIDAY_SET)).cast(pl.Int8).alias("is_holiday"))
        bkt.append(pl.min_horizontal([(pl.col("flight_date").cast(pl.Date)-pl.lit(h)).dt.total_days().abs() for h in _HOLIDAY_DATES]).cast(pl.Int32).alias("days_to_nearest_holiday"))
    if bkt: df=df.with_columns(bkt); cols=set(df.columns)
    # Traffic
    print("  [eng] traffic volumes …")
    if "origin" in cols and "dest" in cols:
        if "route_key" not in cols: df=df.with_columns((pl.col("origin")+"_"+pl.col("dest")).alias("route_key")); cols.add("route_key")
        vol=[]
        if "route_frequency"      not in cols: vol.append(pl.col("route_key").count().over("route_key").alias("route_frequency"))
        if "origin_flight_volume" not in cols: vol.append(pl.col("origin").count().over("origin").alias("origin_flight_volume"))
        if "dest_flight_volume"   not in cols: vol.append(pl.col("dest").count().over("dest").alias("dest_flight_volume"))
        if vol: df=df.with_columns(vol); cols=set(df.columns)
    # Sort
    sb=[c for c in ("tail_number","dep_ts_actual_utc") if c in cols]
    if sb: print("  [eng] sorting …"); df=df.sort(sb)
    # Lags
    print("  [eng] lag features …")
    lag=[]
    for src,dst,n in [("arr_delay","prev_arr_delay",1),("dep_delay","prev_dep_delay",1),("arr_del15","prev_arr_del15",1),("dep_del15","prev_dep_del15",1),
                      ("arr_delay","prev1_arr_delay",1),("dep_delay","prev1_dep_delay",1),("arr_del15","prev1_arr_del15",1),("dep_del15","prev1_dep_del15",1),
                      ("arr_delay","prev2_arr_delay",2),("dep_delay","prev2_dep_delay",2),("arr_del15","prev2_arr_del15",2),("dep_del15","prev2_dep_del15",2)]:
        if src in cols and dst not in cols:
            lag.append(pl.col(src).shift(n).over("tail_number").alias(dst) if "tail_number" in cols else pl.col(src).shift(n).alias(dst))
    if lag: df=df.with_columns(lag); cols=set(df.columns)
    drv=[]
    if "prev_arr_delay" in cols and "prev_arr_delayed_flag" not in cols: drv.append((pl.col("prev_arr_delay").fill_null(0)>15).cast(pl.Int8).alias("prev_arr_delayed_flag"))
    if "prev_arr_delay" in cols and "prev_dep_delay" in cols and "prev_total_delay" not in cols: drv.append((pl.col("prev_arr_delay").fill_null(0)+pl.col("prev_dep_delay").fill_null(0)).alias("prev_total_delay"))
    if drv: df=df.with_columns(drv); cols=set(df.columns)
    # Turnaround
    if "dep_ts_actual_utc" in cols and "arr_ts_actual_utc" in cols:
        print("  [eng] turnaround …")
        ta=[]
        if "prev1_turnaround_minutes" not in cols: ta.append((pl.col("dep_ts_actual_utc")-pl.col("arr_ts_actual_utc").shift(1).over("tail_number")).dt.total_minutes().cast(pl.Float32).alias("prev1_turnaround_minutes"))
        if "turnaround_minutes"       not in cols: ta.append((pl.col("dep_ts_actual_utc")-pl.col("arr_ts_actual_utc").shift(1).over("tail_number")).dt.total_minutes().cast(pl.Float32).alias("turnaround_minutes"))
        if "time_since_prev2_arrival_minutes" not in cols: ta.append((pl.col("dep_ts_actual_utc")-pl.col("arr_ts_actual_utc").shift(2).over("tail_number")).dt.total_minutes().cast(pl.Float32).alias("time_since_prev2_arrival_minutes"))
        if ta: df=df.with_columns(ta); cols=set(df.columns)
    # Day stats
    if "tail_number" in cols and "flight_date" in cols:
        print("  [eng] per-aircraft-day stats …")
        dy=[]
        if "aircraft_leg_number_day" not in cols: dy.append(pl.cum_count("tail_number").over(["tail_number","flight_date"]).cast(pl.Int8).alias("aircraft_leg_number_day"))
        if "cum_dep_delay_aircraft_day" not in cols and "dep_delay" in cols: dy.append(pl.col("dep_delay").cum_sum().over(["tail_number","flight_date"]).shift(1).fill_null(0).cast(pl.Float32).alias("cum_dep_delay_aircraft_day"))
        if "cum_arr_delay_aircraft_day" not in cols and "arr_delay" in cols: dy.append(pl.col("arr_delay").cum_sum().over(["tail_number","flight_date"]).shift(1).fill_null(0).cast(pl.Float32).alias("cum_arr_delay_aircraft_day"))
        if dy: df=df.with_columns(dy); cols=set(df.columns)
    # Flags
    fl=[]
    if "turnaround_minutes"       in cols and "tight_turnaround_flag"    not in cols: fl.append((pl.col("turnaround_minutes").fill_null(999)<60).cast(pl.Int8).alias("tight_turnaround_flag"))
    if "aircraft_leg_number_day"  in cols and "relative_leg_position"    not in cols: fl.append((pl.col("aircraft_leg_number_day").cast(pl.Float32)/pl.col("aircraft_leg_number_day").max().over("tail_number").cast(pl.Float32).replace(0,1)).alias("relative_leg_position"))
    if "prev_arr_delay"           in cols and "rotation_continuity_flag" not in cols: fl.append((pl.col("prev_arr_delay").fill_null(0)<60).cast(pl.Int8).alias("rotation_continuity_flag"))
    if fl: df=df.with_columns(fl)
    print("  [eng] done.")
    return df

def collect_all_model_features(models_dir):
    import joblib as _jl
    models_dir=Path(models_dir); all_f=set()
    for p in sorted(models_dir.glob("xgb_classifier_*.joblib")):
        try:
            clf=_jl.load(p); names=clf.get_booster().feature_names or []
            all_f.update(names); print(f"  [features] {p.name}: {len(names)} features")
        except Exception as e: print(f"  [features] Could not read {p.name}: {e}")
    return sorted(all_f)

def load_and_engineer(parquet_path, cache_suffix="_cache", models_dir=None):
    src=Path(parquet_path); cache=src.with_name(src.stem+cache_suffix+".parquet")
    if cache.exists() and cache.stat().st_mtime>=src.stat().st_mtime:
        print(f"[cache] Reading {cache.name} …")
        import pyarrow.parquet as pq
        df_pd = pq.read_table(str(cache)).to_pandas(timestamp_as_object=False)
        print(f"[cache] {len(df_pd):,} rows · {df_pd.shape[1]} cols ready.")
        return df_pd
    print(f"[cache] Processing {src.name} with Polars …")
    df=pl.read_parquet(src)
    print(f"[raw]   {df.height:,} rows × {df.width} cols")
    df=df.rename({c:_clean_col(c) for c in df.columns})
    if "is_cancelled" in df.columns: df=df.filter(pl.col("is_cancelled")==0)
    if "is_diverted"  in df.columns: df=df.filter(pl.col("is_diverted")==0)
    if TARGET_CLASS   in df.columns: df=df.filter(pl.col(TARGET_CLASS).is_not_null())
    print(f"[raw]   {df.height:,} rows after filter")
    for col in ("dep_ts_actual_utc","arr_ts_actual_utc"):
        if col in df.columns:
            try: df=df.with_columns(pl.col(col).cast(pl.Datetime("us","UTC")))
            except: pass
    df=_engineer_polars(df)
    extra=collect_all_model_features(models_dir) if models_dir else []
    all_needed=set(XGB_FULL_FEATURES)|set(extra)
    fill=[pl.lit(0.0).cast(pl.Float32).alias(c) for c in all_needed if c not in df.columns]
    if fill: df=df.with_columns(fill)
    for col in all_needed:
        if col in df.columns and df[col].null_count()>0:
            med=df[col].median()
            df=df.with_columns(pl.col(col).fill_null(med if med is not None else 0).cast(pl.Float32))
    keep=[c for c in df.columns if c in (set(_IDENTITY_COLS)|all_needed)]
    print(f"  [slim] Keeping {len(keep)} cols, dropping {df.width-len(keep)} unused")
    df=df.select(keep)
    print(f"[cache] Writing {cache.name} …")
    df.write_parquet(cache,compression="snappy")
    print(f"[cache] Written ({cache.stat().st_size/1e6:.0f} MB) — {df.height:,} rows × {df.width} cols.")
    return df.to_pandas()

class FlightIndex:
    def __init__(self,df):
        self._df=df
        print("[index] Building tail-number index …")
        self.tail_index: dict[str, "np.ndarray"] = {}
        if "tail_number" in df.columns:
            _tmp: dict[str, list] = {}
            for pos,val in enumerate(df["tail_number"].astype(str).str.upper()):
                _tmp.setdefault(val,[]).append(pos)
            self.tail_index = {k: np.array(v, dtype=np.int32) for k,v in _tmp.items()}
            del _tmp
        print("[index] Building flight-number index …")
        self.flight_index: dict[str, "np.ndarray"] = {}
        for col in ("flight_number_reporting_airline","flight_number","fl_num","op_carrier_fl_num"):
            if col in df.columns:
                print(f"[index] Flight number column: '{col}'")
                _tmp2: dict[str, list] = {}
                for pos,val in enumerate(df[col].astype(str).str.strip()):
                    _tmp2.setdefault(val,[]).append(pos)
                    s=val.lstrip("0")
                    if s and s!=val: _tmp2.setdefault(s,[]).append(pos)
                self.flight_index = {k: np.array(v, dtype=np.int32) for k,v in _tmp2.items()}
                del _tmp2
                break
        print(f"[index] {len(self.tail_index):,} tails | {len(self.flight_index):,} flight numbers — ready.")

    def search(self,tail_number=None,date=None,flight_number=None,origin=None,dest=None):
        df=self._df
        if tail_number:
            pos=self.tail_index.get(tail_number.strip().upper(),np.array([],dtype=np.int32))
            if len(pos)==0: return pd.DataFrame()
            sub=df.iloc[pos].copy()
        elif flight_number:
            raw=str(flight_number).strip().split()[-1]
            _p1=self.flight_index.get(raw,None); _p2=self.flight_index.get(raw.lstrip("0"),None)
            pos = _p1 if (_p1 is not None and len(_p1)>0) else _p2
            if pos is None or len(pos)==0: return pd.DataFrame()
            sub=df.iloc[pos].copy()
        else: return pd.DataFrame()
        if date:
            d=_parse_date(date)
            if d is not None and "flight_date" in sub.columns:
                sub=sub[sub["flight_date"].apply(lambda x:x.date() if hasattr(x,"date") else x)==d]
        if origin: sub=sub[sub["origin"].str.upper()==origin.strip().upper()]
        if dest:   sub=sub[sub["dest"].str.upper()==dest.strip().upper()]
        return sub.reset_index(drop=True)

    def get_chain(self,row):
        out={"prev2_origin":"","prev2_dest":"","prev1_origin":"","prev1_dest":""}
        p_orig=str(row.get("prev_origin","") or "")
        p_dest=str(row.get("prev_dest","")   or "")
        if p_orig and p_orig not in ("nan","None",""):
            out["prev1_origin"]=p_orig; out["prev1_dest"]=p_dest
            # prev2: look up prev1 row via tail index to get its prev_origin
            tail   = str(row.get("tail_number","")).upper()
            dep_ts = row.get("dep_ts_actual_utc")
            if tail and not pd.isna(dep_ts):
                pos = self.tail_index.get(tail,[])
                if len(pos) >= 2:
                    tdf = self._df.iloc[pos]
                    if "dep_ts_actual_utc" in tdf.columns:
                        bef = tdf[tdf["dep_ts_actual_utc"] < dep_ts]
                        if len(bef) >= 1:
                            prev1_row = bef.iloc[-1]
                            p2o = str(prev1_row.get("prev_origin","") or "")
                            p2d = str(prev1_row.get("prev_dest","")   or "")
                            if p2o not in ("nan","None",""): out["prev2_origin"]=p2o; out["prev2_dest"]=p2d
            return out
        tail=str(row.get("tail_number","")).upper(); dep_ts=row.get("dep_ts_actual_utc")
        if not tail or pd.isna(dep_ts): return out
        pos=self.tail_index.get(tail,[])
        if len(pos)<2: return out
        tdf=self._df.iloc[pos]
        if "dep_ts_actual_utc" not in tdf.columns: return out
        bef=tdf[tdf["dep_ts_actual_utc"]<dep_ts]
        if len(bef)>=1: p1=bef.iloc[-1]; out["prev1_origin"]=str(p1.get("origin","")); out["prev1_dest"]=str(p1.get("dest",""))
        if len(bef)>=2: p2=bef.iloc[-2]; out["prev2_origin"]=str(p2.get("origin","")); out["prev2_dest"]=str(p2.get("dest",""))
        return out

def find_flight(df,flight_id=None,tail_number=None,origin=None,dest=None):
    r=df
    if flight_id:
        fid=str(flight_id).strip().upper()
        for col in ("flight_id","fl_num","op_carrier_fl_num"):
            if col in r.columns:
                mask=r[col].astype(str).str.strip().str.upper()==fid
                if mask.any(): r=r[mask]; break
    if tail_number:
        tn=str(tail_number).strip().upper()
        for col in ("tail_number","tail_num"):
            if col in r.columns: r=r[r[col].astype(str).str.strip().str.upper()==tn]; break
    if origin: r=r[r["origin"].astype(str).str.upper()==origin.strip().upper()]
    if dest:   r=r[r["dest"].astype(str).str.upper()==dest.strip().upper()]
    return r.reset_index(drop=True)

def get_propagation_chain(df,row):
    chain={"prev2_origin":"","prev2_dest":"","prev1_origin":"","prev1_dest":""}
    if "tail_number" not in df.columns or "dep_ts_actual_utc" not in df.columns: return chain
    tail=row.get("tail_number",""); dep_ts=row.get("dep_ts_actual_utc")
    if not tail or pd.isna(dep_ts): return chain
    tdf=df[df["tail_number"]==tail].sort_values("dep_ts_actual_utc")
    bef=tdf[tdf["dep_ts_actual_utc"]<dep_ts]
    if len(bef)>=1: p1=bef.iloc[-1]; chain["prev1_origin"]=str(p1.get("origin","")); chain["prev1_dest"]=str(p1.get("dest",""))
    if len(bef)>=2: p2=bef.iloc[-2]; chain["prev2_origin"]=str(p2.get("origin","")); chain["prev2_dest"]=str(p2.get("dest",""))
    return chain