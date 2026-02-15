from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence, Union, Literal

import pandas as pd
from sqlalchemy import func

from pyopensky.trino import Trino
from pyopensky.schema import StateVectorsData4


DateLike = Union[str, datetime]
AirportFilterMode = Literal["none", "airport", "arrival", "departure"]


@dataclass(frozen=True)
class Airport:
    icao: str
    lat: float
    lon: float


def _to_dt(x: DateLike) -> datetime:
    """Accepts datetime or ISO-like string; returns naive datetime (UTC assumed)."""
    if isinstance(x, datetime):
        return x
    # Let pandas parse lots of formats safely
    return pd.to_datetime(x).to_pydatetime()


def _chunks(start: datetime, stop: datetime, chunk: timedelta) -> Iterable[tuple[datetime, datetime]]:
    cur = start
    while cur < stop:
        nxt = min(cur + chunk, stop)
        yield cur, nxt
        cur = nxt


def fetch_opensky_history_airport(
    start: DateLike,
    stop: DateLike,
    airport: Airport,
    *,
    radius_nm: float = 50.0,
    selected_columns: Sequence[Union[str, object]] = (
        "time",
        "icao24",
        "lat",
        "lon",
        "baroaltitude",
        "geoaltitude",
        "velocity",
        "heading",
        "vertrate",
        "callsign",
        "onground",
        "squawk",
    ),
    mode: AirportFilterMode = "none",
    time_buffer: Optional[str] = None,
    altitude_max_m: Optional[float] = None,
    altitude_min_m: Optional[float] = None,
    limit: Optional[int] = None,
    chunk_hours: int = 6,
    cached: bool = True,
) -> pd.DataFrame:
    """
    Pull historical OpenSky state-vector-like data near an airport over a time range.

    Parameters
    ----------
    start, stop:
        Datetime or string. (UTC assumed if naive.)
    airport:
        Airport(icao, lat, lon)
    radius_nm:
        Circle radius around airport in nautical miles.
    selected_columns:
        Columns to fetch (strings like "lat" or schema attributes).
    mode:
        "none"       -> only spatial filter
        "airport"    -> adds airport=<ICAO> constraint (pyopensky airport logic)
        "arrival"    -> adds arrival_airport=<ICAO>
        "departure"  -> adds departure_airport=<ICAO>
    time_buffer:
        e.g., "30m", "1h" to extend airport query window.
    altitude_max_m / altitude_min_m:
        Optional altitude filters in meters (baro altitude).
    limit:
        Optional limit per chunk (useful for debugging).
    chunk_hours:
        Query chunk size to avoid huge responses.
    cached:
        Use pyopensky caching.

    Returns
    -------
    pd.DataFrame
    """
    start_dt = _to_dt(start)
    stop_dt = _to_dt(stop)
    if stop_dt <= start_dt:
        raise ValueError("stop must be after start")

    # Distance filter (meters)
    radius_m = radius_nm * 1852.0

    # SQLAlchemy distance expression: airport point -> aircraft point
    dist_expr = func.ST_Distance(
        func.to_spherical_geography(func.ST_Point(airport.lon, airport.lat)),
        func.to_spherical_geography(func.ST_Point(StateVectorsData4.lon, StateVectorsData4.lat)),
    )

    # Additional predicates to pass to trino.history()
    predicates = [dist_expr <= radius_m]

    if altitude_max_m is not None:
        predicates.append(StateVectorsData4.baroaltitude < float(altitude_max_m))
    if altitude_min_m is not None:
        predicates.append(StateVectorsData4.baroaltitude > float(altitude_min_m))

    # Optional airport constraints (pyopensky supports these keyword args)
    history_kwargs = dict(
        selected_columns=selected_columns,
        cached=cached,
    )
    if time_buffer is not None:
        history_kwargs["time_buffer"] = time_buffer

    if mode == "airport":
        history_kwargs["airport"] = airport.icao
    elif mode == "arrival":
        history_kwargs["arrival_airport"] = airport.icao
    elif mode == "departure":
        history_kwargs["departure_airport"] = airport.icao
    elif mode != "none":
        raise ValueError(f"Unknown mode: {mode}")

    if limit is not None:
        history_kwargs["limit"] = int(limit)

    trino = Trino()

    out: list[pd.DataFrame] = []
    chunk = timedelta(hours=int(chunk_hours))

    for s, e in _chunks(start_dt, stop_dt, chunk):
        # pyopensky accepts strings or datetimes; strings are fine
        s_str = pd.Timestamp(s).strftime("%Y-%m-%d %H:%M:%S")
        e_str = pd.Timestamp(e).strftime("%Y-%m-%d %H:%M:%S")

        df = trino.history(
            s_str,
            e_str,
            *predicates,
            **history_kwargs,
        )

        if df is None or len(df) == 0:
            continue

        # Normalize join key for FAA join
        if "icao24" in df.columns:
            df["icao24"] = df["icao24"].astype(str).str.lower().str.strip()

        out.append(df)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


def join_with_faa_dim(
    opensky_df: pd.DataFrame,
    faa_dim_path: str,
    *,
    how: str = "left",
) -> pd.DataFrame:
    """
    Join OpenSky states to your FAA aircraft dimension on icao24.
    faa_dim_path: parquet or csv created by your FAA script.
    """
    if faa_dim_path.endswith(".parquet"):
        faa = pd.read_parquet(faa_dim_path)
    else:
        faa = pd.read_csv(faa_dim_path)

    faa["icao24"] = faa["icao24"].astype(str).str.lower().str.strip()
    opensky_df["icao24"] = opensky_df["icao24"].astype(str).str.lower().str.strip()

    return opensky_df.merge(faa, on="icao24", how=how)
