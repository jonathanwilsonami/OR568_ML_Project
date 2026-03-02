# ------------------------------------------------------------
# BWI OpenSky State Vectors → traced lines on a map (Folium)
#   - Handles Unix time seconds properly
#   - Uses `heading` for rotation (new schema)
# ------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import folium
from dataclasses import dataclass
from branca.colormap import linear


# -----------------------------
# 1) Airport + bounding box
# -----------------------------
@dataclass(frozen=True)
class Airport:
    icao: str
    lat: float
    lon: float


# BWI terminal area bbox
LAT_MIN, LAT_MAX = 38.9191667, 39.4166667
LON_MIN, LON_MAX = -77.0597222, -76.3075000


def make_bbox(buffer_deg: float = 0.20) -> tuple[float, float, float, float]:
    """
    Returns (west, south, east, north) bounding box.
    Keep buffer small to reduce API credits when collecting data.
    """
    lats = [LAT_MIN, LAT_MAX]
    lons = [LON_MIN, LON_MAX]
    south = min(lats) - buffer_deg
    north = max(lats) + buffer_deg
    west  = min(lons) - buffer_deg
    east  = max(lons) + buffer_deg
    return (west, south, east, north)


# -----------------------------
# 2) Helper cleanup
# -----------------------------
def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a time-like column to UTC datetime.
    Supports:
      - Unix seconds (int/float or numeric strings)
      - ISO datetime strings
    """
    # Try to coerce to numeric (Unix seconds). Non-numeric -> NaN.
    s_num = pd.to_numeric(series, errors="coerce")

    # If we got ANY numeric values, treat as Unix seconds.
    # (Works even if the column is strings like "1656364200")
    if s_num.notna().any():
        return pd.to_datetime(s_num, unit="s", utc=True, errors="coerce")

    # Otherwise parse as datetime strings
    return pd.to_datetime(series, utc=True, errors="coerce")


def clean_track_points(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Cleans and normalizes the dataframe for mapping.
    Returns (clean_df, time_column_used)
    """
    pts = df.dropna(subset=["lat", "lon"]).copy()

    # Pick a time column for ordering
    if "time" in pts.columns:
        tcol = "time"
    elif "lastcontact" in pts.columns:
        tcol = "lastcontact"
    elif "lastposupdate" in pts.columns:
        tcol = "lastposupdate"
    else:
        raise ValueError("Need a time column: time, lastcontact, or lastposupdate")

    pts[tcol] = _to_utc_datetime(pts[tcol])
    pts = pts.dropna(subset=[tcol])

    # Normalize types (important for folium)
    pts["lat"] = pd.to_numeric(pts["lat"], errors="coerce")
    pts["lon"] = pd.to_numeric(pts["lon"], errors="coerce")
    pts = pts.dropna(subset=["lat", "lon"])

    pts = pts.sort_values(["icao24", tcol])

    # Drop exact duplicates (common in snapshot sampling)
    subset = ["icao24", tcol, "lat", "lon"]
    pts = pts.drop_duplicates(subset=subset)

    return pts, tcol


# -----------------------------
# 3) Map: dots colored by altitude (optional sanity check)
# -----------------------------
def map_states_colored_by_altitude(
    df: pd.DataFrame,
    out_html: str = "bwi_altitude_dots.html",
    max_points: int = 4000,
) -> str:
    pts = df.dropna(subset=["lat", "lon"]).copy()
    if len(pts) == 0:
        raise ValueError("No valid latitude/longitude points to map.")

    # Reduce points if large
    if len(pts) > max_points:
        pts = pts.sample(max_points, random_state=0)

    # Choose an altitude column
    alt = None
    for c in ["geoaltitude", "baroaltitude"]:
        if c in pts.columns:
            alt = c
            break

    if alt is None:
        pts["alt_m"] = 0.0
    else:
        pts["alt_m"] = pd.to_numeric(pts[alt], errors="coerce").fillna(0.0)

    center = [pts["lat"].mean(), pts["lon"].mean()]
    m = folium.Map(location=center, zoom_start=9, control_scale=True)

    cmap = linear.YlOrRd_09.scale(pts["alt_m"].min(), pts["alt_m"].max())
    cmap.caption = "Altitude (m)"
    cmap.add_to(m)

    for _, r in pts.iterrows():
        color = cmap(r["alt_m"])
        cs = str(r.get("callsign", "")).strip()
        icao24 = str(r.get("icao24", ""))
        label = cs if cs else icao24

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"{label} | alt_m={r['alt_m']:.0f}",
        ).add_to(m)

    m.save(out_html)
    return out_html


# -----------------------------
# 4) Map: lines (tracks) per aircraft
# -----------------------------
def map_tracks_by_aircraft(
    df: pd.DataFrame,
    out_html: str = "bwi_tracks.html",
    max_aircraft: int = 30,
    min_points_per_track: int = 4,
    max_points_total: int = 30000,
    show_plane_markers: bool = True,
    show_labels: bool = True,
    label_always_on: bool = True,
) -> str:
    pts, tcol = clean_track_points(df)

    if len(pts) > max_points_total:
        pts = pts.sample(max_points_total, random_state=0)

    # Pick top aircraft by number of points
    counts = pts.groupby("icao24").size().sort_values(ascending=False)
    chosen_icao24 = counts[counts >= min_points_per_track].head(max_aircraft).index
    pts = pts[pts["icao24"].isin(chosen_icao24)]

    if len(pts) == 0:
        raise ValueError(
            "No tracks with enough points to draw. "
            "Collect more snapshots or lower min_points_per_track."
        )

    center = [pts["lat"].mean(), pts["lon"].mean()]
    m = folium.Map(location=center, zoom_start=9, control_scale=True)

    palette = [
        "red","blue","green","purple","orange","darkred","lightred","beige",
        "darkblue","darkgreen","cadetblue","darkpurple","pink","lightblue",
        "lightgreen","gray","black"
    ]

    def _label_html(text: str) -> str:
        return f"""
        <div style="
            font-size: 12px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 6px;
            border: 1px solid rgba(0,0,0,0.25);
            background: rgba(255,255,255,0.85);
            box-shadow: 0 1px 4px rgba(0,0,0,0.15);
            white-space: nowrap;">
            {text}
        </div>
        """

    # Draw one polyline per aircraft
    for i, (icao24, g) in enumerate(pts.groupby("icao24")):
        g = g.sort_values(tcol)

        line = list(zip(g["lat"].astype(float), g["lon"].astype(float)))
        if len(line) < min_points_per_track:
            continue

        color = palette[i % len(palette)]

        callsign_series = g.get("callsign", pd.Series([], dtype=str)).dropna().astype(str).str.strip()
        callsign = callsign_series.iloc[-1] if len(callsign_series) else ""
        label = f"{callsign} ({icao24})" if callsign else f"{icao24}"

        folium.PolyLine(
            locations=line,
            color=color,
            weight=4,
            opacity=0.9,
            tooltip=label,
        ).add_to(m)

        folium.CircleMarker(
            line[0],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"START {label}",
        ).add_to(m)

        end_lat, end_lon = line[-1]

        if show_plane_markers:
            # NEW: use `heading` first (your schema), then fallback
            heading = None
            for hcol in ["heading", "true_track", "track"]:
                if hcol in g.columns:
                    h = pd.to_numeric(g[hcol], errors="coerce").dropna()
                    if len(h):
                        heading = float(h.iloc[-1])
                        break

            if heading is not None:
                plane_html = f"""
                <div style="
                    transform: rotate({heading}deg);
                    color: {color};
                    font-size: 18px;
                    line-height: 18px;
                    ">
                    ✈
                </div>
                """
                folium.Marker(
                    location=[end_lat, end_lon],
                    icon=folium.DivIcon(html=plane_html),
                    tooltip=f"END {label}",
                ).add_to(m)
            else:
                plane_icon = folium.Icon(icon="plane", prefix="fa", color="blue")
                folium.Marker(
                    location=[end_lat, end_lon],
                    icon=plane_icon,
                    tooltip=f"END {label}",
                ).add_to(m)

            if show_labels:
                if label_always_on:
                    folium.Marker(
                        location=[end_lat, end_lon],
                        icon=folium.DivIcon(
                            icon_size=(250, 36),
                            icon_anchor=(0, 0),
                            html=_label_html(label),
                        ),
                    ).add_to(m)
                else:
                    folium.CircleMarker(
                        location=[end_lat, end_lon],
                        radius=6,
                        color=color,
                        fill=True,
                        fill_opacity=0.2,
                        tooltip=label,
                    ).add_to(m)
        else:
            folium.CircleMarker(
                line[-1],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"END {label}",
            ).add_to(m)

    m.save(out_html)
    return out_html


# -----------------------------
# 5) Usage
# -----------------------------
if __name__ == "__main__":
    icao24 = "abbe71"
    df = pd.read_csv(
        "/home/jon/Documents/grad_school/OR568/project/OR568_ML_Project/data_pipeline/raw_data/bwi_states_2022_06_27_raw.csv"
    )

    # Optional: force clean types (helps if booleans came in as strings)
    df["icao24"] = df["icao24"].astype(str).str.lower().str.strip()
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].astype(str)

    df = df[df["icao24"] == icao24]

    map_tracks_by_aircraft(
        df,
        out_html=f"bwi_tracks_for_{icao24}.html",
        min_points_per_track=4
    )