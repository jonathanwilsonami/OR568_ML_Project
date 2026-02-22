# ------------------------------------------------------------
# IAD + DCA OpenSky State Vectors → traced lines on a map (Folium)
# ------------------------------------------------------------
# Usage (example):
#   df = pd.read_parquet("iad_dca_states_smoketest.parquet")
#   map_tracks_by_aircraft(df, out_html="iad_dca_tracks.html")
#
# Optional:
#   map_states_colored_by_altitude(df, out_html="iad_dca_altitude_dots.html")
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

IAD = Airport("KIAD", 38.9475, -77.4599)
DCA = Airport("KDCA", 38.8514, -77.0377)

def make_bbox(buffer_deg: float = 0.20) -> tuple[float, float, float, float]:
    """
    Returns (west, south, east, north) bounding box.
    Keep buffer small to reduce API credits when collecting data.
    """
    lats = [IAD.lat, DCA.lat]
    lons = [IAD.lon, DCA.lon]
    south = min(lats) - buffer_deg
    north = max(lats) + buffer_deg
    west  = min(lons) - buffer_deg
    east  = max(lons) + buffer_deg
    return (west, south, east, north)


# -----------------------------
# 2) Helper cleanup
# -----------------------------
def clean_track_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the dataframe for mapping.
    Expects columns: latitude, longitude, icao24, callsign, and a time column.
    """
    pts = df.dropna(subset=["latitude", "longitude"]).copy()

    # Pick a time column for ordering
    if "snapshot_time" in pts.columns:
        tcol = "snapshot_time"
    elif "last_contact" in pts.columns:
        tcol = "last_contact"
    elif "time_position" in pts.columns:
        tcol = "time_position"
    else:
        raise ValueError("Need a time column: snapshot_time, last_contact, or time_position")

    pts[tcol] = pd.to_datetime(pts[tcol], utc=True, errors="coerce")
    pts = pts.dropna(subset=[tcol])
    pts = pts.sort_values(["icao24", tcol])

    # Drop exact duplicates (common in snapshot sampling)
    subset = ["icao24", tcol, "latitude", "longitude"]
    pts = pts.drop_duplicates(subset=subset)

    return pts


# -----------------------------
# 3) Map: dots colored by altitude (optional sanity check)
# -----------------------------
def map_states_colored_by_altitude(
    df: pd.DataFrame,
    out_html: str = "iad_dca_altitude_dots.html",
    max_points: int = 4000,
) -> str:
    pts = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(pts) == 0:
        raise ValueError("No valid latitude/longitude points to map.")

    # Reduce points if large
    if len(pts) > max_points:
        pts = pts.sample(max_points, random_state=0)

    # Choose an altitude column
    alt = None
    for c in ["geo_altitude", "baro_altitude", "altitude", "geoaltitude"]:
        if c in pts.columns:
            alt = c
            break
    if alt is None:
        pts["alt_m"] = 0.0
    else:
        pts["alt_m"] = pd.to_numeric(pts[alt], errors="coerce").fillna(0.0)

    center = [pts["latitude"].mean(), pts["longitude"].mean()]
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
            location=[float(r["latitude"]), float(r["longitude"])],
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
# 4) Map: lines (tracks) per aircraft — clearer with plane icons + labels
# -----------------------------
def map_tracks_by_aircraft(
    df: pd.DataFrame,
    out_html: str = "iad_dca_tracks.html",
    max_aircraft: int = 30,          # limit number of aircraft drawn
    min_points_per_track: int = 4,   # ignore very short tracks
    max_points_total: int = 30000,   # safety cap
    show_plane_markers: bool = True, # show plane icon at end of track
    show_labels: bool = True,        # show a readable label near the plane icon
    label_always_on: bool = True,    # True=label always visible, False=tooltip only
) -> str:
    pts = clean_track_points(df)

    # Decide time column
    if "snapshot_time" in pts.columns:
        tcol = "snapshot_time"
    elif "last_contact" in pts.columns:
        tcol = "last_contact"
    else:
        tcol = "time_position"

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

    center = [pts["latitude"].mean(), pts["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=9, control_scale=True)

    palette = [
        "red","blue","green","purple","orange","darkred","lightred","beige",
        "darkblue","darkgreen","cadetblue","darkpurple","pink","lightblue",
        "lightgreen","gray","black"
    ]

    # Helper: small readable label box
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

        line = list(zip(g["latitude"].astype(float), g["longitude"].astype(float)))
        if len(line) < min_points_per_track:
            continue

        color = palette[i % len(palette)]

        callsign_series = g["callsign"].dropna().astype(str).str.strip()
        callsign = callsign_series.iloc[-1] if len(callsign_series) else ""
        label = f"{callsign} ({icao24})" if callsign else f"{icao24}"

        # Track line (thicker for clarity)
        folium.PolyLine(
            locations=line,
            color=color,
            weight=4,
            opacity=0.9,
            tooltip=label,
        ).add_to(m)

        # Start marker (small)
        folium.CircleMarker(
            line[0],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"START {label}",
        ).add_to(m)

        # End marker: plane icon + label
        end_lat, end_lon = line[-1]
        if show_plane_markers:
            # Use heading if available for rotation
            heading = None
            if "true_track" in g.columns:
                heading = pd.to_numeric(g["true_track"], errors="coerce").dropna()
                heading = float(heading.iloc[-1]) if len(heading) else None
            elif "track" in g.columns:
                heading = pd.to_numeric(g["track"], errors="coerce").dropna()
                heading = float(heading.iloc[-1]) if len(heading) else None

            # Plane icon marker (FontAwesome "plane")
            plane_icon = folium.Icon(
                icon="plane",
                prefix="fa",
                color="blue"  # folium icon colors are limited; keep it readable
            )

            # If you want heading-based rotation, use DivIcon instead of Icon (more flexible)
            if heading is not None:
                # Simple rotated plane using CSS transform
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
                folium.Marker(
                    location=[end_lat, end_lon],
                    icon=plane_icon,
                    tooltip=f"END {label}",
                ).add_to(m)

            # Label marker next to plane (always visible if label_always_on=True)
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
                    # Tooltip-only label (less clutter)
                    folium.CircleMarker(
                        location=[end_lat, end_lon],
                        radius=6,
                        color=color,
                        fill=True,
                        fill_opacity=0.2,
                        tooltip=label,
                    ).add_to(m)
        else:
            # fallback end circle
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
# 5) Usage (exact style you requested)
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_parquet("iad_dca_states_smoketest.parquet")

    # Lines (easier to see which plane is which)
    # map_tracks_by_aircraft(df, out_html="iad_dca_tracks.html")

    # Optional: dots colored by altitude
    # map_states_colored_by_altitude(df, out_html="iad_dca_altitude_dots.html")
    df = pd.read_parquet("iad_dca_states_10min_30s.parquet")
    map_tracks_by_aircraft(df, out_html="iad_dca_tracks.html", min_points_per_track=4)
