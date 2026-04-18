"""
charts.py  —  All Plotly figure factories for the demo app.
"""
from __future__ import annotations
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from airports import lookup

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _delay_color(delay: float) -> str:
    if delay >= 30:  return "#E24B4A"
    if delay >= 15:  return "#EF9F27"
    return "#1D9E75"

def _prob_color(prob: float) -> str:
    if prob > 0.6:   return "#E24B4A"
    if prob > 0.3:   return "#EF9F27"
    return "#1D9E75"


# ---------------------------------------------------------------------------
# 1. Geographic route map  (real data version)
# ---------------------------------------------------------------------------

def route_map(
    origin: str,
    dest: str,
    prev1_origin: str = "",
    prev1_dest: str = "",
    prev2_origin: str = "",
    prev2_dest: str = "",
    actual_arr_delay: float = 0,
    prev1_arr_delay: float = 0,
    prev2_arr_delay: float = 0,
    flight_id: str = "",
    tail_number: str = "",
) -> go.Figure:
    """
    Show the chain of up to 3 legs on a US geo map.
    Leg colours = arrival delay at destination (green/amber/red).
    """
    # Resolve coordinates
    p2o = lookup(prev2_origin) if prev2_origin else None
    p2d = lookup(prev2_dest)   if prev2_dest   else None
    p1o = lookup(prev1_origin) if prev1_origin else None
    # prev1_dest == origin of current flight
    org = lookup(origin)
    dst = lookup(dest)

    if org is None or dst is None:
        missing = origin if org is None else dest
        fig = go.Figure()
        fig.add_annotation(text=f"Unknown airport: {missing} — add to airports.py",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=13))
        fig.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    # Build legs: (start_coords, end_coords, arr_delay_at_dest, label)
    legs: list[tuple] = []
    if p2o and p2d:
        legs.append((p2o, p2d, prev2_arr_delay,
                     f"Leg 1 · {prev2_origin}→{prev2_dest}"))
    if p2d and org:
        # prev2_dest == prev1_origin; dest of that leg == current origin
        legs.append((p2d, org, prev1_arr_delay,
                     f"Leg 2 · {prev2_dest}→{origin}"))
    elif p1o and org:
        legs.append((p1o, org, prev1_arr_delay,
                     f"Leg 2 · {prev1_origin}→{origin}"))
    legs.append((org, dst, actual_arr_delay,
                 f"Leg 3 · {origin}→{dest}  ← this flight"))

    fig = go.Figure()

    for i, (start, end, arr_delay, label) in enumerate(legs):
        s_lat, s_lon, _ = start
        e_lat, e_lon, _ = end
        is_current = (i == len(legs) - 1)
        n = 50
        lats = [s_lat + (e_lat - s_lat) * t - 3.5 * t * (1 - t)
                for t in (k / (n - 1) for k in range(n))]
        lons = [s_lon + (e_lon - s_lon) * t
                for t in (k / (n - 1) for k in range(n))]
        color = _delay_color(arr_delay)

        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=4 if is_current else 2,
                      color=color, dash="solid" if is_current else "dot"),
            name=label, hoverinfo="name", showlegend=True,
        ))
        fig.add_trace(go.Scattergeo(
            lat=[lats[-1]], lon=[lons[-1]], mode="markers",
            marker=dict(size=7 if is_current else 5, color=color,
                        symbol="triangle-right"),
            hoverinfo="skip", showlegend=False,
        ))

    # Airport markers
    stops, seen = [], set()
    def _add(code, info, delay, size):
        if code and info and code not in seen:
            seen.add(code)
            stops.append((info[0], info[1], code, info[2], delay, size))

    _add(prev2_origin, p2o, 0.0,              8)
    _add(prev2_dest,   p2d, prev2_arr_delay, 10)
    _add(origin,       org, prev1_arr_delay, 12)
    _add(dest,         dst, actual_arr_delay,18)

    for lat, lon, code, city, delay, size in stops:
        is_dest = (code == dest)
        color = _delay_color(delay)
        hover = (f"<b>{code}</b><br>{city}<br>"
                 + (f"Arr delay: {delay:.0f} min" if delay else "Departure airport")
                 + ("<br><b>← THIS FLIGHT'S DESTINATION</b>" if is_dest else ""))
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=2, color="white")),
            text=[code], textposition="top center",
            textfont=dict(size=11, color="#222"),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

    # Delay labels at midpoints
    mid_lats, mid_lons, mid_texts, mid_colors = [], [], [], []
    for start, end, arr_delay, _ in legs:
        if arr_delay >= 1:
            mid_lats.append((start[0] + end[0]) / 2 - 1.5)
            mid_lons.append((start[1] + end[1]) / 2)
            mid_texts.append(f"+{arr_delay:.0f}m")
            mid_colors.append(_delay_color(arr_delay))
    if mid_lats:
        fig.add_trace(go.Scattergeo(
            lat=mid_lats, lon=mid_lons, mode="text",
            text=mid_texts, textfont=dict(size=11, color=mid_colors),
            hoverinfo="skip", showlegend=False,
        ))

    all_coords = [c for c in [p2o, p2d, org, dst] if c]
    all_lats = [c[0] for c in all_coords]
    all_lons = [c[1] for c in all_coords]

    fig.update_layout(
        title=dict(text=f"Route chain — {flight_id}  ·  {tail_number}",
                   font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        height=460, margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            scope="usa", projection_type="albers usa",
            showland=True,       landcolor="rgba(235,235,230,1)",
            showlakes=True,      lakecolor="rgba(200,220,240,0.6)",
            showcoastlines=True, coastlinecolor="rgba(160,160,160,0.8)",
            showsubunits=True,   subunitcolor="rgba(200,200,200,0.7)",
            showcountries=False, bgcolor="rgba(210,228,245,0.5)",
            lataxis=dict(range=[min(all_lats) - 4, max(all_lats) + 4]),
            lonaxis=dict(range=[min(all_lons) - 6, max(all_lons) + 6]),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Actual vs predicted — classification + regression side by side
# ---------------------------------------------------------------------------

def actual_vs_predicted(
    actual_delay: float,
    actual_del15: int,
    pred_prob: float,
    pred_delay: float,
    threshold: float = 0.5,
) -> go.Figure:
    """Gauge-style comparison: probability bar + delay bar."""
    pred_del15 = int(pred_prob >= threshold)
    cls_correct = pred_del15 == actual_del15
    reg_error   = pred_delay - actual_delay

    fig = go.Figure()

    # ── Probability gauge ─────────────────────────────────────────────
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=round(pred_prob * 100, 1),
        title={"text": "P(delay ≥ 15 min)", "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": _prob_color(pred_prob)},
            "steps": [
                {"range": [0, 30],  "color": "rgba(29,158,117,0.15)"},
                {"range": [30, 60], "color": "rgba(239,159,39,0.15)"},
                {"range": [60, 100],"color": "rgba(226,75,74,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 2},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
        domain={"x": [0, 0.45], "y": [0, 1]},
    ))

    # ── Delay bar ─────────────────────────────────────────────────────
    actual_color = _delay_color(actual_delay)
    pred_color   = _prob_color(pred_prob)

    fig.add_trace(go.Bar(
        x=["Actual", "Predicted"],
        y=[actual_delay, pred_delay],
        marker_color=[actual_color, pred_color],
        text=[f"{actual_delay:.0f} min", f"{pred_delay:.0f} min"],
        textposition="outside",
        cliponaxis=False,
        xaxis="x2", yaxis="y2",
        showlegend=False,
    ))

    # ── Annotations ───────────────────────────────────────────────────
    cls_label = "✓ Correct" if cls_correct else "✗ Incorrect"
    cls_color = "#1D9E75"  if cls_correct else "#E24B4A"
    err_sign  = "+" if reg_error > 0 else ""

    fig.add_annotation(
        text=f"<b>Classification:</b> {cls_label}",
        x=0.22, y=-0.12, xref="paper", yref="paper",
        showarrow=False, font=dict(size=12, color=cls_color),
    )
    fig.add_annotation(
        text=f"<b>Reg. error:</b> {err_sign}{reg_error:.1f} min",
        x=0.78, y=-0.12, xref="paper", yref="paper",
        showarrow=False, font=dict(size=12,
                                    color="#E24B4A" if abs(reg_error) > 15 else "#1D9E75"),
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2", showgrid=False),
        yaxis2=dict(domain=[0.05, 0.95], anchor="x2",
                    title="Delay (min)",
                    gridcolor="rgba(128,128,128,0.15)",
                    range=[0, max(actual_delay, pred_delay) * 1.3 + 5]),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Propagation chain  (real data — actual values)
# ---------------------------------------------------------------------------

def propagation_chain(
    prev2_origin: str, prev2_dest: str,
    prev1_origin: str, prev1_dest: str,
    origin: str, dest: str,
    prev2_arr_delay: float, prev2_dep_delay: float,
    prev1_arr_delay: float, prev1_dep_delay: float,
    actual_arr_delay: float, actual_dep_delay: float,
) -> go.Figure:
    steps  = ["prev2", "prev1", "current"]
    p2_route = f"{prev2_origin}→{prev2_dest}" if prev2_origin else "prev2"
    p1_route = f"{prev1_origin}→{prev1_dest}" if prev1_origin else "prev1"
    cur_route = f"{origin}→{dest}"
    labels = [f"{s}<br><sub>{r}</sub>"
              for s, r in zip(steps, [p2_route, p1_route, cur_route])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=[prev2_arr_delay, prev1_arr_delay, actual_arr_delay],
        mode="lines+markers", name="Arrival delay",
        line=dict(color="#E24B4A", width=2),
        marker=dict(size=10, color="#E24B4A"),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=[prev2_dep_delay, prev1_dep_delay, actual_dep_delay],
        mode="lines+markers", name="Departure delay",
        line=dict(color="#378ADD", width=2),
        marker=dict(size=10, color="#378ADD"),
    ))
    fig.add_hline(y=15, line_dash="dot", line_color="#888",
                  annotation_text="15-min threshold",
                  annotation_position="bottom right",
                  annotation_font_size=11)
    fig.update_layout(
        title="Delay propagation — prev2 → prev1 → current",
        yaxis_title="Delay (minutes)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300, margin=dict(l=50, r=20, t=55, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


# ---------------------------------------------------------------------------
# 4. Top feature contributions for this specific flight
# ---------------------------------------------------------------------------

def feature_contributions(row: pd.Series, feature_cols: list[str]) -> go.Figure:
    """
    Show the raw feature values that most influence the model,
    ranked by absolute deviation from the dataset median.
    (Without SHAP we proxy importance by |z-score| of the value.)
    """
    vals = []
    for col in feature_cols:
        v = float(row.get(col, 0) or 0)
        vals.append({"feature": col, "value": v})

    # Highlight the most informative propagation + weather features
    priority = [
        "prev1_arr_delay", "prev1_dep_delay", "prev2_arr_delay", "prev2_dep_delay",
        "prev1_turnaround_minutes", "time_since_prev2_arrival_minutes",
        "dep_temp_c", "dep_wind_speed_m_s", "dep_ceiling_height_m",
        "cum_arr_delay_aircraft_day", "cum_dep_delay_aircraft_day",
        "tight_turnaround_flag", "relative_leg_position",
        "route_frequency", "origin_flight_volume", "dep_hour_local",
    ]
    shown = [v for v in vals if v["feature"] in priority]

    names  = [v["feature"] for v in shown]
    values = [v["value"]   for v in shown]
    colors = []
    for feat, val in zip(names, values):
        if "delay" in feat and val > 15:
            colors.append("#E24B4A")
        elif "delay" in feat and val > 0:
            colors.append("#EF9F27")
        elif val < 0:
            colors.append("#378ADD")
        else:
            colors.append("#888780")

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        title="Key feature values for this flight",
        xaxis_title="Feature value",
        height=380, margin=dict(l=220, r=60, t=55, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)",
                     zeroline=True, zerolinecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(showgrid=False)
    return fig