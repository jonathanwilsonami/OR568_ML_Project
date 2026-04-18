"""
app.py  —  Flight delay model production demo

Search modes:
  • Tail number  +  date       (e.g. N215NV  +  2019-03-14)
  • Flight number [+ date / origin / dest]   (e.g. 1294)

Run:  python app.py   →   http://localhost:8050
"""
from __future__ import annotations

import traceback
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html

from charts import actual_vs_predicted, feature_contributions, propagation_chain, route_map
from data import (
    TARGET_CLASS, TARGET_REG, XGB_FULL_FEATURES,
    FlightIndex, load_and_engineer,
)
from models.loader import discover_models, load_model_pair, predict_row

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_FILE  = ROOT / "data" / "flights_canonical_2019.parquet"
DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Startup:  load data  →  build index  (runs once, before any request)
# ---------------------------------------------------------------------------

_DF:    pd.DataFrame | None = None
_INDEX: FlightIndex  | None = None
_LOAD_ERROR: str = ""

def _startup() -> None:
    global _DF, _INDEX, _LOAD_ERROR
    if not DATA_FILE.exists():
        _LOAD_ERROR = f"Data file not found: {DATA_FILE}"
        print(f"[WARN] {_LOAD_ERROR}")
        return
    try:
        _DF    = load_and_engineer(DATA_FILE)          # cache-aware
        _INDEX = FlightIndex(_DF)                      # O(1) lookup index
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        print(f"[ERROR] {_LOAD_ERROR}\n{traceback.format_exc()}")

_startup()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Flight delay demo",
    suppress_callback_exceptions=True,
)
server = app.server   # exposed for gunicorn / production deployment

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _lbl(text: str) -> html.Label:
    return html.Label(text, className="form-label fw-semibold small text-muted mb-1 d-block")

def _field(label: str, ctrl, md: int = 12) -> dbc.Col:
    return dbc.Col([_lbl(label), ctrl], md=md, className="mb-3")

def _sel(id_: str, opts, val=None) -> dcc.Dropdown:
    return dcc.Dropdown(id=id_, options=opts, value=val, clearable=False,
                        style={"fontSize": "13px"})

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> dbc.Col:
    # Data status banner
    if _LOAD_ERROR:
        banner = dbc.Alert([html.B("Data not loaded  "), html.Br(),
                            html.Span(_LOAD_ERROR, className="small")],
                           color="danger", className="p-2 mb-3")
    elif _DF is None:
        banner = dbc.Alert("No data file.", color="warning", className="p-2 mb-3")
    else:
        banner = dbc.Alert(
            [html.B(f"✓ {len(_DF):,} flights  "),
             html.Span(DATA_FILE.name, className="small text-muted")],
            color="success", className="p-2 mb-3")

    # Model dropdown
    models_found = discover_models(MODELS_DIR)
    model_opts   = ([{"label": n, "value": n} for n in models_found]
                    if models_found else
                    [{"label": "No .joblib models found", "value": "__none__"}])

    return dbc.Col([
        banner,

        # ── Search ────────────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(html.B("Flight search")),
            dbc.CardBody([

                # Mode tabs
                dbc.Tabs([
                    dbc.Tab(label="Tail + date", tab_id="tab-tail"),
                    dbc.Tab(label="Flight number", tab_id="tab-fnum"),
                ], id="search-tabs", active_tab="tab-tail",
                   className="mb-3"),

                # Tail + date panel
                html.Div(id="panel-tail", children=[
                    dbc.Row([
                        _field("Tail number",
                               dbc.Input(id="s-tail", value="N215NV",
                                         placeholder="e.g. N215NV", size="sm")),
                        _field("Date  (YYYY-MM-DD)",
                               dbc.Input(id="s-date", placeholder="e.g. 2019-03-14",
                                         size="sm")),
                        _field("Origin  (optional)",
                               dbc.Input(id="s-orig-tail",
                                         placeholder="e.g. RDM", size="sm")),
                        _field("Dest  (optional)",
                               dbc.Input(id="s-dest-tail",
                                         placeholder="e.g. AZA", size="sm")),
                    ]),
                ]),

                # Flight number panel
                html.Div(id="panel-fnum", style={"display":"none"}, children=[
                    html.P("Use the airline flight number (e.g. 552, 2684).",
                           className="text-muted small mb-2"),
                    dbc.Row([
                        _field("Flight number",
                               dbc.Input(id="s-fnum",
                                         placeholder="e.g. 552", size="sm")),
                        _field("Date  (optional)",
                               dbc.Input(id="s-date-fnum",
                                         placeholder="e.g. 2019-03-14", size="sm")),
                        _field("Origin  (optional)",
                               dbc.Input(id="s-orig-fnum",
                                         placeholder="e.g. AZA", size="sm")),
                        _field("Dest  (optional)",
                               dbc.Input(id="s-dest-fnum",
                                         placeholder="e.g. SBN", size="sm")),
                    ]),
                ]),
            ]),
        ], className="mb-3 shadow-sm"),

        # ── Model ─────────────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(html.B("Model")),
            dbc.CardBody([
                _lbl("Select model"),
                _sel("model-select", model_opts, model_opts[0]["value"]),
                html.Br(),
                _lbl("Decision threshold"),
                dcc.Slider(
                    id="thresh-slider", min=0.1, max=0.9, step=0.05,
                    value=DEFAULT_THRESHOLD,
                    marks={v/10: f"{v/10:.1f}" for v in range(1, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]),
        ], className="mb-3 shadow-sm"),

        dbc.Button("Run prediction", id="run-btn",
                   color="primary", size="lg", className="w-100 mb-3",
                   disabled=(_DF is None)),

        html.Div(id="run-error"),

    ], md=3)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

n_models = len(discover_models(MODELS_DIR))

app.layout = dbc.Container([
    dbc.Row(dbc.Col([
        html.H4("✈  Flight delay model — production demo",
                className="fw-bold mt-4 mb-1"),
        html.P(
            f"{DATA_FILE.name}  ·  "
            + (f"{len(_DF):,} flights ready" if _DF is not None else "data not loaded")
            + f"  ·  {n_models} model{'s' if n_models != 1 else ''} available",
            className="text-muted small mb-4",
        ),
    ])),
    dbc.Row([
        _sidebar(),
        dbc.Col(
            html.Div(id="report-panel", children=[
                dbc.Alert(
                    [html.B("Ready.  "),
                     "Default: tail N215NV on route RDM → AZA.  "
                     "Enter a date and click Run prediction."],
                    color="secondary", className="mt-4",
                )
            ]),
            md=9,
        ),
    ]),
], fluid=True, className="px-4")


# ---------------------------------------------------------------------------
# Callback: toggle search panels
# ---------------------------------------------------------------------------

@app.callback(
    Output("panel-tail", "style"),
    Output("panel-fnum", "style"),
    Input("search-tabs", "active_tab"),
)
def _toggle_panels(tab):
    if tab == "tab-tail":
        return {}, {"display": "none"}
    return {"display": "none"}, {}


# ---------------------------------------------------------------------------
# Callback: run prediction
# ---------------------------------------------------------------------------

@app.callback(
    Output("report-panel", "children"),
    Output("run-error",    "children"),
    Input("run-btn",       "n_clicks"),
    State("search-tabs",   "active_tab"),
    # Tail mode
    State("s-tail",        "value"),
    State("s-date",        "value"),
    State("s-orig-tail",   "value"),
    State("s-dest-tail",   "value"),
    # Flight number mode
    State("s-fnum",        "value"),
    State("s-date-fnum",   "value"),
    State("s-orig-fnum",   "value"),
    State("s-dest-fnum",   "value"),
    # Model
    State("model-select",  "value"),
    State("thresh-slider", "value"),
    prevent_initial_call=True,
)
def run_prediction(
    n_clicks, active_tab,
    s_tail, s_date, s_orig_tail, s_dest_tail,
    s_fnum, s_date_fnum, s_orig_fnum, s_dest_fnum,
    model_name, threshold,
):
    def _err(msg: str, detail: str = ""):
        body = [html.B(msg)]
        if detail:
            body.append(html.Pre(detail, className="small mt-1 mb-0"))
        return dash.no_update, dbc.Alert(body, color="danger", className="p-2")

    if _DF is None or _INDEX is None:
        return _err("Data not loaded.", _LOAD_ERROR)
    if model_name == "__none__":
        return _err("No models found — add .joblib files to models/.")

    threshold = float(threshold or DEFAULT_THRESHOLD)

    # ── Search via index ──────────────────────────────────────────────
    try:
        if active_tab == "tab-tail":
            if not s_tail:
                return _err("Enter a tail number.")
            matches = _INDEX.search(
                tail_number=s_tail or None,
                date=s_date or None,
                origin=s_orig_tail or None,
                dest=s_dest_tail or None,
            )
        else:
            if not s_fnum:
                return _err("Enter a flight number.")
            matches = _INDEX.search(
                flight_number=s_fnum or None,
                date=s_date_fnum or None,
                origin=s_orig_fnum or None,
                dest=s_dest_fnum or None,
            )
    except Exception:
        return _err("Search failed.", traceback.format_exc())

    if matches.empty:
        return (
            dbc.Alert("No flights found — try different criteria.", color="warning"),
            "",
        )

    row = matches.iloc[0]
    multi_note = (
        dbc.Alert(
            f"{len(matches):,} flights matched — showing the first. "
            "Add date / origin / dest to narrow down.",
            color="info", className="mb-3",
        ) if len(matches) > 1 else None
    )

    # ── Load model (cached after first call) ──────────────────────────
    models_found = discover_models(MODELS_DIR)
    if model_name not in models_found:
        return _err(f"Model '{model_name}' not found.")
    try:
        clf, reg = load_model_pair(
            models_found[model_name]["clf_path"],
            models_found[model_name]["reg_path"],
        )
    except Exception:
        return _err("Could not load model.", traceback.format_exc())

    # ── Predict ───────────────────────────────────────────────────────
    try:
        result = predict_row(row, clf, reg, threshold=threshold)
    except Exception:
        return _err("Prediction failed.", traceback.format_exc())

    pred_prob    = result["prob"]
    pred_delay   = result["delay_est"]
    pred_verdict = result["verdict"]

    # ── Actuals ───────────────────────────────────────────────────────
    actual_delay   = float(row.get(TARGET_REG,    0) or 0)
    actual_del15   = int(row.get(TARGET_CLASS,    0) or 0)
    actual_dep_del = float(row.get("dep_delay",   0) or 0)
    actual_verdict = "Delayed" if actual_del15 else "On time"
    cls_correct    = (pred_verdict == "Delayed") == bool(actual_del15)

    # ── Propagation chain ─────────────────────────────────────────────
    chain        = _INDEX.get_chain(row)           # fast — only touches ~20 rows
    prev2_origin = chain["prev2_origin"]
    prev2_dest   = chain["prev2_dest"]
    prev1_origin = chain["prev1_origin"]

    prev2_arr = float(row.get("prev2_arr_delay", 0) or 0)
    prev2_dep = float(row.get("prev2_dep_delay", 0) or 0)
    prev1_arr = float(row.get("prev1_arr_delay", 0) or 0)
    prev1_dep = float(row.get("prev1_dep_delay", 0) or 0)

    origin = str(row.get("origin", ""))
    dest_  = str(row.get("dest",   ""))

    # ── Display metadata ──────────────────────────────────────────────
    fid_disp   = str(row.get("flight_id", row.get("fl_num",
                     row.get("op_carrier_fl_num", s_fnum or "—"))))
    tail_disp  = str(row.get("tail_number", s_tail or "—"))
    carrier    = str(row.get("op_carrier", row.get("carrier", "—")))
    dep_date   = str(row.get("flight_date",
                     row.get("dep_ts_actual_utc", "—")))[:10]
    dep_hour   = row.get("dep_hour_local", "—")
    dist_raw   = row.get("distance")
    dist_disp  = f"{dist_raw:.0f} mi" if pd.notna(dist_raw) else "—"

    t_c  = row.get("dep_temp_c");         w_ms = row.get("dep_wind_speed_m_s")
    ceil = row.get("dep_ceiling_height_m")
    weather_str = (f"{t_c:.1f}°C  ·  {w_ms:.1f} m/s  ·  ceil {ceil:.0f} m"
                   if pd.notna(t_c) else "Weather not in file")

    prev1_turn = row.get("prev1_turnaround_minutes")
    tight_flag = row.get("tight_turnaround_flag", 0)
    leg_pos    = row.get("relative_leg_position")
    cum_dep    = row.get("cum_dep_delay_aircraft_day", 0)

    # ── Risk tier ─────────────────────────────────────────────────────
    if pred_prob < 0.3:
        tier, tier_color, tier_badge = "Low risk",    "#1D9E75", "success"
    elif pred_prob < 0.6:
        tier, tier_color, tier_badge = "Medium risk", "#EF9F27", "warning"
    else:
        tier, tier_color, tier_badge = "High risk",   "#E24B4A", "danger"

    reg_err   = pred_delay - actual_delay
    err_color = "#E24B4A" if abs(reg_err) > 20 else "#1D9E75"

    # ── Report components ─────────────────────────────────────────────

    header = dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([
            html.H5(f"{fid_disp}  —  {origin} → {dest_}", className="fw-bold mb-1"),
            html.Span(f"{tail_disp}  ·  {carrier}  ·  {dep_date}  ·  "
                      f"dep {dep_hour}:00  ·  {dist_disp}",
                      className="text-muted small"),
        ], md=8),
        dbc.Col([
            dbc.Badge(tier,       color=tier_badge,  className="me-2 fs-6"),
            dbc.Badge(model_name, color="secondary", className="me-2 fs-6"),
            dbc.Badge("✓ Correct" if cls_correct else "✗ Incorrect",
                      color="success" if cls_correct else "danger", className="fs-6"),
        ], md=4, className="text-end d-flex align-items-center justify-content-end"),
    ])), className="mb-3 shadow-sm border-start border-4 "
                   + ("border-success" if cls_correct else "border-danger"))

    def _mc(label, value, color="#000"):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.P(label, className="text-muted small mb-1"),
            html.H4(value, className="mb-0 fw-bold", style={"color": color}),
        ]), className="shadow-sm h-100"), md=3)

    metrics = dbc.Row([
        _mc("Actual outcome",    actual_verdict,
            "#E24B4A" if actual_del15 else "#1D9E75"),
        _mc("Predicted outcome", pred_verdict,
            "#E24B4A" if pred_verdict == "Delayed" else "#1D9E75"),
        _mc("Actual delay",      f"{actual_delay:.0f} min", "#E24B4A"),
        _mc("Predicted delay",   f"{pred_delay:.0f} min",   tier_color),
    ], className="mb-3 g-2")

    pred_card = dbc.Card([
        dbc.CardHeader(html.B("Prediction vs actual")),
        dbc.CardBody(dbc.Row([
            dbc.Col([
                html.P("Classification", className="fw-semibold small text-muted mb-2"),
                html.Div([
                    dbc.Badge(pred_verdict,
                              color="danger" if pred_verdict == "Delayed" else "success",
                              className="me-2"),
                    html.Span(f"P(delay≥15) = {pred_prob:.3f}", className="small"),
                ], className="mb-1"),
                html.Div([
                    html.Span("Actual: ", className="small text-muted me-1"),
                    dbc.Badge(actual_verdict,
                              color="danger" if actual_del15 else "success",
                              className="me-2"),
                    dbc.Badge("✓ Correct" if cls_correct else "✗ Incorrect",
                              color="success" if cls_correct else "danger"),
                ]),
            ], md=5),
            dbc.Col([
                html.P("Regression", className="fw-semibold small text-muted mb-2"),
                html.P(f"Predicted:  {pred_delay:.1f} min", className="small mb-1"),
                html.P(f"Actual:     {actual_delay:.1f} min", className="small mb-1"),
                html.P(f"Error:      {reg_err:+.1f} min",
                       className="small mb-0 fw-bold", style={"color": err_color}),
            ], md=4),
            dbc.Col([
                html.P("Settings", className="fw-semibold small text-muted mb-2"),
                html.P(f"Threshold: {threshold:.2f}", className="small mb-1"),
                html.P(f"Model: {model_name}", className="small mb-0 text-muted"),
            ], md=3),
        ])),
    ], className="mb-3 shadow-sm")

    context_card = dbc.Card([
        dbc.CardHeader(html.B("Flight context")),
        dbc.CardBody(dbc.Row([
            dbc.Col([
                html.P("Propagation", className="fw-semibold small text-muted mb-1"),
                html.P(f"prev1  arr {prev1_arr:.0f} min  dep {prev1_dep:.0f} min",
                       className="small mb-0"),
                html.P(f"prev2  arr {prev2_arr:.0f} min  dep {prev2_dep:.0f} min",
                       className="small mb-0"),
            ], md=3),
            dbc.Col([
                html.P("Turnaround", className="fw-semibold small text-muted mb-1"),
                html.P(f"{prev1_turn:.0f} min" if pd.notna(prev1_turn) else "—",
                       className="small mb-0"),
                html.P(f"Tight: {'Yes ⚠' if tight_flag else 'No'}",
                       className="small mb-0"),
            ], md=3),
            dbc.Col([
                html.P("Aircraft day", className="fw-semibold small text-muted mb-1"),
                html.P(f"Leg pos: {leg_pos:.0%}" if pd.notna(leg_pos) else "—",
                       className="small mb-0"),
                html.P(f"Cum dep delay: {cum_dep:.0f} min", className="small mb-0"),
            ], md=3),
            dbc.Col([
                html.P("Weather at dep", className="fw-semibold small text-muted mb-1"),
                html.P(weather_str, className="small mb-0"),
            ], md=3),
        ])),
    ], className="mb-3 shadow-sm")

    def _chart(fig):
        return dbc.Card(dbc.CardBody(
            dcc.Graph(figure=fig, config={"displayModeBar": False})
        ), className="mb-3 shadow-sm")

    map_fig  = route_map(
        origin=origin, dest=dest_,
        prev1_origin=prev1_origin, prev1_dest=origin,
        prev2_origin=prev2_origin, prev2_dest=prev2_dest,
        actual_arr_delay=actual_delay,
        prev1_arr_delay=prev1_arr, prev2_arr_delay=prev2_arr,
        flight_id=fid_disp, tail_number=tail_disp,
    )
    avp_fig  = actual_vs_predicted(
        actual_delay=actual_delay, actual_del15=actual_del15,
        pred_prob=pred_prob, pred_delay=pred_delay, threshold=threshold,
    )
    prop_fig = propagation_chain(
        prev2_origin=prev2_origin, prev2_dest=prev2_dest,
        prev1_origin=prev1_origin, prev1_dest=origin,
        origin=origin, dest=dest_,
        prev2_arr_delay=prev2_arr, prev2_dep_delay=prev2_dep,
        prev1_arr_delay=prev1_arr, prev1_dep_delay=prev1_dep,
        actual_arr_delay=actual_delay, actual_dep_delay=actual_dep_del,
    )
    feat_fig = feature_contributions(row, XGB_FULL_FEATURES)

    return (
        [
            *([multi_note] if multi_note else []),
            header, metrics, pred_card, context_card,
            _chart(map_fig),
            dbc.Row([
                dbc.Col(_chart(avp_fig),  md=5),
                dbc.Col(_chart(prop_fig), md=7),
            ], className="g-2 mb-3"),
            _chart(feat_fig),
        ],
        "",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)