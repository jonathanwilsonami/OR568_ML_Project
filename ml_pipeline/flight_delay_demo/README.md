# Flight delay model — production demo  (v2)

Interactive Dash app that loads a real parquet file, engineers features,
runs your trained XGBoost models, and shows a full flight report with
geographic map, actual-vs-predicted charts, and propagation chain.

## Quick start

```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

cd flight_delay_demo
poetry install
poetry run python app.py
# → http://localhost:8050
```

Without Poetry:
```bash
pip install dash dash-bootstrap-components plotly pandas numpy pyarrow joblib xgboost
python app.py
```

## How to use the app

1. **Load data** — paste the path to a `.parquet` file (e.g. `data/flights_2024.parquet`) and click **Load file**.
2. **Search** — enter a Flight ID, tail number, origin, or destination (any combination).
3. **Run** — choose a model, adjust the threshold if needed, click **Run prediction**.

The report shows:
- Geographic route map with all 3 legs coloured by delay severity
- Actual vs predicted gauge + bar chart
- Propagation chain (prev2 → prev1 → current)
- Key feature values for this specific flight

## Model files

Place joblib files in the `models/` folder with this naming pattern:

```
models/
├── xgb_classifier_xgb_full.joblib
└── xgb_regressor_xgb_full.joblib
```

Any `xgb_classifier_<name>.joblib` / `xgb_regressor_<name>.joblib` pair is
auto-discovered and appears in the model selector dropdown.

## Project structure

```
flight_delay_demo/
├── app.py          Dash layout + callbacks
├── data.py         Parquet loader + feature engineering
├── charts.py       Plotly figure factories
├── airports.py     IATA → lat/lon lookup
├── pyproject.toml
└── models/
    ├── __init__.py
    ├── loader.py   Model discovery + inference
    └── *.joblib    Your trained model files go here
```

## Adding more models

Drop additional `xgb_classifier_<name>.joblib` + `xgb_regressor_<name>.joblib`
pairs into `models/` — they appear in the dropdown automatically on next restart.

## Parquet column requirements

The parquet file should contain at minimum:

| Column | Description |
|--------|-------------|
| `tail_number` | Aircraft identifier for propagation chain |
| `origin` / `dest` | IATA airport codes |
| `dep_ts_actual_utc` | Actual departure timestamp (UTC) |
| `arr_ts_actual_utc` | Actual arrival timestamp (UTC) |
| `arr_delay` | Actual arrival delay in minutes (regression target) |
| `arr_del15` | Binary delay flag ≥15 min (classification target) |

All other features (weather, propagation, traffic volume) are derived automatically.
Missing columns are filled with 0 — the model will still run but accuracy may be lower.