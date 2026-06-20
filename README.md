# Australian Housing & Rental Forecasting for International Students

End-to-end ML pipeline forecasting Australian dwelling prices and rents by
capital city, with an affordability layer aimed at international students
choosing where to study.

## Status: Phase 1 - Data Acquisition (in progress)

## Project structure

```
housing-forecast-au/
├── data/
│   ├── raw/              # original ABS / SQM downloads
│   └── processed/         # cleaned, merged, feature-engineered data
├── notebooks/              # EDA, experimentation
├── src/
│   ├── data/                # acquisition + cleaning scripts
│   ├── features/             # feature engineering (spatio-temporal lag etc.)
│   ├── models/                # ARIMA, Prophet, Random Forest, XGBoost, TFT
│   └── api/                    # FastAPI / Streamlit serving layer
├── tests/
├── reports/                  # evaluation outputs, figures
├── docker/
└── .github/workflows/        # CI/CD
```

## Data sources

| Data | Source | Access method |
|---|---|---|
| Dwelling prices (RPPI) | ABS Data API | Programmatic (SDMX REST, see `src/data/fetch_abs_rppi.py`) |
| Rents | ABS CPI rent series + SQM Research | Manual download (see `src/data/fetch_rents.py` docstring) |
| Cost of living | Numbeo (planned) | TBD |

## Model lineup (Phase 3)

| Model | Scope | Role |
|---|---|---|
| ARIMA | Per-city | Statistical baseline |
| Prophet | Per-city | Trend/seasonality baseline |
| Random Forest | Pooled | Ensemble baseline |
| XGBoost | Pooled | Primary best-evidenced model |
| Temporal Fusion Transformer | Pooled | Novel/differentiated model, probabilistic multi-horizon forecasts |

Model choice is grounded in literature: tree-based ensembles with
spatio-temporal lag features outperformed linear models in a 32-year,
428,000-transaction Adelaide housing study (Soltani et al.), and XGBoost has
outperformed Random Forest in most direct comparisons in the broader house
price prediction literature.

## Setup

```bash
pip install -r requirements.txt
python src/data/fetch_abs_rppi.py     # pulls dwelling price data
python src/data/fetch_rents.py         # combines manually-downloaded rent data
```

## Disclaimer

This project produces trend-based forecasts with uncertainty ranges, not
precise point predictions. It is intended as a directional guide for
international students comparing cities, not financial advice.
