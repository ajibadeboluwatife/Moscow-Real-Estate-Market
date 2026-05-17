# Moscow Real Estate Market

Exploratory Data Analysis (EDA) of Moscow real-estate listings across the secondary market, new-build sales, and rentals.

## Project overview
This repository contains a single end-to-end Python analysis script (`realestate.py`) that:
- loads multiple market datasets,
- performs data preparation and feature engineering,
- explores pricing dynamics by market, geography, and time,
- produces a large set of visualizations for comparative analysis.

The project is designed in a notebook-style workflow but is stored as a `.py` script.

## What the analysis covers
Based on the current script, the analysis includes:
- Secondary vs new-build vs rental market comparisons
- Price-per-square-meter and headline-price distributions
- District/Okrug-level market structure
- Metro and distance-to-center effects
- Time-series market pulse and listing liquidity trends
- New-build class/subsidy comparisons
- Correlation-based early feature signal checks

## Repository structure
```text
.
├── README.md
├── realestate.py
├── requirements.txt
└── dataset/
    └── README.md
```

## Dataset requirements
`realestate.py` expects the following CSV files:
- `secondary_market.csv`
- `rentals.csv`
- `new_builds.csv`
- `district_prices_monthly.csv`
- `metro_stations.csv`

### Expected locations
The script automatically searches in:
1. Kaggle input locations, including:
   - `/kaggle/input/datasets/sergionefedov/moscow-real-estate-sales-and-rentals-20202026`
   - `/kaggle/input/moscow-real-estate-sales-and-rentals-20202026`
2. Local paths:
   - `./dataset`
   - `../dataset`
   - current working directory

For local use, place the CSV files in `dataset/`.

## Setup
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to run
### Option A (recommended): Jupyter/Notebook-style execution
Because the script is notebook-oriented (visual-heavy and display-first), running in Jupyter is the best experience.

### Option B: Run as a Python script
```bash
python realestate.py
```
This executes the full analysis pipeline and renders plots using your active Matplotlib backend.

## Key dependencies
- Python 3.10+ (tested locally here with Python 3.12)
- pandas
- numpy
- matplotlib
- seaborn
- ipython

(See `requirements.txt` for installable package names.)

## Kaggle/local execution notes
- The script includes built-in path resolution for Kaggle and local folders.
- If files are missing, it raises a `FileNotFoundError` with searched locations.
- The script is intentionally analysis-centric and does not currently expose a CLI with arguments.

## Suggested future improvements
- Split `realestate.py` into logical modules (loading, cleaning, plotting, utilities).
- Add a small `--data-dir` CLI argument for explicit local path control.
- Add lightweight checks/tests for dataset schema validation.
- Export key figures/tables to an `outputs/` directory for reproducibility.
