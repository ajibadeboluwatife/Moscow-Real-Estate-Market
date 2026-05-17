# Moscow Real Estate Market

Exploratory Data Analysis (EDA) of the Moscow real estate market using listing data for the **secondary market**, **new builds**, and **rentals**. The project combines data preparation, quality checks, and visualization to explore pricing patterns across geography, transit access, building characteristics, and time.

## Overview

This repository contains a Python analysis script that investigates how Moscow property prices and rents vary by:

- market segment (secondary, new build, rental)
- district and okrug
- distance to the city center
- proximity to metro stations
- room count and total area
- renovation level or project class
- time and macro indicators such as mortgage and key rates

The analysis is designed for exploratory work and presentation-quality charts rather than as a packaged Python library.

## What the project does

The main script performs several stages of analysis:

1. **Locate the dataset** from common local and Kaggle-style paths.
2. **Load multiple CSV files** for the different market segments.
3. **Create derived features** such as:
   - listing month and listing year
   - floor ratio
   - log-transformed size and distance variables
   - building age / years to completion
4. **Run data quality checks** including:
   - missing-value inspection
   - duplicate and ID checks
   - arithmetic consistency checks for price vs. price-per-square-meter
5. **Generate visualizations** for:
   - price distributions
   - district price gradients
   - okrug market comparisons
   - metro-distance pricing relationships
   - room-count pricing patterns
   - renovation / class premiums
   - market trends over time

## Repository structure

```text
.
├── README.md
├── realestate.py
├── requirements.txt
└── .gitignore
```

### Files

- `realestate.py` — main exploratory analysis script
- `README.md` — project documentation
- `requirements.txt` — Python dependencies used by the analysis
- `.gitignore` — excludes Python caches, notebook artifacts, virtual environments, and local dataset files

## Dataset requirements

The script expects the following CSV files to be available in a local `dataset/` folder or a Kaggle input directory:

- `secondary_market.csv`
- `rentals.csv`
- `new_builds.csv`
- `district_prices_monthly.csv`
- `metro_stations.csv`

The script attempts to discover data in these locations:

- `/kaggle/input/datasets/sergionefedov/moscow-real-estate-sales-and-rentals-20202026`
- `/kaggle/input/moscow-real-estate-sales-and-rentals-20202026`
- `./dataset`
- `../dataset`
- current working directory

If the files are not found, it raises a `FileNotFoundError` with the searched paths.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ajibadeboluwatife/Moscow-Real-Estate-Market.git
cd Moscow-Real-Estate-Market
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Create a `dataset/` directory in the project root and place the required CSV files inside it.

```text
Moscow-Real-Estate-Market/
├── dataset/
│   ├── secondary_market.csv
│   ├── rentals.csv
│   ├── new_builds.csv
│   ├── district_prices_monthly.csv
│   └── metro_stations.csv
├── README.md
└── realestate.py
```

## How to run

This project currently uses a **notebook-style Python script**. It includes IPython-specific behavior (for example, inline display configuration), so it is best suited for:

- **Jupyter Notebook**
- **JupyterLab**
- **Kaggle Notebooks**
- an IPython-enabled environment

To run in Jupyter, you can either:

- paste/adapt the script into a notebook, or
- run it in an environment that supports IPython magics and inline plotting

If you want to make the project fully command-line runnable later, a good next step would be refactoring notebook-specific parts into standard Python functions and adding a `main()` entry point.

## Dependencies

The script imports the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `ipython`

Standard-library modules used include:

- `math`
- `warnings`
- `pathlib`
- `textwrap`

## Key analysis themes

Some of the main questions this project explores are:

- How do prices per square meter differ across market segments?
- How strongly do prices vary with distance to the city center?
- How does metro accessibility relate to pricing?
- Which districts and okrugs trade above or below their market medians?
- How do renovation quality and new-build class affect price premiums?
- How have prices, rents, and listing volumes changed over time?

## Notes and caveats

- The current repository is focused on **exploratory analysis**, not model deployment or production packaging.
- The script is relatively large and monolithic, so future refactoring could improve readability and reuse.
- Because the workflow is notebook-oriented, some parts may need adjustment for plain terminal execution.
- Results depend on the availability and structure of the source CSV files.

## Suggested future improvements

Possible next steps for the project include:

- splitting the large analysis script into reusable modules
- adding a notebook version of the analysis
- exporting figures to a dedicated `reports/` or `figures/` folder
- adding sample outputs or screenshots to the README
- introducing data validation checks with clearer reporting
- adding a `main()` function for easier script execution
- packaging helper functions into a utilities module

## Author

Created by **ajibadeboluwatife**.
