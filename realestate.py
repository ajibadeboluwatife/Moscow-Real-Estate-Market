from __future__ import annotations

import math
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from IPython import get_ipython
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 80)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

REQUIRED_FILES = {
    "secondary": "secondary_market.csv",
    "rentals": "rentals.csv",
    "new_builds": "new_builds.csv",
    "district_monthly": "district_prices_monthly.csv",
    "metro": "metro_stations.csv",
}


def has_required_files(path: Path) -> bool:
    return path.exists() and all((path / filename).exists() for filename in REQUIRED_FILES.values())


def resolve_data_dir() -> Path:
    """Resolve data path locally and inside Kaggle notebooks."""
    cwd = Path.cwd()
    candidate_dirs = [
        Path("/kaggle/input/datasets/sergionefedov/moscow-real-estate-sales-and-rentals-20202026"),
        Path("/kaggle/input/moscow-real-estate-sales-and-rentals-20202026"),
        cwd / "dataset",
        cwd.parent / "dataset",
        cwd,
    ]
    for candidate in candidate_dirs:
        if has_required_files(candidate):
            return candidate

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for candidate in kaggle_root.rglob("secondary_market.csv"):
            parent = candidate.parent
            if has_required_files(parent):
                return parent

    searched = "\n".join(f"- {path}" for path in candidate_dirs)
    raise FileNotFoundError(
        "Could not locate the Moscow real estate CSV files. "
        "Attach the Kaggle dataset or place the files under ./dataset.\n"
        f"Searched:\n{searched}"
    )


PROJECT_ROOT = Path.cwd()
DATA_DIR = resolve_data_dir()

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("config", 'InlineBackend.figure_format = "retina"')

PALETTE = {
    "paper": "#F7F1E3",
    "ink": "#101820",
    "muted": "#64748B",
    "grid": "#D7D1C4",
    "secondary": "#B23A48",   # oxblood
    "new_build": "#1D7874",   # oxidized teal
    "rental": "#D99A30",      # brass
    "metro": "#355070",       # deep signal blue
    "violet": "#6D597A",
    "green": "#588157",
    "rose": "#E56B6F",
    "sand": "#D6C6A8",
    "charcoal": "#2B2D42",
}

MARKET_COLORS = {
    "Secondary": PALETTE["secondary"],
    "New build": PALETTE["new_build"],
    "Rental": PALETTE["rental"],
}

OKRUG_COLORS = {
    "CAO": "#B23A48",
    "SAO": "#355070",
    "SVAO": "#1D7874",
    "VAO": "#588157",
    "YuVAO": "#D99A30",
    "YuAO": "#E56B6F",
    "YuZAO": "#6D597A",
    "ZAO": "#457B9D",
    "SZAO": "#A98467",
    "NAO": "#7A9E7E",
    "TAO": "#8D99AE",
    "Zelenograd": "#2A9D8F",
}

mpl.rcParams.update({
    "figure.facecolor": PALETTE["paper"],
    "axes.facecolor": PALETTE["paper"],
    "savefig.facecolor": PALETTE["paper"],
    "axes.edgecolor": PALETTE["ink"],
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"],
    "text.color": PALETTE["ink"],
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.alpha": 0.55,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def rub_mln(x, pos=None):
    return f"{x / 1_000_000:.0f}M"


def rub_k(x, pos=None):
    return f"{x / 1_000:.0f}K"


def pct(x, pos=None):
    return f"{100 * x:.0f}%"


def style_ax(ax, title=None, subtitle=None, xlabel=None, ylabel=None):
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    ax.tick_params(length=0, pad=6)
    if title:
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=34)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 12), textcoords="offset points", ha="left", va="bottom", fontsize=10.5, color=PALETTE["muted"], clip_on=False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=10)
    return ax


def annotate_corner(ax, text, loc="upper right"):
    x = 0.98 if "right" in loc else 0.02
    y = 0.96 if "upper" in loc else 0.04
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9.5,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF9EC", edgecolor=PALETTE["grid"], alpha=0.92),
    )


def robust_iqr(series):
    q1, q2, q3 = series.quantile([0.25, 0.50, 0.75])
    return pd.Series({"q25": q1, "median": q2, "q75": q3})

raw = {name: pd.read_csv(DATA_DIR / filename) for name, filename in REQUIRED_FILES.items()}

secondary = raw["secondary"].copy()
rentals = raw["rentals"].copy()
new_builds = raw["new_builds"].copy()
district_monthly = raw["district_monthly"].copy()
metro = raw["metro"].copy()

for df in [secondary, rentals, new_builds]:
    df["date_posted"] = pd.to_datetime(df["date_posted"])
    df["listing_month"] = df["date_posted"].dt.to_period("M").dt.to_timestamp()
    df["listing_year"] = df["date_posted"].dt.year
    df["floor_ratio"] = df["floor"] / df["total_floors"].replace(0, np.nan)
    df["log_area"] = np.log1p(df["total_area"])
    df["log_metro_distance_min"] = np.log1p(df["metro_distance_min"])
    df["log_to_center_km"] = np.log1p(df["to_center_km"])

secondary["market"] = "Secondary"
secondary["headline_value"] = secondary["price_rub"]
secondary["headline_per_sqm"] = secondary["price_per_sqm"]
secondary["building_age"] = secondary["date_posted"].dt.year - secondary["building_year"]
secondary["log_target"] = np.log1p(secondary["price_rub"])
secondary["log_target_per_sqm"] = np.log1p(secondary["price_per_sqm"])

new_builds["market"] = "New build"
new_builds["headline_value"] = new_builds["price_rub"]
new_builds["headline_per_sqm"] = new_builds["price_per_sqm"]
new_builds["years_to_completion"] = new_builds["completion_year"] - new_builds["date_posted"].dt.year
new_builds["log_target"] = np.log1p(new_builds["price_rub"])
new_builds["log_target_per_sqm"] = np.log1p(new_builds["price_per_sqm"])

rentals["market"] = "Rental"
rentals["headline_value"] = rentals["monthly_rent_rub"]
rentals["headline_per_sqm"] = rentals["rent_per_sqm"]
rentals["building_age"] = rentals["date_posted"].dt.year - rentals["building_year"]
rentals["log_target"] = np.log1p(rentals["monthly_rent_rub"])
rentals["log_target_per_sqm"] = np.log1p(rentals["rent_per_sqm"])

district_monthly["year_month"] = pd.to_datetime(district_monthly["year_month"])
district_monthly["month"] = district_monthly["year_month"].dt.to_period("M").dt.to_timestamp()

listing_frames = {
    "Secondary": secondary,
    "New build": new_builds,
    "Rental": rentals,
}

common_cols = [
    "market", "id", "date_posted", "listing_month", "district", "okrug", "lat", "lon", "total_area", "rooms",
    "floor", "total_floors", "floor_ratio", "metro_station", "metro_line", "metro_distance_min", "to_center_km",
    "headline_value", "headline_per_sqm", "log_target", "log_target_per_sqm",
]
listings = pd.concat([df[common_cols] for df in listing_frames.values()], ignore_index=True)

print({name: df.shape for name, df in raw.items()})
def table_contract(name, df):
    date_col = "date_posted" if "date_posted" in df.columns else "year_month" if "year_month" in df.columns else None
    if date_col:
        dt = pd.to_datetime(df[date_col])
        date_range = f"{dt.min().date()} to {dt.max().date()}"
    else:
        date_range = "n/a"
    return {
        "table": name,
        "rows": len(df),
        "columns": df.shape[1],
        "date_range": date_range,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": df.isna().mean().mean(),
        "districts": df["district"].nunique() if "district" in df.columns else np.nan,
        "okrugs": df["okrug"].nunique() if "okrug" in df.columns else np.nan,
    }

contract = pd.DataFrame([table_contract(name, df) for name, df in raw.items()])
contract_style = (
    contract.style
    .format({"rows": "{:,.0f}", "columns": "{:,.0f}", "duplicate_rows": "{:,.0f}", "missing_cells": "{:,.0f}", "missing_pct": "{:.2%}", "districts": "{:.0f}", "okrugs": "{:.0f}"})
    .background_gradient(subset=["rows"], cmap="YlOrBr")
    .background_gradient(subset=["missing_pct"], cmap="Reds")
)
contract_style
# Column-level missingness, rendered as a compact atlas.
missing_records = []
for name, df in raw.items():
    miss = df.isna().mean().sort_values(ascending=False)
    for col, val in miss.items():
        missing_records.append({"table": name, "column": col, "missing_pct": val})
missing = pd.DataFrame(missing_records)

wide_missing = missing.pivot(index="column", columns="table", values="missing_pct").fillna(np.nan)
wide_missing = wide_missing.loc[wide_missing.max(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, max(5, 0.22 * len(wide_missing))))
sns.heatmap(
    wide_missing,
    ax=ax,
    cmap=sns.light_palette(PALETTE["secondary"], as_cmap=True),
    vmin=0, vmax=max(0.05, float(np.nanmax(wide_missing.values))),
    linewidths=0.6,
    linecolor=PALETTE["paper"],
    cbar_kws={"format": mticker.FuncFormatter(pct), "label": "missing cells"},
)
style_ax(ax, "Missingness atlas", "Only district month-over-month changes should be absent in the first observed month.", xlabel="", ylabel="")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
def arithmetic_check(df, value_col, unit_col, area_col="total_area", tolerance=0.035):
    expected = df[unit_col] * df[area_col]
    rel_error = (df[value_col] - expected).abs() / df[value_col].replace(0, np.nan)
    return pd.Series({
        "median_relative_error": rel_error.median(),
        "p95_relative_error": rel_error.quantile(0.95),
        "share_over_tolerance": (rel_error > tolerance).mean(),
        "max_relative_error": rel_error.max(),
    })

quality_checks = pd.DataFrame({
    "secondary_price_arithmetic": arithmetic_check(secondary, "price_rub", "price_per_sqm"),
    "newbuild_price_arithmetic": arithmetic_check(new_builds, "price_rub", "price_per_sqm"),
    "rental_price_arithmetic": arithmetic_check(rentals, "monthly_rent_rub", "rent_per_sqm"),
}).T

id_checks = []
for market, df in listing_frames.items():
    id_checks.append({
        "market": market,
        "rows": len(df),
        "unique_ids": df["id"].nunique(),
        "duplicate_id_rows": int(df["id"].duplicated().sum()),
        "min_lat": df["lat"].min(),
        "max_lat": df["lat"].max(),
        "min_lon": df["lon"].min(),
        "max_lon": df["lon"].max(),
    })
id_checks = pd.DataFrame(id_checks)

display(Markdown("**Arithmetic coherence of value, area, and per-sqm price**"))
display(quality_checks.style.format("{:.2%}").background_gradient(cmap="YlOrRd"))

display(Markdown("**ID and coordinate checks**"))
display(id_checks.style.format({"rows":"{:,.0f}", "unique_ids":"{:,.0f}", "duplicate_id_rows":"{:,.0f}", "min_lat":"{:.3f}", "max_lat":"{:.3f}", "min_lon":"{:.3f}", "max_lon":"{:.3f}"}))
summary_rows = []
for market, df in listing_frames.items():
    metric = "rent_per_sqm" if market == "Rental" else "price_per_sqm"
    value = "monthly_rent_rub" if market == "Rental" else "price_rub"
    summary_rows.append({
        "market": market,
        "listings": len(df),
        "median_value_rub": df[value].median(),
        "median_per_sqm_rub": df[metric].median(),
        "p10_per_sqm_rub": df[metric].quantile(0.10),
        "p90_per_sqm_rub": df[metric].quantile(0.90),
        "median_area_sqm": df["total_area"].median(),
        "median_center_km": df["to_center_km"].median(),
        "median_metro_min": df["metro_distance_min"].median(),
    })
market_summary = pd.DataFrame(summary_rows).set_index("market")
market_summary.style.format({
    "listings": "{:,.0f}",
    "median_value_rub": "{:,.0f}",
    "median_per_sqm_rub": "{:,.0f}",
    "p10_per_sqm_rub": "{:,.0f}",
    "p90_per_sqm_rub": "{:,.0f}",
    "median_area_sqm": "{:.1f}",
    "median_center_km": "{:.1f}",
    "median_metro_min": "{:.0f}",
}).background_gradient(subset=["median_per_sqm_rub"], cmap="YlGnBu")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=False)
plot_specs = [
    ("Secondary", secondary, "price_per_sqm", "RUB / sqm", (0, secondary["price_per_sqm"].quantile(0.995))),
    ("New build", new_builds, "price_per_sqm", "RUB / sqm", (0, new_builds["price_per_sqm"].quantile(0.995))),
    ("Rental", rentals, "rent_per_sqm", "RUB / sqm / month", (0, rentals["rent_per_sqm"].quantile(0.995))),
]

for ax, (market, df, metric, label, xlim) in zip(axes, plot_specs):
    color = MARKET_COLORS[market]
    x = df[metric]
    bins = np.linspace(x.quantile(0.005), x.quantile(0.995), 44)
    ax.hist(x, bins=bins, color=color, alpha=0.72, edgecolor=PALETTE["paper"], linewidth=0.6)
    for q, lw, alpha in [(0.50, 2.4, 1.0), (0.25, 1.2, 0.65), (0.75, 1.2, 0.65)]:
        ax.axvline(x.quantile(q), color=PALETTE["ink"], lw=lw, alpha=alpha)
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(rub_k))
    style_ax(ax, market, f"median {x.median():,.0f}; IQR {x.quantile(.25):,.0f}-{x.quantile(.75):,.0f}", xlabel=label, ylabel="listings")

fig.suptitle("Price density by market", x=0.06, ha="left", y=0.98, fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
def district_market_profile(df, metric, market):
    out = (
        df.groupby(["district", "okrug"], observed=True)
        .agg(
            listings=(metric, "size"),
            median_per_sqm=(metric, "median"),
            q25=(metric, lambda s: s.quantile(0.25)),
            q75=(metric, lambda s: s.quantile(0.75)),
            median_center_km=("to_center_km", "median"),
            median_metro_min=("metro_distance_min", "median"),
            median_area=("total_area", "median"),
        )
        .reset_index()
    )
    out["market"] = market
    return out

profiles = pd.concat([
    district_market_profile(secondary, "price_per_sqm", "Secondary"),
    district_market_profile(new_builds, "price_per_sqm", "New build"),
    district_market_profile(rentals, "rent_per_sqm", "Rental"),
], ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True)
for ax, market in zip(axes, ["Secondary", "New build", "Rental"]):
    prof = profiles.query("market == @market").copy()
    sizes = np.interp(prof["listings"], (prof["listings"].min(), prof["listings"].max()), (28, 260))
    for okrug, g in prof.groupby("okrug", observed=True):
        ax.scatter(
            g["median_center_km"], g["median_per_sqm"],
            s=np.interp(g["listings"], (prof["listings"].min(), prof["listings"].max()), (28, 260)),
            c=OKRUG_COLORS.get(okrug, PALETTE["muted"]),
            alpha=0.76,
            edgecolors=PALETTE["paper"],
            linewidths=0.8,
            label=okrug if market == "Secondary" else None,
        )
    # Smooth median trend using distance bins.
    prof["distance_bin"] = pd.qcut(prof["median_center_km"], q=12, duplicates="drop")
    trend = prof.groupby("distance_bin", observed=True).agg(x=("median_center_km", "median"), y=("median_per_sqm", "median")).dropna()
    ax.plot(trend["x"], trend["y"], color=PALETTE["ink"], lw=2.2, alpha=0.82)
    ax.set_xlim(0, profiles["median_center_km"].quantile(0.995) + 2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(rub_k))
    style_ax(ax, market, "district medians; bubble area = listing count", xlabel="median distance to center, km", ylabel="RUB / sqm" if market != "Rental" else "RUB / sqm / month")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=9)
fig.suptitle("District price gradients", x=0.055, ha="left", y=0.98, fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0.08, 1, 0.88])
plt.show()
# Okrug market fingerprint: normalized medians expose relative positioning across markets.
okrug_market = []
for market, df, metric in [
    ("Secondary", secondary, "price_per_sqm"),
    ("New build", new_builds, "price_per_sqm"),
    ("Rental", rentals, "rent_per_sqm"),
]:
    temp = df.groupby("okrug", observed=True).agg(median_per_sqm=(metric, "median"), listings=(metric, "size")).reset_index()
    temp["market"] = market
    okrug_market.append(temp)
okrug_market = pd.concat(okrug_market, ignore_index=True)
okrug_market["market_index"] = okrug_market.groupby("market")["median_per_sqm"].transform(lambda s: s / s.median())
heat = okrug_market.pivot(index="okrug", columns="market", values="market_index")
heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index, ["Secondary", "New build", "Rental"]]

fig, ax = plt.subplots(figsize=(7.7, 6.6))
sns.heatmap(
    heat,
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(15, 175, s=80, l=45, center="light", as_cmap=True),
    center=1,
    linewidths=1,
    linecolor=PALETTE["paper"],
    cbar_kws={"label": "index vs market median"},
    ax=ax,
)
style_ax(ax, "Okrug market fingerprint", "Values above 1.00 trade richer than the city median for that market.", xlabel="", ylabel="")
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
fig, ax = plt.subplots(figsize=(9.5, 9))

sample = secondary.sample(min(30000, len(secondary)), random_state=42)
hb = ax.hexbin(
    sample["lon"], sample["lat"],
    C=sample["price_per_sqm"],
    reduce_C_function=np.median,
    gridsize=55,
    mincnt=4,
    cmap=mpl.colors.LinearSegmentedColormap.from_list("price_field", ["#ECE2C6", "#D99A30", "#B23A48", "#2B2D42"]),
    linewidths=0,
    alpha=0.92,
)

for line, g in metro.groupby("line", observed=True):
    ax.scatter(g["lon"], g["lat"], s=15, color=PALETTE["paper"], edgecolor=PALETTE["charcoal"], linewidth=0.5, alpha=0.9)

cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(rub_k))
cb.set_label("median secondary RUB / sqm")

style_ax(ax, "Moscow secondary-market price field", "Hex cells aggregate resale listings; station points show the metro skeleton.", xlabel="longitude", ylabel="latitude")
ax.set_aspect("equal", adjustable="box")
annotate_corner(ax, f"{len(sample):,} sampled listings\n{len(metro):,} metro stations", loc="lower right")
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
def distance_ribbon(df, metric, market, max_min=300, bins=18):
    d = df.loc[df["metro_distance_min"].between(0, max_min), ["metro_distance_min", metric]].copy()
    d["bin"] = pd.cut(d["metro_distance_min"], np.linspace(0, max_min, bins + 1), include_lowest=True)
    out = d.groupby("bin", observed=True).agg(
        x=("metro_distance_min", "median"),
        q25=(metric, lambda s: s.quantile(0.25)),
        median=(metric, "median"),
        q75=(metric, lambda s: s.quantile(0.75)),
        n=(metric, "size"),
    ).dropna().reset_index(drop=True)
    out["market"] = market
    return out

ribbons = pd.concat([
    distance_ribbon(secondary, "price_per_sqm", "Secondary"),
    distance_ribbon(new_builds, "price_per_sqm", "New build"),
    distance_ribbon(rentals, "rent_per_sqm", "Rental"),
], ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True)
for ax, market in zip(axes, ["Secondary", "New build", "Rental"]):
    d = ribbons.query("market == @market")
    c = MARKET_COLORS[market]
    ax.fill_between(d["x"].to_numpy(), d["q25"].to_numpy(), d["q75"].to_numpy(), color=c, alpha=0.18, linewidth=0)
    ax.plot(d["x"].to_numpy(), d["median"].to_numpy(), color=c, lw=3)
    ax.scatter(d["x"], d["median"], s=np.clip(d["n"] / d["n"].max() * 120, 20, 120), color=c, edgecolor=PALETTE["paper"], linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(rub_k))
    style_ax(ax, market, "ribbon = IQR; point size = bin listings", xlabel="minutes to nearest metro", ylabel="RUB / sqm" if market != "Rental" else "RUB / sqm / month")

fig.suptitle("Transit distance gradient", x=0.055, ha="left", y=0.98, fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
for ax, (market, df, metric) in zip(
    axes,
    [("Secondary", secondary, "price_per_sqm"), ("New build", new_builds, "price_per_sqm"), ("Rental", rentals, "rent_per_sqm")],
):
    room_order = sorted(df["rooms"].unique())
    sns.boxenplot(
        data=df[df["rooms"].isin(room_order)],
        x="rooms",
        y=metric,
        order=room_order,
        color=MARKET_COLORS[market],
        linewidth=0.9,
        saturation=0.88,
        ax=ax,
        showfliers=False,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(rub_k))
    style_ax(ax, market, "tail suppressed; focus on the central price architecture", xlabel="rooms", ylabel="RUB / sqm" if market != "Rental" else "RUB / sqm / month")

fig.suptitle("Room-count price architecture", x=0.055, ha="left", y=0.98, fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
def premium_table(df, group_col, metric, min_count=100):
    city_median = df[metric].median()
    out = (
        df.groupby(group_col, observed=True)
        .agg(listings=(metric, "size"), median_per_sqm=(metric, "median"))
        .query("listings >= @min_count")
        .assign(premium_vs_city=lambda x: x["median_per_sqm"] / city_median - 1)
        .sort_values("premium_vs_city")
        .reset_index()
    )
    return out

renov_secondary = premium_table(secondary, "renovation", "price_per_sqm", min_count=100)
renov_rental = premium_table(rentals, "renovation", "rent_per_sqm", min_count=100)
class_new = premium_table(new_builds, "complex_class", "price_per_sqm", min_count=50)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
for ax, (title, d, group_col, color) in zip(
    axes,
    [
        ("Secondary renovation", renov_secondary, "renovation", MARKET_COLORS["Secondary"]),
        ("Rental renovation", renov_rental, "renovation", MARKET_COLORS["Rental"]),
        ("New-build class", class_new, "complex_class", MARKET_COLORS["New build"]),
    ],
):
    ax.axvline(0, color=PALETTE["ink"], lw=1.2)
    ax.barh(d[group_col], d["premium_vs_city"], color=color, alpha=0.82)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:+.0%}"))
    style_ax(ax, title, "median per-sqm premium vs same-market city median", xlabel="premium", ylabel="")

fig.suptitle("Amenity and class premiums", x=0.055, ha="left", y=0.98, fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
def weighted_mean(x, value_col, weight_col):
    weights = x[weight_col].clip(lower=0)
    if weights.sum() == 0:
        return np.nan
    return np.average(x[value_col], weights=weights)

monthly = (
    district_monthly.groupby("month", observed=True)
    .apply(lambda g: pd.Series({
        "secondary_price_per_sqm": weighted_mean(g, "secondary_price_per_sqm", "n_listings_secondary"),
        "newbuild_price_per_sqm": weighted_mean(g, "newbuild_price_per_sqm", "n_listings_newbuild"),
        "rental_price_per_sqm_monthly": weighted_mean(g, "rental_price_per_sqm_monthly", "n_listings_rental"),
        "secondary_listings": g["n_listings_secondary"].sum(),
        "newbuild_listings": g["n_listings_newbuild"].sum(),
        "rental_listings": g["n_listings_rental"].sum(),
        "cbr_key_rate_pct": g["cbr_key_rate_pct"].mean(),
        "avg_mortgage_rate_pct": g["avg_mortgage_rate_pct"].mean(),
    }))
    .reset_index()
)

for col in ["secondary_price_per_sqm", "newbuild_price_per_sqm", "rental_price_per_sqm_monthly"]:
    monthly[f"{col}_idx"] = monthly[col] / monthly[col].iloc[0] * 100

fig, ax = plt.subplots(figsize=(15.5, 7))
series_specs = [
    ("Secondary", "secondary_price_per_sqm_idx"),
    ("New build", "newbuild_price_per_sqm_idx"),
    ("Rental", "rental_price_per_sqm_monthly_idx"),
]
for label, col in series_specs:
    ax.plot(monthly["month"], monthly[col], color=MARKET_COLORS[label], lw=3, label=label)
    ax.scatter(monthly["month"].iloc[::6], monthly[col].iloc[::6], color=MARKET_COLORS[label], s=24, zorder=3)

ax2 = ax.twinx()
ax2.plot(monthly["month"], monthly["avg_mortgage_rate_pct"], color=PALETTE["charcoal"], lw=1.8, alpha=0.68, linestyle="--", label="Mortgage rate")
ax2.plot(monthly["month"], monthly["cbr_key_rate_pct"], color=PALETTE["violet"], lw=1.5, alpha=0.60, linestyle=":", label="CBR key rate")
ax2.set_ylabel("rate, %", color=PALETTE["muted"])
ax2.tick_params(colors=PALETTE["muted"], length=0)
ax2.grid(False)

style_ax(ax, "Citywide market pulse", "Weighted district panel; price and rent indices set to 100 in Jan 2020.", xlabel="", ylabel="index, Jan 2020 = 100")
ax.axhline(100, color=PALETTE["ink"], lw=1, alpha=0.55)
ax.legend(loc="upper left", ncol=3)
ax2.legend(loc="upper right", ncol=2)
fig.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()
# Liquidity rhythm: listing counts by market over time.
fig, ax = plt.subplots(figsize=(15.5, 5.8))
ax.stackplot(
    monthly["month"],
    monthly["secondary_listings"],
    monthly["newbuild_listings"],
    monthly["rental_listings"],
    labels=["Secondary", "New build", "Rental"],
    colors=[MARKET_COLORS["Secondary"], MARKET_COLORS["New build"], MARKET_COLORS["Rental"]],
    alpha=0.78,
)
style_ax(ax, "Listing liquidity by month", "District panel volumes expose whether price moves coincide with changing market depth.", xlabel="", ylabel="district-level listing count")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ax.legend(loc="upper left", ncol=3)
fig.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

class_order = new_builds.groupby("complex_class")["price_per_sqm"].median().sort_values().index
sns.violinplot(
    data=new_builds,
    x="complex_class",
    y="price_per_sqm",
    order=class_order,
    palette=sns.color_palette(["#AEC3B0", "#1D7874", "#D99A30", "#B23A48"]),
    cut=0,
    inner="quartile",
    linewidth=1,
    ax=axes[0],
)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(rub_k))
style_ax(axes[0], "Project class distribution", "quartile bands inside each violin", xlabel="", ylabel="RUB / sqm")

sns.pointplot(
    data=new_builds,
    x="completion_year",
    y="price_per_sqm",
    hue="subsidized_mortgage",
    estimator="median",
    errorbar=("pi", 50),
    palette={False: PALETTE["charcoal"], True: MARKET_COLORS["New build"]},
    dodge=0.32,
    markers=["o", "D"],
    linestyles=["-", "--"],
    ax=axes[1],
)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(rub_k))
style_ax(axes[1], "Completion horizon x mortgage subsidy", "points are medians; bands show interquartile intervals", xlabel="completion year", ylabel="RUB / sqm")
axes[1].legend(title="subsidized mortgage")

fig.suptitle("New-build pricing mechanics", x=0.055, ha="left", y=0.98, fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
def spearman_target(df, target, candidates):
    cols = [target] + [c for c in candidates if c in df.columns]
    corr = df[cols].corr(method="spearman", numeric_only=True)[target].drop(target)
    return corr.sort_values(key=lambda s: s.abs(), ascending=False)

feature_candidates = [
    "total_area", "rooms", "floor", "total_floors", "floor_ratio", "metro_distance_min", "to_center_km",
    "log_area", "log_metro_distance_min", "log_to_center_km", "building_age", "ceiling_height",
    "living_area", "kitchen_area", "mortgage_rate_at_listing", "years_to_completion",
]

corr_frames = []
for market, df, target in [
    ("Secondary", secondary, "log_target"),
    ("New build", new_builds, "log_target"),
    ("Rental", rentals, "log_target"),
]:
    s = spearman_target(df, target, feature_candidates)
    corr_frames.append(s.rename(market))

corr_table = pd.concat(corr_frames, axis=1).fillna(0)
corr_table = corr_table.loc[corr_table.abs().max(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(8.2, 7.2))
sns.heatmap(
    corr_table,
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(240, 10, s=85, l=45, center="light", as_cmap=True),
    center=0,
    linewidths=0.8,
    linecolor=PALETTE["paper"],
    cbar_kws={"label": "Spearman correlation with log headline value"},
    ax=ax,
)
style_ax(ax, "Early feature signal map", "Use as a triage lens, not causal evidence.", xlabel="", ylabel="")
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
