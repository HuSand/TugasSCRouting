"""
Data exploration and profiling.
Generates interactive map, distribution charts, and a quality report.

Outputs (in settings.DATA_DIR):
    surabaya_facilities_map.html
    facility_distribution.png
    snap_distance_histogram.png
    exploration_report.txt
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, HeatMap

log = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "healthcare": "#C44E52",
    "education":  "#4C72B0",
    "emergency":  "#DD8452",
    "government": "#8172B3",
    "transport":  "#55A868",
    "community":  "#937860",
    "other":      "#777777",
}
FOLIUM_COLORS = {
    "healthcare": "red",   "education": "blue",  "emergency": "orange",
    "government": "purple","transport": "green", "community": "gray", "other": "black",
}
FOLIUM_ICONS = {
    "healthcare": "plus-sign", "education": "education", "emergency": "fire",
    "government": "tower",     "transport": "road",      "community": "home",
    "other": "map-marker",
}


# ──────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────

def _profile(fac: gpd.GeoDataFrame):
    log.info(f"Rows: {len(fac)}  |  Columns: {len(fac.columns)}")
    log.info("Column completeness:")
    for col in fac.columns:
        null_pct = fac[col].isna().mean() * 100
        if null_pct > 1:
            log.info(f"  {col:<25} {null_pct:5.1f}% null")


def _quality_check(fac: gpd.GeoDataFrame) -> list:
    issues = []

    # Out-of-bounds coordinates
    oob = fac[
        (fac["lat"] < -7.40) | (fac["lat"] > -7.15) |
        (fac["lon"] < 112.55) | (fac["lon"] > 112.85)
    ]
    if len(oob):
        issues.append(f"{len(oob)} facilities outside Surabaya bbox")

    # Snap distance
    if "snap_distance_m" in fac.columns:
        far = (fac["snap_distance_m"] > 500).sum()
        if far:
            issues.append(f"{far} facilities >500m from road network")

    # Node sharing
    if "nearest_node" in fac.columns:
        dupes = fac["nearest_node"].duplicated().sum()
        if dupes > len(fac) * 0.3:
            issues.append("High road-node sharing — many facilities near same intersection")

    # Unnamed
    unnamed = fac["name"].isna().sum()
    if unnamed > len(fac) * 0.3:
        issues.append(f"{unnamed} unnamed facilities ({unnamed/len(fac)*100:.0f}%)")

    if issues:
        log.warning(f"Quality issues ({len(issues)}):")
        for i, msg in enumerate(issues, 1):
            log.warning(f"  {i}. {msg}")
    else:
        log.info("No critical quality issues found.")
    return issues


def _distributions(fac: gpd.GeoDataFrame):
    log.info("Category distribution:")
    for cat, count in fac["category"].value_counts().items():
        pct = count / len(fac) * 100
        bar = "█" * int(pct / 2)
        log.info(f"  {cat:<15} {count:5d} ({pct:4.1f}%) {bar}")


def _charts(fac: gpd.GeoDataFrame, data_dir: Path):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                          "axes.titlesize": 13, "axes.titleweight": "bold"})

    # Category bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = fac["category"].value_counts().sort_values(ascending=True)
    colors = [CATEGORY_COLORS.get(c, "#777") for c in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 3, bar.get_y() + bar.get_height() / 2,
                str(int(w)), ha="left", va="center", fontsize=10)
    ax.set_title(f"Public Facilities by Category")
    ax.set_xlabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(data_dir / "facility_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved -> facility_distribution.png")

    # Snap distance histogram
    if "snap_distance_m" in fac.columns:
        snap = fac["snap_distance_m"].dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(snap[snap < 500], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(snap.median(), color="red",    linestyle="--", lw=1.5,
                   label=f"Median: {snap.median():.0f}m")
        ax.axvline(snap.quantile(0.95), color="orange", linestyle="--", lw=1.5,
                   label=f"P95: {snap.quantile(0.95):.0f}m")
        ax.set_title("Distance from Facility to Nearest Road Node")
        ax.set_xlabel("Snap Distance (m)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(data_dir / "snap_distance_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved -> snap_distance_histogram.png")


def _interactive_map(fac: gpd.GeoDataFrame, data_dir: Path):
    m = folium.Map(
        location=[fac["lat"].mean(), fac["lon"].mean()],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    for category in sorted(fac["category"].unique()):
        cat_data = fac[fac["category"] == category]
        cluster = MarkerCluster(name=f"{category} ({len(cat_data)})")
        for _, row in cat_data.iterrows():
            name = row.get("name", "Unnamed")
            if pd.isna(name):
                name = "Unnamed"
            popup = (f"<b>{name}</b><br>"
                     f"Type: {row.get('facility_type','?')}<br>"
                     f"Category: {category}<br>"
                     f"Snap: {row.get('snap_distance_m', 0):.0f}m")
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup, max_width=250),
                icon=folium.Icon(
                    color=FOLIUM_COLORS.get(category, "black"),
                    icon=FOLIUM_ICONS.get(category, "map-marker"),
                    prefix="glyphicon",
                ),
            ).add_to(cluster)
        cluster.add_to(m)

    heat_data = [[r["lat"], r["lon"]] for _, r in fac.iterrows()]
    HeatMap(heat_data, name="Density Heatmap", radius=15).add_to(m)
    folium.LayerControl().add_to(m)

    out = data_dir / "surabaya_facilities_map.html"
    m.save(str(out))
    log.info(f"Saved -> surabaya_facilities_map.html  (open in browser)")


def _write_report(fac, issues, data_dir):
    from datetime import datetime
    out = data_dir / "exploration_report.txt"
    with open(out, "w") as f:
        f.write(f"Exploration Report\nGenerated: {datetime.now().isoformat()}\n{'='*50}\n\n")
        f.write(f"Total facilities: {len(fac)}\n\n")
        f.write("CATEGORY BREAKDOWN\n")
        for cat, count in fac["category"].value_counts().items():
            f.write(f"  {cat}: {count}\n")
        f.write(f"\nQUALITY ISSUES ({len(issues)})\n")
        for msg in issues:
            f.write(f"  - {msg}\n")
        f.write("\nATTRIBUTE COMPLETENESS\n")
        for col in ["name", "addr:street", "phone", "opening_hours"]:
            if col in fac.columns:
                pct = fac[col].notna().mean() * 100
                f.write(f"  {col}: {pct:.1f}%\n")
    log.info("Saved -> exploration_report.txt")


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def run_exploration(cfg):
    fac_path = cfg.DATA_DIR / "facilities_with_network.geojson"
    if not fac_path.exists():
        raise FileNotFoundError(
            f"Data not found: {fac_path}\n"
            "Run 'python main.py extract' first."
        )

    fac = gpd.read_file(fac_path)
    log.info(f"Loaded {len(fac)} facilities from {fac_path.name}")

    _profile(fac)
    issues = _quality_check(fac)
    _distributions(fac)
    _charts(fac, cfg.DATA_DIR)
    _interactive_map(fac, cfg.DATA_DIR)
    _write_report(fac, issues, cfg.DATA_DIR)

    log.info("Exploration complete.")
