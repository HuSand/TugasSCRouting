"""
Visualisation: per-scenario route overlay maps and comparison charts.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import folium
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.routing.base import RouteResult, Scenario

log = logging.getLogger(__name__)

ROUTE_COLORS = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]


class ResultVisualiser:

    def __init__(self, output_dir: Path):
        self.out = output_dir

    # ──────────────────────────────────────────────────────
    # Per-scenario HTML map
    # ──────────────────────────────────────────────────────

    def map_scenario(self,
                     G,
                     scenario: Scenario,
                     results: List[RouteResult]):
        m = folium.Map(
            location=list(scenario.source_coords),
            zoom_start=14,
            tiles="cartodbpositron",
        )

        for i, r in enumerate(results):
            if not r.found or len(r.route) < 2:
                continue
            color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            tip   = (f"{r.algorithm_name} | "
                     f"{r.total_time_s/60:.1f} min | "
                     f"{r.total_distance_m/1000:.2f} km")
            try:
                ox.plot_route_folium(
                    G, r.route, route_map=m,
                    color=color, weight=4 + i, opacity=0.78,
                    tooltip=tip,
                )
            except Exception:
                pass

        folium.Marker(
            list(scenario.source_coords),
            popup=f"FROM: {scenario.source_label}",
            icon=folium.Icon(color="green", icon="play", prefix="glyphicon"),
        ).add_to(m)
        folium.Marker(
            list(scenario.target_coords),
            popup=f"TO: {scenario.target_label}",
            icon=folium.Icon(color="red", icon="stop", prefix="glyphicon"),
        ).add_to(m)

        # Legend
        legend = ("<div style='position:fixed;bottom:30px;left:30px;z-index:9999;"
                  "background:white;padding:10px 14px;border-radius:6px;"
                  "border:1px solid #ccc;font-size:12px;line-height:1.8;'>"
                  f"<b>{scenario.name}</b><br>")
        for i, r in enumerate(results):
            if not r.found:
                continue
            c = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            legend += (f"<span style='color:{c};font-size:18px;'>&#9644;</span> "
                       f"{r.algorithm_name}<br>")
        legend += "</div>"
        m.get_root().html.add_child(folium.Element(legend))

        path = self.out / f"comparison_map_{scenario.name}.html"
        m.save(str(path))
        log.info(f"  Map -> comparison_map_{scenario.name}.html")

    # ──────────────────────────────────────────────────────
    # Comparison charts
    # ──────────────────────────────────────────────────────

    def chart_comparison(self, df: pd.DataFrame):
        ok = df[df["found"] == True].copy()
        if ok.empty:
            log.warning("No successful routes — skipping charts.")
            return

        scenarios  = ok["scenario"].unique()
        algorithms = ok["algorithm"].unique()
        x     = np.arange(len(scenarios))
        width = 0.8 / max(len(algorithms), 1)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plt.rcParams.update({"font.size": 10})

        # Left: travel time per scenario
        ax = axes[0]
        for i, algo in enumerate(algorithms):
            sub    = ok[ok["algorithm"] == algo]
            times  = [sub[sub["scenario"] == s]["travel_time_min"].mean()
                      for s in scenarios]
            offset = (i - len(algorithms) / 2 + 0.5) * width
            ax.bar(x + offset, times, width,
                   label=algo,
                   color=ROUTE_COLORS[i % len(ROUTE_COLORS)],
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.set_ylabel("Travel Time (min)")
        ax.set_title("Travel Time by Algorithm & Scenario")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Right: avg computation time
        ax2 = axes[1]
        avg_cpu = [ok[ok["algorithm"] == a]["computation_ms"].mean() for a in algorithms]
        bars = ax2.barh(
            list(algorithms), avg_cpu,
            color=[ROUTE_COLORS[i % len(ROUTE_COLORS)] for i in range(len(algorithms))],
            alpha=0.85,
        )
        for bar in bars:
            w = bar.get_width()
            ax2.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                     f"{w:.1f}ms", ha="left", va="center", fontsize=9)
        ax2.set_xlabel("Avg Computation Time (ms)  — lower is faster")
        ax2.set_title("Algorithm Speed")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        path = self.out / "comparison_chart.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Chart -> comparison_chart.png")
