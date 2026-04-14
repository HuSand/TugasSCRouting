"""
Visualisation: per-scenario route overlay maps and comparison charts.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.routing.base import RouteResult, Scenario

log = logging.getLogger(__name__)

ROUTE_COLORS = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    "#D81B60",
]

ROUTE_COLOR_BY_ALGO = {
    "dijkstra_time": "#9C27B0",
    "dijkstra_distance": "#795548",
    "astar_time": "#00BCD4",
    "astar_distance": "#607D8B",
    "christofides": "#2ECC71",
    "sandy_ga": "#E63946",
    "burhan_ga": "#2196F3",
    "bimo_ga": "#4CAF50",
    "gerald_ga": "#FF9800",
    "gerald_sa": "#D81B60",
}


def _route_street_names(G, route: list) -> list:
    """
    Kembalikan daftar nama jalan yang dilalui (deduplicated berurutan).
    Dipakai untuk turn-by-turn log dan map popup.
    """
    streets = []
    prev = None
    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        name = best.get("name") or best.get("highway") or "unnamed"
        if isinstance(name, list):
            name = name[0]
        name = str(name).strip()
        if name and name != prev:
            streets.append(name)
            prev = name
    return streets


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

        drawn = []
        for i, r in enumerate(results):
            if not r.found or len(r.route) < 2:
                continue
            color = ROUTE_COLOR_BY_ALGO.get(
                r.algorithm_name,
                ROUTE_COLORS[i % len(ROUTE_COLORS)],
            )

            # Ambil koordinat (lat, lon) tiap node dalam route
            coords = []
            for n in r.route:
                node = G.nodes.get(n)
                if node:
                    coords.append((float(node["y"]), float(node["x"])))

            if len(coords) < 2:
                continue

            streets    = _route_street_names(G, r.route)
            street_str = " -> ".join(streets) if streets else "-"
            popup_html = (
                f"<b>{r.algorithm_name}</b><br>"
                f"Waktu : {r.total_time_s/60:.1f} menit<br>"
                f"Jarak : {r.total_distance_m/1000:.2f} km<br>"
                f"<hr style='margin:4px 0'>"
                f"<small><b>Rute:</b><br>{street_str}</small>"
            )
            tooltip = (f"{r.algorithm_name} | "
                       f"{r.total_time_s/60:.1f}min | "
                       f"{r.total_distance_m/1000:.2f}km")

            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.80,
                tooltip=tooltip,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(m)
            drawn.append((i, r, color))

        # Marker awal, waypoint, dan tujuan
        for idx, (label, coords) in enumerate(zip(scenario.label_sequence,
                                                  scenario.coord_sequence), start=1):
            if scenario.optimize_order:
                title = f"DEST {idx}"
                icon = folium.Icon(color="gray", icon="map-marker", prefix="glyphicon")
            elif idx == 1:
                title = "START"
                icon = folium.Icon(color="green", icon="play", prefix="glyphicon")
            elif idx == len(scenario.label_sequence):
                title = "END"
                icon = folium.Icon(color="red", icon="stop", prefix="glyphicon")
            else:
                title = f"STOP {idx}"
                icon = folium.Icon(color="blue", icon="flag", prefix="glyphicon")

            folium.Marker(
                list(coords),
                popup=folium.Popup(f"<b>{title}</b><br>{label}", max_width=220),
                tooltip=f"{title}: {label}",
                icon=icon,
            ).add_to(m)

        if len(scenario.coord_sequence) > 1:
            try:
                m.fit_bounds([list(c) for c in scenario.coord_sequence], padding=(30, 30))
            except Exception:
                pass

        # Legend
        route_label = " -> ".join(scenario.label_sequence)
        if scenario.optimize_order:
            route_label = "unordered destinations; each algorithm chooses visit order"
        legend = (
            "<div style='position:fixed;bottom:30px;left:30px;z-index:9999;"
            "background:white;padding:10px 14px;border-radius:6px;"
            "border:1px solid #ccc;font-size:12px;line-height:1.9;max-width:320px;'>"
            f"<b>{scenario.name}</b><br>"
            f"<small>{route_label}</small><br>"
            "<hr style='margin:4px 0'>"
        )
        for i, r, c in drawn:
            legend += (
                f"<span style='color:{c};font-size:18px;'>&#9644;</span> "
                f"<b>{r.algorithm_name}</b> &nbsp;"
                f"{r.total_time_s/60:.1f}min &nbsp;"
                f"{r.total_distance_m/1000:.2f}km<br>"
            )
        legend += "<hr style='margin:4px 0'><small>Klik rute untuk detail nama jalan</small></div>"
        m.get_root().html.add_child(folium.Element(legend))

        path = self.out / f"comparison_map_{scenario.name}.html"
        m.save(str(path))
        log.info(f"  Map -> comparison_map_{scenario.name}.html  ({len(drawn)} routes drawn)")

    # ──────────────────────────────────────────────────────
    # Turn-by-turn log (dipanggil dari benchmark)
    # ──────────────────────────────────────────────────────

    @staticmethod
    def log_route_streets(G, result: RouteResult):
        """
        Log nama jalan yang dilalui untuk satu result.
        Contoh output:
          [sandy_ga] Jl. Darmo -> Jl. Raya Wonokromo -> Jl. Ahmad Yani -> ...
        """
        if not result.found or len(result.route) < 2:
            return
        streets = _route_street_names(G, result.route)
        if streets:
            log.info(
                f"    route [{result.algorithm_name}] "
                f"{result.total_time_s/60:.1f}min | "
                f"{result.total_distance_m/1000:.2f}km: "
                f"{' -> '.join(streets)}"
            )

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
                   color=ROUTE_COLOR_BY_ALGO.get(algo, ROUTE_COLORS[i % len(ROUTE_COLORS)]),
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
            color=[
                ROUTE_COLOR_BY_ALGO.get(a, ROUTE_COLORS[i % len(ROUTE_COLORS)])
                for i, a in enumerate(algorithms)
            ],
            alpha=0.85,
        )
        for bar in bars:
            w = bar.get_width()
            ax2.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                     f"{w:.1f}ms", ha="left", va="center", fontsize=9)
        ax2.set_xlabel("Avg Computation Time (ms)  - lower is faster")
        ax2.set_title("Algorithm Speed")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        path = self.out / "comparison_chart.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Chart -> comparison_chart.png")
