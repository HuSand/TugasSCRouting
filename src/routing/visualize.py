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
    "ga":             "#E63946",
    "aco_routing":    "#FF9800",
    "gerald_sa":      "#D81B60",
    "particle_swarm": "#2196F3",
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

        # ── Stop markers ──────────────────────────────────────────────────
        # If any algorithm recorded a visit_order in its metadata, use the
        # first such order to number the pins — it reflects an algorithm's
        # actual chosen visit sequence rather than the arbitrary input order.
        visit_order_nodes = None
        for r in results:
            vo = r.metadata.get("visit_order")
            if vo:
                visit_order_nodes = vo
                break

        # Build a node→visit-rank lookup from the chosen order (skip the
        # closing duplicate in a round-trip tour, e.g. [A,B,C,A] → ranks for A,B,C)
        node_visit_rank: dict = {}
        if visit_order_nodes:
            seen = set()
            rank = 1
            for n in visit_order_nodes:
                if n not in seen:
                    node_visit_rank[n] = rank
                    seen.add(n)
                    rank += 1

        n_stops = len(scenario.label_sequence)
        for idx, (label, coords) in enumerate(zip(scenario.label_sequence,
                                                  scenario.coord_sequence), start=1):
            lbl_lower  = label.lower()
            node       = scenario.node_sequence[idx - 1]
            visit_rank = node_visit_rank.get(node, idx)   # fallback to input order

            if idx == 1:
                title = "BASE / START" if scenario.round_trip else "START"
                icon  = folium.Icon(color="green", icon="home", prefix="glyphicon")
            elif idx == n_stops and not scenario.round_trip:
                title = "END"
                icon  = folium.Icon(color="red", icon="stop", prefix="glyphicon")
            else:
                # Show the actual GA visit rank next to the stop number
                rank_note = f" (visit #{visit_rank})" if visit_order_nodes else ""
                title = f"STOP {idx}{rank_note}"
                if any(k in lbl_lower for k in ("polisi", "polsek", "polres", "polda",
                                                  "brimob", "samsat", "polantas",
                                                  "satlantas", "satpas")):
                    stop_color = "red"
                elif any(k in lbl_lower for k in ("kebakaran", "pemadam")):
                    stop_color = "orange"
                elif any(k in lbl_lower for k in ("terminal", "stasiun", "pelabuhan")):
                    stop_color = "blue"
                elif any(k in lbl_lower for k in ("spbu", "pertamina", "shell", "bp")):
                    stop_color = "gray"
                else:
                    stop_color = "blue"
                icon = folium.Icon(color=stop_color, icon="flag", prefix="glyphicon")

            popup_body = f"<b>{title}</b><br>{label}"
            if idx == 1 and scenario.round_trip:
                popup_body += "<br><i>(route returns here)</i>"
            if visit_order_nodes and idx > 1:
                popup_body += f"<br><small>GA visit order: #{visit_rank}</small>"

            folium.Marker(
                list(coords),
                popup=folium.Popup(popup_body, max_width=240),
                tooltip=f"{title}: {label}",
                icon=icon,
            ).add_to(m)

        if len(scenario.coord_sequence) > 1:
            try:
                m.fit_bounds([list(c) for c in scenario.coord_sequence], padding=(30, 30))
            except Exception:
                pass

        # Legend
        # Algorithms with _route_multi_stop (GA, Christofides) choose their own
        # visit order — flag this in the legend so the viewer knows the pins
        # are destination nodes, not a fixed route.
        any_self_ordered = any(
            hasattr(type(r), "_route_multi_stop") or "visit_order" in r.metadata
            for r in results if r.found
        )
        route_label = " -> ".join(scenario.label_sequence)
        if scenario.round_trip:
            route_label += " -> (return to start)"
        if any_self_ordered:
            route_label = "Destinations (pins); each algorithm chooses its own visit order"
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
