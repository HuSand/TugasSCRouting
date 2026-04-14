"""
Algorithm registry, benchmark runner, and scenario builder.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

from src.routing.base import BaseRoutingAlgorithm, RouteResult, Scenario

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

class AlgorithmRegistry:
    """Central list of algorithms the benchmark will run."""

    def __init__(self):
        self._algos: Dict[str, BaseRoutingAlgorithm] = {}

    def register(self, algo: BaseRoutingAlgorithm):
        if algo.name in self._algos:
            log.warning(f"Registry: overwriting '{algo.name}'")
        self._algos[algo.name] = algo

    def unregister(self, name: str):
        self._algos.pop(name, None)

    def all(self) -> List[BaseRoutingAlgorithm]:
        return list(self._algos.values())

    def names(self) -> List[str]:
        return list(self._algos.keys())

    def summary(self):
        log.info(f"Registered algorithms ({len(self._algos)}):")
        for name, algo in self._algos.items():
            log.info(f"  [{name}]  {algo.description}")


# ──────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """Runs every registered algorithm on every scenario."""

    def __init__(self, registry: AlgorithmRegistry):
        self.registry = registry
        self.scenarios: List[Scenario] = []
        self.results:   List[RouteResult] = []

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)

    def run(self, G: nx.MultiDiGraph) -> pd.DataFrame:
        self.results = []
        algos = self.registry.all()

        log.info(f"\nBenchmark: {len(algos)} algorithms × {len(self.scenarios)} scenarios")

        for scenario in self.scenarios:
            log.info(f"\n  Scenario [{scenario.name}]: "
                     f"{scenario.source_label} → {scenario.target_label}")
            for algo in algos:
                result = algo.safe_run(G, scenario.source_node,
                                       scenario.target_node, scenario.name)
                self.results.append(result)
                status = "OK   " if result.found else "FAIL "
                log.info(f"    {status} [{algo.name:<22}]  "
                         f"time={result.total_time_s/60:5.1f}min  "
                         f"dist={result.total_distance_m/1000:5.2f}km  "
                         f"cpu={result.computation_ms:6.1f}ms")

        return self._to_dataframe()

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        ok = df[df["found"] == True]
        if ok.empty:
            log.warning("No successful routes to summarise.")
            return pd.DataFrame()
        summary = ok.groupby("algorithm").agg(
            solved          = ("scenario",        "count"),
            avg_time_min    = ("travel_time_min",  "mean"),
            avg_dist_km     = ("distance_km",      "mean"),
            avg_cpu_ms      = ("computation_ms",   "mean"),
            best_time_min   = ("travel_time_min",  "min"),
            worst_time_min  = ("travel_time_min",  "max"),
        ).round(3)
        log.info(f"\nBENCHMARK SUMMARY\n{'='*60}\n{summary.to_string()}")
        return summary

    def _to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {
                "scenario":        r.scenario_name,
                "algorithm":       r.algorithm_name,
                "found":           r.found,
                "travel_time_s":   r.total_time_s   if r.found else None,
                "travel_time_min": r.total_time_s / 60 if r.found else None,
                "distance_m":      r.total_distance_m if r.found else None,
                "distance_km":     r.total_distance_m / 1000 if r.found else None,
                "nodes_in_route":  r.nodes_in_route,
                "computation_ms":  r.computation_ms,
                "error":           r.error,
            }
            row.update({f"meta_{k}": v for k, v in r.metadata.items()})
            rows.append(row)
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Scenario builder
# ──────────────────────────────────────────────────────────────

def build_scenarios(G: nx.MultiDiGraph,
                    fac: gpd.GeoDataFrame,
                    n: int = 5) -> List[Scenario]:
    """Auto-generate diverse scenarios from the facility data."""

    def make(name, desc, src, dst):
        return Scenario(
            name=name, description=desc,
            source_node=int(src["nearest_node"]),
            target_node=int(dst["nearest_node"]),
            source_label=str(src.get("name") or src["facility_type"]),
            target_label=str(dst.get("name") or dst["facility_type"]),
            source_coords=(float(src["lat"]), float(src["lon"])),
            target_coords=(float(dst["lat"]), float(dst["lon"])),
        )

    hosp   = fac[fac["facility_type"] == "hospital"].dropna(subset=["nearest_node"])
    school = fac[fac["facility_type"] == "school"].dropna(subset=["nearest_node"])
    police = fac[fac["facility_type"] == "police"].dropna(subset=["nearest_node"])
    p1     = fac[fac["priority"] == 1].dropna(subset=["nearest_node"])

    scenarios = []

    if len(hosp) >= 1 and len(school) >= 1:
        scenarios.append(make("hosp_to_school",
                               "Hospital → school (monitoring patrol)",
                               hosp.iloc[0], school.iloc[0]))
    if len(hosp) >= 2:
        scenarios.append(make("hosp_to_hosp",
                               "Hospital A → Hospital B (inter-facility)",
                               hosp.iloc[0], hosp.iloc[1]))
    if len(police) >= 1 and len(hosp) >= 1:
        scenarios.append(make("police_to_hosp",
                               "Police → hospital (emergency route)",
                               police.iloc[0], hosp.iloc[0]))
    if len(p1) >= 4:
        scenarios.append(make("priority_cross_a",
                               "High-priority cross-city A",
                               p1.iloc[0], p1.iloc[-1]))
        scenarios.append(make("priority_cross_b",
                               "High-priority cross-city B",
                               p1.iloc[1], p1.iloc[-2]))

    return scenarios[:n]


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def run_platform(cfg):
    from src.routing.algorithms import (
        DijkstraTime, DijkstraDistance, AStarTime,
        TeamAGA, TeamBGA,
        EXAMPLE_SCENARIOS,
    )
    from src.routing.visualize import ResultVisualiser

    # Load data
    fac_path = cfg.DATA_DIR / "facilities_with_network.geojson"
    net_path = cfg.DATA_DIR / "road_network.graphml"
    if not fac_path.exists() or not net_path.exists():
        raise FileNotFoundError(
            "Required data files missing. Run 'python main.py extract' first."
        )

    log.info("Loading road network and facilities...")
    G   = ox.load_graphml(str(net_path))
    fac = gpd.read_file(str(fac_path))

    # Cast numeric edge attributes (GraphML stores as strings)
    for _, _, d in G.edges(data=True):
        for key in ("travel_time", "length", "speed_kph"):
            if key in d:
                try:
                    d[key] = float(d[key])
                except (ValueError, TypeError):
                    pass

    fac["nearest_node"] = pd.to_numeric(fac["nearest_node"], errors="coerce")
    fac = fac.dropna(subset=["nearest_node"])
    fac["nearest_node"] = fac["nearest_node"].astype(int)
    log.info(f"Graph: {G.number_of_nodes()} nodes  |  Facilities: {len(fac)}")

    # ── Register algorithms ──────────────────────────────────
    # Add or remove algorithms here.
    registry = AlgorithmRegistry()
    registry.register(DijkstraTime())       # baseline: rute tercepat
    registry.register(DijkstraDistance())   # baseline: rute terpendek
    registry.register(AStarTime())          # baseline: A* tercepat
    registry.register(TeamAGA())            # Team A — GA (tuning di algorithms.py)
    registry.register(TeamBGA())            # Team B — GA (tuning di algorithms.py)
    # registry.register(YourNewAlgorithm()) # tambah algoritma lain di sini
    registry.summary()

    # ── Build scenarios ──────────────────────────────────────
    # Gunakan skenario konkret dari EXAMPLE_SCENARIOS (fasilitas Surabaya nyata).
    # Kalau mau pakai auto-generate, ganti baris di bawah dengan:
    #   scenarios = build_scenarios(G, fac, n=cfg.N_SCENARIOS)
    log.info("\nBuilding benchmark scenarios...")
    scenarios = EXAMPLE_SCENARIOS[:cfg.N_SCENARIOS]
    if not scenarios:
        log.error("Could not build scenarios — check that extraction ran successfully.")
        return
    for s in scenarios:
        log.info(f"  [{s.name}]  {s.source_label} → {s.target_label}")

    # ── Run benchmark ────────────────────────────────────────
    runner = BenchmarkRunner(registry)
    for s in scenarios:
        runner.add_scenario(s)
    df      = runner.run(G)
    summary = runner.summary(df)

    df.to_csv(cfg.DATA_DIR / "comparison_results.csv", index=False)
    log.info(f"Saved -> comparison_results.csv")
    if not summary.empty:
        summary.to_csv(cfg.DATA_DIR / "comparison_summary.csv")
        log.info(f"Saved -> comparison_summary.csv")

    # ── Visualise ────────────────────────────────────────────
    log.info("\nGenerating visualisations...")
    vis = ResultVisualiser(cfg.DATA_DIR)
    for scenario in scenarios:
        scene_results = [r for r in runner.results if r.scenario_name == scenario.name]
        vis.map_scenario(G, scenario, scene_results)
    vis.chart_comparison(df)

    log.info("\nOutputs in data/:")
    log.info("  comparison_results.csv    raw results per algorithm/scenario")
    log.info("  comparison_summary.csv    aggregate stats per algorithm")
    log.info("  comparison_chart.png      travel time + speed bar charts")
    log.info("  comparison_map_*.html     route overlay maps (open in browser)")
