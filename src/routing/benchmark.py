"""
Algorithm registry, benchmark runner, and scenario builder.
"""

import logging
import time
from itertools import permutations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

from src.routing.base import BaseRoutingAlgorithm, RouteResult, Scenario

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Evolution log writer
# ──────────────────────────────────────────────────────────────

def _write_evolution_log(result: RouteResult, log_dir: Path):
    """
    Tulis file teks per-generasi untuk satu GA result.
    File: logs/evolution_<algo>_<scenario>.txt
    Hanya dipanggil kalau result punya gen_history di metadata.
    """
    history = result.metadata.get("gen_history")
    if not history:
        return

    log_dir.mkdir(exist_ok=True)
    fname = log_dir / f"evolution_{result.algorithm_name}_{result.scenario_name}.txt"

    first_min  = history[0]["min"]
    final_min  = history[-1]["min"]
    total_impr = (first_min - final_min) / first_min * 100 if first_min > 0 else 0.0

    # Find gens where improvement happened
    prev_best = first_min
    improved_gens = []
    for frame in history:
        if frame["min"] < prev_best - 1e-6:
            improved_gens.append(frame["gen"])
            prev_best = frame["min"]

    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"GA Evolution Log\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Algorithm : {result.algorithm_name}\n")
        f.write(f"Scenario  : {result.scenario_name}\n")
        m = result.metadata
        f.write(f"Config    : pop={m.get('population','?')}  "
                f"gen={m.get('generations','?')}  "
                f"xover={m.get('crossover_rate','?')}  "
                f"mut={m.get('mutation_rate','?')}\n")
        f.write(f"CPU time  : {result.computation_ms:.1f} ms\n")
        f.write(f"{'='*60}\n\n")

        prev      = first_min
        prev_streets: list = []
        for frame in history:
            gen     = frame["gen"]
            t       = frame["min"]
            dist    = frame.get("dist")
            impr    = (first_min - t) / first_min * 100 if first_min > 0 else 0.0
            delta   = prev - t
            streets = frame.get("streets", [])

            # ── status tag ──
            if gen == 1:
                tag = "initial"
            elif delta > 1e-6:
                tag = f"IMPROVED  -{delta:.3f} min"
            else:
                tag = ""
            if gen == len(history):
                tag = (tag + " [FINAL]").strip() if tag else "[FINAL]"

            # ── route changed? ──
            route_changed = streets != prev_streets

            f.write(f"\n{'─'*60}\n")
            f.write(f"Gen {gen:>3}  |  {t:.4f} min")
            if dist is not None:
                f.write(f"  |  {float(dist):.3f} km")
            f.write(f"  |  {impr:.2f}% improved")
            if tag:
                f.write(f"  |  {tag}")
            f.write("\n")

            if streets:
                if route_changed:
                    route_str = " -> ".join(streets)
                    # Wrap long route strings at 56 chars
                    f.write(f"  Route : {route_str[:56]}\n")
                    if len(route_str) > 56:
                        # Continue remaining segments indented
                        rest = route_str[56:]
                        while rest:
                            f.write(f"          {rest[:56]}\n")
                            rest = rest[56:]
                else:
                    f.write(f"  Route : (unchanged)\n")
            else:
                f.write(f"  Route : -\n")

            cand_min = frame.get("candidate_min")
            cand_dist = frame.get("candidate_dist")
            if cand_min is not None:
                f.write(f"  Candidate : {float(cand_min):.4f} min")
                if cand_dist is not None:
                    f.write(f"  |  {float(cand_dist):.3f} km")
                f.write("\n")

            prev         = t
            prev_streets = streets

        # ── Summary ──
        final_streets = history[-1].get("streets", [])
        f.write(f"\n{'='*60}\n")
        f.write(f"SUMMARY\n")
        f.write(f"  Gen 1 best  : {first_min:.4f} min\n")
        f.write(f"  Final best  : {final_min:.4f} min\n")
        f.write(f"  Total impr  : {total_impr:.2f}%\n")
        f.write(f"  # improved  : {len(improved_gens)} generations\n")
        if improved_gens:
            f.write(f"  Improved at : gen {', '.join(str(g) for g in improved_gens)}\n")
        f.write(f"  Route nodes : {result.nodes_in_route}\n")
        f.write(f"  Distance    : {result.total_distance_m/1000:.3f} km\n")
        if final_streets:
            f.write(f"\nFINAL ROUTE\n")
            for i, s in enumerate(final_streets, 1):
                f.write(f"  {i:>2}. {s}\n")

    log.debug(f"    evolution log -> {fname.name}")


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

    def __init__(self, registry: AlgorithmRegistry, log_dir: Path = None):
        self.registry = registry
        self.log_dir  = log_dir
        self.scenarios: List[Scenario] = []
        self.results:   List[RouteResult] = []

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)

    @staticmethod
    def _order_weight(algo: BaseRoutingAlgorithm) -> str:
        return "length" if "distance" in algo.name else "travel_time"

    def _best_visit_order(self,
                          G: nx.MultiDiGraph,
                          algo: BaseRoutingAlgorithm,
                          nodes: list) -> tuple:
        weight = self._order_weight(algo)
        if len(nodes) <= 2:
            return nodes, weight, 0.0

        pair_cost = {}
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                try:
                    pair_cost[(src, dst)] = nx.shortest_path_length(
                        G, src, dst, weight=weight
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pair_cost[(src, dst)] = float("inf")

        best_order = None
        best_cost = float("inf")
        for order in permutations(nodes):
            cost = sum(pair_cost[(src, dst)]
                       for src, dst in zip(order[:-1], order[1:]))
            if cost < best_cost:
                best_order = list(order)
                best_cost = cost

        return best_order or nodes, weight, best_cost

    def _run_algorithm_on_scenario(self,
                                   algo: BaseRoutingAlgorithm,
                                   G: nx.MultiDiGraph,
                                   scenario: Scenario) -> RouteResult:
        nodes = scenario.node_sequence
        labels = scenario.label_sequence
        coords = scenario.coord_sequence

        # Christofides is only meaningful for multi-stop routing.
        # Route it once as a tour instead of splitting into per-leg shortest paths.
        if getattr(algo, "name", "") == "christofides" and scenario.is_multi_stop:
            if hasattr(algo, "_route_multi_stop"):
                return algo._route_multi_stop(
                    G,
                    nodes,
                    scenario.name,
                    source_node=scenario.source_node,
                    target_node=scenario.target_node,
                )

        if len(nodes) <= 2:
            return algo.safe_run(G, scenario.source_node,
                                 scenario.target_node, scenario.name)

        order_objective = "fixed"
        order_score = None
        if scenario.optimize_order:
            ordered_nodes, order_objective, order_score = self._best_visit_order(
                G, algo, nodes
            )
            idx_by_node = {node: i for i, node in enumerate(nodes)}
            order_idx = [idx_by_node[n] for n in ordered_nodes]
            nodes = ordered_nodes
            labels = [labels[i] for i in order_idx]
            coords = [coords[i] for i in order_idx]

        t0 = time.perf_counter()
        full_route: list = []
        leg_rows: list = []
        leg_results: list = []

        for idx, (src, dst) in enumerate(zip(nodes[:-1], nodes[1:]), start=1):
            leg_name = f"{scenario.name}_leg_{idx}"
            result = algo.safe_run(G, src, dst, leg_name)
            leg_results.append(result)
            leg_rows.append({
                "leg": idx,
                "from": labels[idx - 1] if idx - 1 < len(labels) else str(src),
                "to": labels[idx] if idx < len(labels) else str(dst),
                "found": result.found,
                "travel_time_s": result.total_time_s if result.found else None,
                "distance_m": result.total_distance_m if result.found else None,
                "computation_ms": result.computation_ms,
                "error": result.error,
            })

            if not result.found:
                elapsed = (time.perf_counter() - t0) * 1000
                err = (f"leg {idx} failed: "
                       f"{leg_rows[-1]['from']} -> {leg_rows[-1]['to']} | "
                       f"{result.error}")
                return RouteResult.failure(algo.name, scenario.name,
                                           nodes[0],
                                           nodes[-1],
                                           err, elapsed)

            if not full_route:
                full_route.extend(result.route)
            else:
                full_route.extend(result.route[1:])

        elapsed = (time.perf_counter() - t0) * 1000
        metadata = {
            "multi_stop": True,
            "stop_count": len(nodes),
            "stops": labels,
            "visit_order_nodes": nodes,
            "visit_order_coords": coords,
            "order_objective": order_objective,
            "order_score": order_score,
            "legs": leg_rows,
        }
        histories = [r.metadata.get("gen_history") for r in leg_results]
        if histories and all(histories):
            gen_count = min(len(h) for h in histories)
            combined_history = []
            for gen_idx in range(gen_count):
                coords = []
                candidate_coords = []
                streets = []
                candidate_streets = []
                total_min = 0.0
                total_dist = 0.0
                candidate_min = 0.0
                candidate_dist = 0.0
                for history in histories:
                    frame = history[gen_idx]
                    frame_coords = frame.get("coords", [])
                    if coords and frame_coords:
                        coords.extend(frame_coords[1:])
                    else:
                        coords.extend(frame_coords)
                    streets.extend(frame.get("streets", []))
                    total_min += float(frame.get("min", 0.0))
                    total_dist += float(frame.get("dist", 0.0))

                    cand_coords = frame.get("candidate_coords", frame_coords)
                    if candidate_coords and cand_coords:
                        candidate_coords.extend(cand_coords[1:])
                    else:
                        candidate_coords.extend(cand_coords)
                    candidate_streets.extend(frame.get("candidate_streets",
                                                        frame.get("streets", [])))
                    candidate_min += float(frame.get("candidate_min",
                                                     frame.get("min", 0.0)))
                    candidate_dist += float(frame.get("candidate_dist",
                                                      frame.get("dist", 0.0)))
                combined_history.append({
                    "gen": gen_idx + 1,
                    "min": round(total_min, 3),
                    "dist": round(total_dist, 3),
                    "coords": coords,
                    "streets": streets,
                    "candidate_min": round(candidate_min, 3),
                    "candidate_dist": round(candidate_dist, 3),
                    "candidate_coords": candidate_coords,
                    "candidate_streets": candidate_streets,
                })

            first_meta = leg_results[0].metadata
            metadata.update({
                "generations": first_meta.get("generations"),
                "population": first_meta.get("population"),
                "crossover_rate": first_meta.get("crossover_rate"),
                "mutation_rate": first_meta.get("mutation_rate"),
                "gen_history": combined_history,
            })
        return RouteResult.build(G, algo.name, scenario.name,
                                 nodes[0], nodes[-1],
                                 full_route, elapsed, metadata)

    def run(self, G: nx.MultiDiGraph) -> pd.DataFrame:
        self.results = []
        algos = self.registry.all()

        log.info(f"\nBenchmark: {len(algos)} algorithms × {len(self.scenarios)} scenarios")

        for scenario in self.scenarios:
            route_label = " -> ".join(scenario.label_sequence)
            log.info(f"\n  Scenario [{scenario.name}]: {route_label}")
            from src.routing.visualize import ResultVisualiser
            scenario_results = []
            for algo in algos:
                result = self._run_algorithm_on_scenario(algo, G, scenario)
                self.results.append(result)
                scenario_results.append(result)
                status = "OK   " if result.found else "FAIL "
                log.info(f"    {status} [{algo.name:<22}]  "
                         f"time={result.total_time_s/60:5.1f}min  "
                         f"dist={result.total_distance_m/1000:5.2f}km  "
                         f"cpu={result.computation_ms:6.1f}ms")
                ResultVisualiser.log_route_streets(G, result)
                if self.log_dir and "gen_history" in result.metadata:
                    _write_evolution_log(result, self.log_dir)

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
                               "Hospital -> school (monitoring patrol)",
                               hosp.iloc[0], school.iloc[0]))
    if len(hosp) >= 2:
        scenarios.append(make("hosp_to_hosp",
                               "Hospital A -> Hospital B (inter-facility)",
                               hosp.iloc[0], hosp.iloc[1]))
    if len(police) >= 1 and len(hosp) >= 1:
        scenarios.append(make("police_to_hosp",
                               "Police -> hospital (emergency route)",
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
        DijkstraTime, DijkstraDistance, AStarTime, AStarDistance,
        ChristofidesAlgorithm,
        SandyGA, BurhanGA, BimoGA, GeraldGA, GeraldSimulatedAnnealing,
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
    registry.register(DijkstraTime())      # baseline: rute tercepat
    registry.register(DijkstraDistance()) # baseline: rute terpendek
    registry.register(AStarTime())        # baseline: A* tercepat
    registry.register(AStarDistance())    # baseline: A* terpendek
    registry.register(ChristofidesAlgorithm())  # Christofides approximation
    registry.register(SandyGA())          # Sandy
    registry.register(BurhanGA())         # Burhan
    registry.register(BimoGA())           # Bimo
    registry.register(GeraldGA())         # Gerald
    registry.register(GeraldSimulatedAnnealing())  # Gerald SA shortest path
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
        log.info(f"  [{s.name}]  {s.source_label} -> {s.target_label}")

    # ── Run benchmark ────────────────────────────────────────
    runner = BenchmarkRunner(registry, log_dir=cfg.LOG_DIR)
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

    # ── GA Evolution Viewer ──────────────────────────────────
    from src.routing.evolve_viz import build_evolution_viewer
    build_evolution_viewer(G, scenarios, runner.results, cfg.DATA_DIR)

    log.info("\nOutputs in data/:")
    log.info("  comparison_results.csv    raw results per algorithm/scenario")
    log.info("  comparison_summary.csv    aggregate stats per algorithm")
    log.info("  comparison_chart.png      travel time + speed bar charts")
    log.info("  comparison_map_*.html     route overlay maps (open in browser)")
    log.info("  evolution_viewer.html     GA evolution timeline (open in browser)")
    log.info("  (logs/)evolution_*.txt    per-generation log per GA algorithm")
