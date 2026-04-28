"""
Algorithm registry, benchmark runner, and scenario builder.
"""

import logging
import os
import pickle
import time
from multiprocessing import Pool
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

    @staticmethod
    def _best_visit_order(G: nx.MultiDiGraph,
                          algo: BaseRoutingAlgorithm,
                          nodes: list) -> tuple:
        weight = BenchmarkRunner._order_weight(algo)
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

    @staticmethod
    def _run_algorithm_on_scenario(algo: BaseRoutingAlgorithm,
                                   G: nx.MultiDiGraph,
                                   scenario: Scenario) -> RouteResult:
        nodes = scenario.node_sequence
        labels = scenario.label_sequence
        coords = scenario.coord_sequence

        # Any algorithm that implements _route_multi_stop handles the full
        # multi-stop tour itself — it decides visit order and returns to start.
        # The benchmark runner does NOT pre-order stops for these algorithms.
        if scenario.is_multi_stop and hasattr(algo, "_route_multi_stop"):
            return algo._route_multi_stop(
                G,
                nodes,
                scenario.name,
                source_node=scenario.source_node,
                target_node=scenario.target_node,
            )

        if len(nodes) <= 2 and not scenario.round_trip:
            return algo.safe_run(G, scenario.source_node,
                                 scenario.target_node, scenario.name)

        order_objective = "fixed"
        order_score = None
        if scenario.optimize_order:
            ordered_nodes, order_objective, order_score = BenchmarkRunner._best_visit_order(
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

        # ── Return leg (round trip) ──────────────────────────────
        if scenario.round_trip and full_route:
            ret_name = f"{scenario.name}_return"
            ret = algo.safe_run(G, nodes[-1], nodes[0], ret_name)
            leg_results.append(ret)
            leg_rows.append({
                "leg": len(nodes),
                "from": labels[-1] if labels else str(nodes[-1]),
                "to":   labels[0]  if labels else str(nodes[0]),
                "found": ret.found,
                "travel_time_s":  ret.total_time_s    if ret.found else None,
                "distance_m":     ret.total_distance_m if ret.found else None,
                "computation_ms": ret.computation_ms,
                "error":          ret.error,
            })
            if ret.found:
                full_route.extend(ret.route[1:])

        elapsed = (time.perf_counter() - t0) * 1000
        metadata = {
            "multi_stop": True,
            "round_trip": scenario.round_trip,
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

    def _run_scenario_parallel_legs(self,
                                    G: nx.MultiDiGraph,
                                    scenario: Scenario,
                                    algos: list,
                                    n_workers: int) -> dict:
        """
        Flat-pool variant: every (algo × leg) pair is an independent task so
        all available cores stay busy instead of one core per algo.
        Returns dict[algo_name -> RouteResult].
        """
        # ── 1. Prep: compute visit order per algo, build flat task list ───
        algo_meta = {}
        flat_tasks = []

        for algo in algos:
            # Algorithms with _route_multi_stop handle the full tour themselves —
            # skip leg decomposition and run them in the main process instead.
            if scenario.is_multi_stop and hasattr(algo, "_route_multi_stop"):
                algo_meta[algo.name] = {"mode": "christofides"}
                continue

            nodes  = list(scenario.node_sequence)
            labels = list(scenario.label_sequence)
            coords = list(scenario.coord_sequence)

            if len(nodes) <= 2 and not scenario.round_trip:
                algo_meta[algo.name] = {"mode": "simple"}
                continue

            order_objective = "fixed"
            order_score     = None
            if scenario.optimize_order:
                ordered, order_objective, order_score = BenchmarkRunner._best_visit_order(
                    G, algo, nodes
                )
                idx_by_node = {n: i for i, n in enumerate(nodes)}
                order_idx   = [idx_by_node[n] for n in ordered]
                nodes       = ordered
                labels      = [labels[i] for i in order_idx]
                coords      = [coords[i] for i in order_idx]

            task_start = len(flat_tasks)
            legs_info  = []
            for idx, (src, dst) in enumerate(zip(nodes[:-1], nodes[1:]), start=1):
                leg_name = f"{scenario.name}_leg_{idx}"
                flat_tasks.append((algo, src, dst, leg_name))
                legs_info.append({
                    "idx": idx,
                    "from": labels[idx - 1] if idx - 1 < len(labels) else str(src),
                    "to":   labels[idx]     if idx     < len(labels) else str(dst),
                })

            if scenario.round_trip:
                flat_tasks.append((algo, nodes[-1], nodes[0],
                                   f"{scenario.name}_return"))
                legs_info.append({
                    "idx": len(nodes),
                    "from": labels[-1] if labels else str(nodes[-1]),
                    "to":   labels[0]  if labels else str(nodes[0]),
                })

            algo_meta[algo.name] = {
                "mode":             "multi",
                "nodes":            nodes,
                "labels":           labels,
                "coords":           coords,
                "order_objective":  order_objective,
                "order_score":      order_score,
                "task_start":       task_start,
                "task_count":       len(legs_info),
                "legs_info":        legs_info,
            }

        # ── 2. Run all tasks in a flat pool ───────────────────────────────
        results_by_name: dict = {}
        flat_results: list    = []
        if flat_tasks:
            actual_workers = min(len(flat_tasks), n_workers)
            g_bytes = pickle.dumps(G)
            with Pool(processes=actual_workers,
                      initializer=_worker_init,
                      initargs=(g_bytes,)) as pool:
                flat_results = pool.map(_leg_task, flat_tasks)

        # ── 3. Handle special-case algos in main process ──────────────────
        for algo in algos:
            meta = algo_meta.get(algo.name, {"mode": "simple"})

            if meta["mode"] == "christofides":
                if hasattr(algo, "_route_multi_stop"):
                    results_by_name[algo.name] = algo._route_multi_stop(
                        G, scenario.node_sequence, scenario.name,
                        source_node=scenario.source_node,
                        target_node=scenario.target_node,
                    )
                continue

            if meta["mode"] == "simple":
                results_by_name[algo.name] = algo.safe_run(
                    G, scenario.source_node, scenario.target_node, scenario.name
                )
                continue

            # ── 4. Assemble multi-stop result from leg results ────────────
            leg_results = flat_results[meta["task_start"]:
                                       meta["task_start"] + meta["task_count"]]
            nodes      = meta["nodes"]
            full_route: list = []
            leg_rows:   list = []
            total_ms         = sum(r.computation_ms for r in leg_results)
            failed           = False

            for result, info in zip(leg_results, meta["legs_info"]):
                leg_rows.append({
                    "leg":           info["idx"],
                    "from":          info["from"],
                    "to":            info["to"],
                    "found":         result.found,
                    "travel_time_s": result.total_time_s    if result.found else None,
                    "distance_m":    result.total_distance_m if result.found else None,
                    "computation_ms": result.computation_ms,
                    "error":         result.error,
                })
                if not result.found:
                    failed = True
                    err = (f"leg {info['idx']} failed: "
                           f"{info['from']} -> {info['to']} | {result.error}")
                    results_by_name[algo.name] = RouteResult.failure(
                        algo.name, scenario.name,
                        nodes[0], nodes[-1], err, total_ms
                    )
                    break
                if not full_route:
                    full_route.extend(result.route)
                else:
                    full_route.extend(result.route[1:])

            if failed:
                continue

            metadata = {
                "multi_stop":        True,
                "round_trip":        scenario.round_trip,
                "stop_count":        len(nodes),
                "stops":             meta["labels"],
                "visit_order_nodes": nodes,
                "visit_order_coords": meta["coords"],
                "order_objective":   meta["order_objective"],
                "order_score":       meta["order_score"],
                "legs":              leg_rows,
            }

            # Combine gen_history across legs (same logic as _run_algorithm_on_scenario)
            histories = [r.metadata.get("gen_history") for r in leg_results]
            if histories and all(histories):
                gen_count = min(len(h) for h in histories)
                combined  = []
                for gi in range(gen_count):
                    coords_acc = []; cand_coords = []; streets = []; cand_streets = []
                    t_min = t_dist = c_min = c_dist = 0.0
                    for h in histories:
                        fr = h[gi]
                        fc = fr.get("coords", [])
                        coords_acc   = coords_acc + fc[1:] if coords_acc and fc else coords_acc + fc
                        streets     += fr.get("streets", [])
                        t_min       += float(fr.get("min",  0.0))
                        t_dist      += float(fr.get("dist", 0.0))
                        cc = fr.get("candidate_coords", fc)
                        cand_coords  = cand_coords + cc[1:] if cand_coords and cc else cand_coords + cc
                        cand_streets += fr.get("candidate_streets", fr.get("streets", []))
                        c_min       += float(fr.get("candidate_min", fr.get("min",  0.0)))
                        c_dist      += float(fr.get("candidate_dist", fr.get("dist", 0.0)))
                    combined.append({
                        "gen": gi + 1,
                        "min": round(t_min, 3),  "dist": round(t_dist, 3),
                        "coords": coords_acc,     "streets": streets,
                        "candidate_min":  round(c_min,  3),
                        "candidate_dist": round(c_dist, 3),
                        "candidate_coords":   cand_coords,
                        "candidate_streets":  cand_streets,
                    })
                fm = leg_results[0].metadata
                metadata.update({
                    "generations":    fm.get("generations"),
                    "population":     fm.get("population"),
                    "crossover_rate": fm.get("crossover_rate"),
                    "mutation_rate":  fm.get("mutation_rate"),
                    "gen_history":    combined,
                })

            results_by_name[algo.name] = RouteResult.build(
                G, algo.name, scenario.name,
                nodes[0], nodes[-1], full_route, total_ms, metadata
            )

        return results_by_name

    def run(self, G: nx.MultiDiGraph, parallel_legs: bool = False) -> pd.DataFrame:
        from src.routing.visualize import ResultVisualiser

        self.results = []
        algos     = self.registry.all()
        n_workers = min(len(algos), os.cpu_count() or 1)
        mode      = "algo+leg" if parallel_legs else "algo"
        log.info(f"\nBenchmark: {len(algos)} algorithms × {len(self.scenarios)} scenarios "
                 f"(parallel mode={mode}, workers={n_workers})")

        for scenario in self.scenarios:
            route_label = " -> ".join(scenario.label_sequence)
            log.info(f"\n  Scenario [{scenario.name}]: {route_label}")

            if parallel_legs:
                n_leg_workers = min(
                    len(algos) * max(len(scenario.node_sequence) - 1, 1),
                    os.cpu_count() or 1,
                )
                results_by_name = self._run_scenario_parallel_legs(
                    G, scenario, algos, n_leg_workers
                )
            else:
                results_by_name = {}
                g_bytes = pickle.dumps(G)
                tasks   = [(algo, scenario) for algo in algos]
                with Pool(processes=n_workers,
                          initializer=_worker_init,
                          initargs=(g_bytes,)) as pool:
                    for algo, result in zip(algos, pool.map(_algo_task, tasks)):
                        results_by_name[algo.name] = result

            for algo in algos:
                result = results_by_name[algo.name]
                self.results.append(result)
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


_worker_G = None

def _worker_init(g_bytes: bytes):
    global _worker_G
    _worker_G = pickle.loads(g_bytes)

def _algo_task(args):
    algo, scenario = args
    return BenchmarkRunner._run_algorithm_on_scenario(algo, _worker_G, scenario)


def _leg_task(args):
    algo, src, dst, leg_name = args
    return algo.safe_run(_worker_G, src, dst, leg_name)


# ──────────────────────────────────────────────────────────────
# Category-based scenario builder (two focused scenarios)
# ──────────────────────────────────────────────────────────────

def build_category_scenarios(
    G: nx.MultiDiGraph,
    fac: gpd.GeoDataFrame,
    max_emergency: int = 50,
) -> List[Scenario]:
    """
    Build two focused scenarios from real facility data:

    1. emergency_patrol_circuit
       Seluruh pos polisi + pemadam kebakaran (maks 50 titik unik).
       Insight: konektivitas jaringan layanan darurat, waktu respons
       antar kecamatan, dan gap coverage di wilayah Surabaya.

    2. terminal_circuit
       Seluruh terminal bus + terminal feri + SPBU (maks tersedia).
       Insight: aksesibilitas terminal transportasi publik, waktu
       sirkuit inspeksi terminal, dan bottleneck koneksi antar hub.

    Kedua skenario bersifat sirkular (round_trip=True). Urutan titik
    ditentukan dengan heuristik nearest-neighbour geografis sehingga
    rute membentuk loop yang masuk akal secara spasial.
    """

    def _dedup(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Satu fasilitas per nearest_node; prioritaskan yang punya nama."""
        df = df.copy()
        df["_named"] = df["name"].notna() & (df["name"].str.strip() != "")
        df = df.sort_values("_named", ascending=False)
        return (df.drop_duplicates(subset="nearest_node")
                  .drop(columns="_named")
                  .reset_index(drop=True))

    def _diverse_subset(df: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
        """
        Greedy farthest-point sampling: pilih n titik yang paling
        tersebar secara geografis, memastikan coverage seluruh kota.
        """
        if len(df) <= n:
            return df
        lats = df["lat"].values
        lons = df["lon"].values
        clat, clon = lats.mean(), lons.mean()
        selected = [int(np.argmax((lats - clat) ** 2 + (lons - clon) ** 2))]
        for _ in range(n - 1):
            min_d = np.full(len(df), np.inf)
            for s in selected:
                d = (lats - lats[s]) ** 2 + (lons - lons[s]) ** 2
                min_d = np.minimum(min_d, d)
            min_d[selected] = -1.0
            selected.append(int(np.argmax(min_d)))
        return df.iloc[sorted(selected)].reset_index(drop=True)

    def _nn_order(df: gpd.GeoDataFrame, start_node: int = None) -> gpd.GeoDataFrame:
        """
        Urutkan titik-titik dengan nearest-neighbour tour starting dari
        start_node, membentuk rute sirkular yang geografis koheren.
        """
        df = df.reset_index(drop=True)
        n = len(df)
        if n <= 2:
            return df
        lats = df["lat"].values
        lons = df["lon"].values
        if start_node is not None and start_node in df["nearest_node"].values:
            start_idx = int(df.index[df["nearest_node"] == start_node][0])
        else:
            start_idx = int(np.argmin(lons))   # mulai dari ujung barat
        visited = [False] * n
        order = [start_idx]
        visited[start_idx] = True
        for _ in range(n - 1):
            cur = order[-1]
            best_d, best_j = np.inf, -1
            for j in range(n):
                if not visited[j]:
                    d = (lats[cur] - lats[j]) ** 2 + (lons[cur] - lons[j]) ** 2
                    if d < best_d:
                        best_d, best_j = d, j
            order.append(best_j)
            visited[best_j] = True
        return df.iloc[order].reset_index(drop=True)

    def _make_scenario(df: gpd.GeoDataFrame, name: str, description: str) -> Scenario:
        nodes  = df["nearest_node"].astype(int).tolist()
        labels = df["name"].where(df["name"].notna() & (df["name"].str.strip() != ""),
                                   df["facility_type"]).tolist()
        coords = list(zip(df["lat"].astype(float), df["lon"].astype(float)))
        return Scenario(
            name=name,
            description=description,
            source_node=nodes[0],
            target_node=nodes[-1],
            source_label=labels[0],
            target_label=labels[-1],
            source_coords=coords[0],
            target_coords=coords[-1],
            route_nodes=nodes,
            route_labels=labels,
            route_coords=coords,
            round_trip=True,
        )

    fac = fac.dropna(subset=["nearest_node"]).copy()
    fac["nearest_node"] = fac["nearest_node"].astype(int)

    # ── 1. EMERGENCY PATROL CIRCUIT ──────────────────────────────
    # Polisi + pemadam kebakaran → maks 50 titik unik
    # Titik awal: Kepolisian Daerah Jawa Timur (Polda Jatim) — markas utama
    POLDA_NODE = 9156956728

    emg = fac[fac["category"] == "emergency"].copy()
    emg = _dedup(emg)
    log.info(f"Emergency: {len(emg)} unique nodes after dedup (raw={len(fac[fac['category']=='emergency'])})")
    if len(emg) > max_emergency:
        emg = _diverse_subset(emg, max_emergency)
        log.info(f"Emergency: reduced to {len(emg)} via farthest-point geographic sampling")
    emg = _nn_order(emg, start_node=POLDA_NODE)
    n_emg = len(emg)

    emergency_scenario = _make_scenario(
        emg,
        name="emergency_patrol_circuit",
        description=(
            f"{n_emg}-stop sirkuit patroli darurat (polisi + pemadam kebakaran) — "
            "coverage seluruh Surabaya. "
            "Insight: konektivitas jaringan darurat, waktu tempuh antar pos, "
            "dan identifikasi area dengan gap coverage layanan darurat."
        ),
    )

    # ── 2. TERMINAL CIRCUIT ───────────────────────────────────────
    # Terminal bus + terminal feri + SPBU → semua titik unik (< 50)
    # Titik awal: Terminal Intermoda Joyoboyo — hub utama kota
    JOYOBOYO_NODE = 8148987377

    trans = fac[fac["category"] == "transport"].copy()
    trans = _dedup(trans)
    log.info(f"Terminal: {len(trans)} unique nodes after dedup (raw={len(fac[fac['category']=='transport'])})")
    trans = _nn_order(trans, start_node=JOYOBOYO_NODE)
    n_trans = len(trans)

    terminal_scenario = _make_scenario(
        trans,
        name="terminal_circuit",
        description=(
            f"{n_trans}-stop sirkuit terminal transportasi (terminal bus + feri + SPBU) — "
            "seluruh jaringan transportasi publik Surabaya. "
            "Insight: aksesibilitas terminal, waktu sirkuit inspeksi, "
            "dan bottleneck koneksi antar hub transportasi."
        ),
    )

    log.info(f"Scenarios built: [{emergency_scenario.name}] {n_emg} stops | "
             f"[{terminal_scenario.name}] {n_trans} stops")

    return [emergency_scenario, terminal_scenario]


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def run_platform(cfg):
    from src.routing.algorithms import (
        GeneticAlgorithm,
        ChristofidesAlgorithm,
        AntColonyRouting,
        GeraldSimulatedAnnealing,
        ParticleSwarmRouting,
        AntColonyRouting, AntColonyElite,
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
    # GA and Christofides handle multi-stop tours via _route_multi_stop.
    # ACO, SA, and PSO route leg-by-leg using the benchmark's leg decomposition.
    registry = AlgorithmRegistry()
    registry.register(GeneticAlgorithm())         # TSP-GA: evolves visit order
    registry.register(ChristofidesAlgorithm())    # TSP approximation (1.5x bound)
    registry.register(AntColonyRouting())         # pheromone-based path search
    registry.register(GeraldSimulatedAnnealing()) # distance-minimising SA
    registry.register(ParticleSwarmRouting())     # swarm path optimisation
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
    registry.register(AntColonyRouting())
    registry.register(AntColonyElite())
    registry.summary()

    # ── Build scenarios ──────────────────────────────────────
    # Two focused category-based circular scenarios:
    #   1. emergency_patrol_circuit  — police + fire stations (up to 50 stops)
    #   2. terminal_circuit          — bus/ferry terminals + fuel (all available)
    log.info("\nBuilding benchmark scenarios...")
    scenarios = build_category_scenarios(G, fac, max_emergency=50)
    if not scenarios:
        log.error("Could not build scenarios — check that extraction ran successfully.")
        return
    for s in scenarios:
        log.info(f"  [{s.name}]  {len(s.node_sequence)} stops | "
                 f"start={s.source_label} | round_trip={s.round_trip}")

    # ── Run benchmark ────────────────────────────────────────
    runner = BenchmarkRunner(registry, log_dir=cfg.LOG_DIR)
    for s in scenarios:
        runner.add_scenario(s)
    df      = runner.run(G, parallel_legs=getattr(cfg, "PARALLEL_LEGS", False))
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
