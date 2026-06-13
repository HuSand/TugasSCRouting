"""
Core data contracts for the routing platform.

RouteResult  — standardised output every algorithm must return
Scenario     — a source→target routing problem
BaseRoutingAlgorithm — abstract class all algorithms inherit
"""

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx


# ──────────────────────────────────────────────────────────────
# Vehicle
# ──────────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    """
    Representasi kendaraan dengan constraint bahan bakar.

    Attributes
    ----------
    vehicle_type        : str   — "motor" atau "mobil"
    label               : str   — nama tampilan
    tank_liters         : float — kapasitas tangki (liter)
    fuel_efficiency_kpl : float — km per liter
    range_km            : float — jarak maksimum dengan tangki penuh (km)
    refill_threshold_km : float — isi bensin kalau sisa range < nilai ini (km)
    """
    vehicle_type:        str
    label:               str
    tank_liters:         float
    fuel_efficiency_kpl: float
    range_km:            float
    refill_threshold_km: float = 20.0
    # ── Team Orienteering Problem (TOP) ──
    width_m:             float = 0.8    # lebar fisik kendaraan (m) untuk constraint jalan
    avg_speed_kph:       float = 40.0   # kecepatan rata-rata (untuk insight)
    service_time_s:      float = 600.0  # 10 menit per titik
    shift_seconds:       float = 21600.0  # 6 jam per shift

    @staticmethod
    def from_settings(cfg, vehicle_type: str = None) -> "Vehicle":
        """
        Buat Vehicle dari settings.py.
        Kalau vehicle_type tidak di-set, pakai cfg.DEFAULT_VEHICLE.
        """
        vtype = vehicle_type or cfg.DEFAULT_VEHICLE
        info  = cfg.VEHICLE_TYPES[vtype]
        return Vehicle(
            vehicle_type        = vtype,
            label               = info["label"],
            tank_liters         = info["tank_liters"],
            fuel_efficiency_kpl = info["fuel_efficiency_kpl"],
            range_km            = info["range_km"],
            refill_threshold_km = cfg.FUEL_REFILL_THRESHOLD_KM,
            width_m             = info.get("width_m", 0.8),
            avg_speed_kph       = info.get("avg_speed_kph", 40.0),
            service_time_s      = float(getattr(cfg, "SERVICE_TIME_S", 600)),
            shift_seconds       = float(getattr(cfg, "SHIFT_SECONDS", 21600)),
        )
    
    
# ──────────────────────────────────────────────────────────────
# RouteResult
# ──────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """Standardised output from any routing algorithm."""

    algorithm_name:  str
    scenario_name:   str
    source_node:     int
    target_node:     int

    route:           List[int]    # ordered node IDs source → target
    total_time_s:    float        # estimated travel time (seconds)
    total_distance_m: float       # route length (metres)
    nodes_in_route:  int          # len(route) — path length proxy
    computation_ms:  float        # wall-clock time inside find_route()

    metadata:  Dict[str, Any] = field(default_factory=dict)
    found:     bool            = True
    error:     Optional[str]   = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @staticmethod
    def build(G: nx.MultiDiGraph,
              algo_name: str,
              scenario_name: str,
              source: int,
              target: int,
              route: List[int],
              elapsed_ms: float,
              metadata: Dict = None) -> "RouteResult":
        """
        Build a successful RouteResult.
        Computes total_time_s and total_distance_m from the graph so
        individual algorithms don't have to repeat that arithmetic.
        """
        if metadata is None:
            metadata = {}
        total_time = total_dist = 0.0
        for u, v in zip(route[:-1], route[1:]):
            data = G.get_edge_data(u, v)
            if data:
                best = min(data.values(),
                           key=lambda d: float(d.get("travel_time", 9999)))
                total_time += float(best.get("travel_time", 0))
                total_dist += float(best.get("length", 0))
        return RouteResult(
            algorithm_name=algo_name,
            scenario_name=scenario_name,
            source_node=source,
            target_node=target,
            route=route,
            total_time_s=total_time,
            total_distance_m=total_dist,
            nodes_in_route=len(route),
            computation_ms=elapsed_ms,
            metadata=metadata,
            found=True,
        )

    @staticmethod
    def failure(algo_name: str,
                scenario_name: str,
                source: int,
                target: int,
                error: str,
                elapsed_ms: float) -> "RouteResult":
        """Build a failed RouteResult (no path found / exception)."""
        return RouteResult(
            algorithm_name=algo_name,
            scenario_name=scenario_name,
            source_node=source,
            target_node=target,
            route=[],
            total_time_s=float("inf"),
            total_distance_m=float("inf"),
            nodes_in_route=0,
            computation_ms=elapsed_ms,
            found=False,
            error=error,
        )


# ──────────────────────────────────────────────────────────────
# Scenario
# ──────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """A routing problem: get from source_node to target_node, optionally via stops."""
    name:          str
    description:   str
    source_node:   int
    target_node:   int
    source_label:  str   = ""
    target_label:  str   = ""
    source_coords: tuple = (0.0, 0.0)   # (lat, lon) for map pins
    target_coords: tuple = (0.0, 0.0)
    route_nodes:   Optional[List[int]] = None
    route_labels:  Optional[List[str]] = None
    route_coords:  Optional[List[tuple]] = None
    optimize_order: bool = False
    round_trip:     bool = False   # if True, add a return leg back to source after all stops

    @property
    def node_sequence(self) -> List[int]:
        return self.route_nodes or [self.source_node, self.target_node]

    @property
    def label_sequence(self) -> List[str]:
        return self.route_labels or [self.source_label, self.target_label]

    @property
    def coord_sequence(self) -> List[tuple]:
        return self.route_coords or [self.source_coords, self.target_coords]

    @property
    def is_multi_stop(self) -> bool:
        return len(self.node_sequence) > 2


# ──────────────────────────────────────────────────────────────
# BaseRoutingAlgorithm
# ──────────────────────────────────────────────────────────────

class BaseRoutingAlgorithm(ABC):
    """
    Contract every routing algorithm must satisfy.

    Your team subclasses this, sets `name` + `description`,
    and implements `find_route`. The platform handles everything
    else: benchmarking, logging, comparison, visualisation.

    Quick-start skeleton
    --------------------
    class MyAlgorithm(BaseRoutingAlgorithm):
        name        = "my_algo"
        description = "One line explaining your approach"

        def find_route(self, G, source_node, target_node, scenario_name=""):
            start = time.perf_counter()

            # ── YOUR ROUTING LOGIC ──────────────────────────
            route = []        # list of node IDs, source → target
            metadata = {}     # anything extra you want to record
            # ────────────────────────────────────────────────

            elapsed_ms = (time.perf_counter() - start) * 1000
            return RouteResult.build(G, self.name, scenario_name,
                                     source_node, target_node,
                                     route, elapsed_ms, metadata)
    """

    name:        str = "base"
    description: str = "Base — do not register"

    @abstractmethod
    def find_route(self,
                   G: nx.MultiDiGraph,
                   source_node: int,
                   target_node: int,
                   scenario_name: str = "") -> RouteResult:
        """
        Find the best route from source_node to target_node.

        Parameters
        ----------
        G            NetworkX MultiDiGraph.
                     Edge attributes: length (m), travel_time (s), speed_kph.
                     Access via G.get_edge_data(u, v).
        source_node  Starting road-network node ID (int).
        target_node  Destination road-network node ID (int).
        scenario_name  Label for result tagging (pass through as-is).

        Returns
        -------
        RouteResult  Use RouteResult.build() on success,
                     RouteResult.failure() on no-path / exception.
        """

    def safe_run(self, G, source_node, target_node, scenario_name="") -> RouteResult:
        """Wraps find_route with exception handling so one bad algo can't crash the benchmark."""
        t0 = time.perf_counter()
        try:
            return self.find_route(G, source_node, target_node, scenario_name)
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(
                self.name, scenario_name, source_node, target_node,
                traceback.format_exc(), elapsed,
            )

    # ──────────────────────────────────────────────────────────
    # Team Orienteering Problem (TOP) entry point
    # ──────────────────────────────────────────────────────────

    def solve_orienteering(self,
                           problem,
                           scenario_name: str = "",
                           seed: int = 42) -> RouteResult:
        """
        Maksimasi jumlah titik distinct yang dikunjungi dari depot kembali ke
        depot, di bawah time budget shift (travel_time + service_time per titik).

        Default: greedy insertion (cheapest feasible insertion). Tiap model
        (GA/ACO/SA/PSO) meng-override method ini dengan strategi pencarian
        masing-masing. Semua memakai helper feasibility/expand yang sama di
        src/routing/orienteering.py supaya konsisten.

        Parameters
        ----------
        problem : OrienteeringProblem  (src/routing/orienteering.py)
            Berisi G (sudah terfilter lebar jalan), depot, pool_nodes, budget_s,
            service_s, exclude set, pair_cost, koordinat.
        scenario_name : label untuk tagging hasil.

        Returns
        -------
        RouteResult dengan metadata: visited_stops, visited_count, shift, vehicle.
        """
        from src.routing.orienteering import greedy_insertion
        # Default greedy bersifat deterministik → seed diabaikan.
        return greedy_insertion(problem, self.name, scenario_name)

    def safe_orienteering(self, problem, scenario_name: str = "", seed: int = 42) -> RouteResult:
        """Wrap solve_orienteering dengan exception handling."""
        t0 = time.perf_counter()
        try:
            return self.solve_orienteering(problem, scenario_name, seed)
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(
                self.name, scenario_name,
                problem.depot, problem.depot,
                traceback.format_exc(), elapsed,
            )
