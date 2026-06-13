"""
src/routing/orienteering.py
===========================
Engine bersama untuk Team Orienteering Problem (TOP).

Objektif
--------
Maksimasi jumlah titik distinct yang dikunjungi dari depot kembali ke depot,
di bawah time budget shift:

    total_time = sum(travel_time antar stop) + service_time * jumlah_titik
    feasible   <=> total_time <= budget_s   (dan rute balik ke depot)

Semua model (GA/ACO/SA/PSO) memakai helper di sini supaya konsisten:
- OrienteeringProblem : state masalah + cost matrix (precompute sekali per kendaraan)
- greedy_insertion    : baseline deterministik (juga dipakai insight report)
- random_feasible_tour: seed acak feasible untuk metaheuristik
- build_route_result  : expand tur stop -> rute node penuh + RouteResult + metadata
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import networkx as nx

from src.routing.base import RouteResult

log = logging.getLogger(__name__)

INF = float("inf")


# ──────────────────────────────────────────────────────────────
# Problem definition
# ──────────────────────────────────────────────────────────────

@dataclass
class OrienteeringProblem:
    """
    Satu instance TOP untuk satu kendaraan + satu shift.

    Cost matrix (pair_cost) di-precompute SEKALI per (graph kendaraan) lalu
    di-reuse untuk kedua shift dan ke-10 iterasi. `exclude` membedakan titik
    yang sudah dikunjungi (shift sebelumnya / kendaraan lain) — no-overlap.
    """
    G:          nx.MultiDiGraph
    depot:      int
    pool_nodes: List[int]                       # semua kandidat (sebelum exclude)
    budget_s:   float
    service_s:  float
    pair_cost:  Dict[tuple, float]              # (a,b) -> travel_time detik
    exclude:    set = field(default_factory=set)
    labels:     Dict[int, str]   = field(default_factory=dict)
    coords:     Dict[int, tuple] = field(default_factory=dict)
    node_cat:   Dict[int, str]   = field(default_factory=dict)
    vehicle:    Optional[object] = None
    shift:      int = 1

    # ── helpers ──
    def available(self) -> List[int]:
        """Titik kandidat yang belum di-exclude."""
        return [n for n in self.pool_nodes if n not in self.exclude]

    def leg_cost(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        return self.pair_cost.get((a, b), INF)

    def travel_time(self, order: Sequence[int]) -> float:
        """Total travel time (detik) untuk [depot] + order + [depot]."""
        tour = [self.depot] + list(order) + [self.depot]
        return sum(self.leg_cost(a, b) for a, b in zip(tour[:-1], tour[1:]))

    def total_time(self, order: Sequence[int]) -> float:
        """Travel time + service time (10 menit per titik)."""
        return self.travel_time(order) + self.service_s * len(order)

    def is_feasible(self, order: Sequence[int]) -> bool:
        return self.total_time(order) <= self.budget_s + 1e-6

    def repair(self, order: Sequence[int]) -> List[int]:
        """
        Buang titik dari ekor sampai feasible, lalu buang duplikat & node
        ter-exclude. Dipakai metaheuristik (GA/PSO) untuk menjaga validitas.
        """
        seen = set()
        clean = []
        for n in order:
            if n == self.depot or n in self.exclude or n in seen:
                continue
            seen.add(n)
            clean.append(n)
        while clean and not self.is_feasible(clean):
            clean.pop()
        return clean

    def with_exclude(self, exclude: set) -> "OrienteeringProblem":
        """Problem baru (shift berikutnya) yang berbagi cost matrix yang sama."""
        return OrienteeringProblem(
            G=self.G, depot=self.depot, pool_nodes=self.pool_nodes,
            budget_s=self.budget_s, service_s=self.service_s,
            pair_cost=self.pair_cost, exclude=set(exclude),
            labels=self.labels, coords=self.coords, node_cat=self.node_cat,
            vehicle=self.vehicle, shift=self.shift,
        )


# ──────────────────────────────────────────────────────────────
# Precompute cost matrix
# ──────────────────────────────────────────────────────────────

def precompute_pair_cost(G: nx.MultiDiGraph,
                         nodes: List[int],
                         budget_s: float,
                         weight: str = "travel_time") -> Dict[tuple, float]:
    """
    Cost matrix travel_time antar semua node (depot + pool) via Dijkstra.

    Pakai cutoff=budget_s: leg yang lebih panjang dari satu shift tidak mungkin
    masuk solusi, jadi tidak perlu dihitung (memangkas ekspansi Dijkstra).
    Pair yang tak terjangkau / > cutoff dianggap INF (tidak dimasukkan).
    """
    pair_cost: Dict[tuple, float] = {}
    n = len(nodes)
    for i, src in enumerate(nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, src, cutoff=budget_s, weight=weight
            )
        except (nx.NodeNotFound, nx.NetworkXError):
            lengths = {}
        for dst in nodes:
            if src != dst and dst in lengths:
                pair_cost[(src, dst)] = float(lengths[dst])
        if (i + 1) % 25 == 0 or i == n - 1:
            log.debug(f"    pair_cost precompute {i+1}/{n} sources")
    return pair_cost


def build_problem(G: nx.MultiDiGraph,
                  depot: int,
                  pool_nodes: List[int],
                  budget_s: float,
                  service_s: float,
                  vehicle=None,
                  labels: Dict[int, str] = None,
                  coords: Dict[int, tuple] = None,
                  node_cat: Dict[int, str] = None,
                  shift: int = 1) -> OrienteeringProblem:
    """
    Bangun OrienteeringProblem + precompute cost matrix sekali.
    Hanya pool_nodes yang ada di graph (terfilter lebar jalan) yang dipakai.
    """
    pool_in_graph = [n for n in pool_nodes if n in G and n != depot]
    all_nodes = [depot] + pool_in_graph
    log.info(f"  TOP precompute: depot + {len(pool_in_graph)} titik "
             f"(budget={budget_s/3600:.1f}h, service={service_s/60:.0f}min/titik)")
    pair_cost = precompute_pair_cost(G, all_nodes, budget_s)
    return OrienteeringProblem(
        G=G, depot=depot, pool_nodes=pool_in_graph,
        budget_s=budget_s, service_s=service_s, pair_cost=pair_cost,
        exclude=set(), labels=labels or {}, coords=coords or {},
        node_cat=node_cat or {}, vehicle=vehicle, shift=shift,
    )


# ──────────────────────────────────────────────────────────────
# Solvers / generators
# ──────────────────────────────────────────────────────────────

def _score(problem: OrienteeringProblem, order: Sequence[int]) -> float:
    """
    Skor objektif TOP: utamakan jumlah titik, tie-break waktu lebih kecil.
    Lebih besar = lebih baik. Solusi tak-feasible diberi penalti besar.
    """
    if not problem.is_feasible(order):
        # penalti proporsional sejauh mana melebihi budget
        over = problem.total_time(order) - problem.budget_s
        return -1.0 - over / max(problem.budget_s, 1.0)
    return len(order) - 1e-6 * problem.total_time(order)


def _greedy_order(problem: OrienteeringProblem) -> List[int]:
    """
    Sisip titik dengan insertion cost termurah yang masih feasible, berulang.
    insertion_cost(x di antara a,b) = c(a,x) + c(x,b) - c(a,b)
    """
    order: List[int] = []
    candidates = set(problem.available())
    while candidates:
        best_node = None
        best_pos = None
        best_inc = INF
        tour = [problem.depot] + order + [problem.depot]
        for x in candidates:
            for pos in range(len(tour) - 1):
                a, b = tour[pos], tour[pos + 1]
                inc = (problem.leg_cost(a, x) + problem.leg_cost(x, b)
                       - problem.leg_cost(a, b))
                if inc < best_inc:
                    best_inc, best_node, best_pos = inc, x, pos
        if best_node is None or best_inc == INF:
            break
        trial = order[:best_pos] + [best_node] + order[best_pos:]
        candidates.discard(best_node)
        if problem.is_feasible(trial):
            order = trial
        elif not _any_feasible_addition(problem, order, candidates):
            break
    return order


def greedy_insertion(problem: OrienteeringProblem,
                     algo_name: str = "greedy",
                     scenario_name: str = "") -> RouteResult:
    """Baseline deterministik (juga dipakai insight report)."""
    t0 = time.perf_counter()
    order = _greedy_order(problem)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return build_route_result(problem, order, algo_name, scenario_name, elapsed_ms)


def _any_feasible_addition(problem, order, candidates) -> bool:
    for x in candidates:
        if problem.is_feasible(order + [x]):
            return True
    return False


def random_feasible_tour(problem: OrienteeringProblem, rng) -> List[int]:
    """Seed acak: shuffle kandidat, tambah selama masih feasible."""
    cand = problem.available()
    rng.shuffle(cand)
    order: List[int] = []
    for x in cand:
        if problem.is_feasible(order + [x]):
            order.append(x)
    return order


# ──────────────────────────────────────────────────────────────
# Operator bersama untuk metaheuristik
# ──────────────────────────────────────────────────────────────

def _tournament(problem, population, k, rng) -> List[int]:
    cands = rng.sample(population, min(k, len(population)))
    return list(max(cands, key=lambda o: _score(problem, o)))


def _ox_subset(p1, p2, rng) -> List[int]:
    """Order-crossover untuk subset variabel: prefix p1 + sisa dari p2."""
    if not p1:
        return list(p2)
    cut = rng.randint(0, len(p1))
    child = list(p1[:cut])
    seen = set(child)
    for n in p2:
        if n not in seen:
            child.append(n)
            seen.add(n)
    return child


def _mutate_order(problem, order, rng) -> List[int]:
    """Mutasi: add titik baru / swap / remove (subset+order berubah)."""
    order = list(order)
    avail = [n for n in problem.available() if n not in order]
    op = rng.random()
    if op < 0.45 and avail:
        order.insert(rng.randint(0, len(order)), rng.choice(avail))
    elif op < 0.75 and len(order) > 1:
        i, j = rng.sample(range(len(order)), 2)
        order[i], order[j] = order[j], order[i]
    elif order:
        order.pop(rng.randrange(len(order)))
    return order


def _empty_guard(problem, algo_name, scenario_name):
    if not problem.available():
        return build_route_result(problem, [], algo_name, scenario_name, 0.0)
    return None


# ──────────────────────────────────────────────────────────────
# Metaheuristik TOP (dipakai oleh model di algorithms.py)
# ──────────────────────────────────────────────────────────────

def ga_orienteering(problem, algo_name, scenario_name, *,
                    pop_size=40, generations=120, tournament=3,
                    mutation_rate=0.35, patience=25, seed=42) -> RouteResult:
    guard = _empty_guard(problem, algo_name, scenario_name)
    if guard:
        return guard
    rng = random.Random(seed)
    t0 = time.perf_counter()

    population = [_greedy_order(problem)]
    while len(population) < pop_size:
        population.append(random_feasible_tour(problem, rng))

    best = max(population, key=lambda o: _score(problem, o))
    history, no_improve = [], 0

    for gen in range(1, generations + 1):
        cur_best = max(population, key=lambda o: _score(problem, o))
        if _score(problem, cur_best) > _score(problem, best) + 1e-9:
            best, no_improve = list(cur_best), 0
        else:
            no_improve += 1
        history.append(_frame(problem, gen, best))
        if no_improve >= patience:
            break
        nextpop = [list(best)]  # elitism
        while len(nextpop) < pop_size:
            p1 = _tournament(problem, population, tournament, rng)
            p2 = _tournament(problem, population, tournament, rng)
            child = _ox_subset(p1, p2, rng)
            if rng.random() < mutation_rate:
                child = _mutate_order(problem, child, rng)
            nextpop.append(problem.repair(child))
        population = nextpop

    elapsed = (time.perf_counter() - t0) * 1000
    return build_route_result(problem, best, algo_name, scenario_name, elapsed, history)


def sa_orienteering(problem, algo_name, scenario_name, *,
                    T0=1.5, cooling=0.9985, iters=5000, seed=42) -> RouteResult:
    guard = _empty_guard(problem, algo_name, scenario_name)
    if guard:
        return guard
    rng = random.Random(seed)
    t0 = time.perf_counter()
    cur = _greedy_order(problem)
    best, T = list(cur), T0
    history, log_every = [], max(1, iters // 120)

    for it in range(1, iters + 1):
        cand = problem.repair(_mutate_order(problem, cur, rng))
        d = _score(problem, cand) - _score(problem, cur)
        if d > 0 or rng.random() < math.exp(d / max(T, 1e-9)):
            cur = cand
            if _score(problem, cur) > _score(problem, best):
                best = list(cur)
        T *= cooling
        if it % log_every == 0:
            history.append(_frame(problem, len(history) + 1, best))

    elapsed = (time.perf_counter() - t0) * 1000
    return build_route_result(problem, best, algo_name, scenario_name, elapsed, history)


def aco_orienteering(problem, algo_name, scenario_name, *,
                     n_ants=20, n_iter=30, alpha=1.0, beta=3.0,
                     rho=0.1, seed=42) -> RouteResult:
    guard = _empty_guard(problem, algo_name, scenario_name)
    if guard:
        return guard
    rng = random.Random(seed)
    t0 = time.perf_counter()
    avail = problem.available()
    tau: Dict[tuple, float] = {}

    def ph(a, b):
        return tau.get((a, b), 1.0)

    best, history = [], []
    for it in range(1, n_iter + 1):
        iter_best = []
        for _ in range(n_ants):
            order, cur, remaining = [], problem.depot, set(avail)
            while remaining:
                feas = [x for x in remaining if problem.is_feasible(order + [x])]
                if not feas:
                    break
                weights = [(ph(cur, x) ** alpha) *
                           ((1.0 / (problem.leg_cost(cur, x) + 1.0)) ** beta)
                           for x in feas]
                tot = sum(weights)
                if tot <= 0:
                    nxt = rng.choice(feas)
                else:
                    r, acc = rng.random() * tot, 0.0
                    nxt = feas[-1]
                    for x, w in zip(feas, weights):
                        acc += w
                        if acc >= r:
                            nxt = x
                            break
                order.append(nxt)
                remaining.discard(nxt)
                cur = nxt
            if _score(problem, order) > _score(problem, iter_best):
                iter_best = order
        if _score(problem, iter_best) > _score(problem, best):
            best = list(iter_best)
        for k in list(tau):
            tau[k] *= (1 - rho)
        tour = [problem.depot] + best + [problem.depot]
        for a, b in zip(tour[:-1], tour[1:]):
            tau[(a, b)] = ph(a, b) + len(best)
        history.append(_frame(problem, it, best))

    elapsed = (time.perf_counter() - t0) * 1000
    return build_route_result(problem, best, algo_name, scenario_name, elapsed, history)


def pso_orienteering(problem, algo_name, scenario_name, *,
                     n_particles=30, n_iter=80, gbest_pull=0.6, seed=42) -> RouteResult:
    guard = _empty_guard(problem, algo_name, scenario_name)
    if guard:
        return guard
    rng = random.Random(seed)
    t0 = time.perf_counter()
    # Satu partikel di-seed greedy (seperti GA) supaya kualitas konsisten.
    parts = [_greedy_order(problem)]
    parts += [random_feasible_tour(problem, rng) for _ in range(n_particles - 1)]
    pbest = [list(p) for p in parts]
    gbest = list(max(pbest, key=lambda o: _score(problem, o)))
    history = []

    for it in range(1, n_iter + 1):
        for i in range(n_particles):
            p = list(parts[i])
            missing = [n for n in gbest if n not in p]
            if missing and rng.random() < gbest_pull:
                p.insert(rng.randint(0, len(p)), rng.choice(missing))
            p = problem.repair(_mutate_order(problem, p, rng))
            parts[i] = p
            if _score(problem, p) > _score(problem, pbest[i]):
                pbest[i] = list(p)
                if _score(problem, p) > _score(problem, gbest):
                    gbest = list(p)
        history.append(_frame(problem, it, gbest))

    elapsed = (time.perf_counter() - t0) * 1000
    return build_route_result(problem, gbest, algo_name, scenario_name, elapsed, history)


def _frame(problem, gen, order) -> dict:
    """Frame ringan per-generasi untuk kurva evolusi viewer/insight."""
    return {
        "gen":       gen,
        "visited":   len(order),
        "total_min": round(problem.total_time(order) / 60, 1),
        "travel_min": round(problem.travel_time(order) / 60, 1),
    }


# ──────────────────────────────────────────────────────────────
# Result builder
# ──────────────────────────────────────────────────────────────

def build_route_result(problem: OrienteeringProblem,
                        order: List[int],
                        algo_name: str,
                        scenario_name: str,
                        elapsed_ms: float,
                        gen_history: list = None) -> RouteResult:
    """
    Expand tur stop -> rute node penuh di graph (terfilter), bangun RouteResult
    + metadata lengkap untuk CSV, viewer, dan insight.
    """
    G = problem.G
    depot = problem.depot
    full_stops = [depot] + list(order) + [depot]

    full_route: List[int] = []
    legs: List[dict] = []
    feasible = True

    for idx, (a, b) in enumerate(zip(full_stops[:-1], full_stops[1:]), start=1):
        try:
            leg = nx.shortest_path(G, a, b, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            feasible = False
            leg = [a, b]
        leg_streets = _street_names(G, leg)
        legs.append({
            "leg":     idx,
            "from":    problem.labels.get(a, str(a)),
            "to":      problem.labels.get(b, str(b)),
            "from_node": a,
            "to_node":   b,
            "from_cat": problem.node_cat.get(a, ""),
            "to_cat":   problem.node_cat.get(b, ""),
            "streets": leg_streets,
            "coords":  [_node_coord(G, n) for n in leg],
        })
        if not full_route:
            full_route.extend(leg)
        else:
            full_route.extend(leg[1:])

    travel_s  = problem.travel_time(order)
    service_s = problem.service_s * len(order)
    total_s   = travel_s + service_s

    metadata = {
        "objective":       "team_orienteering",
        "visited_count":   len(order),
        "visited_nodes":   list(order),
        "visited_stops":   [problem.labels.get(n, str(n)) for n in order],
        "visited_cats":    [problem.node_cat.get(n, "") for n in order],
        "visited_coords":  [_node_coord(G, n) for n in order],
        "shift":           problem.shift,
        "vehicle":         getattr(problem.vehicle, "vehicle_type", None),
        "vehicle_label":   getattr(problem.vehicle, "label", None),
        "depot_node":      depot,
        "budget_s":        problem.budget_s,
        "service_s_total": service_s,
        "travel_time_s":   round(travel_s, 1),
        "total_time_s":    round(total_s, 1),
        "feasible":        feasible and problem.is_feasible(order),
        "legs":            legs,
        "route_coords":    [_node_coord(G, n) for n in full_route],
    }
    if gen_history:
        metadata["gen_history"] = gen_history

    return RouteResult.build(
        G, algo_name, scenario_name, depot, depot,
        full_route, elapsed_ms, metadata,
    )


def _node_coord(G: nx.MultiDiGraph, n: int) -> list:
    """[lat, lon] dari node, atau [0,0] kalau tak ada."""
    data = G.nodes.get(n, {})
    return [float(data.get("y", 0.0)), float(data.get("x", 0.0))]


def _street_names(G: nx.MultiDiGraph, route: list) -> list:
    """
    Nama jalan yang dilalui (deduplicated berurutan) untuk turn-by-turn viewer.
    Versi ringan (tanpa import folium/matplotlib dari visualize.py).
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
