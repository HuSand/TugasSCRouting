"""
════════════════════════════════════════════════════════════════
 TEAM ALGORITHM FILE
 Add your routing algorithms here.
════════════════════════════════════════════════════════════════

HOW TO ADD YOUR ALGORITHM
--------------------------
1.  Copy the skeleton at the bottom of this file.
2.  Give it a unique `name` and `description`.
3.  Write your routing logic inside `find_route`.
4.  Register it at the bottom: REGISTRY.register(YourAlgorithm())
5.  Run:  python main.py compare

Your algorithm will automatically appear in the benchmark results,
comparison map, and charts — no other file needs to change.

WHAT YOU HAVE ACCESS TO
-----------------------
G            : NetworkX MultiDiGraph of the Surabaya road network.
               Nodes are road intersections (int IDs).
               Edges have attributes:
                 length        (float)  metres
                 travel_time   (float)  seconds
                 speed_kph     (float)  km/h
                 highway       (str)    OSM road type
               Access: G.get_edge_data(u, v) → dict of edges

source_node  : int — starting node ID
target_node  : int — destination node ID
G.nodes[n]   : dict with 'x' (lon) and 'y' (lat) for node n

Useful NetworkX functions:
  nx.shortest_path(G, src, dst, weight="travel_time")
  nx.shortest_path_length(G, src, dst, weight="travel_time")
  nx.astar_path(G, src, dst, heuristic=fn, weight="travel_time")
  list(G.predecessors(n))   — incoming neighbours
  list(G.successors(n))     — outgoing neighbours

Return RouteResult.build(...) on success.
Return RouteResult.failure(...) on no-path or exception.
════════════════════════════════════════════════════════════════
"""

import time
import math
import random
import networkx as nx

from src.routing.base import BaseRoutingAlgorithm, RouteResult, Scenario


# ──────────────────────────────────────────────────────────────
# Built-in Baselines
# (do not remove — used as reference lines in comparisons)
# used as comparison algorithm to AStar(GA Algorithm)
# ──────────────────────────────────────────────────────────────

class DijkstraTime(BaseRoutingAlgorithm):
    """Dijkstra minimising travel time. Standard fastest-route baseline."""
    name        = "dijkstra_time"
    description = "Dijkstra — minimise travel time (fastest route)"

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()
        try:
            route = nx.shortest_path(G, source_node, target_node, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name, source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, self.name, scenario_name, source_node, target_node, route, ms)


class DijkstraDistance(BaseRoutingAlgorithm):
    """Dijkstra minimising physical distance. Shortest-path baseline."""
    name        = "dijkstra_distance"
    description = "Dijkstra — minimise distance in metres (shortest path)"

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()
        try:
            route = nx.shortest_path(G, source_node, target_node, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name, source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, self.name, scenario_name, source_node, target_node, route, ms)

# ──────────────────────────────────────────────────────────────
# Built-in Baselines
# (do not remove — used as lines in comparisons)
# ──────────────────────────────────────────────────────────────
class AStarTime(BaseRoutingAlgorithm):
    """
    A* with straight-line (haversine) heuristic, minimising travel time.
    Typically faster than Dijkstra on large graphs — useful speed comparison.
    """
    name        = "astar_time"
    description = "A* — minimise travel time with haversine heuristic"

    @staticmethod
    def _heuristic(u, v, G):
        """Straight-line travel time estimate (haversine / avg speed)."""
        try:
            nu, nv = G.nodes[u], G.nodes[v]
            lat1, lon1 = math.radians(nu["y"]), math.radians(nu["x"])
            lat2, lon2 = math.radians(nv["y"]), math.radians(nv["x"])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            dist_m = 2 * 6_371_000 * math.asin(math.sqrt(a))
            avg_speed_ms = 30 / 3.6   # conservative 30 km/h
            return dist_m / avg_speed_ms
        except Exception:
            return 0.0

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()
        try:
            route = nx.astar_path(
                G, source_node, target_node,
                heuristic=lambda u, v: self._heuristic(u, v, G),
                weight="travel_time",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name, source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, self.name, scenario_name, source_node, target_node, route, ms)

# ──────────────────────────────────────────────────────────────
class AStarDistance(BaseRoutingAlgorithm):
    """
    A* with straight-line (haversine) heuristic, minimising physical distance.
    Direct comparison for DijkstraDistance, usually with fewer graph expansions.
    """
    name        = "astar_distance"
    description = "A* — minimise distance in metres with haversine heuristic"

    @staticmethod
    def _heuristic(u, v, G):
        """Straight-line distance estimate in metres."""
        try:
            nu, nv = G.nodes[u], G.nodes[v]
            lat1, lon1 = math.radians(nu["y"]), math.radians(nu["x"])
            lat2, lon2 = math.radians(nv["y"]), math.radians(nv["x"])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            return 2 * 6_371_000 * math.asin(math.sqrt(a))
        except Exception:
            return 0.0

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()
        try:
            route = nx.astar_path(
                G, source_node, target_node,
                heuristic=lambda u, v: self._heuristic(u, v, G),
                weight="length",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name, source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, self.name, scenario_name, source_node, target_node, route, ms)



# ──────────────────────────────────────────────────────────────
# GA SHARED HELPERS
# Dipakai oleh semua GA (Sandy, Burhan, Bimo, Gerald).
# Jangan diubah — kalau mau custom, override di class masing-masing.
# ──────────────────────────────────────────────────────────────

def _route_streets(G, path: list) -> list:
    """Deduplicated street names for a path (same logic as visualize.py)."""
    streets, prev = [], None
    for u, v in zip(path[:-1], path[1:]):
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


def _ga_path_cost(G, path: list) -> float:
    """Total travel_time sepanjang path (lower = better)."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if data is None:
            return float("inf")
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        total += float(best.get("travel_time", 9999))
    return total


def _ga_path_distance(G, path: list) -> float:
    """Total distance sepanjang path dalam meter."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if data is None:
            return float("inf")
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        total += float(best.get("length", 0))
    return total


def _ga_random_path(G, source: int, target: int, rng: random.Random):
    """
    Hasilkan 1 path dengan Dijkstra + noise acak pada bobot edge.
    Beda seed → beda path → populasi awal jadi beragam.
    """
    def noisy_weight(u, v, data):
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        t = float(best.get("travel_time", 9999))
        return t * rng.uniform(0.7, 1.5)   # noise 70%-150%

    try:
        return nx.shortest_path(G, source, target, weight=noisy_weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _ga_crossover(p1: list, p2: list, rng: random.Random) -> list:
    """
    Common-node crossover:
      cari node tengah yang ada di kedua parent,
      ambil p1[:pivot] + p2[pivot:].
    Kalau tidak ada node bersama, kembalikan kopi p1.
    """
    set1 = set(p1[1:-1])          # exclude source & target
    common = [n for n in p2[1:-1] if n in set1]
    if not common:
        return p1[:]
    pivot = rng.choice(common)
    i1 = p1.index(pivot)
    i2 = p2.index(pivot)
    return p1[:i1] + p2[i2:]


def _ga_mutate(G, path: list, rng: random.Random) -> list:
    """
    Mutasi: pilih dua node acak dalam path,
    ganti sub-segmen di antaranya dengan shortest path baru.
    Efek: 'jalan pintas' baru yang mungkin lebih efisien.
    """
    if len(path) < 3:
        return path
    i = rng.randint(0, len(path) - 2)
    j = rng.randint(i + 1, min(i + max(len(path) // 3, 2), len(path) - 1))
    try:
        seg = nx.shortest_path(G, path[i], path[j], weight="travel_time")
        return path[:i] + seg + path[j + 1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return path


def _ga_tournament(population: list, fitness: list,
                   k: int, rng: random.Random) -> list:
    """Pilih 1 individu terbaik dari k kandidat acak (tournament selection)."""
    candidates = rng.sample(range(len(population)), min(k, len(population)))
    winner_idx = min(candidates, key=lambda i: fitness[i])
    return population[winner_idx]


def _sa_noisy_shortest_path(G, source: int, target: int,
                            rng: random.Random, noise_min=0.85, noise_max=1.35):
    """
    Hasilkan kandidat rute untuk SA dengan Dijkstra berbobot length + noise.
    Fokusnya shortest path, jadi objective dasarnya jarak fisik, bukan waktu.
    """
    def noisy_length(u, v, data):
        best = min(data.values(), key=lambda d: float(d.get("length", 999999)))
        dist = float(best.get("length", 999999))
        return dist * rng.uniform(noise_min, noise_max)

    try:
        return nx.shortest_path(G, source, target, weight=noisy_length)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _sa_neighbor_path(G, path: list, rng: random.Random) -> list:
    """
    Tetangga SA: pilih sub-rute acak, lalu cari penggantinya dengan
    shortest path bernoise supaya eksplorasi tetap valid di graph jalan.
    """
    if len(path) < 3:
        return path[:]

    i = rng.randint(0, len(path) - 2)
    max_span = max(2, len(path) // 3)
    j = rng.randint(i + 1, min(i + max_span, len(path) - 1))
    segment = _sa_noisy_shortest_path(
        G, path[i], path[j], rng, noise_min=0.75, noise_max=1.45
    )
    if not segment:
        return path[:]
    return path[:i] + segment + path[j + 1:]


# ──────────────────────────────────────────────────────────────
# INDIVIDUAL GA SLOTS
# Sandy  : DONE  -- populasi dan cross-over rate tinggi
# Burhan : TODO  -- isi TUNING ZONE kamu
# Bimo   : TODO  -- isi TUNING ZONE kamu
# Gerald : TODO  -- isi TUNING ZONE kamu
#
# Yang perlu diubah per orang: HANYA blok TUNING ZONE.
# Kalau mau lebih advanced, boleh override _crossover/_mutate.
# Jangan ubah find_route kecuali kamu tau yang kamu lakukan.
# ──────────────────────────────────────────────────────────────

def _ga_run(algo, G, source_node, target_node, scenario_name):
    """
    Shared GA loop — dipakai semua slot supaya tidak duplikat kode.
    Dipanggil dari find_route masing-masing class.
    """
    t0  = time.perf_counter()
    rng = random.Random(algo.RANDOM_SEED)

    # 1. Populasi awal
    population = []
    for _ in range(algo.POPULATION_SIZE):
        p = _ga_random_path(G, source_node, target_node, rng)
        if p:
            population.append(p)

    # Fallback jika graph terlalu sparse
    if not population:
        try:
            route = nx.shortest_path(G, source_node, target_node,
                                     weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(algo.name, scenario_name,
                                       source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, algo.name, scenario_name,
                                 source_node, target_node, route, ms)

    # 2. Evolusi
    gen_history = []
    for gen_idx in range(algo.GENERATIONS):
        # Gunakan _fitness() milik masing-masing algo — ini yang membedakan tiap orang
        fitness  = [algo._fitness(G, p) for p in population]
        best_idx = min(range(len(population)), key=lambda i: fitness[i])
        elite    = population[best_idx]

        # Rekam koordinat + nama jalan rute terbaik generasi ini
        coords = []
        for n in elite:
            node = G.nodes.get(n)
            if node:
                coords.append([round(float(node["y"]), 5),
                                round(float(node["x"]), 5)])
        gen_history.append({
            "gen":     gen_idx + 1,
            "min":     round(_ga_path_cost(G, elite) / 60, 3),  # selalu tampilkan travel_time asli
            "dist":    round(_ga_path_distance(G, elite) / 1000, 3),
            "coords":  coords,
            "streets": _route_streets(G, elite),
        })

        new_pop = [elite]
        while len(new_pop) < algo.POPULATION_SIZE:
            p1 = _ga_tournament(population, fitness, algo.TOURNAMENT_SIZE, rng)
            if rng.random() < algo.CROSSOVER_RATE:
                p2    = _ga_tournament(population, fitness, algo.TOURNAMENT_SIZE, rng)
                child = algo._crossover(p1, p2, rng)
            else:
                child = p1[:]
            if rng.random() < algo.MUTATION_RATE:
                child = algo._mutate(G, child, rng)
            new_pop.append(child)
        population = new_pop

    # 3. Return terbaik (pakai _fitness untuk konsistensi)
    fitness = [algo._fitness(G, p) for p in population]
    best    = population[min(range(len(population)), key=lambda i: fitness[i])]
    ms      = (time.perf_counter() - t0) * 1000
    return RouteResult.build(
        G, algo.name, scenario_name, source_node, target_node, best, ms,
        metadata={
            "generations":    algo.GENERATIONS,
            "population":     algo.POPULATION_SIZE,
            "crossover_rate": algo.CROSSOVER_RATE,
            "mutation_rate":  algo.MUTATION_RATE,
            "gen_history":    gen_history,
        },
    )


# ──────────────────────────────────────────────────────────────
# CHRISTOFIDES ALGORITHM
# Approximation algorithm untuk TSP pada metric space (complete graph).
# Dipakai untuk skenario multi-stop; point-to-point tetap aman lewat fallback.
# ──────────────────────────────────────────────────────────────

def _build_metric_closure(G: nx.MultiDiGraph, nodes: list, weight: str = "travel_time") -> tuple:
    """
    Build complete graph dengan edge weights = shortest path antar node.
    Return (metric_closure, pair_paths) supaya hasil TSP bisa di-expand
    kembali ke road network asli.
    """
    source_graph = G.to_undirected()
    closure = nx.Graph()
    closure.add_nodes_from(nodes)
    pair_paths = {}
    
    for i, src in enumerate(nodes):
        for dst in nodes[i+1:]:
            try:
                path = nx.shortest_path(source_graph, src, dst, weight=weight)
                path_len = nx.shortest_path_length(source_graph, src, dst, weight=weight)
                closure.add_edge(src, dst, weight=float(path_len))
                pair_paths[(src, dst)] = path
                pair_paths[(dst, src)] = list(reversed(path))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Jika tidak ada, gunakan infinity (tidak akan dipilih)
                closure.add_edge(src, dst, weight=float("inf"))
    
    return closure, pair_paths


def _christofides_tour(closure: nx.Graph, start_node=None) -> list:
    """
    Christofides algorithm untuk metric TSP.

    Output berupa urutan node kunjungan. Kalau start_node diberikan,
    hasil akan diputar agar dimulai dari node tersebut.
    
    Catatan: NetworkX menjalankan Christofides di graph lengkap berbobot.
    """
    tour = nx.approximation.traveling_salesman_problem(
        closure,
        weight="weight",
        cycle=False,
        method=nx.approximation.christofides,
    )
    if tour and tour[0] == tour[-1]:
        tour = tour[:-1]
    if start_node in tour:
        idx = tour.index(start_node)
        tour = tour[idx:] + tour[:idx]
    return tour


def _expand_tsp_tour_to_road_path(G: nx.MultiDiGraph, tsp_tour: list, weight: str = "travel_time") -> list:
    """
    Expand urutan kunjungan TSP menjadi path nyata di road network.
    Setiap pasangan node dalam tour dihubungkan dengan shortest path di G.
    
    Returns: full path (list of node IDs) dari tsp_tour[0] → ... → tsp_tour[-1].
    """
    if len(tsp_tour) < 2:
        return tsp_tour
    
    full_path = []
    for src, dst in zip(tsp_tour[:-1], tsp_tour[1:]):
        leg = nx.shortest_path(G, src, dst, weight=weight)
        if not full_path:
            full_path.extend(leg)
        else:
            full_path.extend(leg[1:])
    
    return full_path if full_path else tsp_tour


class ChristofidesAlgorithm(BaseRoutingAlgorithm):
    """
    Christofides approximation untuk multi-stop routing.

    Manual singkat:
    - Dipakai untuk skenario dengan 3+ waypoint.
    - Input node diambil dari scenario.route_nodes / node_sequence.
    - Urutan stop dihitung dari metric closure + Christofides.
    - Urutan itu lalu di-expand kembali ke road network asli.
    - Untuk point-to-point biasa tetap aman memakai shortest path.
    """
    name        = "christofides"
    description = "Christofides — multi-stop TSP approximation"

    def _route_multi_stop(self, G, nodes: list, scenario_name="", source_node=None, target_node=None):
        t0 = time.perf_counter()
        nodes = list(dict.fromkeys(nodes))
        if len(nodes) < 3:
            if source_node is None:
                source_node = nodes[0] if nodes else None
            if target_node is None:
                target_node = nodes[-1] if nodes else None
            if source_node is None or target_node is None:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(self.name, scenario_name, -1, -1, "Not enough nodes for routing", ms)
            try:
                route = nx.shortest_path(G, source_node, target_node, weight="travel_time")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(self.name, scenario_name, source_node, target_node, str(e), ms)
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.build(G, self.name, scenario_name, source_node, target_node, route, ms, {
                "algorithm_variant": "point_to_point_fallback",
                "reason": "fewer_than_three_waypoints",
            })

        closure, pair_paths = _build_metric_closure(G, nodes, weight="travel_time")
        tour = _christofides_tour(closure, start_node=source_node or nodes[0])
        if len(tour) < 2:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name, source_node or nodes[0], target_node or nodes[-1], "Christofides returned an empty tour", ms)

        try:
            route = _expand_tsp_tour_to_road_path(G, tour, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name, source_node or nodes[0], target_node or nodes[-1], str(e), ms)

        ms = (time.perf_counter() - t0) * 1000
        metadata = {
            "algorithm_variant": "christofides_tsp",
            "multi_stop": True,
            "stop_count": len(nodes),
            "stops": nodes,
            "tour_nodes": tour,
            "metric_closure_nodes": len(closure.nodes()),
            "expanded_path_nodes": len(route),
            "valid_solution": bool(route),
        }
        return RouteResult.build(
            G,
            self.name,
            scenario_name,
            source_node if source_node is not None else nodes[0],
            target_node if target_node is not None else nodes[-1],
            route,
            ms,
            metadata,
        )

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()
        try:
            route = nx.shortest_path(G, source_node, target_node, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name,
                                       source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        metadata = {
            "algorithm_variant": "point_to_point_fallback",
            "reason": "point_to_point_routing",
        }
        return RouteResult.build(G, self.name, scenario_name,
                                 source_node, target_node, route, ms, metadata)


# ──────────────────────────────────────────────────────────────
# SANDY
# Fitness: balanced waktu + jarak (50/50)
# → cenderung pilih rute lebih pendek meski sedikit lebih lambat
# ──────────────────────────────────────────────────────────────

class SandyGA(BaseRoutingAlgorithm):
    """
    Sandy — GA dengan fitness balanced time+distance.

    Strategi:
    - Fitness = 50% travel_time + 50% physical_distance (dinormalisasi).
      Efeknya: GA memilih rute yang tidak terlalu memutar meski sedikit
      lebih lambat dari rute tercepat murni. Berbeda dari Dijkstra time.
    - Populasi besar (50) + tournament ketat (5) → konvergen cepat.
    - Custom crossover: pivot di tengah path → segmen lebih seimbang.
    """
    name        = "sandy_ga"
    description = "Sandy — GA balanced time+distance (50/50)"

    # ── TUNING ZONE Sandy ─────────────────────────────────────
    POPULATION_SIZE = 100
    GENERATIONS     = 100
    CROSSOVER_RATE  = 0.85
    MUTATION_RATE   = 0.6
    TOURNAMENT_SIZE = 5
    RANDOM_SEED     = 42
    # ─────────────────────────────────────────────────────────

    def _fitness(self, G, path: list) -> float:
        """
        Balanced fitness: 50% waktu perjalanan + 50% jarak fisik.
        Jarak dinormalisasi ke satuan detik pakai kecepatan referensi 40 km/h.
        Rute yang lebih pendek (walau sedikit lebih lambat) tetap diunggulkan.
        """
        REF_SPEED_MS = 40 / 3.6   # 40 km/h dalam m/s
        time_s = 0.0
        dist_m = 0.0
        for u, v in zip(path[:-1], path[1:]):
            data = G.get_edge_data(u, v)
            if data is None:
                return float("inf")
            best    = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
            time_s += float(best.get("travel_time", 9999))
            dist_m += float(best.get("length", 0))
        dist_as_time = dist_m / REF_SPEED_MS   # konversi jarak → "waktu ekuivalen"
        return 0.5 * time_s + 0.5 * dist_as_time

    def _crossover(self, p1: list, p2: list, rng: random.Random) -> list:
        # Pivot = node bersama paling dekat ke tengah p1.
        set1   = set(p1[1:-1])
        common = [n for n in p2[1:-1] if n in set1]
        if not common:
            return p1[:]
        mid      = len(p1) // 2
        min_dist = min(abs(p1.index(n) - mid) for n in common)
        best     = [n for n in common if abs(p1.index(n) - mid) == min_dist]
        pivot    = rng.choice(best)
        i1       = p1.index(pivot)
        i2       = p2.index(pivot)
        return p1[:i1] + p2[i2:]

    def _mutate(self, G, path: list, rng: random.Random) -> list:
        return _ga_mutate(G, path, rng)

    def find_route(self, G, source_node, target_node, scenario_name=""):
        return _ga_run(self, G, source_node, target_node, scenario_name)


# ──────────────────────────────────────────────────────────────
# BURHAN — TODO: isi bagian ini
# ──────────────────────────────────────────────────────────────

class BurhanGA(BaseRoutingAlgorithm):
    """
    Burhan — tulis strategimu di sini setelah kamu tentukan.

    Yang WAJIB diubah:
      1. Angka-angka di TUNING ZONE
      2. Isi _fitness() dengan objective function milikmu

    Lihat SandyGA di atas sebagai contoh _fitness() yang sudah jadi.
    """
    name        = "burhan_ga"
    description = "Burhan — GA optimized (time + road quality + simplicity)"
    description = "Burhan — GA optimized (time + road quality + simplicity)"

    # ── TUNING ZONE Burhan -- UBAH ANGKA INI ─────────────────
    POPULATION_SIZE = 80    # TODO: coba variasikan
    GENERATIONS     = 120    # TODO: coba variasikan
    CROSSOVER_RATE  = 0.9   # TODO: coba variasikan
    MUTATION_RATE   = 0.4   # TODO: coba variasikan
    TOURNAMENT_SIZE = 5     # TODO: coba variasikan
    RANDOM_SEED     = 99
    # ─────────────────────────────────────────────────────────

    def _fitness(self, G, path: list) -> float:
        total_time = 0.0
        total_dist = 0.0
        total_speed = 0.0
        edges_count = 0

        for u, v in zip(path[:-1], path[1:]):
            data = G.get_edge_data(u, v)
            if data is None:
                return float("inf")

            best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))

            tt = float(best.get("travel_time", 9999))
            dist = float(best.get("length", 0))
            speed = float(best.get("speed_kph", 30))

            total_time += tt
            total_dist += dist
            total_speed += speed
            edges_count += 1

        if edges_count == 0:
            return float("inf")

        avg_speed = total_speed / edges_count

        # 🔥 NORMALIZATION (ini yang bikin beda)
        norm_time = total_time / 1000
        norm_dist = total_dist / 5000
        norm_complexity = edges_count / 50
        norm_speed = avg_speed / 50

        # 🔥 WEIGHTED MULTI-OBJECTIVE
        return (
            0.55 * norm_time +
            0.20 * norm_dist +
            0.15 * norm_complexity -
            0.25 * norm_speed
        )

        # # ── FITNESS FORMULA ─────────────────────────────
        # # 1. waktu = prioritas utama
        # # 2. penalti kompleksitas (banyak belokan)
        # # 3. reward jalan cepat

        # complexity_penalty = edges_count * 2.0
        # speed_reward = avg_speed * 5.0

        # return total_time + complexity_penalty - speed_reward

    # def _fitness(self, G, path: list) -> float:
    #     """
    #   TODO: ganti dengan objective function milikmu.

    #     Nilai return harus berupa float — semakin kecil = semakin baik.
    #     Defaultnya minimasi travel_time (sama seperti Dijkstra).
    #     Nilai return harus berupa float — semakin kecil = semakin baik.
    #     Defaultnya minimasi travel_time (sama seperti Dijkstra).

    #     Edge attributes yang bisa kamu pakai per edge (u, v):
    #       best = min(G.get_edge_data(u,v).values(),
    #                  key=lambda d: float(d.get("travel_time", 9999)))
    #       best.get("travel_time")  # detik
    #       best.get("length")       # meter
    #       best.get("speed_kph")    # km/h
    #       best.get("highway")      # tipe jalan: primary/secondary/residential/...
    #       best.get("name")         # nama jalan
    #     """
    #     return _ga_path_cost(G, path)   # default — ganti dengan idemu
    #     Edge attributes yang bisa kamu pakai per edge (u, v):
    #       best = min(G.get_edge_data(u,v).values(),
    #                  key=lambda d: float(d.get("travel_time", 9999)))
    #       best.get("travel_time")  # detik
    #       best.get("length")       # meter
    #       best.get("speed_kph")    # km/h
    #       best.get("highway")      # tipe jalan: primary/secondary/residential/...
    #       best.get("name")         # nama jalan
    #     """
    #     return _ga_path_cost(G, path)   # default — ganti dengan idemu

    def _crossover(self, p1: list, p2: list, rng: random.Random) -> list:
        return _ga_crossover(p1, p2, rng)   # TODO: boleh override

    def _mutate(self, G, path: list, rng: random.Random) -> list:
        return _ga_mutate(G, path, rng)     # TODO: boleh override

    def find_route(self, G, source_node, target_node, scenario_name=""):
        return _ga_run(self, G, source_node, target_node, scenario_name)

class AntColonyRouting(BaseRoutingAlgorithm):
     
    name        = "aco_routing"
    description = "Ant Colony Optimization — feromon + visibilitas travel_time"
 
    # ------------------------------------------------------------------
    # PARAMETER TUNING
    # ------------------------------------------------------------------
    N_ANTS        = 20     # jumlah semut per iterasi
    N_ITERATIONS  = 30     # jumlah iterasi koloni
    ALPHA         = 1.0    # bobot feromon τ — naikkan → lebih eksploitatif
    BETA          = 2.0    # bobot visibilitas η — naikkan → lebih greedy
    RHO           = 0.1    # laju evaporasi feromon (0.0–1.0)
                           # kecil → feromon bertahan lama (memori panjang)
                           # besar → feromon cepat hilang (eksplorasi lebih)
    Q             = 100.0  # konstanta deposit feromon
    TAU_INIT      = 1.0    # nilai feromon awal semua edge
    RANDOM_SEED   = 42
 
    # ------------------------------------------------------------------
    # HELPER 1: bangun graph sederhana (node → neighbors dengan cost)
    #           dari subset node yang relevan sekitar jalur source-target
    # ------------------------------------------------------------------
    def _get_candidates(self, G, node: int, visited: set) -> list:
        """
        Kembalikan list tetangga yang belum dikunjungi dari node ini,
        beserta travel_time edge terbaik ke masing-masing tetangga.
        Format: [(neighbor, travel_time), ...]
        """
        candidates = []
        for neighbor in G.successors(node):
            if neighbor in visited:
                continue
            edge_dict = G.get_edge_data(node, neighbor)
            if not edge_dict:
                continue
            best_tt = min(
                float(d.get("travel_time", 9999))
                for d in edge_dict.values()
            )
            if best_tt < 9999:
                candidates.append((neighbor, best_tt))
        return candidates
 
    # ------------------------------------------------------------------
    # HELPER 2: satu semut membangun rute dari source ke target
    # ------------------------------------------------------------------
    def _build_ant_path(
        self,
        G,
        source: int,
        target: int,
        pheromone: dict,
        rng: random.Random,
        max_steps: int
    ) -> list | None:
        """
        Semut bergerak dari source ke target dengan memilih node
        berikutnya secara probabilistik.
 
        Strategi:
          - Jika target ada di antara kandidat → langsung pilih target
          - Jika tidak ada kandidat → gunakan Dijkstra sebagai fallback
            untuk melanjutkan ke node terdekat menuju target
          - Jika melebihi max_steps → batalkan (path terlalu panjang)
 
        Returns: list node IDs, atau None jika gagal
        """
        path    = [source]
        visited = {source}
        current = source
 
        for _ in range(max_steps):
            if current == target:
                return path
 
            candidates = self._get_candidates(G, current, visited)
 
            # Tidak ada kandidat → coba Dijkstra lokal sebagai bridge
            if not candidates:
                try:
                    bridge = nx.shortest_path(
                        G, current, target, weight="travel_time"
                    )
                    # Gabungkan path yang sudah ada dengan sisa bridge
                    path += bridge[1:]
                    return path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None
 
            # Shortcut: jika target langsung bisa dicapai
            target_candidates = [(n, tt) for n, tt in candidates if n == target]
            if target_candidates:
                path.append(target)
                return path
 
            # ── Hitung probabilitas transisi ──────────────────────
            scores = []
            for neighbor, tt in candidates:
                tau = pheromone.get((current, neighbor), self.TAU_INIT)
                eta = 1.0 / tt if tt > 0 else 1.0
                score = (tau ** self.ALPHA) * (eta ** self.BETA)
                scores.append(score)
 
            total = sum(scores)
            if total == 0:
                # Semua skor nol → pilih acak (fallback uniform)
                chosen = rng.choice(candidates)[0]
            else:
                # Roulette wheel selection
                probs   = [s / total for s in scores]
                r       = rng.random()
                cumul   = 0.0
                chosen  = candidates[-1][0]  # default: kandidat terakhir
                for (neighbor, _), prob in zip(candidates, probs):
                    cumul += prob
                    if r <= cumul:
                        chosen = neighbor
                        break
 
            path.append(chosen)
            visited.add(chosen)
            current = chosen
 
        return None  # melebihi max_steps
 
    # ------------------------------------------------------------------
    # HELPER 3: hitung total travel_time sebuah path
    # ------------------------------------------------------------------
    def _path_cost(self, G, path: list) -> float:
        """
        Total travel_time (detik) sepanjang path.
        Pakai edge terbaik (travel_time terkecil) untuk setiap hop.
        """
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge_dict = G.get_edge_data(u, v)
            if not edge_dict:
                return float("inf")
            best_tt = min(
                float(d.get("travel_time", 9999))
                for d in edge_dict.values()
            )
            total += best_tt
        return total
 
    # ------------------------------------------------------------------
    # HELPER 4: deposit feromon pada path terbaik iterasi
    # ------------------------------------------------------------------
    def _deposit_pheromone(
        self,
        pheromone: dict,
        path: list,
        cost: float
    ) -> None:
        """
        Semut terbaik deposit feromon di sepanjang jalurnya.
        Jalur lebih pendek → deposit lebih banyak (Q / cost).
        """
        deposit = self.Q / cost if cost > 0 else 0.0
        for u, v in zip(path[:-1], path[1:]):
            key = (u, v)
            pheromone[key] = pheromone.get(key, self.TAU_INIT) + deposit
 
    # ------------------------------------------------------------------
    # CORE: jalankan koloni ACO
    # ------------------------------------------------------------------
    def _run_aco(self, G, source: int, target: int) -> list:
        """
        Jalankan N_ITERATIONS iterasi koloni ACO.
 
        Setiap iterasi:
          1. N_ANTS semut membangun path masing-masing
          2. Feromon menguap (evaporasi global)
          3. Semut terbaik iterasi deposit feromon
          4. Update global best jika ada yang lebih baik
 
        Returns: path terbaik yang ditemukan (list node IDs)
        Raises : nx.NetworkXNoPath jika tidak ada path sama sekali
        """
        rng        = random.Random(self.RANDOM_SEED)
        pheromone  = {}  # sparse dict: (u,v) → tau value
        best_path  = None
        best_cost  = float("inf")
 
        # Estimasi max_steps: 3× jumlah node di subgraph lokal
        # (batas atas agar semut tidak loop selamanya)
        max_steps = min(G.number_of_nodes(), 5000)
 
        for iteration in range(self.N_ITERATIONS):
            iter_best_path = None
            iter_best_cost = float("inf")
 
            # ── Setiap semut bangun satu path ──────────────────
            for _ in range(self.N_ANTS):
                path = self._build_ant_path(
                    G, source, target, pheromone, rng, max_steps
                )
                if path is None or path[-1] != target:
                    continue
 
                cost = self._path_cost(G, path)
                if cost < iter_best_cost:
                    iter_best_path = path
                    iter_best_cost = cost
 
            # ── Evaporasi feromon (semua edge) ─────────────────
            for key in list(pheromone.keys()):
                pheromone[key] *= (1.0 - self.RHO)
                if pheromone[key] < 1e-6:
                    del pheromone[key]  # bersihkan nilai sangat kecil
 
            # ── Deposit feromon dari semut terbaik iterasi ─────
            if iter_best_path is not None:
                self._deposit_pheromone(pheromone, iter_best_path, iter_best_cost)
 
                # Update global best
                if iter_best_cost < best_cost:
                    best_path = iter_best_path
                    best_cost = iter_best_cost
 
        # Fallback: jika tidak ada semut berhasil, pakai Dijkstra
        if best_path is None:
            try:
                best_path = nx.shortest_path(
                    G, source, target, weight="travel_time"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                raise nx.NetworkXNoPath(
                    f"ACO: tidak ada jalur dari {source} ke {target}"
                )
 
        return best_path
 
    # ------------------------------------------------------------------
    # INTERFACE WAJIB — dipanggil oleh framework benchmark
    # ------------------------------------------------------------------
    def find_route(self, G, source_node, target_node, scenario_name=""):
        """
        Entry point yang dipanggil oleh benchmark.
        Menjalankan ACO dan membungkus hasilnya dalam RouteResult.
        """
        t0 = time.perf_counter()
        try:
            route = self._run_aco(G, source_node, target_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(
                self.name, scenario_name,
                source_node, target_node, str(e), ms
            )
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name,
            source_node, target_node, route, ms,
            metadata={
                "n_ants":       self.N_ANTS,
                "n_iterations": self.N_ITERATIONS,
                "alpha":        self.ALPHA,
                "beta":         self.BETA,
                "rho":          self.RHO,
                "strategy":     "pheromone × (1/travel_time) visibility, no highway weighting",
            }
        )
 
 
# ──────────────────────────────────────────────────────────────
# CARA REGISTRASI
# Tambahkan baris ini di bagian paling bawah algorithms.py:
#
#   REGISTRY.register(AntColonyRouting())
#
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# BIMO — TODO: isi bagian ini
# ──────────────────────────────────────────────────────────────

class BimoGA(BaseRoutingAlgorithm):
    """
    Bimo — tulis strategimu di sini setelah kamu tentukan.

    Yang WAJIB diubah:
      1. Angka-angka di TUNING ZONE
      2. Isi _fitness() dengan objective function milikmu

    Lihat SandyGA di atas sebagai contoh _fitness() yang sudah jadi.
    """
    name        = "bimo_ga"
    description = "Bimo — GA (belum dituning)"

    # ── TUNING ZONE Bimo -- UBAH ANGKA INI ───────────────────
    POPULATION_SIZE = 80    # lebih besar supaya eksplorasi rute lebih beragam
    GENERATIONS     = 90    # iterasi lebih banyak untuk stabilisasi fitness
    CROSSOVER_RATE  = 0.88  # tetap tinggi, tapi tidak agresif penuh
    MUTATION_RATE   = 0.42  # cukup eksploratif untuk menghindari stagnasi
    TOURNAMENT_SIZE = 4     # seleksi sedikit lebih ketat dari default
    RANDOM_SEED     = 20
    # ─────────────────────────────────────────────────────────

    def _fitness(self, G, path: list) -> float:
        """
        TODO: ganti dengan objective function milikmu.

        Nilai return harus berupa float — semakin kecil = semakin baik.
        Defaultnya minimasi travel_time (sama seperti Dijkstra).

        Edge attributes yang bisa kamu pakai per edge (u, v):
          best = min(G.get_edge_data(u,v).values(),
                     key=lambda d: float(d.get("travel_time", 9999)))
          best.get("travel_time")  # detik
          best.get("length")       # meter
          best.get("speed_kph")    # km/h
          best.get("highway")      # tipe jalan: primary/secondary/residential/...
          best.get("name")         # nama jalan
        """
        return _ga_path_cost(G, path)   # default — ganti dengan idemu

    def _crossover(self, p1: list, p2: list, rng: random.Random) -> list:
        return _ga_crossover(p1, p2, rng)   # TODO: boleh override

    def _mutate(self, G, path: list, rng: random.Random) -> list:
        return _ga_mutate(G, path, rng)     # TODO: boleh override

    def find_route(self, G, source_node, target_node, scenario_name=""):
        return _ga_run(self, G, source_node, target_node, scenario_name)


# ──────────────────────────────────────────────────────────────
# GERALD — TODO: isi bagian ini
# ──────────────────────────────────────────────────────────────

class GeraldGA(BaseRoutingAlgorithm):
    """
    Gerald — tulis strategimu di sini setelah kamu tentukan.

    Yang WAJIB diubah:
      1. Angka-angka di TUNING ZONE
      2. Isi _fitness() dengan objective function milikmu

    Lihat SandyGA di atas sebagai contoh _fitness() yang sudah jadi.
    """
    name        = "gerald_ga"
    description = "Gerald — GA (belum dituning)"

    # ── TUNING ZONE Gerald -- UBAH ANGKA INI ─────────────────
    POPULATION_SIZE = 30    # TODO: coba variasikan
    GENERATIONS     = 50    # TODO: coba variasikan
    CROSSOVER_RATE  = 0.8   # TODO: coba variasikan
    MUTATION_RATE   = 0.3   # TODO: coba variasikan
    TOURNAMENT_SIZE = 3     # TODO: coba variasikan
    RANDOM_SEED     = 30
    # ─────────────────────────────────────────────────────────

    def _fitness(self, G, path: list) -> float:
        """
        TODO: ganti dengan objective function milikmu.

        Nilai return harus berupa float — semakin kecil = semakin baik.
        Defaultnya minimasi travel_time (sama seperti Dijkstra).

        Edge attributes yang bisa kamu pakai per edge (u, v):
          best = min(G.get_edge_data(u,v).values(),
                     key=lambda d: float(d.get("travel_time", 9999)))
          best.get("travel_time")  # detik
          best.get("length")       # meter
          best.get("speed_kph")    # km/h
          best.get("highway")      # tipe jalan: primary/secondary/residential/...
          best.get("name")         # nama jalan
        """
        return _ga_path_cost(G, path)   # default — ganti dengan idemu

    def _crossover(self, p1: list, p2: list, rng: random.Random) -> list:
        return _ga_crossover(p1, p2, rng)   # TODO: boleh override

    def _mutate(self, G, path: list, rng: random.Random) -> list:
        return _ga_mutate(G, path, rng)     # TODO: boleh override

    def find_route(self, G, source_node, target_node, scenario_name=""):
        return _ga_run(self, G, source_node, target_node, scenario_name)


# ──────────────────────────────────────────────────────────────
# GERALD - Simulated Annealing untuk shortest path

class GeraldSimulatedAnnealing(BaseRoutingAlgorithm):
    """
    Gerald - Simulated Annealing untuk mencari shortest path.

    Objective utama = total distance (meter). Kandidat tetangga dibuat
    dengan mengganti satu sub-rute memakai shortest path berbobot length
    plus noise, lalu diterima memakai probabilitas annealing.
    """
    name        = "gerald_sa"
    description = "Gerald - Simulated Annealing shortest path (distance)"

    ITERATIONS          = 100
    INITIAL_TEMPERATURE = 1200.0
    COOLING_RATE        = 0.94
    MIN_TEMPERATURE     = 0.01
    RANDOM_SEED         = 31

    def _fitness(self, G, path: list) -> float:
        return _ga_path_distance(G, path)

    @staticmethod
    def _frame(G, gen_idx: int, best: list, candidate: list) -> dict:
        coords = []
        for n in best:
            node = G.nodes.get(n)
            if node:
                coords.append([round(float(node["y"]), 5),
                               round(float(node["x"]), 5)])

        candidate_coords = []
        for n in candidate:
            node = G.nodes.get(n)
            if node:
                candidate_coords.append([round(float(node["y"]), 5),
                                         round(float(node["x"]), 5)])

        return {
            "gen": gen_idx + 1,
            "min": round(_ga_path_cost(G, best) / 60, 3),
            "dist": round(_ga_path_distance(G, best) / 1000, 3),
            "coords": coords,
            "streets": _route_streets(G, best),
            "candidate_min": round(_ga_path_cost(G, candidate) / 60, 3),
            "candidate_dist": round(_ga_path_distance(G, candidate) / 1000, 3),
            "candidate_coords": candidate_coords,
            "candidate_streets": _route_streets(G, candidate),
        }

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()
        rng = random.Random(self.RANDOM_SEED)

        current = _sa_noisy_shortest_path(G, source_node, target_node, rng)
        if not current:
            try:
                current = nx.shortest_path(
                    G, source_node, target_node, weight="length"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(
                    self.name, scenario_name, source_node, target_node, str(e), ms
                )

        best = current[:]
        current_score = self._fitness(G, current)
        best_score = current_score
        temperature = self.INITIAL_TEMPERATURE
        gen_history = []

        for idx in range(self.ITERATIONS):
            candidate = _sa_neighbor_path(G, current, rng)
            candidate_score = self._fitness(G, candidate)
            delta = candidate_score - current_score

            accept = delta <= 0
            if not accept and temperature > self.MIN_TEMPERATURE:
                accept_prob = math.exp(-delta / temperature)
                accept = rng.random() < accept_prob

            if accept:
                current = candidate
                current_score = candidate_score

            if current_score < best_score:
                best = current[:]
                best_score = current_score

            gen_history.append(self._frame(G, idx, best, candidate))
            temperature = max(self.MIN_TEMPERATURE,
                              temperature * self.COOLING_RATE)

        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name, source_node, target_node, best, ms,
            metadata={
                "algorithm_family": "simulated_annealing",
                "generations": self.ITERATIONS,
                "population": 1,
                "initial_temperature": self.INITIAL_TEMPERATURE,
                "cooling_rate": self.COOLING_RATE,
                "min_temperature": self.MIN_TEMPERATURE,
                "mutation_rate": 1.0,
                "gen_history": gen_history,
            },
        )


# EXAMPLE SCENARIOS
# Rute konkret pakai fasilitas publik Surabaya beneran.
# Dipakai benchmark sebagai gantinya auto-generate random.
#
# Node ID diambil dari data/facilities_with_network.csv
# Bisa cek di data/surabaya_facilities_map.html untuk visualnya.
#
# PETA SKENARIO:
#   1. darmo_to_rsu_haji          RS Darmo (tengah-barat)
#                                    → RSU Haji Surabaya (timur)
#                                    Kasus: transfer pasien antar RS
#
#   2. polsek_genteng_to_rs_darmo  Polsek Genteng (pusat kota)
#                                    → RS Darmo
#                                    Kasus: respons darurat polisi → RS
#
#   3. national_to_rs_ramelan      National Hospital (barat jauh)
#                                    → RS Ramelan (selatan)
#                                    Kasus: lintas kota barat→selatan
#
#   4. polsek_rungkut_to_rs_onkologi  Polsek Rungkut (timur-selatan)
#                                      → RS Onkologi (timur)
#                                      Kasus: respons darurat area timur
#
#   5. ciputra_to_rsu_haji         Ciputra Hospital (barat jauh)
#                                    → RSU Haji Surabaya (timur)
#                                    Kasus: rute terpanjang lintas kota
# ──────────────────────────────────────────────────────────────

_LEGACY_SINGLE_STOP_SCENARIOS = [
    Scenario(
        name="darmo_to_rsu_haji",
        description="Transfer pasien: RS Darmo (tengah-barat) → RSU Haji Surabaya (timur)",
        source_node=1685220157,
        target_node=4332874690,
        source_label="RS Darmo",
        target_label="RSU Haji Surabaya",
        source_coords=(-7.2874, 112.7382),
        target_coords=(-7.2828, 112.7798),
    ),
    Scenario(
        name="polsek_genteng_to_rs_darmo",
        description="Respons darurat: Polsek Genteng (pusat) → RS Darmo",
        source_node=5589485735,
        target_node=1685220157,
        source_label="Polsek Genteng",
        target_label="RS Darmo",
        source_coords=(-7.2556, 112.7483),
        target_coords=(-7.2874, 112.7382),
    ),
    Scenario(
        name="national_to_rs_ramelan",
        description="Lintas kota barat→selatan: National Hospital → RS Ramelan",
        source_node=1721014942,
        target_node=1719470350,
        source_label="National Hospital",
        target_label="RS Angkatan Laut Dr. Ramelan",
        source_coords=(-7.2993, 112.6764),
        target_coords=(-7.3093, 112.7382),
    ),
    Scenario(
        name="polsek_rungkut_to_rs_onkologi",
        description="Respons darurat area timur: Polsek Rungkut → RS Onkologi",
        source_node=4574365996,
        target_node=7059452149,
        source_label="Polsek Rungkut",
        target_label="RS Onkologi",
        source_coords=(-7.3384, 112.7712),
        target_coords=(-7.2909, 112.7893),
    ),
    Scenario(
        name="ciputra_to_rsu_haji",
        description="Rute terpanjang lintas kota: Ciputra Hospital (barat jauh) → RSU Haji (timur)",
        source_node=4163428113,
        target_node=4332874690,
        source_label="Ciputra Hospital",
        target_label="RSU Haji Surabaya",
        source_coords=(-7.2809, 112.6346),
        target_coords=(-7.2828, 112.7798),
    ),
    Scenario(
        name="benowo_to_onkologi",
        description="EXTREME lintas kota: Polsek Benowo (barat jauh) → RS Onkologi (timur) ~19km",
        source_node=5539027568,
        target_node=7059452149,
        source_label="Polsek Benowo",
        target_label="RS Onkologi",
        source_coords=(-7.2359, 112.6076),
        target_coords=(-7.2909, 112.7893),
    ),
]


# ──────────────────────────────────────────────────────────────
# PSO HELPER FUNCTIONS
# Used exclusively by ParticleSwarmRouting.
# ──────────────────────────────────────────────────────────────

def _pso_path_cost(G, path: list) -> float:
    """Total travel_time along path (lower = better)."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if data is None:
            return float("inf")
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        total += float(best.get("travel_time", 9999))
    return total


def _pso_random_path(G, source: int, target: int, rng: random.Random):
    """
    Generate one path using Dijkstra with random noise on edge weights.
    Different RNG states → different paths → diverse initial swarm.
    """
    def noisy_weight(u, v, data):
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        t = float(best.get("travel_time", 9999))
        return t * rng.uniform(0.5, 1.8)

    try:
        return nx.shortest_path(G, source, target, weight=noisy_weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _pso_repair_path(G, partial_nodes: list):
    """
    Repair a potentially disconnected sequence of nodes into a valid path
    by connecting consecutive nodes with shortest_path segments.
    Returns None if any segment is unreachable.
    """
    if len(partial_nodes) < 2:
        return partial_nodes if partial_nodes else None

    # Deduplicate consecutive identical nodes
    cleaned = [partial_nodes[0]]
    for n in partial_nodes[1:]:
        if n != cleaned[-1]:
            cleaned.append(n)
    if len(cleaned) < 2:
        return cleaned

    full_path = []
    for u, v in zip(cleaned[:-1], cleaned[1:]):
        try:
            seg = nx.shortest_path(G, u, v, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        if not full_path:
            full_path.extend(seg)
        else:
            full_path.extend(seg[1:])    # avoid duplicating junction node

    return full_path if full_path else None


def _pso_mix_paths(G, current: list, personal_best: list, global_best: list,
                   inertia_w: float, cognitive_w: float, social_w: float,
                   rng: random.Random):
    """
    Discrete PSO velocity update via path segment mixing.

    Strategy:
      1. Keep a random middle segment from current path      (inertia)
      2. Splice in a segment from personal_best              (cognitive)
      3. Splice in a segment from global_best                (social)
      4. Repair the assembled waypoints into a valid path

    The probability of each splice is governed by the corresponding weight.
    """
    source, target = current[0], current[-1]

    # Collect waypoints — always start from source
    waypoints = [source]

    # --- inertia: keep some nodes from current path ---
    if len(current) > 2 and rng.random() < inertia_w:
        n_keep = max(1, int(len(current) * rng.uniform(0.2, 0.5)))
        start_idx = rng.randint(1, max(1, len(current) - n_keep - 1))
        waypoints.extend(current[start_idx:start_idx + n_keep])

    # --- cognitive: pull toward personal best ---
    if len(personal_best) > 2 and rng.random() < cognitive_w:
        n_take = max(1, int(len(personal_best) * rng.uniform(0.15, 0.4)))
        start_idx = rng.randint(1, max(1, len(personal_best) - n_take - 1))
        waypoints.extend(personal_best[start_idx:start_idx + n_take])

    # --- social: pull toward global best ---
    if len(global_best) > 2 and rng.random() < social_w:
        n_take = max(1, int(len(global_best) * rng.uniform(0.15, 0.4)))
        start_idx = rng.randint(1, max(1, len(global_best) - n_take - 1))
        waypoints.extend(global_best[start_idx:start_idx + n_take])

    # Always end at target
    waypoints.append(target)

    # Deduplicate consecutive identical nodes
    cleaned = [waypoints[0]]
    for n in waypoints[1:]:
        if n != cleaned[-1]:
            cleaned.append(n)

    repaired = _pso_repair_path(G, cleaned)
    return repaired  # may be None if repair fails


def _pso_mutate(G, path: list, rng: random.Random) -> list:
    """
    Mutate a path by re-routing a random sub-segment via shortest_path.
    Introduces diversity to avoid premature convergence.
    """
    if len(path) < 3:
        return path
    i = rng.randint(0, len(path) - 2)
    j = rng.randint(i + 1, min(i + max(len(path) // 4, 2), len(path) - 1))
    try:
        # Re-route sub-segment with noisy weights for variety
        def noisy_weight(u, v, data):
            best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
            t = float(best.get("travel_time", 9999))
            return t * rng.uniform(0.6, 1.6)
        seg = nx.shortest_path(G, path[i], path[j], weight=noisy_weight)
        return path[:i] + seg + path[j + 1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return path


# ──────────────────────────────────────────────────────────────
# PARTICLE SWARM OPTIMIZATION
# Discrete path-based PSO for network routing.
# Each particle = one candidate path from source → target.
# Fitness = total travel_time.
# ──────────────────────────────────────────────────────────────

class ParticleSwarmRouting(BaseRoutingAlgorithm):
    """
    Particle Swarm Optimization — routing based on swarm candidate paths.

    Approach (discrete / path-based):
    - Each particle represents one candidate path from source to target.
    - Initial swarm generated via shortest path with randomised edge weights.
    - Fitness = total travel_time (lower is better).
    - Personal best = best path ever found by each particle.
    - Global best   = best path found by entire swarm.
    - Update step: mix segments from current path, personal best, and
      global best, then repair disconnected segments.
    - Random mutation prevents stagnation.
    """
    name        = "particle_swarm"
    description = "Particle Swarm Optimization — routing based on swarm candidate paths"

    # ── PSO parameters ────────────────────────────────────────
    N_PARTICLES      = 40
    N_ITERATIONS     = 80
    INERTIA_WEIGHT   = 0.5
    COGNITIVE_WEIGHT = 1.2
    SOCIAL_WEIGHT    = 1.4
    MUTATION_RATE    = 0.25
    RANDOM_SEED      = 42
    # ──────────────────────────────────────────────────────────

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0  = time.perf_counter()
        rng = random.Random(self.RANDOM_SEED)

        # ── 1. initialise swarm ───────────────────────────────
        particles      = []   # current path per particle
        personal_bests = []   # best-ever path per particle
        personal_costs = []   # cost of personal best

        for _ in range(self.N_PARTICLES):
            p = _pso_random_path(G, source_node, target_node, rng)
            if p:
                cost = _pso_path_cost(G, p)
                particles.append(p)
                personal_bests.append(p[:])
                personal_costs.append(cost)

        # Fallback: if swarm is empty, try deterministic shortest path
        if not particles:
            try:
                route = nx.shortest_path(G, source_node, target_node,
                                         weight="travel_time")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(self.name, scenario_name,
                                           source_node, target_node, str(e), ms)
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.build(G, self.name, scenario_name,
                                     source_node, target_node, route, ms,
                                     {"strategy": "fallback_dijkstra"})

        # Global best
        global_best_idx  = min(range(len(particles)), key=lambda i: personal_costs[i])
        global_best_path = personal_bests[global_best_idx][:]
        global_best_cost = personal_costs[global_best_idx]

        # ── 2. iterate ────────────────────────────────────────
        for iteration in range(self.N_ITERATIONS):
            for i in range(len(particles)):
                # --- update particle position via path mixing ---
                new_path = _pso_mix_paths(
                    G, particles[i], personal_bests[i], global_best_path,
                    self.INERTIA_WEIGHT, self.COGNITIVE_WEIGHT, self.SOCIAL_WEIGHT,
                    rng,
                )

                # If mixing failed, keep as-is
                if new_path is None:
                    new_path = particles[i]

                # --- random mutation ---
                if rng.random() < self.MUTATION_RATE:
                    new_path = _pso_mutate(G, new_path, rng)

                # --- evaluate fitness ---
                new_cost = _pso_path_cost(G, new_path)

                # Update particle position
                particles[i] = new_path

                # Update personal best
                if new_cost < personal_costs[i]:
                    personal_bests[i] = new_path[:]
                    personal_costs[i] = new_cost

                    # Update global best
                    if new_cost < global_best_cost:
                        global_best_path = new_path[:]
                        global_best_cost = new_cost

        # ── 3. return best path found ─────────────────────────
        ms = (time.perf_counter() - t0) * 1000
        metadata = {
            "n_particles":     self.N_PARTICLES,
            "n_iterations":    self.N_ITERATIONS,
            "inertia_weight":  self.INERTIA_WEIGHT,
            "cognitive_weight": self.COGNITIVE_WEIGHT,
            "social_weight":   self.SOCIAL_WEIGHT,
            "mutation_rate":   self.MUTATION_RATE,
            "random_seed":     self.RANDOM_SEED,
            "strategy":        "discrete_path_pso",
        }
        return RouteResult.build(G, self.name, scenario_name,
                                 source_node, target_node,
                                 global_best_path, ms, metadata)


# ──────────────────────────────────────────────────────────────
# SKELETON — copy this to add a new algorithm
# ──────────────────────────────────────────────────────────────
#
# class MyAlgorithm(BaseRoutingAlgorithm):
#     name        = "my_algo"           # must be unique
#     description = "One-line summary"
#
#     def find_route(self, G, source_node, target_node, scenario_name=""):
#         t0 = time.perf_counter()
#         try:
#             route    = []     # your list of node IDs
#             metadata = {}     # optional extra info
#         except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
#             ms = (time.perf_counter() - t0) * 1000
#             return RouteResult.failure(self.name, scenario_name,
#                                        source_node, target_node, str(e), ms)
#         ms = (time.perf_counter() - t0) * 1000
#         return RouteResult.build(G, self.name, scenario_name,
#                                  source_node, target_node, route, ms, metadata)


_SCENARIO_POINTS = {
    "rs_darmo": {
        "label": "RS Darmo",
        "node": 1685220157,
        "coords": (-7.2874, 112.7382),
    },
    "rsu_haji": {
        "label": "RSU Haji Surabaya",
        "node": 4332874690,
        "coords": (-7.2828, 112.7798),
    },
    "polsek_genteng": {
        "label": "Polsek Genteng",
        "node": 5589485735,
        "coords": (-7.2556, 112.7483),
    },
    "national": {
        "label": "National Hospital",
        "node": 1721014942,
        "coords": (-7.2993, 112.6764),
    },
    "rs_ramelan": {
        "label": "RS Angkatan Laut Dr. Ramelan",
        "node": 1719470350,
        "coords": (-7.3093, 112.7382),
    },
    "polsek_rungkut": {
        "label": "Polsek Rungkut",
        "node": 4574365996,
        "coords": (-7.3384, 112.7712),
    },
    "rs_onkologi": {
        "label": "RS Onkologi",
        "node": 7059452149,
        "coords": (-7.2909, 112.7893),
    },
    "ciputra": {
        "label": "Ciputra Hospital",
        "node": 4163428113,
        "coords": (-7.2809, 112.6346),
    },
    "polsek_benowo": {
        "label": "Polsek Benowo",
        "node": 5539027568,
        "coords": (-7.2359, 112.6076),
    },
}


def _multi_stop_scenario(
    name: str, description: str, point_keys: list, round_trip: bool = False
) -> Scenario:
    points = [_SCENARIO_POINTS[k] for k in point_keys]
    nodes = [p["node"] for p in points]
    labels = [p["label"] for p in points]
    coords = [p["coords"] for p in points]
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
        round_trip=round_trip,
    )


EXAMPLE_SCENARIOS = [
    # ── 5-stop (baseline complexity) ─────────────────────────────
    _multi_stop_scenario(
        "emergency_west_to_east_5",
        "5-stop: emergency corridor from far-west Surabaya to eastern hospitals",
        ["polsek_benowo", "ciputra", "national", "rs_darmo", "rs_onkologi"],
    ),
    _multi_stop_scenario(
        "hospital_transfer_chain_5",
        "5-stop: hospital transfer chain west → central → south → east Surabaya",
        ["ciputra", "national", "rs_darmo", "rs_ramelan", "rsu_haji"],
    ),
    _multi_stop_scenario(
        "police_healthcare_patrol_5",
        "5-stop: police + healthcare patrol, city center to east, returns to base",
        ["polsek_genteng", "rs_darmo", "rs_ramelan", "polsek_rungkut", "rs_onkologi"],
        round_trip=True,
    ),
    # ── 7-stop (complicated) ──────────────────────────────────────
    _multi_stop_scenario(
        "cross_city_zigzag_7",
        "7-stop: full cross-city zigzag — far-west to far-east via north, center, and south",
        ["polsek_benowo", "national", "polsek_genteng", "rs_darmo",
         "rs_ramelan", "rsu_haji", "rs_onkologi"],
    ),
    # ── 9-stop round trip (very complicated) ─────────────────────
    _multi_stop_scenario(
        "full_city_patrol_9",
        "9-stop: all-facilities full-city patrol with return to base — hardest scenario",
        ["polsek_benowo", "ciputra", "national", "polsek_genteng",
         "rs_darmo", "rs_ramelan", "polsek_rungkut", "rs_onkologi", "rsu_haji"],
        round_trip=True,
    ),
]
