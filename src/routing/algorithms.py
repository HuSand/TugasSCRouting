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
# GA SHARED HELPERS
# Dipakai oleh TeamAGA dan TeamBGA — jangan diubah.
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# TEAM GA SLOTS
# Masing-masing tim tuning parameter di bagian "TUNING ZONE".
# Jangan ubah _ga_* helper di atas — ubah di sini saja.
# ──────────────────────────────────────────────────────────────

class TeamAGA(BaseRoutingAlgorithm):
    """
    ── TEAM A — Genetic Algorithm ───────────────────────────────
    Cara kerja:
      1. Buat POPULATION_SIZE path awal (Dijkstra + noise).
      2. Setiap generasi: seleksi → crossover → mutasi.
      3. Elitisme: individu terbaik selalu lolos ke generasi berikut.
      4. Output: path dengan travel_time terkecil setelah GENERATIONS gen.

    Yang bisa Tim A ubah:
      - Parameter di TUNING ZONE (baris 40-an ke bawah)
      - Override _crossover / _mutate kalau mau strategi berbeda
    ─────────────────────────────────────────────────────────────
    """
    name        = "team_a_ga"
    description = "Team A — Genetic Algorithm"

    # ── TUNING ZONE Team A ────────────────────────────────────
    POPULATION_SIZE = 30     # jumlah individu per generasi
    GENERATIONS     = 50     # berapa kali evolusi
    CROSSOVER_RATE  = 0.8    # probabilitas crossover (0.0–1.0)
    MUTATION_RATE   = 0.3    # probabilitas mutasi   (0.0–1.0)
    TOURNAMENT_SIZE = 3      # peserta tournament selection
    RANDOM_SEED     = 42     # set None → non-deterministik
    # ─────────────────────────────────────────────────────────

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0  = time.perf_counter()
        rng = random.Random(self.RANDOM_SEED)

        # ── 1. Populasi awal ──────────────────────────────────
        population = []
        for _ in range(self.POPULATION_SIZE):
            p = _ga_random_path(G, source_node, target_node, rng)
            if p:
                population.append(p)

        # Fallback kalau populasi kosong (graph terlalu sparse)
        if not population:
            try:
                route = nx.shortest_path(G, source_node, target_node,
                                         weight="travel_time")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(self.name, scenario_name,
                                           source_node, target_node, str(e), ms)
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.build(G, self.name, scenario_name,
                                     source_node, target_node, route, ms)

        # ── 2. Evolusi ────────────────────────────────────────
        for _ in range(self.GENERATIONS):
            fitness = [_ga_path_cost(G, p) for p in population]
            best_idx = min(range(len(population)), key=lambda i: fitness[i])
            elite = population[best_idx]

            new_pop = [elite]  # elitisme: simpan yang terbaik
            while len(new_pop) < self.POPULATION_SIZE:
                parent1 = _ga_tournament(population, fitness,
                                         self.TOURNAMENT_SIZE, rng)
                if rng.random() < self.CROSSOVER_RATE:
                    parent2 = _ga_tournament(population, fitness,
                                             self.TOURNAMENT_SIZE, rng)
                    child = _ga_crossover(parent1, parent2, rng)
                else:
                    child = parent1[:]

                if rng.random() < self.MUTATION_RATE:
                    child = _ga_mutate(G, child, rng)

                new_pop.append(child)
            population = new_pop

        # ── 3. Ambil individu terbaik ─────────────────────────
        fitness = [_ga_path_cost(G, p) for p in population]
        best    = population[min(range(len(population)), key=lambda i: fitness[i])]

        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name, source_node, target_node, best, ms,
            metadata={
                "generations":   self.GENERATIONS,
                "population":    self.POPULATION_SIZE,
                "crossover_rate": self.CROSSOVER_RATE,
                "mutation_rate": self.MUTATION_RATE,
            },
        )


class TeamBGA(BaseRoutingAlgorithm):
    """
    ── TEAM B — Genetic Algorithm ───────────────────────────────
    Sama dengan Team A, tapi parameter default berbeda.
    Tim B bebas mengubah TUNING ZONE dan meng-override metode GA
    untuk mencoba strategi yang lebih baik.

    Contoh hal yang bisa dicoba Tim B:
      - Naikkan GENERATIONS atau POPULATION_SIZE
      - Turunkan MUTATION_RATE supaya konvergen lebih cepat
      - Override _crossover → coba crossover berbeda
    ─────────────────────────────────────────────────────────────
    """
    name        = "team_b_ga"
    description = "Team B — Genetic Algorithm"

    # ── TUNING ZONE Team B ────────────────────────────────────
    POPULATION_SIZE = 20     # lebih kecil → tiap generasi lebih cepat
    GENERATIONS     = 80     # lebih banyak generasi untuk kompensasi
    CROSSOVER_RATE  = 0.7
    MUTATION_RATE   = 0.4    # mutasi lebih agresif → eksplorasi lebih luas
    TOURNAMENT_SIZE = 5      # tekanan seleksi lebih tinggi
    RANDOM_SEED     = 7
    # ─────────────────────────────────────────────────────────

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0  = time.perf_counter()
        rng = random.Random(self.RANDOM_SEED)

        # ── 1. Populasi awal ──────────────────────────────────
        population = []
        for _ in range(self.POPULATION_SIZE):
            p = _ga_random_path(G, source_node, target_node, rng)
            if p:
                population.append(p)

        if not population:
            try:
                route = nx.shortest_path(G, source_node, target_node,
                                         weight="travel_time")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(self.name, scenario_name,
                                           source_node, target_node, str(e), ms)
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.build(G, self.name, scenario_name,
                                     source_node, target_node, route, ms)

        # ── 2. Evolusi ────────────────────────────────────────
        for _ in range(self.GENERATIONS):
            fitness = [_ga_path_cost(G, p) for p in population]
            best_idx = min(range(len(population)), key=lambda i: fitness[i])
            elite = population[best_idx]

            new_pop = [elite]
            while len(new_pop) < self.POPULATION_SIZE:
                parent1 = _ga_tournament(population, fitness,
                                         self.TOURNAMENT_SIZE, rng)
                if rng.random() < self.CROSSOVER_RATE:
                    parent2 = _ga_tournament(population, fitness,
                                             self.TOURNAMENT_SIZE, rng)
                    child = _ga_crossover(parent1, parent2, rng)
                else:
                    child = parent1[:]

                if rng.random() < self.MUTATION_RATE:
                    child = _ga_mutate(G, child, rng)

                new_pop.append(child)
            population = new_pop

        # ── 3. Ambil individu terbaik ─────────────────────────
        fitness = [_ga_path_cost(G, p) for p in population]
        best    = population[min(range(len(population)), key=lambda i: fitness[i])]

        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name, source_node, target_node, best, ms,
            metadata={
                "generations":   self.GENERATIONS,
                "population":    self.POPULATION_SIZE,
                "crossover_rate": self.CROSSOVER_RATE,
                "mutation_rate": self.MUTATION_RATE,
            },
        )


# ──────────────────────────────────────────────────────────────
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

EXAMPLE_SCENARIOS = [
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
]


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
