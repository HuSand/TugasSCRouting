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
    description = "Burhan — GA (belum dituning)"

    # ── TUNING ZONE Burhan -- UBAH ANGKA INI ─────────────────
    POPULATION_SIZE = 30    # TODO: coba variasikan
    GENERATIONS     = 50    # TODO: coba variasikan
    CROSSOVER_RATE  = 0.8   # TODO: coba variasikan
    MUTATION_RATE   = 0.3   # TODO: coba variasikan
    TOURNAMENT_SIZE = 3     # TODO: coba variasikan
    RANDOM_SEED     = 10
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
    POPULATION_SIZE = 30    # TODO: coba variasikan
    GENERATIONS     = 50    # TODO: coba variasikan
    CROSSOVER_RATE  = 0.8   # TODO: coba variasikan
    MUTATION_RATE   = 0.3   # TODO: coba variasikan
    TOURNAMENT_SIZE = 3     # TODO: coba variasikan
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
