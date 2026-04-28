"""
════════════════════════════════════════════════════════════════
 ROUTING ALGORITHMS
 All algorithm implementations live here.
 Register each one at the bottom of this file via REGISTRY.register().
════════════════════════════════════════════════════════════════

HOW TO ADD A NEW ALGORITHM
--------------------------
1. Subclass BaseRoutingAlgorithm.
2. Set a unique `name` and `description`.
3. Implement find_route() — or add _route_multi_stop() for TSP-aware routing.
4. Register: REGISTRY.register(YourAlgorithm())
5. Run: python main.py compare

GRAPH ACCESS REFERENCE
-----------------------
G            : NetworkX MultiDiGraph of the Surabaya road network.
               Nodes are road intersections (int IDs).
               Edges have:
                 length        (float)  metres
                 travel_time   (float)  seconds
                 speed_kph     (float)  km/h
                 highway       (str)    OSM road type
               Access: G.get_edge_data(u, v) → dict of parallel edges

source_node  : int — starting node ID
target_node  : int — destination node ID
G.nodes[n]   : dict with 'x' (lon) and 'y' (lat) for node n

Useful NetworkX functions:
  nx.shortest_path(G, src, dst, weight="travel_time")
  nx.shortest_path_length(G, src, dst, weight="travel_time")
  nx.single_source_dijkstra_path_length(G, src, weight="travel_time")
  list(G.successors(n))     — outgoing neighbours
  list(G.predecessors(n))   — incoming neighbours

Return RouteResult.build(...) on success.
Return RouteResult.failure(...) on no-path or exception.
════════════════════════════════════════════════════════════════
"""

import logging
import time
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx

from src.routing.base import BaseRoutingAlgorithm, RouteResult, Scenario

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# SECTION 1: GA SHARED HELPERS
#
# These utility functions are used by GeneticAlgorithm for
# point-to-point (single-leg) path optimisation.
# They operate on lists of road-network node IDs.
# ══════════════════════════════════════════════════════════════════

def _route_streets(G, path: list) -> list:
    """
    Extract a deduplicated list of street names traversed by a path.
    Consecutive repeated names are collapsed to one entry.
    Used for evolution logs and map tooltips.
    """
    streets, prev = [], None
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        # Pick the fastest parallel edge between u and v
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
    """
    Total travel_time (seconds) along a path.
    When an edge is missing (disconnected path), returns infinity
    so the path is ranked last in selection.
    """
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if data is None:
            return float("inf")
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        total += float(best.get("travel_time", 9999))
    return total


def _ga_path_distance(G, path: list) -> float:
    """Total physical distance (metres) along a path."""
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
    Generate one candidate path using Dijkstra with randomised edge weights.

    By adding ±30-50% noise to travel_time weights, each call produces a
    different valid path. Running this N times gives a diverse initial
    population covering different road corridors.
    """
    def noisy_weight(u, v, data):
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        t = float(best.get("travel_time", 9999))
        return t * rng.uniform(0.7, 1.5)   # ±30-50% noise

    try:
        return nx.shortest_path(G, source, target, weight=noisy_weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _ga_crossover(p1: list, p2: list, rng: random.Random) -> list:
    """
    Common-node crossover for road-network paths.

    Find a road intersection that appears in both parent paths (a "pivot"),
    then combine the first half of p1 up to the pivot with the second half
    of p2 from the pivot onwards. If no common node exists, return a copy of p1.

    This is analogous to single-point crossover in classical GA, but adapted
    to the constraint that paths must stay connected on the road graph.
    """
    set1 = set(p1[1:-1])                           # intermediate nodes of p1
    common = [n for n in p2[1:-1] if n in set1]    # shared intersections
    if not common:
        return p1[:]
    pivot = rng.choice(common)
    i1 = p1.index(pivot)
    i2 = p2.index(pivot)
    return p1[:i1] + p2[i2:]


def _ga_mutate(G, path: list, rng: random.Random) -> list:
    """
    Segment re-route mutation.

    Pick two random nodes within the path, then replace the road segment
    between them with a fresh shortest-path computation. This injects
    new road corridors into the population and prevents stagnation.
    """
    if len(path) < 3:
        return path
    i = rng.randint(0, len(path) - 2)
    j = rng.randint(i + 1, min(i + max(len(path) // 3, 2), len(path) - 1))
    try:
        seg = nx.shortest_path(G, path[i], path[j], weight="travel_time")
        return path[:i] + seg + path[j + 1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return path     # mutation failed — keep original


def _ga_tournament(population: list, fitness: list,
                   k: int, rng: random.Random) -> list:
    """
    Tournament selection: draw k candidates at random, return the fittest one.

    A higher k means stronger selection pressure (better individuals win more
    often). Too high a k can cause premature convergence; too low reduces
    directed improvement.
    """
    candidates = rng.sample(range(len(population)), min(k, len(population)))
    winner_idx = min(candidates, key=lambda i: fitness[i])
    return population[winner_idx]


def _ga_run(algo, G, source_node, target_node, scenario_name):
    """
    Core GA loop for point-to-point (single-leg) path optimisation.

    Shared by GeneticAlgorithm.find_route() for simple A→B routing.
    The algo object supplies hyperparameters and _fitness / _crossover / _mutate.

    Flow:
      1. Build a diverse initial population via noisy-Dijkstra.
      2. For each generation: evaluate fitness, select, cross, mutate.
      3. Carry the generation's elite forward (elitism = 1).
      4. Return the best path found, with per-generation history.
    """
    t0  = time.perf_counter()
    rng = random.Random(algo.RANDOM_SEED)

    # ── 1. Initial population ─────────────────────────────────────────────
    population = []
    for _ in range(algo.POPULATION_SIZE):
        p = _ga_random_path(G, source_node, target_node, rng)
        if p:
            population.append(p)

    # If the graph is too sparse to generate any valid path, fall back to Dijkstra
    if not population:
        try:
            route = nx.shortest_path(G, source_node, target_node, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(algo.name, scenario_name,
                                       source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, algo.name, scenario_name,
                                 source_node, target_node, route, ms)

    # ── 2. Generational evolution ─────────────────────────────────────────
    gen_history = []
    for gen_idx in range(algo.GENERATIONS):
        # Evaluate fitness for every individual in the population
        fitness  = [algo._fitness(G, p) for p in population]
        best_idx = min(range(len(population)), key=lambda i: fitness[i])
        elite    = population[best_idx]   # best individual is carried to next gen

        # Record coordinates and street names of the best path this generation
        coords = []
        for n in elite:
            node = G.nodes.get(n)
            if node:
                coords.append([round(float(node["y"]), 5),
                                round(float(node["x"]), 5)])
        gen_history.append({
            "gen":     gen_idx + 1,
            "min":     round(_ga_path_cost(G, elite) / 60, 3),   # minutes
            "dist":    round(_ga_path_distance(G, elite) / 1000, 3),  # km
            "coords":  coords,
            "streets": _route_streets(G, elite),
        })

        # Build next generation
        new_pop = [elite]   # elitism: best always survives unchanged
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

    # ── 3. Return the best individual from the final generation ──────────
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


# ══════════════════════════════════════════════════════════════════
# SECTION 2: SA SHARED HELPERS
#
# Helper functions used by GeraldSimulatedAnnealing.
# ══════════════════════════════════════════════════════════════════

def _sa_noisy_shortest_path(G, source: int, target: int,
                            rng: random.Random,
                            noise_min: float = 0.85,
                            noise_max: float = 1.35):
    """
    Generate a candidate SA path using Dijkstra with noise on edge lengths
    (not travel_time). The physical-distance objective matches SA's fitness.
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
    Generate a neighbouring solution for SA by replacing a random sub-segment
    of the current path with a new noisy shortest-path segment.

    The noise ensures the replacement segment is not always identical to
    the original, providing genuine exploration of nearby solutions.
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


# ══════════════════════════════════════════════════════════════════
# SECTION 3: GENETIC ALGORITHM
#
# A Genetic Algorithm that works at two levels:
#   - Point-to-point (find_route): optimises the road-level path
#     between two nodes using a population of candidate paths.
#   - Multi-stop TSP (_route_multi_stop): optimises the *visit order*
#     of a set of stops, then stitches legs together. The tour always
#     returns to the starting point (circular route).
# ══════════════════════════════════════════════════════════════════

class GeneticAlgorithm(BaseRoutingAlgorithm):
    """
    Genetic Algorithm for routing — balanced time + distance fitness.

    Point-to-point mode (find_route):
      - Each individual is a road-network path from source to target.
      - Fitness = 50% travel_time + 50% physical_distance (normalised to seconds).
      - This trades pure speed for a shorter physical route —
        useful when distance matters as well as time.
      - Crossover: mid-pivot common-node crossover.
      - Mutation: random sub-segment re-route.

    Multi-stop TSP mode (_route_multi_stop):
      - Each individual is a *permutation* of the intermediate stops.
      - The start node is fixed; the tour ends back at start (circular).
      - Fitness = total shortest-path travel_time for the full circular tour.
      - Crossover: Order Crossover (OX) — preserves valid permutations.
      - Mutation: swap two random stops in the visit order.
      - Called automatically by the benchmark when len(stops) > 2.
    """

    name        = "ga"
    description = "Genetic Algorithm — balanced time+distance (50/50), TSP-aware multi-stop"

    # ── Point-to-point hyperparameters (used by find_route) ──────────────
    POPULATION_SIZE = 30
    GENERATIONS     = 20
    CROSSOVER_RATE  = 0.85
    MUTATION_RATE   = 0.6
    TOURNAMENT_SIZE = 5
    RANDOM_SEED     = 42

    # ── TSP multi-stop hyperparameters (used by _route_multi_stop) ───────
    # Separate from point-to-point because TSP search space is smaller
    # (permutations of ~50 stops vs. road paths with thousands of nodes),
    # so fewer individuals and generations are needed to converge.
    TSP_POPULATION_SIZE = 40    # individuals per generation
    TSP_GENERATIONS     = 80    # max generations before forced stop
    TSP_PATIENCE        = 15    # stop early if no improvement for this many gens
    TSP_WORKERS         = 4     # parallel threads for pairwise precomputation

    # ─────────────────────────────────────────────────────────────────────

    def _fitness(self, G, path: list) -> float:
        """
        Balanced fitness for point-to-point path optimisation.

        Combines travel_time (speed) and physical distance (path length)
        equally weighted at 50/50. Distance is normalised to seconds by
        dividing by 40 km/h (a conservative urban driving speed), making
        both objectives comparable on the same scale.

        Effect: a slightly slower route that avoids a long detour can
        outcompete a fast-but-winding route. Different from Dijkstra,
        which minimises only travel_time.
        """
        REF_SPEED_MS = 40 / 3.6   # 40 km/h expressed in m/s for normalisation
        time_s = 0.0
        dist_m = 0.0
        for u, v in zip(path[:-1], path[1:]):
            data = G.get_edge_data(u, v)
            if data is None:
                return float("inf")
            best    = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
            time_s += float(best.get("travel_time", 9999))
            dist_m += float(best.get("length", 0))
        # Convert distance to a "time equivalent" so both terms are in seconds
        dist_as_time = dist_m / REF_SPEED_MS
        return 0.5 * time_s + 0.5 * dist_as_time

    def _crossover(self, p1: list, p2: list, rng: random.Random) -> list:
        """
        Mid-pivot common-node crossover.

        Instead of picking a random shared intersection as the pivot,
        this variant selects the shared node that is geometrically
        closest to the midpoint of p1. This produces children that
        evenly combine the first half of one parent with the second
        half of the other, avoiding lopsided offspring.
        """
        set1   = set(p1[1:-1])
        common = [n for n in p2[1:-1] if n in set1]
        if not common:
            return p1[:]
        mid      = len(p1) // 2
        # Find the pivot closest to the middle index of p1
        min_dist = min(abs(p1.index(n) - mid) for n in common)
        best     = [n for n in common if abs(p1.index(n) - mid) == min_dist]
        pivot    = rng.choice(best)
        i1       = p1.index(pivot)
        i2       = p2.index(pivot)
        return p1[:i1] + p2[i2:]

    def _mutate(self, G, path: list, rng: random.Random) -> list:
        """Standard segment re-route mutation (see _ga_mutate for details)."""
        return _ga_mutate(G, path, rng)

    # ── Multi-stop TSP entry point ────────────────────────────────────────

    def _route_multi_stop(self, G, nodes: list, scenario_name: str = "",
                          source_node: int = None,
                          target_node: int = None) -> RouteResult:
        """
        TSP-GA for multi-stop circular routing.

        The algorithm decides both WHICH ORDER to visit the stops and
        WHICH ROAD to take between each pair — the caller only provides
        the set of stops, not their order.

        Algorithm outline:
          1. Precompute a pairwise travel_time cost matrix between all stops
             using N full Dijkstra sweeps (one per stop as source).
             This is O(N × graph_Dijkstra) — done once, reused every generation.
          2. Evolve a population of visit-order permutations via GA.
             - Chromosome: permutation of the N-1 intermediate stops
               (start node is fixed at nodes[0]).
             - Fitness: total travel_time of the circular tour
               start → stop_1 → stop_2 → ... → stop_{N-1} → start.
             - Crossover: Order Crossover (OX) — always produces a valid
               permutation with no duplicates.
             - Mutation: swap two random stop positions.
          3. Expand the best permutation into a full road-network path
             by running shortest_path for each leg of the final tour.

        The tour is always circular — the algorithm guarantees the route
        returns to the starting point.
        """
        t0 = time.perf_counter()

        # Remove duplicate stops while preserving the original order
        nodes = list(dict.fromkeys(nodes))

        if len(nodes) < 2:
            ms = (time.perf_counter() - t0) * 1000
            n0 = nodes[0] if nodes else -1
            return RouteResult.failure(self.name, scenario_name, n0, n0,
                                       "Need at least 2 stops for a tour", ms)

        # The first node in the list is the fixed starting (and ending) point
        start         = nodes[0]
        intermediates = nodes[1:]   # these are the stops whose order is evolved

        # ── Step 1: Precompute pairwise costs (parallel) ──────────────────
        # Run one Dijkstra sweep per stop as source, then extract costs to
        # all other stops from the result. N sweeps instead of N² calls.
        # ThreadPoolExecutor runs sweeps concurrently — each Dijkstra is
        # independent, so threads don't block each other.
        n_nodes = len(nodes)
        log.info(f"  GA TSP [{scenario_name}]: precomputing {n_nodes}×{n_nodes} "
                 f"cost matrix ({n_nodes} Dijkstra sweeps, "
                 f"{self.TSP_WORKERS} threads)...")

        def _dijkstra_row(src):
            try:
                return src, dict(nx.single_source_dijkstra_path_length(
                    G, src, weight="travel_time"
                ))
            except (nx.NodeNotFound, nx.NetworkXError):
                return src, {}

        pair_cost: dict = {}
        with ThreadPoolExecutor(max_workers=self.TSP_WORKERS) as pool:
            for src, lengths in pool.map(_dijkstra_row, nodes):
                for dst in nodes:
                    if dst != src:
                        pair_cost[(src, dst)] = lengths.get(dst, float("inf"))

        log.info(f"  GA TSP [{scenario_name}]: cost matrix ready — "
                 f"starting evolution (pop={self.TSP_POPULATION_SIZE}, "
                 f"max_gen={self.TSP_GENERATIONS}, patience={self.TSP_PATIENCE})")

        # ── Tour cost helper (used inside the GA loop) ────────────────────
        def tour_cost(perm: list) -> float:
            """
            Sum of pairwise costs for the circular tour:
              start → perm[0] → perm[1] → ... → perm[-1] → start
            All costs come from the precomputed dict — O(N) lookup.
            """
            full_tour = [start] + perm + [start]
            return sum(
                pair_cost.get((a, b), float("inf"))
                for a, b in zip(full_tour[:-1], full_tour[1:])
            )

        # ── Order Crossover (OX) for permutations ────────────────────────
        def ox_crossover(p1: list, p2: list) -> list:
            """
            Standard Order Crossover (OX) — classic TSP operator.

            Copies a random contiguous segment from p1 into the child,
            then fills the remaining positions with elements from p2
            in their original relative order. Guarantees no duplicates.
            """
            size = len(p1)
            if size < 2:
                return p1[:]
            # Choose two cut points
            a, b = sorted(rng.sample(range(size), 2))
            child = [None] * size
            child[a:b + 1] = p1[a:b + 1]   # copy segment from parent 1
            # Fill from parent 2, skipping nodes already in the child
            fill = [x for x in p2 if x not in child]
            j = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = fill[j]
                    j += 1
            return child

        # ── Swap mutation for permutations ────────────────────────────────
        def swap_mutate(perm: list) -> list:
            """
            Randomly swap two stops in the tour order.
            Simple and effective for TSP: a single swap can yield a very
            different total tour cost depending on the graph topology.
            """
            if len(perm) < 2:
                return perm[:]
            p = perm[:]
            i, j = rng.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
            return p

        # ── Step 2: Evolve visit-order permutations ───────────────────────
        rng = random.Random(self.RANDOM_SEED)

        # Initialise population with random shuffles of the intermediate stops
        population = []
        for _ in range(self.TSP_POPULATION_SIZE):
            perm = intermediates[:]
            rng.shuffle(perm)
            population.append(perm)

        best_perm     = intermediates[:]
        best_cost     = tour_cost(best_perm)
        gen_history   = []
        no_improve    = 0   # consecutive generations without improvement

        for gen_idx in range(self.TSP_GENERATIONS):
            fitness  = [tour_cost(p) for p in population]
            best_idx = min(range(len(population)), key=lambda i: fitness[i])
            elite    = population[best_idx]

            if fitness[best_idx] < best_cost - 1e-6:
                best_perm  = elite[:]
                best_cost  = fitness[best_idx]
                no_improve = 0
            else:
                no_improve += 1

            gen_history.append({
                "gen":  gen_idx + 1,
                "min":  round(best_cost / 60, 3),
                "dist": 0.0,    # filled after final expansion
            })

            # Log progress every 10 generations so user can see it's running
            if (gen_idx + 1) % 10 == 0:
                log.info(f"  GA TSP [{scenario_name}]: gen {gen_idx+1}/{self.TSP_GENERATIONS} "
                         f"— best {best_cost/60:.2f} min "
                         f"(no-improve streak: {no_improve}/{self.TSP_PATIENCE})")

            # Early stopping — no point continuing if the population has converged
            if no_improve >= self.TSP_PATIENCE:
                log.info(f"  GA TSP [{scenario_name}]: early stop at gen {gen_idx+1} "
                         f"(no improvement for {self.TSP_PATIENCE} gens)")
                break

            # Build next generation with elitism
            new_pop = [elite[:]]
            while len(new_pop) < self.TSP_POPULATION_SIZE:
                p1 = _ga_tournament(population, fitness, self.TOURNAMENT_SIZE, rng)
                if rng.random() < self.CROSSOVER_RATE:
                    p2    = _ga_tournament(population, fitness, self.TOURNAMENT_SIZE, rng)
                    child = ox_crossover(p1, p2)
                else:
                    child = p1[:]
                if rng.random() < self.MUTATION_RATE:
                    child = swap_mutate(child)
                new_pop.append(child)
            population = new_pop

        # ── Step 3: Expand best permutation into a full road path ─────────
        # The tour is circular: [start, stop_1, stop_2, ..., stop_{N-1}, start]
        # Each leg is expanded to a real road path using shortest_path.
        full_tour_stops = [start] + best_perm + [start]   # first == last → circular
        full_route: list = []

        for src, dst in zip(full_tour_stops[:-1], full_tour_stops[1:]):
            try:
                leg = nx.shortest_path(G, src, dst, weight="travel_time")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(self.name, scenario_name,
                                           start, start, str(e), ms)
            if not full_route:
                full_route.extend(leg)
            else:
                # Skip the first node of each leg to avoid duplicating junctions
                full_route.extend(leg[1:])

        # Backfill the final gen_history entry with actual road-level data
        # now that we have the real expanded path.
        final_dist_km = round(_ga_path_distance(G, full_route) / 1000, 3)
        final_coords  = []
        for n in full_route:
            nd = G.nodes.get(n)
            if nd:
                final_coords.append([round(float(nd["y"]), 5),
                                      round(float(nd["x"]), 5)])
        if gen_history:
            gen_history[-1]["dist"]    = final_dist_km
            gen_history[-1]["coords"]  = final_coords
            gen_history[-1]["streets"] = _route_streets(G, full_route)

        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name,
            start, start,   # source == target because the tour is circular
            full_route, ms,
            metadata={
                "algorithm_variant": "tsp_ga",
                "generations":       self.GENERATIONS,
                "population":        self.POPULATION_SIZE,
                "crossover_rate":    self.CROSSOVER_RATE,
                "mutation_rate":     self.MUTATION_RATE,
                "stop_count":        len(nodes),
                "visit_order":       full_tour_stops,
                "gen_history":       gen_history,
            },
        )
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

        # NORMALIZATION
        norm_time = total_time / 1000
        norm_dist = total_dist / 5000
        norm_complexity = edges_count / 50
        norm_speed = avg_speed / 50

        # WEIGHTED MULTI-OBJECTIVE
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
        """
        Point-to-point routing via path-level GA.
        Delegates to the shared _ga_run loop using this class's hyperparameters
        and fitness / crossover / mutate implementations.
        """
        return _ga_run(self, G, source_node, target_node, scenario_name)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: ANT COLONY OPTIMIZATION (ACO) — renumbered after Christofides removal

# ══════════════════════════════════════════════════════════════════
# SECTION 5: ANT COLONY OPTIMIZATION (ACO)
#
# Mimics how real ant colonies find shortest paths via pheromone trails.
# Ants probabilistically choose edges based on pheromone strength (τ)
# and visibility (η = 1/travel_time). Over iterations, shorter paths
# accumulate more pheromone and are chosen more often — positive feedback.
#
# Parameters:
#   ALPHA  — pheromone weight (higher → more exploitative)
#   BETA   — visibility weight (higher → more greedy/myopic)
#   RHO    — evaporation rate (higher → shorter memory)
#   Q      — pheromone deposit constant
# ══════════════════════════════════════════════════════════════════

class AntColonyRouting(BaseRoutingAlgorithm):
    """
    Ant Colony Optimization (ACO) for point-to-point routing.

    Each ant builds a path by choosing the next node probabilistically,
    weighted by pheromone (learned from past good solutions) and
    visibility (inverse of travel_time — prefers fast edges).

    After all ants complete their paths in an iteration, pheromone evaporates
    globally and the best-performing ant deposits fresh pheromone along its
    route. This reinforces good paths over time.
    """
    name        = "aco_routing"
    description = "Ant Colony Optimization — pheromone × visibility (travel_time)"

    # ── ACO parameters ────────────────────────────────────────────────────
    N_ANTS        = 20     # number of ants per iteration
    N_ITERATIONS  = 30     # total iterations of the colony
    ALPHA         = 1.0    # pheromone influence — raise to exploit known good paths
    BETA          = 2.0    # visibility influence — raise to be more greedy
    RHO           = 0.1    # evaporation rate — low → long pheromone memory
    Q             = 100.0  # deposit constant: shorter path → more deposit (Q/cost)
    TAU_INIT      = 1.0    # initial pheromone on all edges (uniform start)
    RANDOM_SEED   = 42

    def _get_candidates(self, G, node: int, visited: set) -> list:
        """
        Return unvisited neighbours of `node` with their edge travel_time.
        Format: [(neighbour_node, travel_time), ...]
        """
        candidates = []
        for neighbour in G.successors(node):
            if neighbour in visited:
                continue
            edge_dict = G.get_edge_data(node, neighbour)
            if not edge_dict:
                continue
            # Use the fastest parallel edge between node and neighbour
            best_tt = min(float(d.get("travel_time", 9999))
                          for d in edge_dict.values())
            if best_tt < 9999:
                candidates.append((neighbour, best_tt))
        return candidates

    def _build_ant_path(self, G, source: int, target: int,
                        pheromone: dict, rng: random.Random,
                        max_steps: int) -> list | None:
        """
        Build one ant's path from source to target.

        At each step the ant picks the next node via roulette-wheel selection
        weighted by (τ^α × η^β) for each candidate. If the target is directly
        reachable, the ant moves there immediately (greedy shortcut). If stuck
        with no unvisited candidates, Dijkstra bridges the gap to the target.
        """
        path    = [source]
        visited = {source}
        current = source

        for _ in range(max_steps):
            if current == target:
                return path

            candidates = self._get_candidates(G, current, visited)

            # Stuck: bridge to target via Dijkstra and terminate
            if not candidates:
                try:
                    bridge = nx.shortest_path(G, current, target, weight="travel_time")
                    path += bridge[1:]
                    return path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None

            # Shortcut: if target is directly reachable, take it
            if any(n == target for n, _ in candidates):
                path.append(target)
                return path

            # Probabilistic transition: score each candidate
            scores = []
            for neighbour, tt in candidates:
                tau   = pheromone.get((current, neighbour), self.TAU_INIT)
                eta   = 1.0 / tt if tt > 0 else 1.0
                scores.append((tau ** self.ALPHA) * (eta ** self.BETA))

            total = sum(scores)
            if total == 0:
                # All scores zero — choose uniformly at random
                chosen = rng.choice(candidates)[0]
            else:
                # Roulette wheel selection
                probs  = [s / total for s in scores]
                r      = rng.random()
                cumul  = 0.0
                chosen = candidates[-1][0]   # default: last candidate
                for (neighbour, _), prob in zip(candidates, probs):
                    cumul += prob
                    if r <= cumul:
                        chosen = neighbour
                        break

            path.append(chosen)
            visited.add(chosen)
            current = chosen

        return None   # exceeded max_steps

    def _path_cost(self, G, path: list) -> float:
        """Total travel_time (seconds) along a path."""
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge_dict = G.get_edge_data(u, v)
            if not edge_dict:
                return float("inf")
            total += min(float(d.get("travel_time", 9999)) for d in edge_dict.values())
        return total

    def _deposit_pheromone(self, pheromone: dict,
                           path: list, cost: float) -> None:
        """
        Deposit pheromone along the best path of an iteration.
        The deposit amount is inversely proportional to cost:
          better (shorter) paths receive more pheromone → positive feedback.
        """
        deposit = self.Q / cost if cost > 0 else 0.0
        for u, v in zip(path[:-1], path[1:]):
            key = (u, v)
            pheromone[key] = pheromone.get(key, self.TAU_INIT) + deposit

    def _run_aco(self, G, source: int, target: int) -> list:
        """
        Run N_ITERATIONS of the ant colony.

        Each iteration:
          1. Send N_ANTS from source; collect valid paths to target.
          2. Evaporate all pheromone globally by factor (1 - RHO).
          3. Best ant of this iteration deposits fresh pheromone.
          4. Update global best if improved.

        Falls back to Dijkstra if no ant ever reaches the target.
        """
        rng       = random.Random(self.RANDOM_SEED)
        pheromone = {}   # sparse dict: (u, v) → tau value
        best_path = None
        best_cost = float("inf")

        # Cap max_steps to 3× graph size to prevent infinite loops
        max_steps = min(G.number_of_nodes(), 5000)

        for _ in range(self.N_ITERATIONS):
            iter_best_path = None
            iter_best_cost = float("inf")

            # All ants build their paths for this iteration
            for _ in range(self.N_ANTS):
                path = self._build_ant_path(G, source, target, pheromone, rng, max_steps)
                if path is None or path[-1] != target:
                    continue   # ant failed to reach target
                cost = self._path_cost(G, path)
                if cost < iter_best_cost:
                    iter_best_path = path
                    iter_best_cost = cost

            # Global evaporation
            for key in list(pheromone.keys()):
                pheromone[key] *= (1.0 - self.RHO)
                if pheromone[key] < 1e-6:
                    del pheromone[key]   # prune near-zero entries to save memory

            # Deposit pheromone for the best path this iteration
            if iter_best_path is not None:
                self._deposit_pheromone(pheromone, iter_best_path, iter_best_cost)
                if iter_best_cost < best_cost:
                    best_path = iter_best_path
                    best_cost = iter_best_cost

        # Fallback: if the colony never found a path, use Dijkstra
        if best_path is None:
            try:
                best_path = nx.shortest_path(G, source, target, weight="travel_time")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                raise nx.NetworkXNoPath(f"ACO: no path from {source} to {target}")

        return best_path

    def find_route(self, G, source_node, target_node, scenario_name=""):
        """Run ACO and wrap the result in a RouteResult."""
        t0 = time.perf_counter()
        try:
            route = self._run_aco(G, source_node, target_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name,
                                       source_node, target_node, str(e), ms)
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name, source_node, target_node, route, ms,
            metadata={
                "n_ants":       self.N_ANTS,
                "n_iterations": self.N_ITERATIONS,
                "alpha":        self.ALPHA,
                "beta":         self.BETA,
                "rho":          self.RHO,
            },
        )


# ══════════════════════════════════════════════════════════════════
# SECTION 6: SIMULATED ANNEALING (SA)
                "strategy":     "pheromone × (1/travel_time) visibility, no highway weighting",
            }
        )
 
"""
════════════════════════════════════════════════════════════════
 ALGORITMA: ACO-Elite (Ant Colony Optimization — Optimized)
 Nama      : [NAMA KAMU]
 
 CARA PASANG KE algorithms.py:
   1. Copy seluruh class AntColonyElite di bawah ini
   2. Paste ke algorithms.py, di bawah class TeamBGA
   3. Di bagian paling bawah algorithms.py tambahkan:
        REGISTRY.register(AntColonyElite())
   4. Jalankan: python main.py compare
════════════════════════════════════════════════════════════════
"""
 
import time
import math
import random
import heapq
import networkx as nx
 
from src.routing.base import BaseRoutingAlgorithm, RouteResult
 
 
class AntColonyElite(BaseRoutingAlgorithm):
    """
    ── ACO-ELITE: ANT COLONY OPTIMIZATION (VERSI TEROPTIMASI) ───
 
    MASALAH VERSI LAMA & SOLUSINYA
    ────────────────────────────────
    Versi ACO sebelumnya lambat karena 4 masalah utama:
 
    [1] FULL GRAPH WALK → SUBGRAPH PRUNING
        Semut menjelajahi jutaan node. Sekarang kita potong graph
        hanya ke subgraph yang relevan (node dalam radius BFS
        dari jalur Dijkstra awal). Hasilnya: graph 100-300 node
        bukan jutaan.
 
    [2] NO SPATIAL BIAS → HAVERSINE-GUIDED VISIBILITY
        Visibilitas η sebelumnya hanya 1/travel_time, tidak
        mempertimbangkan apakah arah gerak mendekati target.
        Sekarang: η = (1/travel_time) × direction_factor
        Direction factor = jarak_current_ke_target /
                           jarak_neighbor_ke_target
        Jika neighbor lebih dekat ke target → faktor > 1 (bonus).
 
    [3] BLIND ROULETTE O(N) → CANDIDATE LIST TOP-K
        Sebelumnya setiap step mengevaluasi semua tetangga.
        Sekarang: pre-compute TOP-K tetangga terbaik per node
        (berdasarkan travel_time). Semut hanya pilih dari K
        kandidat tersebut → O(K) bukan O(N).
 
    [4] DENSE PHEROMONE DICT → SPARSE + PRUNING
        Dict feromon tumbuh tak terbatas. Sekarang: feromon
        di bawah TAU_MIN dihapus (sparse), dan hanya edge
        di subgraph yang punya feromon (bukan seluruh OSM).
 
    BONUS IMPROVEMENTS (meningkatkan kualitas hasil)
    ─────────────────────────────────────────────────
    [5] ELITIST ANT — hanya semut terbaik iterasi yang deposit
        feromon (bukan semua semut). Ini memperketat eksplorasi
        dan mempercepat konvergensi ke rute berkualitas.
 
    [6] DIJKSTRA WARMUP — populasi awal dari Dijkstra, bukan
        random. ACO mulai dari solusi yang sudah "cukup baik"
        dan feromon awal ditaburkan di jalur Dijkstra.
 
    [7] 2-OPT POST-PROCESSING — rute terbaik yang ditemukan
        ACO diperbaiki dengan 2-opt lokal untuk menghilangkan
        crossing path. Ini yang membuat hasilnya bisa menyamai
        atau mengalahkan GA.
 
    [8] STAGNATION RESET — jika N_STAGNATION iterasi berturutan
        tidak ada perbaikan, feromon direset ke TAU_INIT.
        Mencegah ACO terjebak di local optimum.
 
    PERBANDINGAN LENGKAP
    ─────────────────────
    ┌──────────────────┬──────────┬──────────┬──────────┬──────────┐
    │                  │ GA       │ ACO lama │ ACO-Elite│
    ├──────────────────┼──────────┼──────────┼──────────┤
    │ Kecepatan        │ ★★★★     │ ★★       │ ★★★★     │
    │ Kualitas rute    │ ★★★★     │ ★★★      │ ★★★★★    │
    │ Deterministik    │ ✗ (seed) │ ✗ (seed) │ ✗ (seed) │
    │ Post-processing  │ ✓ (2-opt)│ ✗        │ ✓ (2-opt)│
    │ Memory usage     │ sedang   │ tinggi   │ rendah   │
    └──────────────────┴──────────┴──────────┴──────────┘
    ─────────────────────────────────────────────────────────────
    """
 
    name        = "aco_elite"
    description = "ACO-Elite — subgraph pruning + candidate list + elitist + 2-opt"
 
    # ------------------------------------------------------------------
    # PARAMETER — bisa diubah untuk tuning
    # ------------------------------------------------------------------
    N_ANTS          = 15     # semut per iterasi (lebih sedikit, tapi lebih terarah)
    N_ITERATIONS    = 20     # iterasi total
    ALPHA           = 1.0    # bobot feromon τ
    BETA            = 3.0    # bobot visibilitas η (dinaikkan agar lebih greedy)
    RHO             = 0.15   # laju evaporasi
    Q               = 100.0  # konstanta deposit feromon
    TAU_INIT        = 1.0    # feromon awal
    TAU_MIN         = 1e-4   # feromon minimum (bawah ini dihapus)
    TOP_K           = 8      # kandidat tetangga terbaik per node
    BFS_RADIUS      = 3      # radius BFS untuk membangun subgraph
    N_STAGNATION    = 5      # iterasi tanpa perbaikan → reset feromon
    RANDOM_SEED     = 42
 
    # ------------------------------------------------------------------
    # OPTIMASI 1: Bangun subgraph terbatas sekitar rute Dijkstra
    # ------------------------------------------------------------------
    def _build_subgraph(
        self,
        G,
        source: int,
        target: int,
        dijkstra_path: list
    ) -> set:
        """
        Ambil semua node dalam jangkauan BFS_RADIUS hop dari
        setiap node di jalur Dijkstra.
 
        Hasilnya: subgraph kecil (~100-500 node) yang mencakup
        area relevan di sekitar jalur optimal. Semut hanya
        bergerak dalam subgraph ini.
        """
        relevant = set(dijkstra_path)
        frontier = set(dijkstra_path)
 
        for _ in range(self.BFS_RADIUS):
            next_frontier = set()
            for node in frontier:
                for nb in G.successors(node):
                    if nb not in relevant:
                        relevant.add(nb)
                        next_frontier.add(nb)
                for nb in G.predecessors(node):
                    if nb not in relevant:
                        relevant.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
 
        return relevant
 
    # ------------------------------------------------------------------
    # OPTIMASI 2: Pre-compute candidate list TOP-K per node
    # ------------------------------------------------------------------
    def _build_candidate_lists(
        self,
        G,
        subgraph_nodes: set
    ) -> dict:
        """
        Untuk setiap node di subgraph, simpan TOP-K tetangga
        dengan travel_time terkecil.
 
        Format: {node: [(neighbor, travel_time), ...]}
 
        Dibangun sekali di awal, dipakai oleh semua semut
        di semua iterasi → O(K) per step, bukan O(all neighbors).
        """
        candidates = {}
        for node in subgraph_nodes:
            neighbors = []
            for nb in G.successors(node):
                if nb not in subgraph_nodes:
                    continue
                edge_dict = G.get_edge_data(node, nb)
                if not edge_dict:
                    continue
                best_tt = min(
                    float(d.get("travel_time", 9999))
                    for d in edge_dict.values()
                )
                if best_tt < 9999:
                    neighbors.append((nb, best_tt))
            # Simpan hanya TOP-K terbaik
            neighbors.sort(key=lambda x: x[1])
            candidates[node] = neighbors[:self.TOP_K]
        return candidates
 
    # ------------------------------------------------------------------
    # OPTIMASI 3: Visibilitas dengan Haversine direction factor
    # ------------------------------------------------------------------
    def _haversine(self, G, u: int, v: int) -> float:
        """Jarak garis lurus antara dua node (meter)."""
        try:
            nu = G.nodes[u]
            nv = G.nodes[v]
            lat1 = math.radians(nu["y"])
            lon1 = math.radians(nu["x"])
            lat2 = math.radians(nv["y"])
            lon2 = math.radians(nv["x"])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat / 2) ** 2
                 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
            return 2 * 6_371_000 * math.asin(math.sqrt(a))
        except Exception:
            return 1.0
 
    def _visibility(
        self,
        travel_time: float,
        current: int,
        neighbor: int,
        target: int,
        G
    ) -> float:
        """
        η(current → neighbor) = (1 / travel_time) × direction_factor
 
        direction_factor mengukur seberapa "maju" langkah ini
        ke arah target:
          = dist(current, target) / dist(neighbor, target)
 
        Jika neighbor lebih dekat ke target → faktor > 1 (bonus)
        Jika neighbor lebih jauh → faktor < 1 (penalti)
        """
        if travel_time <= 0:
            return 0.0
        base = 1.0 / travel_time
 
        d_current  = self._haversine(G, current, target)
        d_neighbor = self._haversine(G, neighbor, target)
 
        if d_neighbor <= 0:
            direction_factor = 2.0  # langsung ke target
        elif d_current <= 0:
            direction_factor = 1.0
        else:
            direction_factor = d_current / d_neighbor
 
        return base * direction_factor
 
    # ------------------------------------------------------------------
    # CORE: satu semut membangun path dalam subgraph
    # ------------------------------------------------------------------
    def _build_ant_path(
        self,
        G,
        source: int,
        target: int,
        pheromone: dict,
        candidate_lists: dict,
        rng: random.Random,
        max_steps: int
    ) -> list | None:
        """
        Semut bergerak dari source ke target menggunakan
        candidate list dan visibilitas yang sudah dipre-compute.
 
        Menggunakan roulette wheel selection berbasis
        τ^α × η^β dari TOP-K kandidat saja.
        """
        path    = [source]
        visited = {source}
        current = source
 
        for _ in range(max_steps):
            if current == target:
                return path
 
            # Ambil kandidat dari pre-computed list
            raw_candidates = candidate_lists.get(current, [])
            # Filter yang sudah dikunjungi
            candidates = [(nb, tt) for nb, tt in raw_candidates
                          if nb not in visited]
 
            if not candidates:
                # Semut terjebak — pakai Dijkstra lokal sebagai bridge
                try:
                    bridge = nx.shortest_path(
                        G, current, target, weight="travel_time"
                    )
                    path += bridge[1:]
                    return path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None
 
            # Shortcut: jika target ada di kandidat
            if any(nb == target for nb, _ in candidates):
                path.append(target)
                return path
 
            # Hitung skor probabilistik
            scores = []
            for nb, tt in candidates:
                tau = pheromone.get((current, nb), self.TAU_INIT)
                eta = self._visibility(tt, current, nb, target, G)
                score = (tau ** self.ALPHA) * (eta ** self.BETA)
                scores.append(score)
 
            total = sum(scores)
            if total == 0:
                chosen = rng.choice(candidates)[0]
            else:
                probs  = [s / total for s in scores]
                r      = rng.random()
                cumul  = 0.0
                chosen = candidates[-1][0]
                for (nb, _), prob in zip(candidates, probs):
                    cumul += prob
                    if r <= cumul:
                        chosen = nb
                        break
 
            path.append(chosen)
            visited.add(chosen)
            current = chosen
 
        return None  # melebihi max_steps
 
    # ------------------------------------------------------------------
    # BONUS 3: 2-opt post-processing pada rute terbaik
    # ------------------------------------------------------------------
    def _two_opt(self, G, path: list) -> list:
        """
        Perbaiki rute dengan 2-opt: coba balik setiap sub-segmen,
        pertahankan jika menghasilkan total travel_time lebih kecil.
 
        Ini yang membuat ACO-Elite bisa menyamai kualitas GA.
        """
        if len(path) < 4:
            return path
 
        def path_cost(p):
            total = 0.0
            for u, v in zip(p[:-1], p[1:]):
                ed = G.get_edge_data(u, v)
                if not ed:
                    return float("inf")
                total += min(
                    float(d.get("travel_time", 9999))
                    for d in ed.values()
                )
            return total
 
        improved = True
        best     = path[:]
        best_cost = path_cost(best)
 
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 2, len(best)):
                    candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    c = path_cost(candidate)
                    if c < best_cost - 1e-9:
                        best      = candidate
                        best_cost = c
                        improved  = True
        return best
 
    # ------------------------------------------------------------------
    # HELPER: hitung total travel_time sebuah path
    # ------------------------------------------------------------------
    def _path_cost(self, G, path: list) -> float:
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            ed = G.get_edge_data(u, v)
            if not ed:
                return float("inf")
            total += min(
                float(d.get("travel_time", 9999))
                for d in ed.values()
            )
        return total
 
    # ------------------------------------------------------------------
    # CORE: jalankan koloni ACO-Elite
    # ------------------------------------------------------------------
    def _run_aco_elite(self, G, source: int, target: int) -> list:
        """
        Pipeline lengkap ACO-Elite:
 
        1. Dijkstra warmup → dapat jalur awal + bangun subgraph
        2. Build candidate lists dari subgraph
        3. Iterasi ACO dengan elitist deposit
        4. Stagnation reset jika perlu
        5. 2-opt pada rute terbaik
        """
        rng = random.Random(self.RANDOM_SEED)
 
        # ── BONUS 2: Dijkstra warmup ──────────────────────────
        try:
            dijkstra_path = nx.shortest_path(
                G, source, target, weight="travel_time"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise nx.NetworkXNoPath(
                f"ACO-Elite: tidak ada jalur dari {source} ke {target}"
            )
 
        dijkstra_cost = self._path_cost(G, dijkstra_path)
 
        # ── OPTIMASI 1: Bangun subgraph terbatas ──────────────
        subgraph_nodes = self._build_subgraph(G, source, target, dijkstra_path)
 
        # ── OPTIMASI 2: Pre-compute candidate lists ───────────
        candidate_lists = self._build_candidate_lists(G, subgraph_nodes)
 
        # ── Inisialisasi feromon ──────────────────────────────
        # Taburkan feromon awal di jalur Dijkstra (warmup)
        pheromone = {}
        warmup_deposit = self.Q / dijkstra_cost if dijkstra_cost > 0 else 1.0
        for u, v in zip(dijkstra_path[:-1], dijkstra_path[1:]):
            pheromone[(u, v)] = self.TAU_INIT + warmup_deposit
 
        best_path  = dijkstra_path[:]
        best_cost  = dijkstra_cost
        stagnation = 0
        max_steps  = min(len(subgraph_nodes) * 2, 2000)
 
        # ── Iterasi koloni ────────────────────────────────────
        for iteration in range(self.N_ITERATIONS):
            iter_best_path = None
            iter_best_cost = float("inf")
 
            # Setiap semut bangun satu path
            for _ in range(self.N_ANTS):
                path = self._build_ant_path(
                    G, source, target,
                    pheromone, candidate_lists,
                    rng, max_steps
                )
                if path is None or path[-1] != target:
                    continue
                cost = self._path_cost(G, path)
                if cost < iter_best_cost:
                    iter_best_path = path
                    iter_best_cost = cost
 
            # ── Evaporasi feromon (semua edge di dict) ────────
            for key in list(pheromone.keys()):
                pheromone[key] *= (1.0 - self.RHO)
                # OPTIMASI 4: hapus feromon sangat kecil (sparse)
                if pheromone[key] < self.TAU_MIN:
                    del pheromone[key]
 
            # ── BONUS 1: Elitist deposit (hanya terbaik) ─────
            if iter_best_path is not None:
                deposit = self.Q / iter_best_cost
                for u, v in zip(iter_best_path[:-1], iter_best_path[1:]):
                    key = (u, v)
                    pheromone[key] = pheromone.get(key, self.TAU_INIT) + deposit
 
                if iter_best_cost < best_cost:
                    best_path  = iter_best_path[:]
                    best_cost  = iter_best_cost
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
 
            # ── BONUS 4: Stagnation reset ─────────────────────
            if stagnation >= self.N_STAGNATION:
                # Reset feromon ke nilai awal, pertahankan jalur terbaik
                pheromone = {}
                for u, v in zip(best_path[:-1], best_path[1:]):
                    pheromone[(u, v)] = self.TAU_INIT + warmup_deposit
                stagnation = 0
 
        # ── BONUS 3: 2-opt post-processing ───────────────────
        optimized = self._two_opt(G, best_path)
        if self._path_cost(G, optimized) < best_cost:
            best_path = optimized
 
        return best_path
 
    # ------------------------------------------------------------------
    # INTERFACE WAJIB — dipanggil oleh framework benchmark
    # ------------------------------------------------------------------
    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()
        try:
            route = self._run_aco_elite(G, source_node, target_node)
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
                "top_k":        self.TOP_K,
                "bfs_radius":   self.BFS_RADIUS,
                "optimizations": "subgraph_pruning + candidate_list + "
                                 "direction_visibility + elitist + 2opt + stagnation_reset",
            }
        )
 

 
# ──────────────────────────────────────────────────────────────
# CARA REGISTRASI
# Tambahkan baris ini di bagian paling bawah algorithms.py:
#
#   REGISTRY.register(AntColonyRouting())
#
# Inspired by the metallurgical process of slowly cooling metal.
# Starts with a random valid path and iteratively proposes small
# changes (neighbours). Improvements are always accepted; degradations
# are accepted with probability exp(-Δcost / T), where T decreases
# over time. Early on (high T) it explores widely; later (low T) it
# exploits the best solution found.
#
# Objective: minimise physical distance (metres), not travel_time.
# ══════════════════════════════════════════════════════════════════

class GeraldSimulatedAnnealing(BaseRoutingAlgorithm):
    """
    Simulated Annealing for shortest-distance path finding.

    Unlike the GA (which optimises travel_time), SA here minimises
    the total physical distance in metres. Useful for comparing
    time-optimal vs distance-optimal strategies.
    """
    name        = "gerald_sa"
    description = "Simulated Annealing — minimise physical distance"

    # ── SA parameters ─────────────────────────────────────────────────────
    ITERATIONS          = 100     # total neighbour proposals
    INITIAL_TEMPERATURE = 1200.0  # high T → accept bad moves freely (exploration)
    COOLING_RATE        = 0.94    # multiply T by this each iteration (slow cool)
    MIN_TEMPERATURE     = 0.01    # stop accepting bad moves below this threshold
    RANDOM_SEED         = 31

    def _fitness(self, G, path: list) -> float:
        """Objective: total path length in metres (lower = better)."""
        return _ga_path_distance(G, path)

    @staticmethod
    def _frame(G, gen_idx: int, best: list, candidate: list) -> dict:
        """
        Capture the state of one SA iteration for the evolution log viewer.
        Records coordinates and street names for both the current best path
        and the candidate (proposed neighbour) path.
        """
        def to_coords(path):
            result = []
            for n in path:
                node = G.nodes.get(n)
                if node:
                    result.append([round(float(node["y"]), 5),
                                   round(float(node["x"]), 5)])
            return result

        return {
            "gen":              gen_idx + 1,
            "min":              round(_ga_path_cost(G, best) / 60, 3),
            "dist":             round(_ga_path_distance(G, best) / 1000, 3),
            "coords":           to_coords(best),
            "streets":          _route_streets(G, best),
            "candidate_min":    round(_ga_path_cost(G, candidate) / 60, 3),
            "candidate_dist":   round(_ga_path_distance(G, candidate) / 1000, 3),
            "candidate_coords": to_coords(candidate),
            "candidate_streets": _route_streets(G, candidate),
        }

    def find_route(self, G, source_node, target_node, scenario_name=""):
        """
        Run Simulated Annealing from source_node to target_node.

        Starts from a noisy shortest-path (to avoid a trivial starting point),
        then iteratively proposes sub-segment re-routes. Accepts improvements
        always; accepts degradations with decreasing probability as T cools.
        """
        t0  = time.perf_counter()
        rng = random.Random(self.RANDOM_SEED)

        # Initial solution: noisy shortest path (not the global optimum)
        current = _sa_noisy_shortest_path(G, source_node, target_node, rng)
        if not current:
            # Fallback: clean shortest path if noisy version fails
            try:
                current = nx.shortest_path(G, source_node, target_node, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                ms = (time.perf_counter() - t0) * 1000
                return RouteResult.failure(
                    self.name, scenario_name, source_node, target_node, str(e), ms
                )

        best          = current[:]
        current_score = self._fitness(G, current)
        best_score    = current_score
        temperature   = self.INITIAL_TEMPERATURE
        gen_history   = []

        for idx in range(self.ITERATIONS):
            # Propose a neighbour by re-routing a random sub-segment
            candidate       = _sa_neighbor_path(G, current, rng)
            candidate_score = self._fitness(G, candidate)
            delta           = candidate_score - current_score

            # Accept if better, or probabilistically if worse (Metropolis criterion)
            accept = delta <= 0
            if not accept and temperature > self.MIN_TEMPERATURE:
                accept_prob = math.exp(-delta / temperature)
                accept      = rng.random() < accept_prob

            if accept:
                current       = candidate
                current_score = candidate_score

            # Always track the global best seen so far
            if current_score < best_score:
                best       = current[:]
                best_score = current_score

            gen_history.append(self._frame(G, idx, best, candidate))

            # Cool down — multiplicative geometric cooling
            temperature = max(self.MIN_TEMPERATURE, temperature * self.COOLING_RATE)

        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name, source_node, target_node, best, ms,
            metadata={
                "algorithm_family":   "simulated_annealing",
                "generations":        self.ITERATIONS,
                "population":         1,   # SA has no population — just one current solution
                "initial_temperature": self.INITIAL_TEMPERATURE,
                "cooling_rate":       self.COOLING_RATE,
                "min_temperature":    self.MIN_TEMPERATURE,
                "mutation_rate":      1.0,  # SA always proposes a neighbour each step
                "gen_history":        gen_history,
            },
        )


# ══════════════════════════════════════════════════════════════════
# SECTION 7: PARTICLE SWARM OPTIMIZATION (PSO)
#
# Each "particle" represents a candidate path. Particles move through
# the solution space guided by their own best-known position (personal
# best) and the swarm's overall best (global best). The blend of
# inertia, personal memory, and social influence navigates the search.
#
# Adapted to discrete path spaces: "velocity" is implemented as
# probabilistic segment splicing from personal and global bests,
# followed by repair to ensure the path is road-valid.
# ══════════════════════════════════════════════════════════════════

def _pso_path_cost(G, path: list) -> float:
    """Total travel_time (seconds) for a PSO candidate path."""
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
    Generate one swarm particle (initial path) via noisy Dijkstra.
    Higher noise range than GA to create more initial diversity in the swarm.
    """
    def noisy_weight(u, v, data):
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        t = float(best.get("travel_time", 9999))
        return t * rng.uniform(0.5, 1.8)   # wider noise for more initial diversity

    try:
        return nx.shortest_path(G, source, target, weight=noisy_weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _pso_repair_path(G, partial_nodes: list):
    """
    Repair a potentially disconnected waypoint sequence into a valid road path.

    PSO segment mixing may produce a list of node IDs that are not consecutive
    neighbours on the graph. This function connects each consecutive pair
    with its shortest road path, producing a valid routable path.

    Returns None if any segment is unreachable (broken graph connection).
    """
    if len(partial_nodes) < 2:
        return partial_nodes if partial_nodes else None

    # Deduplicate consecutive identical nodes (they add no new road)
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
            return None   # can't repair this gap
        if not full_path:
            full_path.extend(seg)
        else:
            full_path.extend(seg[1:])   # skip duplicate junction

    return full_path if full_path else None


def _pso_mix_paths(G, current: list, personal_best: list, global_best: list,
                   inertia_w: float, cognitive_w: float, social_w: float,
                   rng: random.Random):
    """
    Discrete PSO position update via path segment mixing.

    Analogous to the continuous PSO velocity equation:
      new_pos = inertia*current + cognitive*(personal_best - current)
                                + social*(global_best - current)

    Here "addition" means probabilistically splicing segments from each source:
      - Inertia:    keep a slice of the current path (momentum)
      - Cognitive:  pull toward the particle's personal best
      - Social:     pull toward the global best

    The combined waypoints are then repaired into a valid road path.
    """
    source, target = current[0], current[-1]
    waypoints = [source]

    # Inertia: keep some intermediate nodes from the current path
    if len(current) > 2 and rng.random() < inertia_w:
        n_keep    = max(1, int(len(current) * rng.uniform(0.2, 0.5)))
        start_idx = rng.randint(1, max(1, len(current) - n_keep - 1))
        waypoints.extend(current[start_idx:start_idx + n_keep])

    # Cognitive: pull toward personal best
    if len(personal_best) > 2 and rng.random() < cognitive_w:
        n_take    = max(1, int(len(personal_best) * rng.uniform(0.15, 0.4)))
        start_idx = rng.randint(1, max(1, len(personal_best) - n_take - 1))
        waypoints.extend(personal_best[start_idx:start_idx + n_take])

    # Social: pull toward global best
    if len(global_best) > 2 and rng.random() < social_w:
        n_take    = max(1, int(len(global_best) * rng.uniform(0.15, 0.4)))
        start_idx = rng.randint(1, max(1, len(global_best) - n_take - 1))
        waypoints.extend(global_best[start_idx:start_idx + n_take])

    waypoints.append(target)

    # Remove consecutive duplicates before repair
    cleaned = [waypoints[0]]
    for n in waypoints[1:]:
        if n != cleaned[-1]:
            cleaned.append(n)

    return _pso_repair_path(G, cleaned)   # may return None if repair fails


def _pso_mutate(G, path: list, rng: random.Random) -> list:
    """
    Random sub-segment mutation for PSO particles.
    Re-routes a random slice via noisy shortest_path to inject diversity
    and prevent swarm stagnation.
    """
    if len(path) < 3:
        return path
    i = rng.randint(0, len(path) - 2)
    j = rng.randint(i + 1, min(i + max(len(path) // 4, 2), len(path) - 1))
    try:
        def noisy_weight(u, v, data):
            best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
            t = float(best.get("travel_time", 9999))
            return t * rng.uniform(0.6, 1.6)
        seg = nx.shortest_path(G, path[i], path[j], weight=noisy_weight)
        return path[:i] + seg + path[j + 1:]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return path   # mutation failed — keep original


class ParticleSwarmRouting(BaseRoutingAlgorithm):
    """
    Particle Swarm Optimization (PSO) for point-to-point routing.

    Each of the N_PARTICLES particles is a candidate path.
    Particles update by blending segments from their personal best
    and the global best, governed by inertia, cognitive, and social weights.
    Random mutation prevents premature convergence.
    """
    name        = "particle_swarm"
    description = "Particle Swarm Optimization — segment-mixing swarm on road paths"

    # ── PSO parameters ────────────────────────────────────────────────────
    N_PARTICLES      = 40    # swarm size — more particles → broader search
    N_ITERATIONS     = 80    # number of update rounds
    INERTIA_WEIGHT   = 0.5   # how much of the current path to retain
    COGNITIVE_WEIGHT = 1.2   # pull toward particle's personal best
    SOCIAL_WEIGHT    = 1.4   # pull toward swarm's global best (slightly stronger)
    MUTATION_RATE    = 0.25  # probability of random sub-segment mutation each step
    RANDOM_SEED      = 42

    def find_route(self, G, source_node, target_node, scenario_name=""):
        """Run PSO and return the best path found by the swarm."""
        t0  = time.perf_counter()
        rng = random.Random(self.RANDOM_SEED)

        # ── 1. Initialise swarm ───────────────────────────────────────────
        particles      = []   # current path per particle
        personal_bests = []   # best-ever path per particle
        personal_costs = []   # cost of each particle's personal best

        for _ in range(self.N_PARTICLES):
            p = _pso_random_path(G, source_node, target_node, rng)
            if p:
                cost = _pso_path_cost(G, p)
                particles.append(p)
                personal_bests.append(p[:])
                personal_costs.append(cost)

        # Fallback: if the entire swarm failed to initialise, use Dijkstra
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

        # Identify the initial global best
        global_best_idx  = min(range(len(particles)), key=lambda i: personal_costs[i])
        global_best_path = personal_bests[global_best_idx][:]
        global_best_cost = personal_costs[global_best_idx]

        # ── 2. Swarm iteration ────────────────────────────────────────────
        for _ in range(self.N_ITERATIONS):
            for i in range(len(particles)):
                # Update particle position by mixing current, personal, and global bests
                new_path = _pso_mix_paths(
                    G, particles[i], personal_bests[i], global_best_path,
                    self.INERTIA_WEIGHT, self.COGNITIVE_WEIGHT, self.SOCIAL_WEIGHT,
                    rng,
                )

                if new_path is None:
                    new_path = particles[i]   # repair failed — keep current path

                # Optional random mutation to escape local optima
                if rng.random() < self.MUTATION_RATE:
                    new_path = _pso_mutate(G, new_path, rng)

                new_cost    = _pso_path_cost(G, new_path)
                particles[i] = new_path

                # Update personal best
                if new_cost < personal_costs[i]:
                    personal_bests[i] = new_path[:]
                    personal_costs[i] = new_cost

                    # Update global best
                    if new_cost < global_best_cost:
                        global_best_path = new_path[:]
                        global_best_cost = new_cost

        # ── 3. Return the global best path found ──────────────────────────
        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(
            G, self.name, scenario_name,
            source_node, target_node, global_best_path, ms,
            metadata={
                "n_particles":      self.N_PARTICLES,
                "n_iterations":     self.N_ITERATIONS,
                "inertia_weight":   self.INERTIA_WEIGHT,
                "cognitive_weight": self.COGNITIVE_WEIGHT,
                "social_weight":    self.SOCIAL_WEIGHT,
                "mutation_rate":    self.MUTATION_RATE,
                "strategy":         "discrete_path_pso",
            },
        )


# ══════════════════════════════════════════════════════════════════
# LEGACY SINGLE-STOP SCENARIOS
# Concrete point-to-point scenarios between real Surabaya facilities.
# Kept as reference; not used by the default benchmark.
# ══════════════════════════════════════════════════════════════════

_LEGACY_SINGLE_STOP_SCENARIOS = [
    Scenario(
        name="darmo_to_rsu_haji",
        description="Transfer pasien: RS Darmo (tengah-barat) → RSU Haji Surabaya (timur)",
        source_node=1685220157,  target_node=4332874690,
        source_label="RS Darmo", target_label="RSU Haji Surabaya",
        source_coords=(-7.2874, 112.7382), target_coords=(-7.2828, 112.7798),
    ),
    Scenario(
        name="polsek_genteng_to_rs_darmo",
        description="Respons darurat: Polsek Genteng (pusat) → RS Darmo",
        source_node=5589485735,  target_node=1685220157,
        source_label="Polsek Genteng", target_label="RS Darmo",
        source_coords=(-7.2556, 112.7483), target_coords=(-7.2874, 112.7382),
    ),
    Scenario(
        name="national_to_rs_ramelan",
        description="Lintas kota barat→selatan: National Hospital → RS Ramelan",
        source_node=1721014942,  target_node=1719470350,
        source_label="National Hospital", target_label="RS Angkatan Laut Dr. Ramelan",
        source_coords=(-7.2993, 112.6764), target_coords=(-7.3093, 112.7382),
    ),
    Scenario(
        name="polsek_rungkut_to_rs_onkologi",
        description="Respons darurat area timur: Polsek Rungkut → RS Onkologi",
        source_node=4574365996,  target_node=7059452149,
        source_label="Polsek Rungkut", target_label="RS Onkologi",
        source_coords=(-7.3384, 112.7712), target_coords=(-7.2909, 112.7893),
    ),
    Scenario(
        name="ciputra_to_rsu_haji",
        description="Rute terpanjang lintas kota: Ciputra Hospital (barat jauh) → RSU Haji (timur)",
        source_node=4163428113,  target_node=4332874690,
        source_label="Ciputra Hospital", target_label="RSU Haji Surabaya",
        source_coords=(-7.2809, 112.6346), target_coords=(-7.2828, 112.7798),
    ),
    Scenario(
        name="benowo_to_onkologi",
        description="EXTREME lintas kota: Polsek Benowo (barat jauh) → RS Onkologi (timur) ~19km",
        source_node=5539027568,  target_node=7059452149,
        source_label="Polsek Benowo", target_label="RS Onkologi",
        source_coords=(-7.2359, 112.6076), target_coords=(-7.2909, 112.7893),
    ),
]


# ══════════════════════════════════════════════════════════════════
# EXAMPLE MULTI-STOP SCENARIOS
# Pre-defined multi-stop tours using known Surabaya facility nodes.
# Can be used instead of (or alongside) the category-based scenarios.
# ══════════════════════════════════════════════════════════════════

# Known facility node IDs extracted from OSM / facilities_with_network.csv
_SCENARIO_POINTS = {
    "rs_darmo":        {"label": "RS Darmo",                       "node": 1685220157, "coords": (-7.2874, 112.7382)},
    "rsu_haji":        {"label": "RSU Haji Surabaya",              "node": 4332874690, "coords": (-7.2828, 112.7798)},
    "polsek_genteng":  {"label": "Polsek Genteng",                 "node": 5589485735, "coords": (-7.2556, 112.7483)},
    "national":        {"label": "National Hospital",              "node": 1721014942, "coords": (-7.2993, 112.6764)},
    "rs_ramelan":      {"label": "RS Angkatan Laut Dr. Ramelan",   "node": 1719470350, "coords": (-7.3093, 112.7382)},
    "polsek_rungkut":  {"label": "Polsek Rungkut",                 "node": 4574365996, "coords": (-7.3384, 112.7712)},
    "rs_onkologi":     {"label": "RS Onkologi",                    "node": 7059452149, "coords": (-7.2909, 112.7893)},
    "ciputra":         {"label": "Ciputra Hospital",               "node": 4163428113, "coords": (-7.2809, 112.6346)},
    "polsek_benowo":   {"label": "Polsek Benowo",                  "node": 5539027568, "coords": (-7.2359, 112.6076)},
}


def _multi_stop_scenario(name: str, description: str,
                         point_keys: list, round_trip: bool = False) -> Scenario:
    """Build a multi-stop Scenario from named facility keys."""
    points = [_SCENARIO_POINTS[k] for k in point_keys]
    nodes  = [p["node"]   for p in points]
    labels = [p["label"]  for p in points]
    coords = [p["coords"] for p in points]
    return Scenario(
        name=name, description=description,
        source_node=nodes[0],  target_node=nodes[-1],
        source_label=labels[0], target_label=labels[-1],
        source_coords=coords[0], target_coords=coords[-1],
        route_nodes=nodes, route_labels=labels, route_coords=coords,
        round_trip=round_trip,
    )


EXAMPLE_SCENARIOS = [
    # ── 5-stop scenarios ──────────────────────────────────────────────────
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
    # ── 7-stop scenario ───────────────────────────────────────────────────
    _multi_stop_scenario(
        "cross_city_zigzag_7",
        "7-stop: full cross-city zigzag — far-west to far-east via north, center, and south",
        ["polsek_benowo", "national", "polsek_genteng", "rs_darmo",
         "rs_ramelan", "rsu_haji", "rs_onkologi"],
    ),
    # ── 9-stop round trip ─────────────────────────────────────────────────
    _multi_stop_scenario(
        "full_city_patrol_9",
        "9-stop: all-facilities full-city patrol with return to base — hardest scenario",
        ["polsek_benowo", "ciputra", "national", "polsek_genteng",
         "rs_darmo", "rs_ramelan", "polsek_rungkut", "rs_onkologi", "rsu_haji"],
        round_trip=True,
    ),
]
