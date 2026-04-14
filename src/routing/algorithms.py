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
import networkx as nx

from src.routing.base import BaseRoutingAlgorithm, RouteResult


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
# TEAM ALGORITHM SLOTS
# Replace the body of find_route with your own logic.
# ──────────────────────────────────────────────────────────────

class TeamAModel(BaseRoutingAlgorithm):
    """
    ── TEAM A ───────────────────────────────────────────────────
    Replace predict_edge_weight() with your model's scoring logic.

    Pattern: your model scores every edge (lower = preferred),
    then Dijkstra finds the optimal path using those scores as weights.

    If your model works differently (e.g. outputs a full path directly),
    just replace the body of find_route entirely.
    ─────────────────────────────────────────────────────────────
    """
    name        = "team_a"
    description = "Team A model"

    def predict_edge_weight(self, u: int, v: int, edge_data: dict) -> float:
        """
        Return a cost for traversing edge (u → v).
        Lower cost = more preferred by the router.

        edge_data keys available:
          length        (float)  metres
          travel_time   (float)  seconds
          speed_kph     (float)  km/h
          highway       (str)    OSM road type, e.g. "residential"

        Replace this with your model's prediction.
        Example: return your_model.predict([features])[0]
        """
        # ── TEAM A: put your edge-scoring logic here ─────────
        return edge_data.get("travel_time", 9999)
        # ─────────────────────────────────────────────────────

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()

        # Build a weight function from your model
        def weight_fn(u, v, edge_dict):
            best = min(edge_dict.values(),
                       key=lambda d: self.predict_edge_weight(u, v, d))
            return self.predict_edge_weight(u, v, best)

        try:
            route = nx.shortest_path(G, source_node, target_node, weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name,
                                       source_node, target_node, str(e), ms)

        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, self.name, scenario_name,
                                 source_node, target_node, route, ms)


class TeamBModel(BaseRoutingAlgorithm):
    """
    ── TEAM B ───────────────────────────────────────────────────
    Same pattern — Team B replaces predict_edge_weight with
    their own model. The benchmark will automatically compare
    Team A vs Team B vs all baselines.
    ─────────────────────────────────────────────────────────────
    """
    name        = "team_b"
    description = "Team B model"

    def predict_edge_weight(self, u: int, v: int, edge_data: dict) -> float:
        """
        Replace this with Team B's model.
        Must return a float — lower = more preferred.
        """
        # ── TEAM B: put your edge-scoring logic here ─────────
        return edge_data.get("travel_time", 9999)
        # ─────────────────────────────────────────────────────

    def find_route(self, G, source_node, target_node, scenario_name=""):
        t0 = time.perf_counter()

        def weight_fn(u, v, edge_dict):
            best = min(edge_dict.values(),
                       key=lambda d: self.predict_edge_weight(u, v, d))
            return self.predict_edge_weight(u, v, best)

        try:
            route = nx.shortest_path(G, source_node, target_node, weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            ms = (time.perf_counter() - t0) * 1000
            return RouteResult.failure(self.name, scenario_name,
                                       source_node, target_node, str(e), ms)

        ms = (time.perf_counter() - t0) * 1000
        return RouteResult.build(G, self.name, scenario_name,
                                 source_node, target_node, route, ms)


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
