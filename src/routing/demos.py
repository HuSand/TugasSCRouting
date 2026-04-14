"""
Baseline routing demonstrations.
Shows the data is working correctly before algorithm comparison.

Outputs (in settings.DATA_DIR):
    demo_shortest_path.html     fastest vs shortest route overlay
    demo_nearest_hospital.html  nearest facility from a random point
    demo_coverage_report.txt    travel-time coverage stats
"""

import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import folium

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Demo 1 — fastest vs shortest path between two facilities
# ──────────────────────────────────────────────────────────────

def _demo_path_comparison(G, fac, cfg):
    log.info("Demo 1: Fastest vs Shortest path")
    hospitals = fac[fac["facility_type"] == "hospital"].dropna(subset=["name"])
    schools   = fac[fac["facility_type"] == "school"].dropna(subset=["name"])

    if hospitals.empty or schools.empty:
        log.warning("Not enough named hospitals/schools — skipping demo 1.")
        return

    src, dst = hospitals.iloc[0], schools.iloc[0]
    sn, dn   = int(src["nearest_node"]), int(dst["nearest_node"])
    log.info(f"  From: {src.get('name','Hospital')} -> {dst.get('name','School')}")

    try:
        fast  = nx.shortest_path(G, sn, dn, weight="travel_time")
        t_sec = nx.shortest_path_length(G, sn, dn, weight="travel_time")
        short = nx.shortest_path(G, sn, dn, weight="length")
        d_m   = nx.shortest_path_length(G, sn, dn, weight="length")
    except nx.NetworkXNoPath:
        log.error("No path found between these facilities.")
        return

    log.info(f"  Fastest : {len(fast)} nodes, {t_sec/60:.1f} min")
    log.info(f"  Shortest: {len(short)} nodes, {d_m/1000:.2f} km")

    m = ox.plot_route_folium(G, fast,  color="red",  weight=5, opacity=0.75)
    ox.plot_route_folium(    G, short, route_map=m,
                             color="blue", weight=5, opacity=0.75)
    folium.Marker([src["lat"], src["lon"]],
                  popup=f"FROM: {src.get('name','Hospital')}",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([dst["lat"], dst["lon"]],
                  popup=f"TO: {dst.get('name','School')}",
                  icon=folium.Icon(color="red")).add_to(m)

    out = cfg.DATA_DIR / "demo_shortest_path.html"
    m.save(str(out))
    log.info(f"  Saved -> demo_shortest_path.html  (red=fastest, blue=shortest)")


# ──────────────────────────────────────────────────────────────
# Demo 2 — nearest hospital from a random point
# ──────────────────────────────────────────────────────────────

def _demo_nearest_facility(G, fac, cfg):
    log.info("Demo 2: Nearest hospital from a random location")
    hospitals = fac[fac["facility_type"] == "hospital"]
    if hospitals.empty:
        log.warning("No hospitals found — skipping demo 2.")
        return

    np.random.seed(42)
    lat = -7.28 + np.random.uniform(-0.04, 0.04)
    lon = 112.73 + np.random.uniform(-0.06, 0.06)
    origin = ox.distance.nearest_nodes(G, lon, lat)
    log.info(f"  Random origin: ({lat:.4f}, {lon:.4f})")

    best_hosp, best_t = None, float("inf")
    for _, h in hospitals.iterrows():
        try:
            t = nx.shortest_path_length(G, origin, int(h["nearest_node"]),
                                        weight="travel_time")
            if t < best_t:
                best_t, best_hosp = t, h
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if best_hosp is None:
        log.error("No reachable hospital found.")
        return

    log.info(f"  Nearest: {best_hosp.get('name','?')}  —  {best_t/60:.1f} min")

    route = nx.shortest_path(G, origin, int(best_hosp["nearest_node"]),
                              weight="travel_time")
    m = ox.plot_route_folium(G, route, color="red", weight=5)
    folium.Marker([lat, lon],
                  popup="YOUR LOCATION",
                  icon=folium.Icon(color="green", icon="user", prefix="glyphicon")).add_to(m)
    folium.Marker([best_hosp["lat"], best_hosp["lon"]],
                  popup=f"NEAREST: {best_hosp.get('name','Hospital')}",
                  icon=folium.Icon(color="red",   icon="plus-sign", prefix="glyphicon")).add_to(m)

    out = cfg.DATA_DIR / "demo_nearest_hospital.html"
    m.save(str(out))
    log.info(f"  Saved -> demo_nearest_hospital.html")


# ──────────────────────────────────────────────────────────────
# Demo 3 — coverage analysis
# ──────────────────────────────────────────────────────────────

def _demo_coverage(G, fac, cfg):
    log.info("Demo 3: Coverage analysis (travel time to nearest facility)")
    np.random.seed(123)
    lats = -7.28 + np.random.uniform(-0.05, 0.05, cfg.N_COVERAGE_SAMPLES)
    lons = 112.73 + np.random.uniform(-0.07, 0.07, cfg.N_COVERAGE_SAMPLES)
    origins = ox.distance.nearest_nodes(G, lons, lats)

    rows = []
    for category in ["healthcare", "education", "emergency"]:
        cat_fac = fac[fac["category"] == category]
        if cat_fac.empty:
            continue
        nodes = [int(n) for n in cat_fac["nearest_node"].unique()][:cfg.MAX_FACILITIES_PER_CAT]
        times = []
        for src in origins:
            best = min(
                (nx.shortest_path_length(G, src, d, weight="travel_time")
                 for d in nodes
                 if nx.has_path(G, src, d)),
                default=None,
            )
            if best is not None:
                times.append(best)

        if times:
            rows.append({
                "category":      category,
                "facilities":    len(cat_fac),
                "mean_min":      np.mean(times) / 60,
                "median_min":    np.median(times) / 60,
                "p95_min":       np.percentile(times, 95) / 60,
                "max_min":       np.max(times) / 60,
            })
            log.info(f"  {category.upper()} ({len(cat_fac)} facilities)"
                     f"  mean={np.mean(times)/60:.1f}min  "
                     f"median={np.median(times)/60:.1f}min  "
                     f"P95={np.percentile(times,95)/60:.1f}min")

    if rows:
        out = cfg.DATA_DIR / "demo_coverage_report.txt"
        with open(out, "w") as f:
            f.write("Coverage Report\n" + "="*50 + "\n\n")
            for r in rows:
                f.write(f"{r['category'].upper()} ({r['facilities']} facilities)\n"
                        f"  Mean:   {r['mean_min']:.1f} min\n"
                        f"  Median: {r['median_min']:.1f} min\n"
                        f"  P95:    {r['p95_min']:.1f} min\n"
                        f"  Max:    {r['max_min']:.1f} min\n\n")
        log.info(f"  Saved -> demo_coverage_report.txt")


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def run_demos(cfg):
    fac_path = cfg.DATA_DIR / "facilities_with_network.geojson"
    net_path = cfg.DATA_DIR / "road_network.graphml"
    if not fac_path.exists() or not net_path.exists():
        raise FileNotFoundError(
            "Required data files missing. Run 'python main.py extract' first."
        )

    log.info("Loading data for demos...")
    G   = ox.load_graphml(str(net_path))
    fac = gpd.read_file(str(fac_path))

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

    _demo_path_comparison(G, fac, cfg)
    _demo_nearest_facility(G, fac, cfg)
    _demo_coverage(G, fac, cfg)

    log.info("Demos complete.")
