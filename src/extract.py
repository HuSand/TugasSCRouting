"""
Data extraction pipeline.
Downloads public facilities and road network from OpenStreetMap.

Outputs (in settings.DATA_DIR):
    facilities_raw.geojson
    facilities_clean.csv / .geojson
    road_network.graphml
    road_network_nodes.geojson / _edges.geojson
    facilities_with_network.csv / .geojson
    extraction_report.txt
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _build_combined_tags(categories: dict) -> dict:
    combined: dict = {}
    for info in categories.values():
        for key, values in info["tags"].items():
            if key not in combined:
                combined[key] = []
            if isinstance(values, list):
                combined[key].extend(values)
            else:
                combined[key] = True
    return combined


def _assign_category(row: pd.Series, categories: dict) -> str:
    for cat_name, info in categories.items():
        for key, values in info["tags"].items():
            cell = row.get(key)
            if cell is None or pd.isna(cell):
                continue
            if isinstance(values, list) and cell in values:
                return cat_name
            if values is True:
                return cat_name
    return "other"


# ──────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────

def _extract_raw(cfg) -> gpd.GeoDataFrame:
    log.info("Querying facilities from OpenStreetMap...")
    tags = _build_combined_tags(cfg.FACILITY_CATEGORIES)
    try:
        gdf = ox.features_from_place(cfg.PLACE, tags=tags)
    except Exception as e:
        log.warning(f"Place query failed ({e}), falling back to bounding box")
        gdf = ox.features_from_bbox(bbox=cfg.BBOX, tags=tags)

    log.info(f"Raw features: {len(gdf)}")
    out = cfg.DATA_DIR / "facilities_raw.geojson"
    gdf.to_file(out, driver="GeoJSON")
    log.info(f"Saved -> {out.name}")
    return gdf


def _clean_facilities(gdf: gpd.GeoDataFrame, cfg) -> gpd.GeoDataFrame:
    log.info("Cleaning and categorizing facilities...")
    df = gdf.copy()
    df["geometry"] = df.geometry.centroid
    df["lat"] = df.geometry.y
    df["lon"] = df.geometry.x

    df["category"] = df.apply(_assign_category, axis=1, categories=cfg.FACILITY_CATEGORIES)
    df["priority"] = df["category"].map(
        lambda c: cfg.FACILITY_CATEGORIES.get(c, {}).get("priority", 5)
    )

    type_col = pd.Series(index=df.index, dtype="object")
    for col in ["amenity", "office", "healthcare"]:
        if col in df.columns:
            type_col = type_col.fillna(df[col])
    df["facility_type"] = type_col.fillna("unknown")

    keep = ["geometry", "lat", "lon", "name", "facility_type", "category", "priority"]
    optional = ["addr:street", "addr:city", "addr:postcode", "phone",
                "website", "opening_hours", "operator", "capacity", "building"]
    keep += [c for c in optional if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["name", "facility_type"], keep="first")
    log.info(f"Deduplicated: {before} → {len(df)} (-{before - len(df)})")

    for cat, count in df["category"].value_counts().items():
        log.info(f"  {cat}: {count}")

    df.to_csv(cfg.DATA_DIR / "facilities_clean.csv", index=False)
    df.to_file(cfg.DATA_DIR / "facilities_clean.geojson", driver="GeoJSON")
    log.info(f"Saved -> facilities_clean.csv + .geojson")
    return df


def _extract_network(cfg):
    log.info("Downloading road network...")
    try:
        G = ox.graph_from_place(cfg.PLACE, network_type=cfg.NETWORK_TYPE)
    except Exception as e:
        log.warning(f"Place query failed ({e}), falling back to bounding box")
        G = ox.graph_from_bbox(bbox=cfg.BBOX, network_type=cfg.NETWORK_TYPE)

    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    ox.save_graphml(G, filepath=str(cfg.DATA_DIR / "road_network.graphml"))
    nodes, edges = ox.graph_to_gdfs(G)
    nodes.to_file(cfg.DATA_DIR / "road_network_nodes.geojson", driver="GeoJSON")
    edges.to_file(cfg.DATA_DIR / "road_network_edges.geojson", driver="GeoJSON")
    log.info("Saved -> road_network.graphml + nodes + edges")
    return G, nodes, edges


def _snap_to_network(fac: gpd.GeoDataFrame, G, cfg) -> gpd.GeoDataFrame:
    log.info("Snapping facilities to nearest road node...")
    fac = fac.copy()
    nearest = ox.distance.nearest_nodes(G, X=fac["lon"].values, Y=fac["lat"].values)
    fac["nearest_node"] = nearest

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    dists = []
    for _, row in fac.iterrows():
        nid = row["nearest_node"]
        if nid in nodes_gdf.index:
            ng = nodes_gdf.loc[nid].geometry
            dx = (row["lon"] - ng.x) * 111_000 * np.cos(np.radians(row["lat"]))
            dy = (row["lat"] - ng.y) * 111_000
            dists.append(np.sqrt(dx**2 + dy**2))
        else:
            dists.append(np.nan)
    fac["snap_distance_m"] = dists

    log.info(f"Snap distance — median: {np.nanmedian(dists):.0f}m  "
             f"P95: {np.nanpercentile(dists, 95):.0f}m  "
             f"max: {np.nanmax(dists):.0f}m")
    far = sum(1 for d in dists if not np.isnan(d) and d > 500)
    if far:
        log.warning(f"{far} facilities are >500m from the road network")

    fac.to_csv(cfg.DATA_DIR / "facilities_with_network.csv", index=False)
    fac.to_file(cfg.DATA_DIR / "facilities_with_network.geojson", driver="GeoJSON")
    log.info("Saved -> facilities_with_network.csv + .geojson")
    return fac


def _write_report(fac, G, edges, cfg):
    out = cfg.DATA_DIR / "extraction_report.txt"
    with open(out, "w") as f:
        f.write(f"Extraction Report — {cfg.PLACE}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n{'='*50}\n\n")
        f.write("FACILITIES\n")
        for cat, n in fac["category"].value_counts().items():
            f.write(f"  {cat}: {n}\n")
        f.write(f"\nROAD NETWORK\n"
                f"  Nodes: {G.number_of_nodes()}\n"
                f"  Edges: {G.number_of_edges()}\n")
        if "speed_kph" in edges.columns:
            f.write(f"  Avg speed: {edges['speed_kph'].mean():.1f} km/h\n")
        f.write(f"\nSNAP QUALITY\n"
                f"  Mean: {fac['snap_distance_m'].mean():.1f}m\n"
                f"  >500m: {(fac['snap_distance_m'] > 500).sum()} facilities\n")
        f.write("\nOUTPUT FILES\n")
        for p in sorted(cfg.DATA_DIR.glob("*")):
            sz = p.stat().st_size
            label = f"{sz/1_048_576:.2f} MB" if sz > 1_048_576 else f"{sz/1024:.1f} KB"
            f.write(f"  {p.name}: {label}\n")
    log.info(f"Saved -> extraction_report.txt")


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def run_extraction(cfg):
    ox.settings.use_cache = cfg.OSM_USE_CACHE
    ox.settings.log_console = False
    ox.settings.timeout = cfg.OSM_TIMEOUT

    gdf_raw   = _extract_raw(cfg)
    gdf_clean = _clean_facilities(gdf_raw, cfg)
    G, nodes, edges = _extract_network(cfg)
    gdf_final = _snap_to_network(gdf_clean, G, cfg)
    _write_report(gdf_final, G, edges, cfg)

    log.info(f"Extraction complete: {len(gdf_final)} facilities, "
             f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
