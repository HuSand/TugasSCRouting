"""
src/routing/training.py
=======================
Orkestrasi Team Orienteering Problem (TOP) multi-vehicle, multi-shift.

Alur
----
1. Load road network + facilities, bangun candidate pool (bus_stop + bus_station
   + emergency), dedup per nearest_node, batasi MAX_POOL_NODES.
2. Untuk tiap kendaraan, filter graph berdasarkan lebar jalan, precompute cost
   matrix SEKALI (di-reuse semua model & iterasi).
3. for model in [GA, ACO, SA, PSO]:
       for iter in range(TRAINING_ITERATIONS):
           visited_global = {}            # no-overlap lintas kendaraan & shift
           for vehicle in fleet:
               shift1 -> exclude visited_global
               shift2 -> exclude visited_global ∪ shift1
               visited_global |= shift1 ∪ shift2
4. Tulis training_results.csv (per shift) + training_log.json (untuk viewer).
5. Ringkasan best/mean/std titik per (model, kendaraan); insight bila < target.
"""

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.routing.base import Vehicle
from src.routing.orienteering import build_problem
from src.routing.width import filter_graph_by_width, width_filter_stats

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Candidate pool
# ──────────────────────────────────────────────────────────────

def _farthest_point_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Greedy farthest-point sampling: n titik paling tersebar geografis."""
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


def build_candidate_pool(cfg, fac, depot: int):
    """
    Bangun pool titik tetap yang TERSEBAR MERATA di Surabaya (eliminasi data
    tak perlu). Untuk tiap grup di cfg.POOL_GROUPS, pilih `count` titik paling
    tersebar geografis (farthest-point sampling). Default: 60 emergency + 60
    transport = 120 titik tetap yang bisa dikunjungi.

    Returns
    -------
    pool_nodes : List[int]
    labels     : Dict[node -> str]
    coords     : Dict[node -> (lat, lon)]
    node_cat   : Dict[node -> str]   nama grup ("emergency" / "transport")
    """
    pool_nodes: List[int] = []
    labels: Dict[int, str] = {}
    coords: Dict[int, tuple] = {}
    node_cat: Dict[int, str] = {}

    for group, spec in cfg.POOL_GROUPS.items():
        mask = pd.Series(False, index=fac.index)
        if spec.get("match_categories"):
            mask = mask | fac["category"].isin(spec["match_categories"])
        if spec.get("match_facility_types"):
            mask = mask | fac["facility_type"].isin(spec["match_facility_types"])

        sub = fac[mask].dropna(subset=["nearest_node"]).copy()
        sub["nearest_node"] = sub["nearest_node"].astype(int)
        sub = sub[sub["nearest_node"] != depot]
        # buang node yang sudah dipakai grup lain (no duplikat antar grup)
        sub = sub[~sub["nearest_node"].isin(pool_nodes)]

        # Dedup per node, prioritaskan yang punya nama
        sub["_named"] = sub["name"].notna() & (sub["name"].astype(str).str.strip() != "")
        sub = (sub.sort_values("_named", ascending=False)
                  .drop_duplicates(subset="nearest_node")
                  .reset_index(drop=True))

        count  = int(spec.get("count", len(sub)))
        before = len(sub)
        sub = _farthest_point_sample(sub, count)
        log.info(f"Pool[{group}]: {len(sub)}/{before} titik (target {count}) "
                 f"tersebar merata via farthest-point")

        for _, r in sub.iterrows():
            node = int(r["nearest_node"])
            if node in labels:
                continue
            name = str(r["name"]).strip() if r.get("_named") else ""
            labels[node] = name if name and name != "nan" else str(r["facility_type"])
            coords[node] = (float(r["lat"]), float(r["lon"]))
            node_cat[node] = group
            pool_nodes.append(node)

    log.info(f"Pool total: {len(pool_nodes)} titik tetap "
             f"({', '.join(f'{g}={sum(1 for c in node_cat.values() if c==g)}' for g in cfg.POOL_GROUPS)})")
    return pool_nodes, labels, coords, node_cat


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def run_multi_vehicle_training(cfg):
    import geopandas as gpd
    import osmnx as ox
    from src.routing.algorithms import (
        GeneticAlgorithm, AntColonyElitePro,
        GeraldSimulatedAnnealing, ParticleSwarmRouting,
    )

    # ── Load data ────────────────────────────────────────────
    fac_path = cfg.DATA_DIR / "facilities_with_network.geojson"
    net_path = cfg.DATA_DIR / "road_network.graphml"
    if not fac_path.exists() or not net_path.exists():
        raise FileNotFoundError("Data hilang. Jalankan 'python main.py extract' dulu.")

    log.info("Loading road network and facilities...")
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

    depot = int(cfg.DEPOT_NODE)
    if depot not in G:
        raise ValueError(f"DEPOT_NODE {depot} tidak ada di graph.")
    log.info(f"Graph: {G.number_of_nodes()} nodes | Depot: {depot}")

    pool_nodes, labels, coords, node_cat = build_candidate_pool(cfg, fac, depot)
    coords[depot] = (float(G.nodes[depot]["y"]), float(G.nodes[depot]["x"]))
    labels[depot] = "DEPOT (Polda Jatim)"
    node_cat[depot] = "depot"

    # ── Armada (fleet) ──────────────────────────────────────
    fleet = [(vtype, unit)
             for vtype, count in cfg.FLEET.items()
             for unit in range(1, count + 1)]
    log.info(f"Fleet: {[f'{v}#{u}' for v, u in fleet]}")

    # ── Precompute cost matrix per kendaraan (sekali) ───────
    base_problems: Dict[str, object] = {}
    for vtype in cfg.FLEET:
        vehicle = Vehicle.from_settings(cfg, vtype)
        stats = width_filter_stats(G, vehicle, cfg.WIDTH_MISSING_PASSABLE)
        log.info(f"Width filter [{vtype}]: {stats['passable']}/{stats['total_edges']} "
                 f"edge lolos ({stats['passable_pct']}%), {stats['blocked']} terblok")
        G_v = filter_graph_by_width(G, vehicle, cfg.WIDTH_MISSING_PASSABLE)
        log.info(f"Precompute cost matrix [{vtype}]...")
        t0 = time.perf_counter()
        base_problems[vtype] = build_problem(
            G_v, depot, pool_nodes,
            budget_s=cfg.SHIFT_SECONDS, service_s=cfg.SERVICE_TIME_S,
            vehicle=vehicle, labels=labels, coords=coords, node_cat=node_cat,
        )
        log.info(f"  done in {time.perf_counter()-t0:.1f}s")

    # ── Models ──────────────────────────────────────────────
    models = [GeneticAlgorithm(), AntColonyElitePro(),
              GeraldSimulatedAnnealing(), ParticleSwarmRouting()]

    rows: List[dict] = []          # per (model, iter, vehicle, shift)
    # best_run[(model, vtype, unit)] = {"total": int, "iter": int, "shifts": [...]}
    best_run: Dict[tuple, dict] = {}

    n_iter = int(getattr(cfg, "TRAINING_ITERATIONS", 10))
    log.info(f"\nTraining: {len(models)} model × {len(fleet)} kendaraan "
             f"× {n_iter} iterasi × {cfg.N_SHIFTS} shift")

    for model in models:
        log.info(f"\n=== MODEL: {model.name} ===")
        for it in range(1, n_iter + 1):
            visited_global: set = set()
            for vtype, unit in fleet:
                base = base_problems[vtype]
                vehicle = base.vehicle
                shift_results = []
                vehicle_visited: set = set()
                veh_row_start = len(rows)   # pointer: rows added for this vehicle this iter

                for shift in range(1, cfg.N_SHIFTS + 1):
                    excl = set(visited_global) | vehicle_visited
                    prob = base.with_exclude(excl)
                    prob.shift = shift
                    seed = it * 10000 + unit * 100 + shift * 10
                    scen = f"{vtype}{unit}_s{shift}"
                    res = model.safe_orienteering(prob, scenario_name=scen, seed=seed)

                    vnodes = set(res.metadata.get("visited_nodes", []))
                    vehicle_visited |= vnodes
                    shift_results.append(res)

                    dist_km = res.total_distance_m / 1000.0 if res.found else 0.0
                    refills = max(0, math.ceil(dist_km / vehicle.range_km) - 1) if dist_km else 0
                    rows.append({
                        "model":          model.name,
                        "iteration":      it,
                        "vehicle":        vtype,
                        "vehicle_unit":   unit,
                        "shift":          shift,
                        "visited_count":  res.metadata.get("visited_count", 0),
                        "travel_min":     round(res.metadata.get("travel_time_s", 0) / 60, 1),
                        "service_min":    round(res.metadata.get("service_s_total", 0) / 60, 1),
                        "total_min":      round(res.metadata.get("total_time_s", 0) / 60, 1),
                        "budget_min":     round(cfg.SHIFT_SECONDS / 60, 1),
                        "distance_km":    round(dist_km, 2),
                        "refills_est":    refills,
                        "computation_ms": round(res.computation_ms, 1),
                        # time_feasible sementara; diperbarui setelah vehicle_total diketahui
                        "feasible":       res.metadata.get("feasible", res.found),
                    })

                vehicle_total = len(vehicle_visited)
                visited_global |= vehicle_visited

                # Feasible = time budget OK  AND  vehicle mencapai target harian (55 titik)
                # Kedua syarat harus terpenuhi agar shift dianggap feasible.
                vehicle_target_met = vehicle_total >= cfg.MIN_POINTS_TARGET
                for row in rows[veh_row_start:]:
                    row["feasible"] = bool(row["feasible"]) and vehicle_target_met

                key = (model.name, vtype, unit)
                if key not in best_run or vehicle_total > best_run[key]["total"]:
                    shifts = [_shift_payload(r) for r in shift_results]
                    for sh in shifts:
                        sh["feasible"] = bool(sh["feasible"]) and vehicle_target_met
                    best_run[key] = {
                        "total":   vehicle_total,
                        "iter":    it,
                        "vehicle": vtype,
                        "unit":    unit,
                        "shifts":  shifts,
                    }

            fleet_total   = len(visited_global)
            fleet_target  = cfg.MIN_POINTS_TARGET * len(fleet)   # 55 × 2 = 110
            fleet_status  = "✓" if fleet_total >= fleet_target else f"✗ (butuh {fleet_target})"
            log.info(f"  iter {it:>2}: fleet total = {fleet_total} titik {fleet_status}")

    # ── Save CSV ─────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = cfg.DATA_DIR / "training_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nSaved -> {csv_path.name}")

    # ── Summary per (model, vehicle) ─────────────────────────
    summary = _summarise(df, cfg)
    summary.to_csv(cfg.DATA_DIR / "training_summary.csv", index=False)
    log.info(f"Saved -> training_summary.csv")
    log.info(f"\nTRAINING SUMMARY (titik per kendaraan, 2 shift)\n{'='*60}\n"
             f"{summary.to_string(index=False)}")

    # ── Analisis feasibility (kondisi mencapai target) ───────
    # Selalu dihitung — dipakai untuk dashboard (Feasibility Roadmap) & txt report.
    from src.routing.insight import (compute_feasibility_conditions,
                                     generate_insight_report)
    feasibility = compute_feasibility_conditions(cfg, base_problems, summary)
    generate_insight_report(cfg, base_problems, summary, data=feasibility)

    # ── Training log JSON (untuk viewer) ─────────────────────
    _write_training_log(cfg, best_run, depot, coords, labels, node_cat, df,
                        feasibility)

    # ── Route viewer ─────────────────────────────────────────
    from src.routing.route_viewer import build_route_viewer
    build_route_viewer(cfg.DATA_DIR)

    log.info("\nOutputs in data/:")
    log.info("  training_results.csv   per (model, iterasi, kendaraan, shift)")
    log.info("  training_summary.csv   ringkasan best/mean/std per (model, kendaraan)")
    log.info("  training_log.json      rute terbaik untuk viewer")
    log.info("  route_viewer.html      dropdown rute (kendaraan/shift/stop) ala Google Maps")
    return df


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _shift_payload(res) -> dict:
    """Ringkas RouteResult satu shift menjadi data untuk viewer/JSON."""
    m = res.metadata
    return {
        "shift":         m.get("shift"),
        "visited_count": m.get("visited_count", 0),
        "visited_stops": m.get("visited_stops", []),
        "visited_nodes": m.get("visited_nodes", []),
        "travel_min":    round(m.get("travel_time_s", 0) / 60, 1),
        "service_min":   round(m.get("service_s_total", 0) / 60, 1),
        "total_min":     round(m.get("total_time_s", 0) / 60, 1),
        "distance_km":   round((res.total_distance_m or 0) / 1000, 2),
        "feasible":      m.get("feasible", res.found),
        "route_coords":  m.get("route_coords", []),
        "legs":          m.get("legs", []),
        "gen_history":   m.get("gen_history", []),
    }


def _summarise(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Ringkasan per (model, vehicle): jumlah titik per kendaraan (gabung 2 shift)
    diagregasi best/mean/std lintas iterasi.
    """
    per_run = (df.groupby(["model", "vehicle", "vehicle_unit", "iteration"])
                 ["visited_count"].sum().reset_index())
    rows = []
    target = cfg.MIN_POINTS_TARGET
    for (model, vehicle), g in per_run.groupby(["model", "vehicle"]):
        totals = g["visited_count"].values
        rows.append({
            "model":        model,
            "vehicle":      vehicle,
            "best_total":   int(totals.max()),
            "mean_total":   round(float(totals.mean()), 1),
            "std_total":    round(float(totals.std(ddof=0)), 1),
            "worst_total":  int(totals.min()),
            "target":       target,
            "meets_target": bool(totals.max() >= target),
        })
    return pd.DataFrame(rows).sort_values(["vehicle", "best_total"],
                                          ascending=[True, False])


def _write_training_log(cfg, best_run: dict, depot: int, coords: dict,
                        labels: dict, node_cat: dict, df: pd.DataFrame,
                        feasibility: dict | None = None):
    """Tulis training_log.json: rute, pool, dan dashboard performa untuk viewer."""
    dashboard = _build_dashboard(cfg, df, best_run)
    if feasibility is not None:
        dashboard["feasibility"] = feasibility
    payload = {
        "generated":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "depot":       {"node": depot, "label": labels.get(depot, "DEPOT"),
                        "coord": list(coords.get(depot, (0, 0)))},
        "target":      cfg.MIN_POINTS_TARGET,
        "shift_min":   round(cfg.SHIFT_SECONDS / 60, 1),
        "service_min": round(cfg.SERVICE_TIME_S / 60, 1),
        "n_iterations": int(getattr(cfg, "TRAINING_ITERATIONS", 10)),
        # semua titik pool (untuk overlay marker + legenda kategori + tab Peta Titik)
        "pool": [
            {"node": n, "label": labels.get(n, str(n)),
             "coord": list(coords.get(n, (0, 0))), "cat": node_cat.get(n, "")}
            for n in coords if n != depot
        ],
        "runs":        [],
        "dashboard":   dashboard,
    }
    for (model, vtype, unit), info in sorted(best_run.items()):
        payload["runs"].append({
            "model":         model,
            "vehicle":       vtype,
            "vehicle_unit":  unit,
            "best_iter":     info["iter"],
            "vehicle_total": info["total"],
            "shifts":        info["shifts"],
        })
    out = cfg.DATA_DIR / "training_log.json"
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log.info(f"Saved -> {out.name}")


# ──────────────────────────────────────────────────────────────
# Dashboard metrics (leaderboard, konvergensi, insight)
# ──────────────────────────────────────────────────────────────

# Penamaan & identitas tampilan tiap model (bukan nama unit-test).
_MODEL_META = {
    "ga":             ("Genetic Algorithm", "Evolusi populasi rute", "🧬", "#e5484d"),
    "aco_elite_pro":  ("Ant Colony Elite Pro", "Koloni semut + feromon MMAS", "🐜", "#f59e0b"),
    "gerald_sa":      ("Simulated Annealing", "Pendinginan probabilistik", "🔥", "#8b5cf6"),
    "particle_swarm": ("Particle Swarm", "Kawanan partikel", "🐝", "#2563eb"),
}


def _build_dashboard(cfg, df: pd.DataFrame, best_run: dict) -> dict:
    """
    Hitung metrik performa kaya per model (objektif utama = jumlah titik):
    coverage, konsistensi, pencapaian target, success rate, throughput,
    waktu komputasi, utilisasi waktu, kecepatan konvergensi, skor & ranking
    gaya leaderboard, kurva konvergensi, dan performa per-iterasi.
    """
    target     = cfg.MIN_POINTS_TARGET
    budget_min = cfg.SHIFT_SECONDS / 60
    n_iter     = int(df["iteration"].max()) if not df.empty else 0

    # Agregasi tingkat "vehicle-run" (gabung 2 shift per kendaraan per iterasi)
    runs = (df.groupby(["model", "vehicle", "vehicle_unit", "iteration"])
              .agg(total_visited=("visited_count", "sum"),
                   total_time=("total_min", "sum"))
              .reset_index())

    models = []
    for name, g in runs.groupby("model"):
        cov          = g["total_visited"].astype(float)
        coverage_avg = float(cov.mean())
        coverage_std = float(cov.std(ddof=0))
        consistency  = max(0.0, min(100.0,
                       (1 - (coverage_std / coverage_avg if coverage_avg else 1)) * 100))
        sub          = df[df["model"] == name]
        mean_total   = float(g["total_time"].mean())
        throughput   = coverage_avg / (mean_total / 60) if mean_total > 0 else 0.0

        # Per-iteration detail (for viewer dropdown: runtime history + convergence)
        _sub_veh = (sub.groupby(["vehicle", "vehicle_unit", "iteration"])
                       .agg(visited=("visited_count", "sum"),
                            runtime_ms=("computation_ms", "mean"),
                            distance_km=("distance_km", "sum"),
                            travel_min=("travel_min", "sum"),
                            service_min=("service_min", "sum"),
                            total_min=("total_min", "sum"),
                            feasible=("feasible", "mean"))
                       .reset_index())
        per_iter_detail = []
        for _it in range(1, n_iter + 1):
            _g = _sub_veh[_sub_veh["iteration"] == _it]
            if not _g.empty:
                _cov   = float(_g["visited"].mean())
                _tmin  = float(_g["total_min"].mean())
                _dist  = float(_g["distance_km"].sum())
                _trav  = float(_g["travel_min"].sum())
                _svc   = float(_g["service_min"].sum())
                _vis   = float(_g["visited"].sum())
                per_iter_detail.append({
                    "iter":        _it,
                    "coverage":    round(_cov, 1),
                    "runtime_ms":  round(float(_g["runtime_ms"].mean()), 1),
                    "distance_km": round(float(_g["distance_km"].mean()), 2),
                    "total_min":   round(_tmin, 1),
                    "speed_kph":   round(_dist / (_trav / 60), 1) if _trav > 0 else 0.0,
                    "throughput":  round(_cov / (_tmin / 60), 2) if _tmin > 0 else 0.0,
                    "spatial_eff": round(_vis / _dist, 2) if _dist > 0 else 0.0,
                    "service_ratio_pct": round(_svc / (_svc + _trav) * 100, 1) if (_svc + _trav) > 0 else 0.0,
                    "time_util_pct": round(_tmin / budget_min * 100, 1) if budget_min else 0.0,
                    "feasible_pct": round(float(_g["feasible"].mean() * 100), 1),
                })
            else:
                per_iter_detail.append({
                    "iter": _it, "coverage": 0.0, "runtime_ms": 0.0,
                    "distance_km": 0.0, "total_min": 0.0, "speed_kph": 0.0,
                    "throughput": 0.0, "spatial_eff": 0.0, "service_ratio_pct": 0.0,
                    "time_util_pct": 0.0, "feasible_pct": 0.0,
                })

        # Per-shift detail granular (identitas: kendaraan/unit/shift per iterasi)
        per_shift_detail = [
            {"iter":        int(r["iteration"]),
             "vehicle":     str(r["vehicle"]),
             "unit":        int(r["vehicle_unit"]),
             "shift":       int(r["shift"]),
             "visited":     int(r["visited_count"]),
             "travel_min":  round(float(r["travel_min"]), 1),
             "total_min":   round(float(r["total_min"]), 1),
             "distance_km": round(float(r["distance_km"]), 2),
             "runtime_ms":  round(float(r["computation_ms"]), 1),
             "feasible":    bool(r["feasible"])}
            for _, r in sub.sort_values(
                ["iteration", "vehicle", "vehicle_unit", "shift"]).iterrows()
        ]

        # Konvergensi: best vehicle-run model ini, gen_history SEMUA shift + identitas
        conv_by_shift, conv_meta, best_total = {}, {}, -1
        for (m, vt, u), info in best_run.items():
            if m == name and info["total"] > best_total:
                best_total = info["total"]
                conv_meta = {"vehicle": vt, "unit": int(u), "iter": int(info["iter"]),
                             "vehicle_total": int(info["total"])}
                conv_by_shift = {}
                for _sh in info.get("shifts", []):
                    frames = [{"x": fr["gen"], "visited": fr["visited"],
                               "total_min": fr.get("total_min", 0),
                               "travel_min": fr.get("travel_min", 0)}
                              for fr in _sh.get("gen_history", [])]
                    if not frames:
                        continue
                    gmax = frames[-1]["x"] or 1
                    for c in frames:
                        c["progress"] = round(c["x"] / gmax * 100, 1)
                    conv_by_shift[str(_sh.get("shift"))] = frames
        # legacy/headline curve = shift-1 (atau shift pertama yang punya data)
        conv = conv_by_shift.get("1") or next(iter(conv_by_shift.values()), [])
        conv_speed = 0.0
        if conv:
            final = conv[-1]["visited"] or 1
            gen95 = next((c["progress"] for c in conv if c["visited"] >= 0.95 * final), 100)
            conv_speed = round(100 - gen95, 1)

        per_iter = (g.groupby("iteration")["total_visited"].mean()
                     .reindex(range(1, n_iter + 1)).fillna(0).round(2).tolist())

        # Jarak, kecepatan, efisiensi spasial & rasio layanan (agregat)
        _veh_dist     = sub.groupby(["vehicle", "vehicle_unit", "iteration"])["distance_km"].sum()
        avg_dist_km   = float(_veh_dist.mean()) if len(_veh_dist) else 0.0
        _tot_dist     = float(sub["distance_km"].sum())
        _tot_travel_h = float(sub["travel_min"].sum()) / 60
        _tot_svc      = float(sub["service_min"].sum())
        _tot_trav     = float(sub["travel_min"].sum())
        avg_speed_kph = _tot_dist / _tot_travel_h if _tot_travel_h > 0 else 0.0
        spatial_eff   = float(sub["visited_count"].sum()) / _tot_dist if _tot_dist > 0 else 0.0
        service_ratio = _tot_svc / (_tot_svc + _tot_trav) * 100 if (_tot_svc + _tot_trav) > 0 else 0.0

        # Rata-rata jarak & waktu per kendaraan-run, dipisah motor vs mobil
        by_vehicle = {}
        for _vt, _vg in sub.groupby("vehicle"):
            _runs_v = (_vg.groupby(["vehicle_unit", "iteration"])
                          .agg(dist=("distance_km", "sum"),
                               tmin=("total_min", "sum"),
                               vis=("visited_count", "sum")))
            by_vehicle[str(_vt)] = {
                "distance_km": round(float(_runs_v["dist"].mean()), 1),
                "total_min":   round(float(_runs_v["tmin"].mean()), 1),
                "visited":     round(float(_runs_v["vis"].mean()), 1),
            }

        dn, tagline, icon, color = _MODEL_META.get(name, (name, "", "⭐", "#64748b"))
        models.append({
            "name": name, "display_name": dn, "tagline": tagline,
            "icon": icon, "color": color,
            "coverage_avg":          round(coverage_avg, 1),
            "coverage_best":         int(cov.max()),
            "coverage_worst":        int(cov.min()),
            "coverage_std":          round(coverage_std, 2),
            "consistency_pct":       round(consistency, 1),
            "target_attainment_pct": round(coverage_avg / target * 100, 1),
            "success_rate_pct":      round(float((cov >= target).mean() * 100), 1),
            "feasible_pct":          round(float(sub["feasible"].mean() * 100), 1),
            "avg_runtime_ms":        round(float(sub["computation_ms"].mean()), 1),
            "throughput":            round(throughput, 2),
            "avg_distance_km":       round(avg_dist_km, 1),
            "avg_speed_kph":         round(avg_speed_kph, 1),
            "spatial_eff":           round(spatial_eff, 2),
            "service_ratio_pct":     round(service_ratio, 1),
            "by_vehicle":            by_vehicle,
            "time_util_pct":         round(float(sub["total_min"].mean()) / budget_min * 100, 1),
            "convergence_speed_pct": conv_speed,
            "per_iteration":         per_iter,
            "per_iter_detail":       per_iter_detail,
            "per_shift_detail":      per_shift_detail,
            "convergence":           conv,
            "convergence_by_shift":  conv_by_shift,
            "convergence_meta":      conv_meta,
        })

    # Skor komposit (objektif utama = coverage, dibobot terbesar)
    cov_vals = [m["coverage_avg"] for m in models] or [1]
    tp_vals  = [m["throughput"]   for m in models] or [1]
    cmin, cmax = min(cov_vals), max(cov_vals)
    tmin, tmax = min(tp_vals),  max(tp_vals)
    norm  = lambda v, lo, hi: (v - lo) / (hi - lo) if hi > lo else 1.0
    for m in models:
        m["overall_score"] = round((
            0.55 * norm(m["coverage_avg"], cmin, cmax) +
            0.20 * (m["consistency_pct"] / 100) +
            0.15 * norm(m["throughput"], tmin, tmax) +
            0.10 * (m["success_rate_pct"] / 100)
        ) * 100, 1)

    models.sort(key=lambda m: m["overall_score"], reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for i, m in enumerate(models):
        m["rank"]  = i + 1
        m["medal"] = medals[i] if i < 3 else ""
        s = m["overall_score"]
        m["tier"] = "S" if s >= 85 else "A" if s >= 70 else "B" if s >= 55 else "C"

    return {
        "models":       models,
        "insights":     _dashboard_insights(models, df, target),
        "metric_guide": _metric_guide(target),
        "n_iterations": n_iter,
        "target":       target,
    }


def _dashboard_insights(models, df: pd.DataFrame, target: int) -> list:
    """Narasi insight bermakna (Indonesia) — performa antar model & iterasi."""
    if not models:
        return ["Tidak ada data model."]
    out = []
    top = models[0]
    out.append(f"🏆 <b>{top['display_name']}</b> memimpin leaderboard (skor {top['overall_score']}) "
               f"dengan rata-rata <b>{top['coverage_avg']} titik/kendaraan</b> "
               f"— {top['target_attainment_pct']}% dari target {target}.")
    mc = max(models, key=lambda m: m["consistency_pct"])
    out.append(f"🎯 Paling stabil antar 10 iterasi: <b>{mc['display_name']}</b> "
               f"(konsistensi {mc['consistency_pct']}%, deviasi cuma {mc['coverage_std']} titik).")
    fast = min(models, key=lambda m: m["avg_runtime_ms"])
    out.append(f"⚡ Komputasi tercepat: <b>{fast['display_name']}</b> "
               f"({fast['avg_runtime_ms']} ms per shift).")
    peak = max(models, key=lambda m: m["coverage_best"])
    out.append(f"📈 Rekor tertinggi: <b>{peak['display_name']}</b> mencapai "
               f"<b>{peak['coverage_best']} titik</b> dalam satu kendaraan.")
    conv = max(models, key=lambda m: m["convergence_speed_pct"])
    out.append(f"🚀 Konvergensi tercepat: <b>{conv['display_name']}</b> "
               f"(mencapai 95% kualitas solusi paling dini).")
    try:
        veh = (df.groupby(["model", "vehicle", "vehicle_unit", "iteration"])["visited_count"]
                 .sum().reset_index().groupby("vehicle")["visited_count"].mean())
        mo, mb = veh.get("motor"), veh.get("mobil")
        if mo is not None and mb is not None:
            out.append(f"🛣️ Jangkauan rata-rata: motor <b>{mo:.1f}</b> titik vs mobil "
                       f"<b>{mb:.1f}</b> titik — mobil terblok di jalan sempit (constraint lebar).")
    except Exception:
        pass
    best_cov = max(m["coverage_best"] for m in models)
    if best_cov < target:
        out.append(f"⚠️ Belum ada model mencapai target {target} titik (terbaik {best_cov}, "
                   f"kurang {target - best_cov}). Lihat <b>insight_report.txt</b> "
                   f"untuk kondisi (kecepatan/service/jam/armada) yang bisa mencapainya.")
    else:
        out.append(f"✅ Target {target} titik tercapai oleh setidaknya satu model.")
    return out


def _metric_guide(target: int) -> list:
    """Penjelasan tiap indikator supaya bermakna (bukan jargon unit-test)."""
    return [
        {"label": "Coverage (Titik Terkunjungi)",
         "desc": f"Objektif UTAMA: rata-rata titik unik yang dikunjungi satu kendaraan dalam 2 shift. Makin tinggi makin baik (target {target})."},
        {"label": "Pencapaian Target",
         "desc": f"Persentase coverage terhadap target {target} titik/kendaraan."},
        {"label": "Konsistensi",
         "desc": "Kestabilan hasil antar 10 iterasi — 100% berarti nyaris tanpa variasi."},
        {"label": "Success Rate",
         "desc": "Persentase run yang mencapai/melebihi target."},
        {"label": "Efisiensi (titik/jam)",
         "desc": "Throughput: titik dikunjungi per jam kerja — efisiensi rute + waktu layanan."},
        {"label": "Waktu Komputasi",
         "desc": "Rata-rata waktu CPU per shift untuk menemukan solusi (ms). Lebih kecil lebih cepat."},
        {"label": "Kecepatan Konvergensi",
         "desc": "Seberapa dini algoritma mencapai 95% kualitas solusinya selama pencarian."},
        {"label": "Feasibility",
         "desc": "Persentase shift yang memenuhi semua constraint (budget 6 jam + lebar jalan)."},
        {"label": "Utilisasi Waktu",
         "desc": "Seberapa penuh jam kerja shift terpakai (travel + service vs 6 jam)."},
        {"label": "Skor Keseluruhan",
         "desc": "Gabungan berbobot: 55% coverage + 20% konsistensi + 15% efisiensi + 10% success rate."},
    ]
