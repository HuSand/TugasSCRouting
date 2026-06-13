"""
src/routing/fuel.py
===================
Post-processing modul untuk fuel constraint.

Cara kerja
----------
1. Algoritma (GA/SA/ACO/PSO) selesai → menghasilkan rute biasa.
2. inject_fuel_stops() dipanggil dengan rute + daftar node SPBU.
3. Fungsi menelusuri rute leg per leg:
   - Kalau sisa fuel cukup → lanjut.
   - Kalau tidak cukup    → cari SPBU paling "searah" (minimum detour),
                             sisipkan ke rute, reset fuel.
4. Return FuelResult berisi rute baru + statistik fuel.

Pemilihan SPBU
--------------
Bukan sekadar SPBU terdekat, tapi SPBU dengan detour cost terkecil:

    detour(spbu) = dist(current → spbu) + dist(spbu → next_stop)

Ini memilih SPBU yang paling "searah" menuju stop berikutnya,
bukan hanya yang paling dekat dari posisi sekarang.

Edge case yang di-handle
-------------------------
- Satu leg > full tank   : loop injeksi SPBU berulang dalam leg yang sama.
- Tidak ada SPBU reachable: fallback ke SPBU terdekat secara euclidean.
- SPBU sudah ada di rute  : tetap disinggahi ulang (re-fuel).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# FuelResult
# ──────────────────────────────────────────────────────────────

@dataclass
class FuelResult:
    """
    Output dari inject_fuel_stops().

    Attributes
    ----------
    route            : List[int]  — rute final (sudah termasuk node SPBU)
    fuel_stops       : List[int]  — node SPBU yang disisipkan (bisa duplikat
                                    kalau SPBU yang sama dikunjungi 2x)
    refill_count     : int        — jumlah pengisian bensin
    fuel_remaining_km: float      — estimasi sisa range saat tiba di tujuan akhir
    vehicle_type     : str        — "motor" / "mobil"
    feasible         : bool       — False kalau ada leg yang tidak bisa diselesaikan
    warning          : str        — pesan warning kalau feasible=False
    """
    route:             List[int]
    fuel_stops:        List[int]
    refill_count:      int
    fuel_remaining_km: float
    vehicle_type:      str
    feasible:          bool  = True
    warning:           str   = ""


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _edge_length_m(G: nx.MultiDiGraph, u: int, v: int) -> float:
    """
    Panjang edge terpendek (meter) antara u dan v.
    Return infinity kalau edge tidak ada.
    """
    data = G.get_edge_data(u, v)
    if data is None:
        return float("inf")
    best = min(data.values(), key=lambda d: float(d.get("length", 999_999)))
    return float(best.get("length", float("inf")))


def _path_length_m(G: nx.MultiDiGraph, u: int, v: int) -> float:
    """
    Jarak jalur terpendek (meter) antara u dan v via Dijkstra.
    Dipakai untuk hitung detour cost SPBU.
    Return infinity kalau tidak ada jalur.
    """
    try:
        return nx.shortest_path_length(G, u, v, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def _euclidean_dist_m(G: nx.MultiDiGraph, u: int, v: int) -> float:
    """
    Jarak euclidean (meter) antara dua node berdasarkan koordinat lat/lon.
    Dipakai sebagai fallback kalau graph tidak terhubung.
    """
    nu = G.nodes.get(u, {})
    nv = G.nodes.get(v, {})
    if not nu or not nv:
        return float("inf")
    lat1, lon1 = float(nu.get("y", 0)), float(nu.get("x", 0))
    lat2, lon2 = float(nv.get("y", 0)), float(nv.get("x", 0))
    # Haversine-lite (cukup akurat untuk jarak pendek di Surabaya)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
           math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 6_371_000 * 2 * math.asin(math.sqrt(a))


def _find_best_spbu(
    G:            nx.MultiDiGraph,
    current_node: int,
    next_node:    int,
    spbu_nodes:   List[int],
    max_candidates: int = 10,
) -> Optional[int]:
    """
    Pilih SPBU dengan detour cost terkecil.

    Detour cost = dist(current → spbu) + dist(spbu → next_stop)
    SPBU yang paling "searah" menuju next_stop akan punya detour terkecil.

    Parameters
    ----------
    current_node    : node posisi kendaraan sekarang
    next_node       : node stop tujuan berikutnya
    spbu_nodes      : list semua node SPBU di Surabaya
    max_candidates  : batasi kandidat ke N SPBU terdekat secara euclidean
                      sebelum hitung Dijkstra (supaya tidak terlalu lambat)

    Returns
    -------
    node ID SPBU terbaik, atau None kalau tidak ada SPBU reachable.
    """
    if not spbu_nodes:
        return None

    # ── 1. Pre-filter: ambil N kandidat terdekat secara euclidean ──
    candidates = sorted(
        spbu_nodes,
        key=lambda s: _euclidean_dist_m(G, current_node, s),
    )[:max_candidates]

    # ── 2. Hitung detour cost via Dijkstra untuk tiap kandidat ──
    best_spbu   = None
    best_detour = float("inf")

    for spbu in candidates:
        d_to_spbu   = _path_length_m(G, current_node, spbu)
        d_from_spbu = _path_length_m(G, spbu, next_node)

        if d_to_spbu == float("inf") or d_from_spbu == float("inf"):
            continue  # SPBU tidak reachable, skip

        detour = d_to_spbu + d_from_spbu
        if detour < best_detour:
            best_detour = detour
            best_spbu   = spbu

    # ── 3. Fallback: kalau tidak ada yang reachable, ambil euclidean nearest ──
    if best_spbu is None:
        log.warning("No reachable SPBU found via Dijkstra, falling back to euclidean nearest")
        best_spbu = min(
            spbu_nodes,
            key=lambda s: _euclidean_dist_m(G, current_node, s),
        )

    return best_spbu


def _path_between(G: nx.MultiDiGraph, u: int, v: int) -> List[int]:
    """
    Shortest path (list of nodes) dari u ke v. Return [u, v] kalau gagal.
    """
    try:
        return nx.shortest_path(G, u, v, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [u, v]


def _route_total_length_m(G: nx.MultiDiGraph, route: List[int]) -> float:
    """Total panjang rute (meter) dari list node."""
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        total += _edge_length_m(G, u, v)
    return total


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def inject_fuel_stops(
    G:           nx.MultiDiGraph,
    route:       List[int],
    spbu_nodes:  List[int],
    vehicle,                     # Vehicle dataclass dari base.py
    verbose:     bool = False,
) -> FuelResult:
    """
    Post-process rute hasil algoritma dengan menyisipkan stop SPBU
    ketika sisa bahan bakar tidak cukup untuk melanjutkan ke stop berikutnya.

    Parameters
    ----------
    G           : graph jalan Surabaya
    route       : list node hasil algoritma (source → ... → target)
    spbu_nodes  : list node SPBU yang tersedia di Surabaya
    vehicle     : Vehicle dataclass (dari base.py)
    verbose     : kalau True, log setiap keputusan fuel

    Returns
    -------
    FuelResult dengan rute baru yang sudah mengandung node SPBU.
    """
    if len(route) < 2:
        return FuelResult(
            route=route, fuel_stops=[], refill_count=0,
            fuel_remaining_km=vehicle.range_km,
            vehicle_type=vehicle.vehicle_type,
        )

    range_km      = vehicle.range_km
    threshold_km  = vehicle.refill_threshold_km
    fuel_remaining = range_km          # mulai dengan tangki penuh (km)

    new_route:   List[int] = [route[0]]
    fuel_stops:  List[int] = []
    refill_count: int      = 0
    feasible:     bool     = True
    warning:      str      = ""

    MAX_SPBU_PER_LEG = 20  # guard: cegah infinite loop kalau range < 1 leg

    for i in range(len(route) - 1):
        current = route[i]
        nxt     = route[i + 1]

        # Hitung jarak leg ini (meter → km)
        leg_dist_m  = _path_length_m(G, current, nxt)
        leg_dist_km = leg_dist_m / 1000.0

        if leg_dist_km == float("inf"):
            # Leg tidak reachable, lanjut saja (algoritma mungkin punya bug)
            new_route.append(nxt)
            continue

        # ── Injeksi SPBU berulang kalau leg sangat panjang ──────────────
        spbu_inserted = 0
        pos           = current

        while fuel_remaining - leg_dist_km < threshold_km:
            # Sisa fuel tidak cukup (termasuk safety buffer)

            if spbu_inserted >= MAX_SPBU_PER_LEG:
                # Terlalu banyak SPBU dalam satu leg — sesuatu tidak beres
                warning = (
                    f"Leg {current}→{nxt} butuh >{MAX_SPBU_PER_LEG} SPBU. "
                    f"Mungkin range kendaraan terlalu kecil."
                )
                log.warning(warning)
                feasible = False
                break

            # Cari SPBU paling searah
            best_spbu = _find_best_spbu(G, pos, nxt, spbu_nodes)
            if best_spbu is None:
                warning = f"Tidak ada SPBU reachable dari node {pos}"
                log.warning(warning)
                feasible = False
                break

            # Hitung jarak dari posisi sekarang ke SPBU
            dist_to_spbu_km = _path_length_m(G, pos, best_spbu) / 1000.0

            if dist_to_spbu_km > fuel_remaining:
                # Bahkan ke SPBU pun tidak cukup bensin — infeasible
                warning = (
                    f"Bensin habis sebelum mencapai SPBU terdekat "
                    f"(sisa={fuel_remaining:.1f}km, "
                    f"jarak SPBU={dist_to_spbu_km:.1f}km)"
                )
                log.warning(warning)
                feasible = False
                break

            # Sisipkan path ke SPBU ke dalam rute
            path_to_spbu = _path_between(G, pos, best_spbu)
            new_route.extend(path_to_spbu[1:])  # skip duplikat head

            fuel_stops.append(best_spbu)
            refill_count += 1
            fuel_remaining = range_km  # isi penuh

            if verbose:
                log.info(
                    f"  ⛽ SPBU disisipkan: node {best_spbu} "
                    f"(leg {i}: {current}→{nxt}, "
                    f"sisa sebelum isi={fuel_remaining:.1f}km)"
                )

            # Update posisi dan sisa jarak leg
            pos          = best_spbu
            leg_dist_km  = _path_length_m(G, pos, nxt) / 1000.0
            spbu_inserted += 1

        # Tambahkan path dari posisi terakhir ke stop berikutnya
        path_to_next = _path_between(G, pos, nxt)
        new_route.extend(path_to_next[1:])

        # Kurangi fuel sesuai jarak yang ditempuh
        actual_km    = _path_length_m(G, pos, nxt) / 1000.0
        fuel_remaining = max(0.0, fuel_remaining - actual_km)

        if verbose:
            log.info(
                f"  Leg {i}: {current}→{nxt} "
                f"({actual_km:.1f}km) | "
                f"sisa fuel: {fuel_remaining:.1f}km"
            )

    return FuelResult(
        route             = new_route,
        fuel_stops        = fuel_stops,
        refill_count      = refill_count,
        fuel_remaining_km = fuel_remaining,
        vehicle_type      = vehicle.vehicle_type,
        feasible          = feasible,
        warning           = warning,
    )


def extract_spbu_nodes(fac_gdf) -> List[int]:
    """
    Ambil node SPBU dari GeoDataFrame fasilitas.
    SPBU di OSM punya facility_type = 'fuel' dan category = 'transport'.
    """
    mask = (
        (fac_gdf["facility_type"] == "fuel") |
        (fac_gdf["category"] == "transport")
    )
    spbu = fac_gdf[mask].dropna(subset=["nearest_node"]).copy()
    spbu["nearest_node"] = spbu["nearest_node"].astype(int)

    nodes = spbu["nearest_node"].unique().tolist()
    log.info(f"SPBU nodes available: {len(nodes)}")
    return nodes
