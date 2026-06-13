"""
src/routing/width.py
=====================
Constraint lebar jalan untuk Team Orienteering Problem.

Aturan
------
Kalau lebar jalan (edge `width`, meter) <= lebar kendaraan (`vehicle.width_m`),
maka edge itu TIDAK bisa dilewati kendaraan tersebut.

Pendekatan
----------
Lebar jalan memengaruhi routing secara langsung (bukan post-processing), jadi
kita bangun *edge-subgraph view* per kendaraan: edge yang terlalu sempit dibuang
sebelum Dijkstra/algoritma dijalankan. Motor (0.8 m) bisa lewat hampir semua
jalan; mobil (1.8 m) terblok di gang/jalan sempit.

Format data width OSM heterogen:
    "4"            -> 4.0
    "3.5"          -> 3.5
    "4 m"          -> 4.0
    ['4', '7']     -> 4.0  (ambil MINIMUM — segmen tersempit yang mengikat)
    "['3', '2']"   -> 2.0  (string repr list, dari graphml)
Edge tanpa data width (~27%): ikut flag missing_passable.
"""

import ast
import logging
import re
from typing import Optional

import networkx as nx

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_width(raw) -> Optional[float]:
    """
    Parse nilai width OSM heterogen menjadi float meter.

    Untuk way dengan beberapa nilai width (list), ambil MINIMUM karena
    segmen tersempit yang menentukan apakah kendaraan bisa lewat.

    Returns
    -------
    float (meter) atau None kalau tidak terparse.
    """
    if raw is None:
        return None

    # Sudah list/tuple Python (osmnx kadang mengembalikan list)
    if isinstance(raw, (list, tuple)):
        vals = [parse_width(x) for x in raw]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None

    s = str(raw).strip()
    if not s:
        return None

    # String repr dari list, mis. "['3', '2']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            return parse_width(parsed)
        except (ValueError, SyntaxError):
            pass  # fallback ke regex di bawah

    # Ambil semua angka di string (handle "4 m", "3.5m", "approx 4")
    nums = [float(m) for m in _NUM_RE.findall(s)]
    nums = [n for n in nums if n > 0]
    return min(nums) if nums else None


def _edge_passable(data: dict, vehicle_width_m: float, missing_passable: bool) -> bool:
    """True kalau edge bisa dilewati kendaraan dengan lebar vehicle_width_m."""
    w = parse_width(data.get("width"))
    if w is None:
        return missing_passable
    return w > vehicle_width_m


def filter_graph_by_width(G: nx.MultiDiGraph,
                          vehicle,
                          missing_passable: bool = True) -> nx.MultiDiGraph:
    """
    Kembalikan graph yang hanya berisi edge yang bisa dilewati kendaraan.

    Edge dibuang kalau parse_width(width) <= vehicle.width_m. Edge tanpa data
    width ikut `missing_passable`. Node yang jadi terisolasi (semua jalannya
    terlalu sempit) ikut hilang — ini benar: kendaraan memang tak bisa ke sana.

    Di-materialkan sebagai copy nyata (bukan lazy view) supaya Dijkstra
    precompute yang dipanggil ratusan kali per kendaraan tetap cepat
    (tanpa overhead callback filter per-edge).
    """
    vw = float(vehicle.width_m)
    keep_edges = [
        (u, v, k)
        for u, v, k, data in G.edges(keys=True, data=True)
        if _edge_passable(data, vw, missing_passable)
    ]
    return G.edge_subgraph(keep_edges).copy()


def width_filter_stats(G: nx.MultiDiGraph,
                       vehicle,
                       missing_passable: bool = True) -> dict:
    """
    Statistik berapa edge yang lolos/terblok untuk kendaraan ini.
    Dipakai untuk logging/QA.
    """
    vw = float(vehicle.width_m)
    total = blocked = no_data = 0
    for _, _, data in G.edges(data=True):
        total += 1
        w = parse_width(data.get("width"))
        if w is None:
            no_data += 1
            if not missing_passable:
                blocked += 1
        elif w <= vw:
            blocked += 1
    passable = total - blocked
    return {
        "vehicle":      vehicle.vehicle_type,
        "width_m":      vw,
        "total_edges":  total,
        "passable":     passable,
        "blocked":      blocked,
        "no_width_data": no_data,
        "passable_pct": round(100.0 * passable / total, 1) if total else 0.0,
    }
