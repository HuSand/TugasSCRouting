"""
src/routing/insight.py
======================
Insight report: kalau model gagal mencapai target minimum titik (default 60
per kendaraan), dalam kondisi apa target itu bisa tercapai?

Lever yang dianalisis (satu per satu, dibanding baseline):
- service_time : durasi cek fasilitas per titik (10 → ... menit)
- speed        : kecepatan rata-rata (multiplier travel_time)
- shift_hours  : panjang shift / jam kerja
- jumlah kendaraan : untuk coverage TOTAL distinct armada

Memakai greedy insertion (deterministik, cepat — tanpa Dijkstra ulang): cukup
men-skala cost matrix yang sudah ada. Jadi sweep banyak skenario tetap ringan.
"""

import logging
from datetime import datetime

from src.routing.orienteering import OrienteeringProblem, _greedy_order

log = logging.getLogger(__name__)


def _scaled_problem(base, service_s, budget_s, speed_mult):
    """Copy problem dengan pair_cost di-skala kecepatan + budget/service baru."""
    pc = {k: v / speed_mult for k, v in base.pair_cost.items()}
    return OrienteeringProblem(
        G=base.G, depot=base.depot, pool_nodes=base.pool_nodes,
        budget_s=budget_s, service_s=service_s, pair_cost=pc,
        exclude=set(), labels=base.labels, coords=base.coords,
        vehicle=base.vehicle,
    )


def _sim_vehicle_total(base, service_s, budget_s, speed_mult, n_shifts,
                       prior_excluded=None):
    """Greedy 2-shift untuk satu kendaraan → total titik (no-overlap antar shift)."""
    prob = _scaled_problem(base, service_s, budget_s, speed_mult)
    visited = set(prior_excluded or set())
    own = set()
    for _ in range(n_shifts):
        order = _greedy_order(prob.with_exclude(visited | own))
        own |= set(order)
    return len(own)


def _sim_fleet_total(base, service_s, budget_s, speed_mult, n_shifts, n_vehicles):
    """Greedy untuk armada n kendaraan (no-overlap) → total titik unik."""
    prob = _scaled_problem(base, service_s, budget_s, speed_mult)
    visited = set()
    for _ in range(n_vehicles):
        own = set()
        for _ in range(n_shifts):
            own |= set(_greedy_order(prob.with_exclude(visited | own)))
        visited |= own
    return len(visited)


def compute_feasibility_conditions(cfg, base_problems, summary) -> dict:
    """
    Analisis terstruktur: pada KONDISI APA (service time, kecepatan, jam kerja,
    jumlah kendaraan) target titik tercapai? Mengembalikan dict siap-render untuk
    dashboard (lihat route_viewer) — selalu dihitung, baik target tercapai maupun
    tidak. Memakai greedy insertion (deterministik) sebagai estimasi batas kondisi.
    """
    target   = cfg.MIN_POINTS_TARGET
    n_shifts = cfg.N_SHIFTS
    base_service = cfg.SERVICE_TIME_S
    base_budget  = cfg.SHIFT_SECONDS

    service_sweep = [600, 540, 480, 420, 360, 300, 240, 180, 120]   # 10..2 menit
    speed_sweep   = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    hours_sweep   = [6, 7, 8, 9, 10, 11, 12]
    vehicle_sweep = list(range(1, 9))

    vehicles = []
    for vtype, base in base_problems.items():
        vlabel = getattr(base.vehicle, "label", vtype)
        baseline = _sim_vehicle_total(base, base_service, base_budget, 1.0, n_shifts)
        best_col = summary[summary["vehicle"] == vtype]["best_total"]
        best_meta = int(best_col.max()) if len(best_col) else 0
        # Status/feasibility mengacu HASIL MODEL nyata (best_meta), bukan estimasi
        # greedy — supaya konsisten dgn leaderboard/tabel per-shift.
        current = best_meta
        reached = best_meta >= target
        greedy_meets = baseline >= target

        # Ambang tiap lever (ubah SATU faktor) — None jika tak tercapai dalam sweep
        svc_hit = next((s for s in service_sweep
                        if _sim_vehicle_total(base, s, base_budget, 1.0, n_shifts) >= target), None)
        spd_hit = next((m for m in speed_sweep
                        if _sim_vehicle_total(base, base_service, base_budget, m, n_shifts) >= target), None)
        hrs_hit = next((h for h in hours_sweep
                        if _sim_vehicle_total(base, base_service, h * 3600, 1.0, n_shifts) >= target), None)

        # Coverage armada (no-overlap) per jumlah kendaraan
        fleet = []
        n_fleet_needed = None
        for nv in vehicle_sweep:
            tot = _sim_fleet_total(base, base_service, base_budget, 1.0, n_shifts, nv)
            meets = tot >= target
            fleet.append({"n": nv, "unique": int(tot), "meets": meets})
            if meets and n_fleet_needed is None:
                n_fleet_needed = nv

        thresholds = {
            "service_min": (svc_hit / 60 if svc_hit is not None else None),
            "speed_pct":   ((spd_hit - 1) * 100 if spd_hit is not None else None),
            "shift_hours": (hrs_hit if hrs_hit is not None else None),
        }
        recommendation = {
            "n_vehicles":  n_fleet_needed if n_fleet_needed is not None else max(vehicle_sweep),
            "service_min": base_service / 60,
            "shift_hours": base_budget / 3600,
            "feasible_at_baseline": reached,
        }
        narrative = _feasibility_narrative(
            vlabel, target, best_meta, int(baseline), reached, greedy_meets,
            thresholds, n_fleet_needed, base_service / 60, base_budget / 3600)

        vehicles.append({
            "vehicle":     vtype,
            "label":       vlabel,
            "baseline":    int(baseline),       # estimasi greedy (teoretis)
            "best_model":  best_meta,           # hasil model nyata
            "current":     int(current),        # = best_model
            "target":      target,
            "status":      "TERCAPAI" if reached else "GAGAL",
            "greedy_meets": greedy_meets,
            "thresholds":  thresholds,
            "fleet":       fleet,
            "recommendation": recommendation,
            "narrative":   narrative,
        })

    return {
        "target":      target,
        "n_shifts":    n_shifts,
        "base_service_min": base_service / 60,
        "base_shift_hours": base_budget / 3600,
        "vehicles":    vehicles,
    }


def _feasibility_narrative(vlabel, target, best_meta, baseline, reached,
                           greedy_meets, thr, n_fleet, base_service_min,
                           base_shift_hours) -> str:
    """Kalimat rekomendasi Bahasa Indonesia dengan angka konkret.

    Status mengacu hasil MODEL nyata (best_meta); greedy hanya estimasi teoretis.
    """
    if reached:
        margin = best_meta - target
        return (f"Model terbaik {vlabel} mencapai {best_meta} titik "
                f"(≥ target {target}, surplus {margin} titik) pada kondisi baseline "
                f"({base_shift_hours:.0f} jam/shift, service {base_service_min:.0f} menit/titik). "
                f"Konfigurasi 1 kendaraan sudah cukup.")

    # Belum tercapai oleh model
    if greedy_meets:
        # Secara teori target dapat dicapai pada baseline → gap di kualitas model
        return (f"Model terbaik {vlabel} baru {best_meta}/{target} titik, padahal estimasi "
                f"greedy ({baseline} titik) menunjukkan target sebenarnya dapat dicapai pada "
                f"kondisi baseline. Gap ada pada kualitas metaheuristik, bukan kapasitas waktu/"
                f"armada — rekomendasi: tambah iterasi/tuning hyperparameter. Untuk margin aman, "
                f"1 kendaraan (estimasi) sudah cukup.")

    levers = []
    if thr["service_min"] is not None:
        levers.append(f"turunkan service ke ≤ {thr['service_min']:.0f} menit/titik")
    if thr["shift_hours"] is not None:
        levers.append(f"perpanjang shift ke ≥ {thr['shift_hours']:.0f} jam")
    if thr["speed_pct"] is not None:
        levers.append(f"naikkan kecepatan ~{thr['speed_pct']:.0f}%")
    parts = [f"Model terbaik {vlabel} baru {best_meta}/{target} titik dengan 1 kendaraan "
             f"(estimasi greedy {baseline})."]
    if n_fleet is not None:
        parts.append(f"Rekomendasi: gunakan {n_fleet} kendaraan {vlabel.lower()} "
                     f"(no-overlap) untuk menutup target.")
    if levers:
        parts.append("Alternatif 1 kendaraan: " + ", atau ".join(levers) + ".")
    else:
        parts.append("Lever tunggal (service/jam/kecepatan) tak cukup sendirian — "
                     "perlu kombinasi atau tambah armada.")
    return " ".join(parts)


def _feasibility_to_lines(data: dict) -> list:
    """Format hasil compute_feasibility_conditions menjadi teks insight_report.txt."""
    lines = []
    lines.append("Insight Report — Team Orienteering (maksimasi titik dikunjungi)")
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 64)
    lines.append(f"Target minimum   : {data['target']} titik per kendaraan "
                 f"({data['n_shifts']} shift)")
    lines.append(f"Baseline shift   : {data['base_shift_hours']:.0f} jam × {data['n_shifts']} shift")
    lines.append(f"Baseline service : {data['base_service_min']:.0f} menit per titik")
    lines.append("")
    lines.append("Catatan: angka di bawah dihitung dengan greedy insertion "
                 "(deterministik), sebagai estimasi batas kondisi. Model "
                 "metaheuristik (lihat training_summary.csv) bisa sedikit lebih baik.")
    lines.append("")
    for v in data["vehicles"]:
        thr = v["thresholds"]
        lines.append("─" * 64)
        lines.append(f"KENDARAAN: {v['label']} ({v['vehicle']})")
        lines.append(f"  Greedy baseline      : {v['baseline']} titik per kendaraan")
        lines.append(f"  Model terbaik (train): {v['best_model']} titik")
        lines.append(f"  Status target {v['target']}    : {v['status']}")
        lines.append("")
        lines.append(f"  REKOMENDASI: {v['narrative']}")
        lines.append("")
        lines.append(f"  Kondisi agar mencapai {v['target']} titik (ubah SATU faktor):")
        lines.append("    • Service time   : "
                     + (f"turunkan ke <= {thr['service_min']:.0f} menit/titik"
                        if thr["service_min"] is not None else "tak cukup sendirian"))
        lines.append("    • Kecepatan      : "
                     + (f"naikkan ~{thr['speed_pct']:.0f}%"
                        if thr["speed_pct"] is not None else "tak cukup sendirian"))
        lines.append("    • Jam kerja      : "
                     + (f"perpanjang shift ke >= {thr['shift_hours']:.0f} jam"
                        if thr["shift_hours"] is not None else "tak cukup sendirian"))
        lines.append("")
        lines.append(f"  Coverage ARMADA (total titik unik vs jumlah kendaraan {v['vehicle']}):")
        for f in v["fleet"]:
            mark = "  <-- >= target" if f["meets"] else ""
            lines.append(f"    {f['n']} kendaraan : {f['unique']} titik unik{mark}")
        lines.append("")
    return lines


def generate_insight_report(cfg, base_problems, summary, data=None):
    """
    Tulis data/insight_report.txt dari hasil compute_feasibility_conditions.
    `data` boleh dilewatkan agar tak menghitung ulang.
    """
    if data is None:
        data = compute_feasibility_conditions(cfg, base_problems, summary)
    out = cfg.DATA_DIR / "insight_report.txt"
    out.write_text("\n".join(_feasibility_to_lines(data)), encoding="utf-8")
    log.info(f"Saved -> {out.name}")
    return out
