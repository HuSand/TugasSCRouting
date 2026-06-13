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


def generate_insight_report(cfg, G, base_problems, summary, depot,
                            pool_nodes, labels, coords):
    """
    Tulis data/insight_report.txt: untuk tiap jenis kendaraan, baseline greedy
    dan ambang minimal tiap lever yang membuat target tercapai.
    """
    target   = cfg.MIN_POINTS_TARGET
    n_shifts = cfg.N_SHIFTS
    base_service = cfg.SERVICE_TIME_S
    base_budget  = cfg.SHIFT_SECONDS

    service_sweep = [600, 540, 480, 420, 360, 300, 240, 180, 120]   # 10..2 menit
    speed_sweep   = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    hours_sweep   = [6, 7, 8, 9, 10, 11, 12]
    vehicle_sweep = list(range(1, 9))

    lines = []
    lines.append("Insight Report — Team Orienteering (maksimasi titik dikunjungi)")
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 64)
    lines.append(f"Target minimum   : {target} titik per kendaraan (2 shift)")
    lines.append(f"Baseline shift   : {base_budget/3600:.0f} jam × {n_shifts} shift")
    lines.append(f"Baseline service : {base_service/60:.0f} menit per titik")
    lines.append("")
    lines.append("Catatan: angka di bawah dihitung dengan greedy insertion "
                 "(deterministik), sebagai estimasi batas kondisi. Model "
                 "metaheuristik (lihat training_summary.csv) bisa sedikit lebih baik.")
    lines.append("")

    for vtype, base in base_problems.items():
        vlabel = getattr(base.vehicle, "label", vtype)
        baseline = _sim_vehicle_total(base, base_service, base_budget, 1.0, n_shifts)
        best_meta = summary[summary["vehicle"] == vtype]["best_total"].max()

        lines.append("─" * 64)
        lines.append(f"KENDARAAN: {vlabel} ({vtype})")
        lines.append(f"  Greedy baseline      : {baseline} titik per kendaraan")
        lines.append(f"  Model terbaik (train): {int(best_meta)} titik")
        status = "TERCAPAI" if max(baseline, best_meta) >= target else "GAGAL"
        lines.append(f"  Status target {target}    : {status}")
        lines.append("")

        if max(baseline, best_meta) >= target:
            lines.append(f"  Target sudah tercapai untuk {vlabel}.")
            lines.append("")
            continue

        lines.append(f"  Kondisi agar mencapai {target} titik (ubah SATU faktor):")

        # 1. Service time
        hit = next((s for s in service_sweep
                    if _sim_vehicle_total(base, s, base_budget, 1.0, n_shifts) >= target), None)
        if hit is not None:
            lines.append(f"    • Service time   : turunkan ke <= {hit/60:.0f} menit/titik "
                         f"(dari {base_service/60:.0f} menit)")
        else:
            lines.append(f"    • Service time   : bahkan {service_sweep[-1]/60:.0f} menit "
                         f"belum cukup sendirian")

        # 2. Speed
        hit = next((m for m in speed_sweep
                    if _sim_vehicle_total(base, base_service, base_budget, m, n_shifts) >= target), None)
        if hit is not None:
            lines.append(f"    • Kecepatan      : naikkan ~{(hit-1)*100:.0f}% "
                         f"(×{hit:.2f} kecepatan rata-rata)")
        else:
            lines.append(f"    • Kecepatan      : ×{speed_sweep[-1]:.1f} pun belum cukup sendirian")

        # 3. Shift hours
        hit = next((h for h in hours_sweep
                    if _sim_vehicle_total(base, base_service, h*3600, 1.0, n_shifts) >= target), None)
        if hit is not None:
            lines.append(f"    • Jam kerja      : perpanjang shift ke >= {hit} jam "
                         f"(dari {base_budget/3600:.0f} jam)")
        else:
            lines.append(f"    • Jam kerja      : {hours_sweep[-1]} jam/shift pun belum cukup sendirian")

        lines.append("")
        lines.append(f"  Coverage ARMADA (total titik unik vs jumlah kendaraan {vtype}):")
        for nv in vehicle_sweep:
            tot = _sim_fleet_total(base, base_service, base_budget, 1.0, n_shifts, nv)
            mark = "  <-- >= target" if tot >= target else ""
            lines.append(f"    {nv} kendaraan : {tot} titik unik{mark}")
        lines.append("")

    out = cfg.DATA_DIR / "insight_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved -> {out.name}")
    return out
