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
import math
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


def compute_feasibility_conditions(cfg, summary, df) -> dict:
    """
    Rekomendasi feasibility berbasis HASIL MODEL nyata (bukan greedy).

    Untuk tiap kendaraan yang belum mencapai target, hitung secara aritmetik
    dari run terbaik model: pada kondisi apa target tercapai —
      • tambah kendaraan (bagi beban armada),
      • perpanjang shift (jam kerja), atau
      • turunkan waktu kunjungan per titik (service time).
    Angka diestimasi dari waktu tempuh + waktu layanan run terbaik.
    """
    target           = cfg.MIN_POINTS_TARGET
    n_shifts         = cfg.N_SHIFTS
    base_service_min = cfg.SERVICE_TIME_S / 60
    base_shift_hours = cfg.SHIFT_SECONDS / 3600
    budget_total_min = n_shifts * base_shift_hours * 60          # mis. 2 × 6 × 60 = 720

    vehicles = []
    for vtype, g in df.groupby("vehicle"):
        vlabel = str(vtype).capitalize()
        # run terbaik = (model, unit, iterasi) dengan total titik tertinggi
        # (gabung 2 shift). WAJIB sertakan "model" — kalau tidak, visited_count
        # ke-sum lintas semua model dan angkanya jadi salah (mis. 4×50≈200).
        per_run = (g.groupby(["model", "vehicle_unit", "iteration"])
                     .agg(vis=("visited_count", "sum"),
                          travel=("travel_min", "sum"),
                          total=("total_min", "sum"))
                     .reset_index())
        if per_run.empty:
            continue
        bestrow      = per_run.loc[per_run["vis"].idxmax()]
        best         = int(bestrow["vis"])
        travel_total = float(bestrow["travel"])
        total_total  = float(bestrow["total"])
        reached      = best >= target
        gap          = max(0, target - best)

        rec = {"add_vehicles": None, "shift_hours": None, "service_min": None}
        if not reached and best > 0:
            # (a) Service time: agar `target` titik muat dengan waktu tempuh ± sama
            svc_needed = (budget_total_min - travel_total) / target
            if 0 < svc_needed < base_service_min:
                rec["service_min"] = int(math.floor(svc_needed))
            # (b) Shift time: perlu waktu untuk `target` titik (rata-rata waktu/titik run ini)
            avg_per_point = total_total / best
            shift_needed  = math.ceil(target * avg_per_point / n_shifts / 60)
            if base_shift_hours < shift_needed <= 24:
                rec["shift_hours"] = int(shift_needed)
            # (c) Tambah kendaraan (bagi beban / naikkan cakupan armada)
            rec["add_vehicles"] = 1

        narrative = _feasibility_narrative(vlabel, target, best, gap, reached, rec)

        vehicles.append({
            "vehicle":    vtype,
            "label":      vlabel,
            "best_model": best,
            "target":     target,
            "gap":        gap,
            "status":     "TERCAPAI" if reached else "GAGAL",
            "travel_min": round(travel_total, 1),
            "total_min":  round(total_total, 1),
            "budget_min": round(budget_total_min, 1),
            "rec":        rec,
            "narrative":  narrative,
        })

    return {
        "target":           target,
        "n_shifts":         n_shifts,
        "base_service_min": base_service_min,
        "base_shift_hours": base_shift_hours,
        "vehicles":         vehicles,
    }


def _feasibility_narrative(vlabel, target, best, gap, reached, rec) -> str:
    """Narasi operasional (Bahasa Indonesia) dari hasil model — tanpa greedy."""
    if reached:
        return (f"Model {vlabel} mencapai {best} titik (≥ target {target}). "
                f"Sudah feasible pada kondisi sekarang — tak perlu perubahan.")
    opts = []
    if rec.get("add_vehicles"):
        opts.append(f"tambah {rec['add_vehicles']} kendaraan")
    if rec.get("shift_hours"):
        opts.append(f"perpanjang shift ke ≥ {rec['shift_hours']} jam")
    if rec.get("service_min"):
        opts.append(f"turunkan waktu kunjungan ke ≤ {rec['service_min']} menit/titik")
    body = ", ATAU ".join(opts) if opts else (
        "perpanjang shift atau tambah kendaraan (menurunkan service saja tak cukup "
        "karena waktu tempuh sudah dominan)")
    return (f"Model {vlabel} baru {best}/{target} titik (kurang {gap}). "
            f"Agar feasible: {body}.")


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
    lines.append("Catatan: rekomendasi di bawah dihitung dari HASIL MODEL terbaik "
                 "(bukan greedy) — estimasi aritmetik dari waktu tempuh + waktu layanan.")
    lines.append("")
    for v in data["vehicles"]:
        r = v["rec"]
        lines.append("─" * 64)
        lines.append(f"KENDARAAN: {v['label']} ({v['vehicle']})")
        lines.append(f"  Model terbaik : {v['best_model']} titik (target {v['target']})")
        lines.append(f"  Status        : {v['status']}")
        lines.append(f"  REKOMENDASI   : {v['narrative']}")
        if v["status"] != "TERCAPAI":
            lines.append("  Pilih salah satu (estimasi dari hasil model):")
            lines.append("    • Tambah kendaraan : "
                         + (f"+{r['add_vehicles']} unit" if r["add_vehicles"] else "-"))
            lines.append("    • Perpanjang shift : "
                         + (f">= {r['shift_hours']} jam" if r["shift_hours"] else "-"))
            lines.append("    • Service/titik     : "
                         + (f"<= {r['service_min']} menit" if r["service_min"] else "-"))
        lines.append("")
    return lines


def generate_insight_report(cfg, data):
    """Tulis data/insight_report.txt dari hasil compute_feasibility_conditions."""
    out = cfg.DATA_DIR / "insight_report.txt"
    out.write_text("\n".join(_feasibility_to_lines(data)), encoding="utf-8")
    log.info(f"Saved -> {out.name}")
    return out
