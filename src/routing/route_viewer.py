"""
src/routing/route_viewer.py
===========================
Visualizer Team Orienteering — 3 tab, light theme, self-contained.

Tab 1  Rute       : peta rute terbaik per model/kendaraan/shift (turn-by-turn).
Tab 2  Dashboard  : leaderboard model gaya game + chart konvergensi & per-iterasi
                    + insight naratif + panduan metrik.
Tab 3  Peta Titik : peta semua titik kandidat (60 emergency + 60 transport) + depot.

Membaca data/training_log.json (ditulis training.py, termasuk blok `dashboard`).
Peta pakai Leaflet + CartoDB Positron (tanpa error Referer saat dibuka file://).
Chart digambar dengan SVG inline (tanpa CDN) supaya tetap jalan offline.
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def build_route_viewer(data_dir: Path) -> "Path | None":
    """Bangun route_viewer.html dari training_log.json."""
    src = Path(data_dir) / "training_log.json"
    if not src.exists():
        log.warning("training_log.json tidak ada — route viewer dilewati.")
        return None

    payload = src.read_text(encoding="utf-8")
    html = _HTML_TEMPLATE.replace("__ROUTE_DATA__", payload)
    out = Path(data_dir) / "route_viewer.html"
    out.write_text(html, encoding="utf-8")
    log.info(f"  Route viewer -> route_viewer.html")
    return out


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Surabaya Orienteering — Visualizer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#f8fafc; --panel:#fff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
    --accent:#2563eb; --accent-soft:#eff6ff; --hi:#10b981; --hi-soft:#ecfdf5;
    --emergency:#ef4444; --transport:#3b82f6; --depot:#10b981;
    --gold:#f59e0b; --silver:#9ca3af; --bronze:#b45309;
    --shadow:0 4px 6px -1px rgba(0,0,0,0.05),0 2px 4px -1px rgba(0,0,0,0.03);
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{font-family:'Inter',system-ui,Arial,sans-serif;color:var(--ink);background:var(--bg);
       height:100vh;display:flex;flex-direction:column;overflow:hidden}

  /* Topbar */
  .topbar{display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:0 20px;min-height:56px;flex:none;
          background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.05);z-index:2000}
  .brand{font-weight:800;font-size:15px;letter-spacing:-.3px}
  .brand span{color:var(--accent)}
  .tabs{display:flex;flex-wrap:wrap;gap:8px;background:#f1f5f9;padding:6px;border-radius:12px;flex:1 1 auto;min-width:220px}
  .tab{border:0;background:transparent;color:var(--muted);font:inherit;font-weight:600;font-size:13px;
       padding:8px 12px;border-radius:9px;cursor:pointer;transition:.15s;white-space:nowrap;flex:1 1 auto;min-width:90px}
  .tab:hover{color:var(--ink)}
  .tab.active{background:#fff;color:var(--accent);box-shadow:0 1px 3px rgba(0,0,0,0.1)}
  .meta-right{margin-left:auto;font-size:11.5px;color:var(--muted);text-align:right;line-height:1.5;flex:0 1 180px;min-width:150px}

  main{flex:1;position:relative;overflow:hidden}
  .panel{position:absolute;inset:0;display:none}
  .panel.active{display:flex}

  /* Widgets */
  .select{position:relative}
  .select select{width:100%;appearance:none;-webkit-appearance:none;padding:10px 34px 10px 12px;
    border:1.5px solid var(--line);border-radius:11px;background:#f8fafc;color:var(--ink);
    font-size:13.5px;font-weight:500;font-family:inherit;cursor:pointer;transition:.15s}
  .select select:hover{border-color:#cbd5e1}
  .select select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
  .select::after{content:"";position:absolute;right:14px;top:50%;width:8px;height:8px;
    border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);
    transform:translateY(-65%) rotate(45deg);pointer-events:none}
  .field{margin-bottom:12px}
  .field>label{display:block;font-size:10.5px;font-weight:600;letter-spacing:.6px;
    text-transform:uppercase;color:var(--muted);margin-bottom:5px}
  .dot{width:11px;height:11px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.12)}
  .legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:11.5px;color:var(--muted)}
  .legend .it{display:flex;align-items:center;gap:5px}
  .num-marker{display:flex;align-items:center;justify-content:center;border-radius:50%;color:#fff;
    font-size:11px;font-weight:700;border:2.5px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.15)}

  /* Tab Rute */
  .route-wrap{display:flex;width:100%;height:100%}
  .route-side{width:380px;flex:none;height:100%;display:flex;flex-direction:column;background:#fff;
    box-shadow:2px 0 8px rgba(0,0,0,0.05);z-index:1000}
  .route-side .head{padding:16px 18px 12px;background:linear-gradient(135deg,#2563eb,#3b82f6);color:#fff}
  .route-side .head .pill{display:inline-block;background:rgba(255,255,255,.18);padding:2px 9px;
    border-radius:999px;font-size:11px;margin-top:8px;font-weight:600}
  .route-side .body{flex:1;overflow-y:auto;padding:14px 16px 24px}
  .stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0 14px}
  .stat{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:10px 12px}
  .stat .k{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
  .stat .v{font-size:18px;font-weight:700;margin-top:2px}
  .stat .v small{font-size:11px;font-weight:500;color:var(--muted)}
  .stat.good .v{color:var(--hi)} .stat.warn .v{color:var(--emergency)}
  .sec-title{font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--muted);
    margin:6px 0 8px;display:flex;justify-content:space-between}
  .toggle{font-size:11px;color:var(--accent);cursor:pointer;font-weight:600;user-select:none}
  .timeline{position:relative}
  .leg{position:relative;padding:8px 10px 10px 30px;border-radius:10px;cursor:pointer;transition:.12s}
  .leg:hover{background:#f1f5f9} .leg.active{background:var(--accent-soft)}
  .leg::before{content:"";position:absolute;left:11px;top:26px;bottom:-6px;width:2px;background:var(--line)}
  .leg:last-child::before{display:none}
  .leg .num{position:absolute;left:2px;top:9px;width:20px;height:20px;border-radius:50%;color:#fff;
    font-size:10.5px;font-weight:700;display:flex;align-items:center;justify-content:center;
    border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.1)}
  .leg .route{font-size:13px;font-weight:600;line-height:1.35}
  .leg .route .arrow{color:var(--muted);font-weight:500;margin:0 3px}
  .leg .streets{font-size:11px;color:var(--muted);margin-top:3px;line-height:1.45}
  #mapRoute,#mapPoints{flex:1;height:100%;background:#e5e7eb}
  .leaflet-container{font-family:'Inter',system-ui,sans-serif}

  /* Mode Rute toggle */
  .vm-wrap{display:flex;gap:3px;background:#f1f4f9;padding:3px;border-radius:9px}
  .vm-btn{flex:1;border:0;background:transparent;color:var(--muted);font:inherit;
    font-size:11px;font-weight:600;padding:6px 4px;border-radius:7px;cursor:pointer;
    transition:.15s;white-space:nowrap;line-height:1.3}
  .vm-btn.active{background:#fff;color:var(--accent);box-shadow:0 1px 6px rgba(20,33,61,.1)}
  .vm-btn:hover:not(.active){color:var(--ink)}
  .leg-veh-hdr{font-size:12px;font-weight:700;padding:8px 0 4px;margin-top:4px;
    border-top:1px solid var(--line);display:flex;align-items:center;gap:7px}
  .leg-veh-line{display:inline-block;width:26px;height:3px;border-radius:2px;flex:none}

  /* Dashboard */
  .dash{width:100%;height:100%;overflow-y:auto;padding:22px 26px 40px}
  .dash-grid{max-width:1240px;margin:0 auto;display:flex;flex-direction:column;gap:26px}
  .card{background:#fff;border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow);padding:24px}
  .card h2{margin:0 0 4px;font-size:17px;font-weight:700;letter-spacing:-.3px}
  .card .sub{font-size:12.5px;color:var(--muted);margin-bottom:20px}

  /* Podium */
  .podium-wrapper{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:26px;
    padding:32px 28px 18px;min-height:300px;background:linear-gradient(180deg,#f8fbff,#eef4ff);
    border-radius:20px;border:1px solid rgba(148,163,184,.22);width:100%;box-sizing:border-box}
  .podium-box{border:1.5px solid var(--line);border-radius:26px;padding:22px 18px 18px;position:relative;
    display:flex;flex-direction:column;align-items:center;text-align:center;background:#fff;
    box-shadow:0 22px 45px rgba(15,23,42,.08);transition:.25s;flex:0 1 280px;max-width:300px;min-width:240px;box-sizing:border-box}
  .podium-box:hover{transform:translateY(-8px);box-shadow:0 32px 68px rgba(15,23,42,.14)}
  .podium-box.rank-1{order:2;border-color:#d97706;background:linear-gradient(180deg,#fff8e1,#fff);min-height:285px}
  .podium-box.rank-2{order:1;border-color:#9ca3af;background:linear-gradient(180deg,#f3f4f6,#fff);min-height:255px}
  .podium-box.rank-3{order:3;border-color:#c47114;background:linear-gradient(180deg,#fff7ed,#fff);min-height:245px}
  .podium-box .podium-footer{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:16px;width:100%}
  .podium-box .podium-stat{border-radius:16px;padding:12px 14px;background:rgba(15,23,42,.05);font-size:13px;font-weight:700;color:var(--ink)}
  .podium-box .podium-stat span{display:block;font-size:10.5px;font-weight:600;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
  .podium-badge{position:absolute;top:-14px;left:50%;transform:translateX(-50%);padding:6px 18px;
    border-radius:999px;color:#fff;font-size:11px;font-weight:900;letter-spacing:.7px;box-shadow:0 4px 14px rgba(0,0,0,.14)}
  .podium-box.rank-1 .podium-badge{background:linear-gradient(135deg,#f59e0b,#d97706)}
  .podium-box.rank-2 .podium-badge{background:linear-gradient(135deg,#94a3b8,#475569)}
  .podium-box.rank-3 .podium-badge{background:linear-gradient(135deg,#d97706,#a16207)}
  .podium-title{font-size:15px;font-weight:700;margin-top:12px;color:var(--ink)}
  .podium-algo{font-size:12px;color:var(--muted);margin-bottom:auto;font-weight:600}
  .podium-opt-metrics{font-size:20px;font-weight:800;color:var(--ink);line-height:1.2}
  .podium-opt-metrics span{font-size:11px;font-weight:600;color:var(--muted);display:block;margin-top:2px;text-transform:uppercase;letter-spacing:.3px}
  .podium-other{margin-top:24px;padding:18px 0 0;border-top:1px solid rgba(148,163,184,.2)}
  .podium-other-title{font-size:13px;font-weight:700;color:var(--ink);margin-bottom:14px;text-transform:uppercase;letter-spacing:.6px}
  .podium-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
  .podium-item{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:14px;box-shadow:0 10px 25px rgba(15,23,42,.06);transition:.15s}
  .podium-item:hover{transform:translateY(-2px);box-shadow:0 14px 30px rgba(15,23,42,.1)}
  .podium-item .rank-chip{display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:12px;background:rgba(251,191,36,.18);color:#b45309;font-weight:800;font-size:14px;margin-bottom:10px}
  .podium-item .algo-name{font-size:13px;font-weight:700;margin-bottom:4px;color:var(--ink)}
  .podium-item .algo-meta{font-size:11px;color:var(--muted);margin-bottom:10px}
  .podium-item .mini-stat{display:flex;justify-content:space-between;font-size:12px;color:var(--ink);margin-top:8px}
  .podium-item .mini-stat span:first-child{color:var(--muted)}

  /* Leaderboard */
  .lb{display:flex;flex-direction:column;gap:18px}
  .lb-card{display:grid;grid-template-columns:46px 52px minmax(0,1fr) minmax(140px,180px);gap:22px;
    align-items:flex-start;border:1.5px solid var(--line);border-radius:20px;padding:24px;
    transition:.15s;position:relative;min-width:0;overflow:hidden}
  .lb-card:hover{box-shadow:0 14px 28px rgba(15,23,42,.08);border-color:#cbd5e1}
  .lb-card > div:last-child{padding-left:16px;min-width:140px;max-width:180px}
  .lb-main{display:flex;flex-direction:column;gap:14px}
  .lb-section{background:#f8fafc;border:1px solid rgba(226,232,240,.9);border-radius:18px;padding:18px}
  .lb-std{font-size:11.5px;color:var(--muted);font-weight:600;margin:-6px 0 8px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
  .lb-std b{color:var(--ink);font-weight:800}
  .metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(208px,1fr));gap:10px;margin-top:14px}
  .metric-pill{display:flex;align-items:center;gap:9px;padding:12px 14px;border-radius:14px;color:#fff;font-size:12.5px;font-weight:700;box-shadow:0 10px 24px rgba(15,23,42,.08)}
  .metric-pill span.icon{font-size:14px}
  .metric-pill.runtime{background:linear-gradient(135deg,#312e81,#4338ca)}
  .metric-pill.convergence{background:linear-gradient(135deg,#0f766e,#115e59)}
  .metric-pill.throughput{background:linear-gradient(135deg,#1d4ed8,#2563eb)}
  .metric-pill.feasibility{background:linear-gradient(135deg,#15803d,#166534)}
  .metric-pill.distance{background:linear-gradient(135deg,#0369a1,#0ea5e9)}
  .metric-pill.speed{background:linear-gradient(135deg,#0e7490,#0891b2)}
  .metric-pill.spatial{background:linear-gradient(135deg,#b45309,#d97706)}
  .metric-pill.serviceratio{background:linear-gradient(135deg,#0f766e,#0d9488)}
  .metric-pill.timeutil{background:linear-gradient(135deg,#6b21a8,#9333ea)}
  .metric-pill.stability{background:linear-gradient(135deg,#115e59,#0f766e)}
  .metric-pill b{font-weight:800}
  .best-run-banner{display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,#eef2ff,#f8fafc);
    color:#0f172a;border:1px solid #cbd5e1;padding:10px 14px;border-radius:14px;font-size:12px;font-weight:700;margin-bottom:14px}
  .best-run-banner strong{color:var(--accent)}
  .lb-card.pos-0{border-color:#fef08a;background:linear-gradient(180deg,#fffdf0,#fff)}
  .lb-card.pos-0 .lb-section{background:rgba(254,240,138,.26);border-color:rgba(254,240,138,.55)}
  .lb-card.pos-1{border-color:#c7d2fe;background:linear-gradient(180deg,#eff6ff,#fff)}
  .lb-card.pos-1 .lb-section{background:rgba(192,232,255,.24);border-color:rgba(96,165,250,.35)}
  .lb-card.pos-2{border-color:#fed7aa;background:linear-gradient(180deg,#fff7ed,#fff)}
  .lb-card.pos-2 .lb-section{background:rgba(254,215,170,.28);border-color:rgba(251,146,60,.35)}
  .lb-card.pos-3{border-color:#c7d2fe;background:linear-gradient(180deg,#f0f9ff,#fff)}
  .lb-card.pos-3 .lb-section{background:rgba(187,222,251,.22);border-color:rgba(148,163,184,.33)}
  .lb-rank{font-size:26px;font-weight:900;text-align:center;margin-top:8px}
  .lb-card.pos-0 .lb-rank{color:#ca8a04}
  .lb-card.pos-1 .lb-rank{color:#475569}
  .lb-card.pos-2 .lb-rank{color:#c2410c}
  .lb-card.pos-3 .lb-rank{color:var(--muted)}
  .lb-avatar{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:26px;margin-top:4px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
  .lb-name{font-size:17px;font-weight:700;display:flex;align-items:center;gap:10px}
  .lb-sub{font-size:12px;color:var(--muted);margin:3px 0 14px}
  .tier{font-size:10px;font-weight:800;padding:3px 9px;border-radius:6px;letter-spacing:.5px;text-transform:uppercase}
  .tier-S{background:linear-gradient(135deg,#7c3aed,#d97706);color:#fff;box-shadow:0 2px 8px rgba(124,58,237,.35);border:1px solid #f59e0b}
  .tier-A{background:#10b981;color:#fff;box-shadow:0 2px 6px rgba(16,185,129,.25)}
  .tier-B{background:#2563eb;color:#fff;box-shadow:0 2px 6px rgba(37,99,235,.25)}
  .tier-C{background:#64748b;color:#fff;box-shadow:0 2px 6px rgba(100,116,139,.25)}
  .lb-score{border-radius:18px;padding:18px 16px;display:flex;flex-direction:column;
    justify-content:center;min-height:92px;box-shadow:0 12px 34px rgba(15,23,42,.12);min-width:120px;max-width:180px;overflow:hidden;align-self:flex-start}
  .lb-card.pos-0 .lb-score{background:linear-gradient(135deg,#fde68a,#f59e0b);border:1px solid #d97706}
  .lb-card.pos-1 .lb-score{background:linear-gradient(135deg,#e2e8f0,#cbd5e1);border:1px solid #94a3b8}
  .lb-card.pos-2 .lb-score{background:linear-gradient(135deg,#fed7aa,#f97316);border:1px solid #c2410c}
  .lb-card.pos-3 .lb-score{background:linear-gradient(135deg,#d9f99d,#16a34a);border:1px solid #15803d}
  .lb-score .sn{font-size:28px;font-weight:900;line-height:1;color:#fff;text-shadow:0 1px 4px rgba(0,0,0,.16)}
  .lb-score .sl{font-size:11px;font-weight:700;color:rgba(255,255,255,.92);letter-spacing:.85px;margin-top:6px;text-transform:uppercase}

  /* Accordion (animated) */
  .accordion-trigger{background:#fff;border:1px solid var(--line);color:var(--ink);font-size:11.5px;font-weight:700;
    cursor:pointer;padding:8px 16px;border-radius:8px;display:inline-flex;align-items:center;gap:6px;outline:none;transition:.12s;box-shadow:0 1px 2px rgba(0,0,0,.03)}
  .accordion-trigger:hover{background:#f8fafc;border-color:#cbd5e1}
  .accordion-trigger .caret{display:inline-block;transition:transform .3s ease}
  .accordion-trigger.open .caret{transform:rotate(180deg)}
  .accordion-content{max-height:0;opacity:0;overflow:hidden;
    transition:max-height .38s cubic-bezier(.4,0,.2,1),opacity .25s ease,margin-top .3s ease;margin-top:0}
  .accordion-content.open{max-height:1600px;opacity:1;margin-top:14px}
  .acc-single{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.03)}
  .acc-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
  .table-container{border:1px solid var(--line);border-radius:12px;overflow:auto;background:#fff;max-height:360px}
  .iter-table{width:100%;border-collapse:collapse;font-size:12.5px;text-align:center}
  .iter-table th,.iter-table td{padding:11px 14px;border-bottom:1px solid var(--line);white-space:nowrap}
  .iter-table th{background:#f1f5f9;color:#475569;font-weight:700;font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;position:sticky;top:0;z-index:1}
  .iter-table tbody tr:nth-child(even){background:#fafcff}
  .iter-table tbody tr:hover{background:#eff6ff}
  .iter-table tr:last-child td{border-bottom:none}
  .iter-table tr.best-row{background:#f0fdf4!important;font-weight:700;color:#15803d}
  .iter-table tr.best-row td{border-top:1px solid #bbf7d0;border-bottom:1px solid #bbf7d0}
  .iter-table .veh-tag{display:inline-block;padding:3px 10px;border-radius:7px;font-size:10.5px;font-weight:700}
  .veh-motor{background:#dbeafe;color:#1d4ed8}
  .veh-mobil{background:#fee2e2;color:#b91c1c}
  .iter-table td.no{color:#dc2626;font-weight:800}.iter-table td.yes{color:#16a34a;font-weight:800}
  .dt-title{font-size:12.5px;font-weight:700;margin-bottom:8px;color:var(--ink)}
  .acc-note{font-size:11px;color:var(--muted);margin-top:10px;line-height:1.5}
  .acc-filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .acc-filters label{font-size:10.5px;color:var(--muted);font-weight:600}
  .acc-filters select{font-size:11.5px;padding:5px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);cursor:pointer}
  .podium-veh{font-size:11.5px;color:var(--muted);font-weight:600;margin-top:6px}
  .podium-veh .veh-tag{display:inline-block;padding:2px 9px;border-radius:6px;font-size:10.5px;font-weight:700}

  /* Sub-card drilldown (klik metric pill -> timeline inline) */
  .metric-pill{cursor:pointer;position:relative;transition:transform .12s,box-shadow .12s}
  .metric-pill:hover{transform:translateY(-1px);box-shadow:0 14px 30px rgba(15,23,42,.16)}
  .metric-pill.sel{outline:2px solid rgba(255,255,255,.85);outline-offset:-3px}
  .metric-pill .chev{margin-left:auto;font-size:10px;opacity:.85;transition:transform .25s ease}
  .metric-pill.sel .chev{transform:rotate(180deg)}
  .subcard-panel{max-height:0;opacity:0;overflow:hidden;
    transition:max-height .38s cubic-bezier(.4,0,.2,1),opacity .25s ease,margin-top .3s ease;margin-top:0}
  .subcard-panel.open{max-height:900px;opacity:1;margin-top:12px}
  .subcard-inner{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px}
  .subcard-stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}
  .subcard-stats .s{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:8px 12px;font-size:11px;color:var(--muted);font-weight:600}
  .subcard-stats .s b{display:block;font-size:15px;color:var(--ink);font-weight:800;margin-top:2px}
  @media(max-width:700px){.subcard-inner{grid-template-columns:1fr}}
  /* Convergence sub-card (multi-curve) */
  .subcard-inner .conv-wrap{grid-column:1/-1}
  .conv-bar{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:12px}
  .conv-charts{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
  .conv-cell{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:10px}
  .conv-cap{font-size:11.5px;font-weight:700;color:var(--ink);margin-bottom:4px;display:flex;align-items:center;gap:6px}
  .conv-cap span{font-size:9.5px;font-weight:600;padding:1px 6px;border-radius:5px}
  .conv-cap.up span{background:#dcfce7;color:#15803d}
  .conv-cap.down span{background:#dbeafe;color:#1d4ed8}
  @media(max-width:820px){.conv-charts{grid-template-columns:1fr}}

  /* SVG chart tooltip + hover dots */
  .svg-wrap{position:relative}
  .chart-tip{position:absolute;pointer-events:none;background:rgba(15,23,42,.92);color:#fff;font-size:11px;
    padding:5px 9px;border-radius:7px;white-space:nowrap;transform:translate(-50%,-130%);opacity:0;transition:opacity .12s;z-index:5;font-weight:600}
  .svg-wrap circle.hot{cursor:pointer}
  @keyframes drawline{to{stroke-dashoffset:0}}

  /* Feasibility Roadmap */
  .feas-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
  .feas-card{border:1px solid var(--line);border-radius:16px;padding:18px;background:#f8fafc}
  .feas-card.ok{border-color:#bbf7d0;background:linear-gradient(180deg,#f0fdf4,#fff)}
  .feas-card.bad{border-color:#fecaca;background:linear-gradient(180deg,#fef2f2,#fff)}
  .feas-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
  .feas-head .name{font-size:15px;font-weight:800;color:var(--ink)}
  .feas-badge{font-size:10px;font-weight:800;padding:3px 9px;border-radius:6px;text-transform:uppercase;letter-spacing:.5px}
  .feas-badge.ok{background:#16a34a;color:#fff}.feas-badge.bad{background:#dc2626;color:#fff}
  .feas-narr{font-size:12.5px;color:var(--ink);line-height:1.6;background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:12px}
  .feas-narr b{color:var(--accent)}
  .feas-levers{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:6px}
  .feas-lever{background:#fff;border:1px solid var(--line);border-top:3px solid #f59e0b;border-radius:12px;padding:14px 10px;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,.05)}
  .feas-lever .lk{font-size:11px;color:var(--muted);font-weight:700;letter-spacing:.2px}
  .feas-lever .lv{font-size:19px;font-weight:900;color:#b45309;margin-top:7px;line-height:1.1}
  .feas-lever.na{border-top-color:#cbd5e1;opacity:.65}
  .feas-lever.na .lv{color:#94a3b8;font-weight:600;font-size:15px}
  .feas-fleet{display:flex;gap:5px;flex-wrap:wrap}
  .feas-fleet .fc{flex:1;min-width:44px;text-align:center;border:1px solid var(--line);border-radius:8px;padding:6px 2px;background:#fff;font-size:11px}
  .feas-fleet .fc.meets{background:#dcfce7;border-color:#86efac;color:#15803d;font-weight:700}
  .feas-fleet .fc .fn{font-size:10px;color:var(--muted)}.feas-fleet .fc.meets .fn{color:#15803d}
  .feas-sub2{font-size:11.5px;color:var(--muted);font-weight:600;margin:-2px 0 10px}
  .feas-tag{font-size:9px;font-weight:700;background:#fef3c7;color:#92400e;padding:1px 7px;border-radius:5px;margin-left:6px;text-transform:none;letter-spacing:0}
  .feas-warn{font-size:12px;line-height:1.6;background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;border-radius:10px;padding:11px 13px;margin-bottom:12px;color:#78350f}
  .feas-warn b{color:#92400e}
  .feas-ok-note{font-size:12px;line-height:1.6;background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;border-radius:10px;padding:11px 13px;margin-bottom:12px;color:#14532d;font-weight:600}
  @media(max-width:560px){.feas-levers{grid-template-columns:1fr}}

  /* Bars */
  .bars-vertical{display:flex;flex-direction:column;gap:12px;margin-bottom:16px}
  .bar .bh{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted);margin-bottom:4px}
  .bar .bh b{color:var(--ink);font-weight:700}
  .track{height:9px;background:#e2e8f0;border-radius:6px;overflow:hidden}
  .fill{height:100%;border-radius:6px;transition:width .5s}

  /* Insights */
  .insights-vertical{display:flex;flex-direction:column;gap:10px}
  .insight-row{display:flex;align-items:flex-start;gap:16px;background:#f8fafc;border:1px solid var(--line);
    border-radius:12px;padding:14px 20px;position:relative}
  .insight-row.active-accent{border-left:4px solid var(--accent)}
  .insight-row.active-success{border-left:4px solid var(--hi)}
  .insight-row.active-warning{border-left:4px solid var(--emergency)}
  .insight-desc{font-size:13px;color:var(--ink);font-weight:500;flex:1;line-height:1.5}
  .insight-desc b{font-weight:700;color:var(--accent)}

  /* Comparison table */
  .table-card{border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#fff;margin-top:10px}
  table.cmp{width:100%;border-collapse:collapse;font-size:12.5px}
  table.cmp th,table.cmp td{padding:12px 16px;text-align:right;border-bottom:1px solid var(--line)}
  table.cmp th:first-child,table.cmp td:first-child{text-align:left}
  table.cmp th{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);font-weight:600;background:#f8fafc}
  table.cmp tr:last-child td{border-bottom:none}
  table.cmp tr:hover td{background:#f8fafc}
  table.cmp td.best{color:var(--hi);font-weight:700;background:#f0fdf4}
  table.cmp th.sortable{cursor:pointer;user-select:none}
  table.cmp th.sortable:hover{color:var(--ink)}
  table.cmp th .arr{opacity:.4;font-size:9px;margin-left:3px}
  table.cmp th.sorted .arr{opacity:1;color:var(--accent)}

  /* Chart selector */
  .chart-selector-wrapper{display:flex;flex-wrap:wrap;justify-content:flex-start;gap:8px;margin-bottom:12px}
  .chart-btn{background:#fff;border:1.5px solid var(--line);padding:7px 13px;border-radius:10px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;transition:.15s;min-width:auto;text-align:center}
  .chart-btn.active{background:var(--accent);color:#fff;border-color:var(--accent);box-shadow:0 8px 20px rgba(37,99,235,.16)}
  .charts-layout{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:22px}
  .chart-box{background:#f8fafc;border:1px solid var(--line);border-radius:16px;padding:18px;min-height:300px;display:flex;flex-direction:column}
  .chart-title{font-size:12.5px;font-weight:700;margin-bottom:12px;color:var(--ink);display:flex;align-items:center;gap:6px}
  .chart-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:12px;justify-content:center}
  .chart-legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}

  /* Metric guide */
  .guide{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
  .guide .g{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px}
  .guide .g b{font-size:13px;font-weight:700;color:var(--ink)} .guide .g p{margin:6px 0 0;font-size:11.5px;color:var(--muted);line-height:1.5}
  .empty{max-width:520px;margin:80px auto;text-align:center;color:var(--muted)}

  /* Tab Peta Titik */
  .points-wrap{display:flex;width:100%;height:100%}
  .points-side{width:320px;flex:none;background:#fff;box-shadow:2px 0 8px rgba(0,0,0,.05);z-index:1000;padding:18px;overflow-y:auto}
  .points-side h2{font-size:15px;margin:0 0 4px}
  .points-side .sub{font-size:12px;color:var(--muted);margin-bottom:16px}
  .pcount{display:flex;align-items:center;gap:10px;background:#f8fafc;border:1px solid var(--line);
    border-radius:12px;padding:12px;margin-bottom:10px}
  .pcount .ic{width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px}
  .pcount .n{font-size:22px;font-weight:800;line-height:1}
  .pcount .l{font-size:11px;color:var(--muted)}

  @media(max-width:1024px){.charts-layout{grid-template-columns:1fr}}
  @media(max-width:900px){
    .lb-card{grid-template-columns:46px 52px 1fr}
    .lb-card > div:last-child{max-width:none;width:100%;padding-left:0}
  }
  @media(max-width:720px){
    .lb-card{grid-template-columns:1fr;grid-template-rows:auto auto}
    .lb-card > div:last-child{width:100%;padding-left:0}
    .chart-selector-wrapper{justify-content:center}
    .chart-btn{min-width:140px}
    .topbar{align-items:flex-start}
    .tabs{width:100%;justify-content:flex-start}
    .tab{flex:1 1 100%;min-width:0;text-align:center}
    .meta-right{margin-left:0;width:100%;text-align:left}
  }
</style>
</head>
<body>
  <div class="topbar">
    <div class="brand">🗺️ Surabaya <span>Orienteering</span> Visualizer</div>
    <div class="tabs">
      <button class="tab active" data-tab="route">🧭 Rute</button>
      <button class="tab" data-tab="dash">🏆 Dashboard</button>
      <button class="tab" data-tab="points">📍 Peta Titik</button>
    </div>
    <div class="meta-right" id="metaRight"></div>
  </div>

  <main>
    <!-- TAB RUTE -->
    <section class="panel active" id="p-route">
      <div class="route-wrap">
        <aside class="route-side">
          <div class="head">
            <div class="field" style="margin:0"><label style="color:#dbe7ff">Model</label>
              <div class="select"><select id="modelSel"></select></div></div>
            <span class="pill" id="targetPill"></span>
          </div>
          <div class="body">
            <div class="field" id="vehicleField"><label>Kendaraan</label><div class="select"><select id="vehicleSel"></select></div></div>
            <div class="field"><label>Shift</label><div class="select"><select id="shiftSel"></select></div></div>
            <div class="field">
              <label>Mode Rute</label>
              <div class="vm-wrap">
                <button class="vm-btn active" id="vmSingle">🚗 Per Kendaraan</button>
                <button class="vm-btn" id="vmAll">🏍🚗 Semua Kendaraan</button>
              </div>
            </div>
            <div class="stats" id="rStats"></div>
            <div id="routeLegend" class="legend" style="margin-bottom:14px">
              <span class="it"><span class="dot" style="background:var(--emergency)"></span>Emergency</span>
              <span class="it"><span class="dot" style="background:var(--transport)"></span>Transport</span>
              <span class="it"><span class="dot" style="background:var(--depot)"></span>Depot</span>
            </div>
            <div class="sec-title"><span>Rute (klik leg untuk highlight)</span>
              <span class="toggle" id="poolToggle">◌ Semua titik</span></div>
            <div class="timeline" id="legs"></div>
          </div>
        </aside>
        <div id="mapRoute"></div>
      </div>
    </section>

    <!-- TAB DASHBOARD -->
    <section class="panel" id="p-dash">
      <div class="dash"><div class="dash-grid" id="dashGrid"></div></div>
    </section>

    <!-- TAB PETA TITIK -->
    <section class="panel" id="p-points">
      <div class="points-wrap">
        <aside class="points-side" id="pointsSide"></aside>
        <div id="mapPoints"></div>
      </div>
    </section>
  </main>

<script>
let DATA = {};
try { DATA = __ROUTE_DATA__; } catch(e) { console.error("Gagal membaca data.", e); }

const CAT = { emergency:'#e5484d', transport:'#2563eb', depot:'#16a34a', '':'#64748b' };

if (DATA && DATA.generated) {
  document.getElementById('metaRight').innerHTML =
    `Generated ${DATA.generated}<br>Target ${DATA.target} titik/kendaraan · shift ${DATA.shift_min}m · service ${DATA.service_min}m`;
  document.getElementById('targetPill').textContent =
    `Target ${DATA.target} titik/sesi · ${DATA.n_iterations||'?'} iterasi`;
}

/* ── Tabs ── */
let mapRoute=null, mapPoints=null, dashDone=false, pointsDone=false;
let activeMetricView = 'stability';

document.querySelectorAll('.tab').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    b.classList.add('active');
    const t = b.dataset.tab;
    document.getElementById('p-' + t).classList.add('active');
    if (t === 'route' && mapRoute) setTimeout(() => mapRoute.invalidateSize(), 100);
    if (t === 'dash' && !dashDone) { renderDashboard(); dashDone = true; }
    if (t === 'points') {
      if (!pointsDone) { renderPoints(); pointsDone = true; }
      else setTimeout(() => mapPoints && mapPoints.invalidateSize(), 100);
    }
  });
});

function baseTiles(map){
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{
    subdomains:'abcd',maxZoom:20,
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
  }).addTo(map);
}
function depotIcon(){return L.divIcon({className:'',iconSize:[30,30],iconAnchor:[15,15],
  html:`<div style="width:30px;height:30px;border-radius:50%;background:${CAT.depot};border:3px solid #fff;
    box-shadow:0 2px 8px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;color:#fff;font-size:15px;">★</div>`});}
function numIcon(n,c){return L.divIcon({className:'',iconSize:[24,24],iconAnchor:[12,12],
  html:`<div class="num-marker" style="width:24px;height:24px;background:${c}">${n}</div>`});}

/* ── TAB RUTE ── */
const VEH_COLORS = { motor:'#f59e0b', mobil:'#8b5cf6' };
const VEH_ICONS  = { motor:'🏍', mobil:'🚗' };

if (document.getElementById('mapRoute')) {
  mapRoute = L.map('mapRoute', {zoomControl:true}); baseTiles(mapRoute);
}

const modelSel    = document.getElementById('modelSel'),
      vehicleSel  = document.getElementById('vehicleSel'),
      shiftSel    = document.getElementById('shiftSel'),
      legsDiv     = document.getElementById('legs'),
      rStats      = document.getElementById('rStats'),
      vehicleField= document.getElementById('vehicleField');

if (DATA && DATA.runs) {
  [...new Set(DATA.runs.map(r => r.model))].forEach(m => {
    const dn = DATA.dashboard && DATA.dashboard.models.find(x => x.name === m);
    if (modelSel) modelSel.add(new Option(dn ? dn.display_name : m, m));
  });
}

function runsFor(m){ return DATA.runs ? DATA.runs.filter(r => r.model === m) : []; }
function curRun(){ return modelSel ? runsFor(modelSel.value).find(r => `${r.vehicle}#${r.vehicle_unit}` === vehicleSel.value) : null; }

function fillVehicles(){
  if (!vehicleSel || !modelSel) return;
  vehicleSel.innerHTML = '';
  runsFor(modelSel.value).forEach(r =>
    vehicleSel.add(new Option(`${r.vehicle} #${r.vehicle_unit}`, `${r.vehicle}#${r.vehicle_unit}`)));
}

function fillShifts(){
  if (!shiftSel) return;
  shiftSel.innerHTML = '';
  if (viewMode === 'single') {
    const r = curRun(); if (!r) return;
    r.shifts.forEach(s => shiftSel.add(new Option(`Shift ${s.shift} — ${s.visited_count} titik`, s.shift)));
  } else {
    const allRuns = runsFor(modelSel.value); if (!allRuns.length) return;
    allRuns[0].shifts.forEach(s => {
      const tot = allRuns.reduce((acc, r) => { const sh = r.shifts.find(x => x.shift === s.shift); return acc + (sh ? sh.visited_count : 0); }, 0);
      shiftSel.add(new Option(`Shift ${s.shift} — ${tot} titik (gabungan)`, s.shift));
    });
  }
}

let rLayers=[], poolLayer=null, hiLayer=null, showPool=false, viewMode='single';

function clearRoute(){ rLayers.forEach(l => mapRoute.removeLayer(l)); rLayers = []; if (hiLayer){ mapRoute.removeLayer(hiLayer); hiLayer = null; } }
function drawPool(){
  if (poolLayer){ mapRoute.removeLayer(poolLayer); poolLayer = null; }
  if (!showPool || !DATA.pool) return;
  poolLayer = L.layerGroup();
  DATA.pool.forEach(p => L.circleMarker(p.coord, {radius:3, color:CAT[p.cat]||'#888', weight:1, fillOpacity:.5, opacity:.5})
    .bindPopup(`${p.label} <small>(${p.cat})</small>`).addTo(poolLayer));
  poolLayer.addTo(mapRoute);
}

function setLegendSingle(){
  const el = document.getElementById('routeLegend'); if (!el) return;
  el.innerHTML = `<span class="it"><span class="dot" style="background:var(--emergency)"></span>Emergency</span>
    <span class="it"><span class="dot" style="background:var(--transport)"></span>Transport</span>
    <span class="it"><span class="dot" style="background:var(--depot)"></span>Depot</span>`;
}

function setLegendAll(runs){
  const el = document.getElementById('routeLegend'); if (!el) return;
  const types = [...new Set(runs.map(r => r.vehicle))];
  el.innerHTML = types.map(v => {
    const col = VEH_COLORS[v] || '#888';
    const dash = v === 'motor'
      ? `border-bottom:2px dashed ${col};background:none;height:0;margin-top:2px`
      : `background:${col}`;
    return `<span class="it"><span class="leg-veh-line" style="${dash}"></span>${VEH_ICONS[v]||''} ${v.charAt(0).toUpperCase()+v.slice(1)}</span>`;
  }).join('') + `<span class="it"><span class="dot" style="background:var(--depot)"></span>Depot</span>`;
}

function renderRoute(){
  if (viewMode === 'all') { renderAllVehicles(); return; }
  if (!mapRoute) return;
  clearRoute();
  const run = curRun(); if (!run) return;
  const sh = run.shifts.find(s => String(s.shift) === String(shiftSel.value)); if (!sh) return;
  rLayers.push(L.marker(DATA.depot.coord, {icon:depotIcon(), zIndexOffset:1000})
    .addTo(mapRoute).bindPopup(`<b>DEPOT</b><br>${DATA.depot.label}`));
  if (sh.route_coords && sh.route_coords.length) {
    rLayers.push(L.polyline(sh.route_coords, {color:'#2563eb', weight:4, opacity:.45, lineJoin:'round'}).addTo(mapRoute));
    mapRoute.fitBounds(L.latLngBounds(sh.route_coords).pad(.18));
  }
  const legs = sh.legs || [];
  legs.forEach((lg, i) => { if (i < legs.length-1) {
    const c = lg.coords[lg.coords.length-1];
    rLayers.push(L.marker(c, {icon:numIcon(i+1, CAT[lg.to_cat]||CAT['']), zIndexOffset:500})
      .addTo(mapRoute).bindPopup(`<b>#${i+1} ${lg.to}</b><br><small>${lg.to_cat||'-'}</small>`));
  }});
  if (rStats) {
    const avgSpeed = sh.travel_min > 0 ? (sh.distance_km / (sh.travel_min/60)) : 0;
    const avgStop  = sh.visited_count > 0 ? ((sh.service_min||0) / sh.visited_count) : 0;
    rStats.innerHTML = `
    <div class="stat good"><div class="k">Titik dikunjungi</div><div class="v">${sh.visited_count}</div></div>
    <div class="stat ${sh.feasible?'':'warn'}"><div class="k">Feasible</div><div class="v">${sh.feasible?'✓ Ya':'⚠ Tidak'}</div></div>
    <div class="stat"><div class="k">Total jarak</div><div class="v">${(sh.distance_km||0).toFixed(1)}<small> km</small></div></div>
    <div class="stat"><div class="k">Kecepatan rata-rata</div><div class="v">${avgSpeed.toFixed(1)}<small> km/jam</small></div></div>
    <div class="stat"><div class="k">Rata-rata berhenti</div><div class="v">${avgStop.toFixed(1)}<small> mnt/titik</small></div></div>
    <div class="stat"><div class="k">Travel</div><div class="v">${sh.travel_min}<small> mnt</small></div></div>
    <div class="stat"><div class="k">Service total</div><div class="v">${(sh.service_min||0).toFixed(0)}<small> mnt</small></div></div>
    <div class="stat"><div class="k">Total waktu</div><div class="v">${sh.total_min}<small> mnt</small></div></div>`;
  }
  if (!legsDiv) return;
  legsDiv.innerHTML = '';
  legs.forEach((lg, i) => {
    const isStop = i < legs.length-1;
    const col = isStop ? (CAT[lg.to_cat]||CAT['']) : CAT.depot;
    const num = isStop ? (i+1) : '★';
    const st = (lg.streets||[]).join(' → ') || '(jalan tak bernama)';
    const row = document.createElement('div'); row.className = 'leg';
    row.innerHTML = `<span class="num" style="background:${col}">${num}</span>
      <div class="route">${lg.from}<span class="arrow">→</span>${lg.to}</div><div class="streets">${st}</div>`;
    row.onclick = () => {
      document.querySelectorAll('.leg').forEach(r => r.classList.remove('active'));
      row.classList.add('active');
      if (hiLayer) mapRoute.removeLayer(hiLayer);
      hiLayer = L.polyline(lg.coords, {color:'#10b981', weight:7, opacity:.95, lineJoin:'round'}).addTo(mapRoute);
      mapRoute.fitBounds(hiLayer.getBounds().pad(.35));
    };
    legsDiv.appendChild(row);
  });
  setLegendSingle();
}

function renderAllVehicles(){
  if (!mapRoute) return;
  clearRoute();
  const allRuns = runsFor(modelSel.value);
  const shiftNum = String(shiftSel.value);
  if (!allRuns.length) return;
  const bounds = []; let totalVisited=0, feasOk=0, feasTotal=0, totalKm=0, totalTravelMin=0;
  allRuns.forEach(run => {
    const sh = run.shifts.find(s => String(s.shift) === shiftNum); if (!sh) return;
    const col = VEH_COLORS[run.vehicle] || '#64748b';
    const isMotor = run.vehicle === 'motor';
    totalVisited += sh.visited_count; feasTotal++;
    totalKm += (sh.distance_km||0); totalTravelMin += (sh.travel_min||0);
    if (sh.feasible) feasOk++;
    if (sh.route_coords && sh.route_coords.length) {
      rLayers.push(L.polyline(sh.route_coords, {
        color:col, weight:isMotor?3:5, opacity:.8, lineJoin:'round',
        dashArray:isMotor?'10 5':null
      }).addTo(mapRoute).bindPopup(
        `<b>${VEH_ICONS[run.vehicle]||''} ${run.vehicle.toUpperCase()} #${run.vehicle_unit}</b> — ${sh.visited_count} titik`));
      bounds.push(...sh.route_coords);
    }
    (sh.legs||[]).forEach((lg, i) => { if (i < (sh.legs||[]).length-1) {
      const c = lg.coords[lg.coords.length-1];
      rLayers.push(L.circleMarker(c, {radius:6, color:'#fff', weight:2, fillColor:col, fillOpacity:.9})
        .addTo(mapRoute)
        .bindPopup(`<b>${VEH_ICONS[run.vehicle]||''} ${run.vehicle} #${run.vehicle_unit}</b><br>#${i+1} ${lg.to}<br><small>${lg.to_cat||'-'}</small>`));
    }});
  });
  rLayers.push(L.marker(DATA.depot.coord, {icon:depotIcon(), zIndexOffset:1000})
    .addTo(mapRoute).bindPopup(`<b>DEPOT</b><br>${DATA.depot.label}`));
  if (bounds.length) mapRoute.fitBounds(L.latLngBounds(bounds).pad(.18));
  const fleetAvgSpeed = totalTravelMin > 0 ? (totalKm / (totalTravelMin/60)) : 0;
  if (rStats) rStats.innerHTML = `
    <div class="stat good"><div class="k">Total dikunjungi</div><div class="v">${totalVisited}</div></div>
    <div class="stat ${feasOk===feasTotal?'good':'warn'}"><div class="k">Feasible</div><div class="v">${feasOk}/${feasTotal}</div></div>
    <div class="stat"><div class="k">Total jarak armada</div><div class="v">${totalKm.toFixed(1)}<small> km</small></div></div>
    <div class="stat"><div class="k">Kecepatan rata-rata</div><div class="v">${fleetAvgSpeed.toFixed(1)}<small> km/jam</small></div></div>` +
    allRuns.map(run => {
      const sh = run.shifts.find(s => String(s.shift) === shiftNum);
      const col = VEH_COLORS[run.vehicle] || '#64748b';
      return sh ? `<div class="stat"><div class="k" style="color:${col}">${VEH_ICONS[run.vehicle]||''} ${run.vehicle} #${run.vehicle_unit}</div><div class="v" style="color:${col}">${sh.visited_count}<small> titik</small></div></div>` : '';
    }).join('');
  if (!legsDiv) return;
  legsDiv.innerHTML = '';
  allRuns.forEach(run => {
    const sh = run.shifts.find(s => String(s.shift) === shiftNum); if (!sh) return;
    const col = VEH_COLORS[run.vehicle] || '#64748b';
    const hdr = document.createElement('div'); hdr.className = 'leg-veh-hdr'; hdr.style.color = col;
    hdr.innerHTML = `<span class="leg-veh-line" style="background:${col}"></span>
      ${VEH_ICONS[run.vehicle]||''} ${run.vehicle.toUpperCase()} #${run.vehicle_unit} — ${sh.visited_count} titik`;
    legsDiv.appendChild(hdr);
    (sh.legs||[]).forEach((lg, i) => {
      const isStop = i < (sh.legs||[]).length-1;
      const num = isStop ? (i+1) : '★';
      const st = (lg.streets||[]).join(' → ') || '(jalan tak bernama)';
      const row = document.createElement('div'); row.className = 'leg';
      row.innerHTML = `<span class="num" style="background:${isStop?col:CAT.depot}">${num}</span>
        <div class="route">${lg.from}<span class="arrow">→</span>${lg.to}</div><div class="streets">${st}</div>`;
      row.onclick = () => {
        document.querySelectorAll('.leg').forEach(r => r.classList.remove('active'));
        row.classList.add('active');
        if (hiLayer) mapRoute.removeLayer(hiLayer);
        hiLayer = L.polyline(lg.coords, {color:col, weight:7, opacity:.95, lineJoin:'round'}).addTo(mapRoute);
        mapRoute.fitBounds(hiLayer.getBounds().pad(.35));
      };
      legsDiv.appendChild(row);
    });
  });
  setLegendAll(allRuns);
}

/* Mode toggle */
const vmSingleBtn = document.getElementById('vmSingle'), vmAllBtn = document.getElementById('vmAll');
if (vmSingleBtn) vmSingleBtn.onclick = () => {
  viewMode = 'single';
  vmSingleBtn.classList.add('active'); vmAllBtn.classList.remove('active');
  if (vehicleField) vehicleField.style.display = '';
  fillVehicles(); fillShifts(); renderRoute();
};
if (vmAllBtn) vmAllBtn.onclick = () => {
  viewMode = 'all';
  vmAllBtn.classList.add('active'); vmSingleBtn.classList.remove('active');
  if (vehicleField) vehicleField.style.display = 'none';
  fillShifts(); renderRoute();
};

const poolToggleEl = document.getElementById('poolToggle');
if (poolToggleEl) poolToggleEl.addEventListener('click', function(){
  showPool = !showPool;
  this.style.color = showPool ? '#10b981' : '';
  this.textContent = (showPool ? '● ' : '◌ ') + 'Semua titik';
  drawPool();
});
if (modelSel) modelSel.onchange = () => { fillVehicles(); fillShifts(); renderRoute(); };
if (vehicleSel) vehicleSel.onchange = () => { fillShifts(); renderRoute(); };
if (shiftSel) shiftSel.onchange = renderRoute;
fillVehicles(); fillShifts(); renderRoute();

/* ── SVG line chart ── */
/* cumulative max (best-so-far) — objektif yang dimaksimasi (titik) */
function bestSoFar(points){
  let mx = -Infinity;
  return points.map(p => { mx = Math.max(mx, p.y); return {x:p.x, y:mx, raw:p.y}; });
}
/* cumulative min (best-so-far) — cost yang diminimasi (jarak/waktu) */
function bestSoFarMin(points){
  let mn = Infinity;
  return points.map(p => { if(p.y>0) mn = Math.min(mn, p.y); return {x:p.x, y:(mn===Infinity?p.y:mn), raw:p.y}; });
}

function lineChart(host, series, opts){
  if (!host) return;
  opts = opts || {};
  const W = opts.w||460, H = opts.h||230, P = {l:42, r:14, t:14, b:34};
  const unit = opts.unit||'', anim = opts.anim !== false, xname = opts.xname||'x';
  const visible = series.filter(s => s.points && s.points.length);
  if (!visible.length){
    let leg = '';
    if (opts.legendAll) leg = '<div class="chart-legend">' + opts.legendAll.map(se => {
      const hidden = opts.hidden && opts.hidden.has(se.name);
      const click = opts.toggle ? ` style="cursor:pointer;opacity:${hidden?.4:1}" onclick="${opts.toggle}('${se.name.replace(/'/g,"")}')"` : '';
      return `<span${click}><i style="background:${se.color}"></i>${se.name}</span>`;
    }).join('') + '</div>';
    host.innerHTML = '<div style="padding:24px;text-align:center;font-size:11px;color:var(--muted)">Tidak ada data</div>' + leg;
    return;
  }
  const xs = visible.flatMap(s => s.points.map(p => p.x));
  const ys = visible.flatMap(s => s.points.map(p => p.y));
  const xmin = opts.xmin != null ? opts.xmin : Math.min(0, ...xs);
  const xmax = opts.xmax != null ? opts.xmax : Math.max(1, ...xs);
  const ymax = opts.ymax != null ? opts.ymax : Math.max(1, ...ys) * 1.12, ymin = 0;
  const sx = x => P.l + (x-xmin)/((xmax-xmin)||1) * (W-P.l-P.r);
  const sy = y => H - P.b - (y-ymin)/((ymax-ymin)||1) * (H-P.t-P.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet">`;
  for (let i=0; i<=4; i++) {
    const v = ymin + (ymax-ymin)*i/4, py = sy(v);
    s += `<line x1="${P.l}" y1="${py}" x2="${W-P.r}" y2="${py}" stroke="#f1f5f9"/>`;
    s += `<text x="${P.l-6}" y="${py+3}" text-anchor="end" font-size="9" font-weight="500" fill="#94a3b8">${Math.round(v)}</text>`;
  }
  s += `<text x="${W/2}" y="${H-6}" text-anchor="middle" font-size="10" font-weight="600" fill="#64748b">${opts.xlabel||''}</text>`;
  const dots = [];
  visible.forEach((se, si) => {
    const d = se.points.map((p,i) => (i?'L':'M') + sx(p.x).toFixed(1) + ' ' + sy(p.y).toFixed(1)).join(' ');
    const len = se.points.length * 60 + 80;
    const animStyle = anim ? `stroke-dasharray:${len};stroke-dashoffset:${len};animation:drawline .9s ${si*.12}s ease forwards` : '';
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2.4" stroke-linejoin="round" style="${animStyle}"/>`;
    se.points.forEach(p => {
      const cx = sx(p.x).toFixed(1), cy = sy(p.y).toFixed(1);
      const lbl = (se.name? se.name+' · ':'') + xname+' '+p.x+' → '+p.y+(unit?(' '+unit):'');
      dots.push(`<circle class="hot" cx="${cx}" cy="${cy}" r="3.2" fill="${se.color}" data-tip="${lbl.replace(/"/g,'&quot;')}" data-cx="${cx}" data-cy="${cy}"></circle>`);
    });
  });
  s += dots.join('');
  s += '</svg>';
  s += '<div class="chart-tip"></div>';
  const legSrc = opts.legendAll || visible;
  s += '<div class="chart-legend">' + legSrc.map(se => {
    const hidden = opts.hidden && opts.hidden.has(se.name);
    const click = opts.toggle ? ` style="cursor:pointer;opacity:${hidden?.4:1}" onclick="${opts.toggle}('${se.name.replace(/'/g,"")}')"` : '';
    return `<span${click}><i style="background:${se.color}"></i>${se.name}</span>`;
  }).join('') + '</div>';
  host.classList.add('svg-wrap');
  host.innerHTML = s;
  const tip = host.querySelector('.chart-tip'), svg = host.querySelector('svg');
  host.querySelectorAll('circle.hot').forEach(c => {
    c.addEventListener('mouseenter', () => {
      const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
      const px = +c.dataset.cx / vb.width * r.width, py = +c.dataset.cy / vb.height * r.height;
      tip.textContent = c.dataset.tip; tip.style.left = px+'px'; tip.style.top = py+'px'; tip.style.opacity = 1;
    });
    c.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
  });
}

function bar(label, pct, value, color){
  const w = Math.max(0, Math.min(100, pct));
  return `<div class="bar"><div class="bh"><span>${label}</span><b>${value}</b></div>
    <div class="track"><div class="fill" style="width:${w}%;background:${color}"></div></div></div>`;
}

function switchMetricView(view){
  activeMetricView = view;
  document.querySelectorAll('.chart-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.metric === view));
  drawSelectedCharts();
}

function capV(v){ return v ? v.charAt(0).toUpperCase()+v.slice(1) : v; }

let _cmpSort = {col:-1, asc:false};
function sortCmpTable(col){
  const tbl = document.getElementById('cmpTable'); if (!tbl) return;
  const ths = tbl.querySelectorAll('thead th');
  const kind = ths[col].dataset.kind;
  const asc = (_cmpSort.col === col) ? !_cmpSort.asc : (kind==='min');
  _cmpSort = {col, asc};
  ths.forEach((th,i)=>{ th.classList.toggle('sorted', i===col);
    const a=th.querySelector('.arr'); if(a) a.textContent = (i===col)?(asc?'▲':'▼'):'▼'; });
  const tb = tbl.querySelector('tbody');
  const rows = Array.from(tb.querySelectorAll('tr'));
  rows.sort((r1,r2) => {
    if (kind==='txt'){
      const t1=r1.cells[col].textContent.trim(), t2=r2.cells[col].textContent.trim();
      return asc? t1.localeCompare(t2) : t2.localeCompare(t1);
    }
    const v1=parseFloat(r1.cells[col].dataset.v)||0, v2=parseFloat(r2.cells[col].dataset.v)||0;
    return asc? v1-v2 : v2-v1;
  });
  rows.forEach(r=>tb.appendChild(r));
}

/* Sub-card drilldown: timeline per iterasi (1..N) */
const SUBCARD_META = {
  runtime:      {key:'runtime_ms',        title:'⚡ Runtime per Iterasi',        unit:'ms',        color:'#4338ca', dec:1},
  throughput:   {key:'throughput',        title:'🛞 Throughput per Iterasi',     unit:'titik/jam', color:'#2563eb', dec:2},
  feasibility:  {key:'feasible_pct',      title:'✅ Feasible per Iterasi',        unit:'%',         color:'#16a34a', dec:1},
  distance:     {key:'distance_km',       title:'🛣️ Jarak per Iterasi',          unit:'km',        color:'#0ea5e9', dec:2},
  speed:        {key:'speed_kph',         title:'💨 Kecepatan per Iterasi',       unit:'km/jam',    color:'#0891b2', dec:1},
  timeutil:     {key:'time_util_pct',     title:'⏱️ Utilisasi Waktu per Iterasi', unit:'%',         color:'#9333ea', dec:1},
  spatial:      {key:'spatial_eff',       title:'🎯 Efisiensi per Iterasi',       unit:'titik/km',  color:'#d97706', dec:2},
  serviceratio: {key:'service_ratio_pct', title:'🧰 Rasio Layanan per Iterasi',   unit:'%',         color:'#0d9488', dec:1},
  stability:    {key:'coverage',          title:'📊 Coverage per Iterasi (sebaran → std)', unit:'titik', color:'#0f766e', dec:1},
};

/* Registry metrik untuk Analisis Multimetrik (per-iterasi + konvergensi bila ada) */
const METRICS = {
  stability:    {label:'📊 Std/Stabilitas', iter:'coverage',         unit:'titik',     conv:'visited',    max:true},
  coverage:     {label:'🎯 Coverage',      iter:'coverage',          unit:'titik',     conv:'visited',    max:true},
  runtime:      {label:'⚡ Runtime',        iter:'runtime_ms',        unit:'ms'},
  throughput:   {label:'🛞 Throughput',     iter:'throughput',        unit:'titik/jam'},
  distance:     {label:'🛣️ Jarak',          iter:'distance_km',       unit:'km',        conv:'travel_min', max:false},
  speed:        {label:'💨 Kecepatan',      iter:'speed_kph',         unit:'km/jam'},
  spatial:      {label:'🎯 Efisiensi',      iter:'spatial_eff',       unit:'titik/km'},
  serviceratio: {label:'🧰 Rasio Layanan',  iter:'service_ratio_pct', unit:'%'},
  timeutil:     {label:'⏱️ Utilisasi',      iter:'time_util_pct',     unit:'%'},
  feasible:     {label:'✅ Feasible',        iter:'feasible_pct',      unit:'%'},
  waktu:        {label:'⏱️ Total Waktu',    iter:'total_min',         unit:'mnt',       conv:'total_min',  max:false},
};
function toggleSubcard(idx, metric, pill){
  const panel = document.getElementById('subcard-'+idx);
  const inner = document.getElementById('subcard-inner-'+idx);
  const m = DATA.dashboard.models[idx]; if (!panel||!inner||!m) return;
  const pills = pill.parentElement.querySelectorAll('.metric-pill');
  const already = panel.classList.contains('open') && panel.dataset.metric === metric;
  if (already){ panel.classList.remove('open'); pills.forEach(p=>p.classList.remove('sel')); panel.dataset.metric=''; return; }
  pills.forEach(p=>p.classList.remove('sel')); pill.classList.add('sel');
  panel.dataset.metric = metric;

  if (metric === 'convergence'){
    const keys = Object.keys(m.convergence_by_shift||{});
    const cmv = m.convergence_meta || {};
    const ident = cmv.vehicle ? `${capV(cmv.vehicle)} unit ${cmv.unit}, iterasi #${cmv.iter}` : '—';
    const shiftBtns = keys.map((k,i)=>`<button class="chart-btn conv-shift-btn ${i===0?'active':''}" style="min-width:auto;padding:5px 14px" onclick="drawConvSub(${idx},'${k}',this)">Shift ${k}</button>`).join('');
    inner.innerHTML = `<div class="conv-wrap">
      <div class="conv-bar">
        <div>
          <div class="dt-title" style="margin:0">🚀 Konvergensi per Generasi (best-so-far)</div>
          <div class="acc-note" style="margin-top:2px">Run terbaik: <b>${ident}</b> · Konvergensi ${m.convergence_speed_pct}% (seberapa dini capai 95% kualitas)</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">${shiftBtns}</div>
      </div>
      <div class="conv-charts">
        <div class="conv-cell"><div class="conv-cap up">🎯 Titik dikunjungi <span>maksimasi ↑</span></div><div id="sc-cv-vis-${idx}" style="min-height:150px"></div></div>
        <div class="conv-cell"><div class="conv-cap down">🛣️ Travel time <span>minimasi ↓</span></div><div id="sc-cv-trav-${idx}" style="min-height:150px"></div></div>
        <div class="conv-cell"><div class="conv-cap down">⏱️ Total waktu <span>minimasi ↓</span></div><div id="sc-cv-tot-${idx}" style="min-height:150px"></div></div>
      </div>
      <div class="acc-note">Objektif (titik) naik → makin baik; cost (travel/total) turun → makin baik. Tiap kurva = nilai best-so-far sepanjang generasi pencarian untuk shift terpilih.</div>
    </div>`;
    drawConvSub(idx, keys[0] || '1', null);
  } else {
    const cfg = SUBCARD_META[metric];
    const pts = (m.per_iter_detail||[]).map(d => ({x:d.iter, y:d[cfg.key]}));
    const vals = pts.map(p=>p.y);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const mean = vals.reduce((a,b)=>a+b,0)/(vals.length||1);
    inner.innerHTML = `<div><div class="dt-title">${cfg.title}</div>
      <div id="sc-ts-${idx}" style="min-height:180px"></div></div>
      <div><div class="dt-title">Ringkasan ${pts.length} iterasi</div>
      <div class="subcard-stats">
        <div class="s">Min<b>${mn.toFixed(cfg.dec)} ${cfg.unit}</b></div>
        <div class="s">Rata-rata<b>${mean.toFixed(cfg.dec)} ${cfg.unit}</b></div>
        <div class="s">Maks<b>${mx.toFixed(cfg.dec)} ${cfg.unit}</b></div>
      </div>
      <div class="table-container" style="max-height:200px"><table class="iter-table"><thead><tr><th>Iter</th><th>${cfg.unit}</th></tr></thead>
      <tbody>${pts.map(p=>`<tr><td>#${p.x}</td><td><b>${(+p.y).toFixed(cfg.dec)}</b></td></tr>`).join('')}</tbody></table></div></div>`;
    lineChart(document.getElementById('sc-ts-'+idx),
      [{name:m.display_name, color:cfg.color, points:pts}],
      {xmin:1, xmax: Math.max(2, pts.length), xlabel:'Iterasi', xname:'iter', unit:cfg.unit, w:380, h:180});
  }
  panel.classList.add('open');
}

function filterShiftTable(idx){
  const veh = (document.getElementById('fveh-'+idx)||{}).value || '';
  const sh  = (document.getElementById('fshift-'+idx)||{}).value || '';
  const tbl = document.getElementById('shifttbl-'+idx); if (!tbl) return;
  tbl.querySelectorAll('tbody tr').forEach(tr => {
    const okV = !veh || tr.dataset.veh === veh;
    const okS = !sh  || tr.dataset.shift === sh;
    tr.style.display = (okV && okS) ? '' : 'none';
  });
}

function drawConvSub(idx, key, btn){
  const m = DATA.dashboard.models[idx]; if (!m) return;
  if (btn){ btn.parentElement.querySelectorAll('.conv-shift-btn').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); }
  const frames = (m.convergence_by_shift||{})[key] || [];
  const hVis = document.getElementById('sc-cv-vis-'+idx);
  const hTrav= document.getElementById('sc-cv-trav-'+idx);
  const hTot = document.getElementById('sc-cv-tot-'+idx);
  if (!frames.length){
    [hVis,hTrav,hTot].forEach(h=>{ if(h) h.innerHTML='<div style="padding:20px;text-align:center;font-size:11px;color:var(--muted)">Tidak ada data</div>'; });
    return;
  }
  const o = {xmin:0, xmax:100, xlabel:'Progres pencarian (%)', xname:'progres', w:320, h:150};
  lineChart(hVis,  [{name:'Shift '+key, color:'#16a34a', points: bestSoFar(frames.map(c=>({x:c.progress,y:c.visited})))}],     Object.assign({unit:'titik'}, o));
  lineChart(hTrav, [{name:'Shift '+key, color:'#2563eb', points: bestSoFarMin(frames.map(c=>({x:c.progress,y:c.travel_min||0})))}], Object.assign({unit:'mnt'}, o));
  lineChart(hTot,  [{name:'Shift '+key, color:'#7c3aed', points: bestSoFarMin(frames.map(c=>({x:c.progress,y:c.total_min||0})))}],  Object.assign({unit:'mnt'}, o));
}

function toggleAccordion(idx, btn){
  const content = document.getElementById(`accordion-${idx}`);
  if (!content) return;
  const wasOpen = content.classList.contains('open');
  content.classList.toggle('open');
  if (btn) btn.classList.toggle('open', !wasOpen);
}

/* ── TAB DASHBOARD ── */
function renderDashboard(){
  const grid = document.getElementById('dashGrid'); if (!grid) return;
  const D = DATA.dashboard;
  if (!D || !D.models || !D.models.length){
    grid.innerHTML = `<div class="empty"><h2>Belum ada data dashboard</h2>
      <p>Jalankan <code>python main.py train</code> untuk menghasilkan metrik performa.</p></div>`;
    return;
  }
  const M = D.models, target = D.target;

  /* ── Podium: tiap entri = 1 kendaraan-run (iterasi × kendaraan, gabung 2 shift) ── */
  let allRuns = [];
  M.forEach(m => {
    const psd = m.per_shift_detail || [];
    if (psd.length) {
      const agg = {};
      psd.forEach(d => {
        const k = d.iter + '|' + d.vehicle + '|' + d.unit;
        (agg[k] = agg[k] || {iter:d.iter, vehicle:d.vehicle, unit:d.unit, score:0, km:0, rt:0, n:0, feas:0});
        const a = agg[k]; a.score += d.visited; a.km += d.distance_km; a.rt += d.runtime_ms; a.n++; a.feas += d.feasible?1:0;
      });
      Object.values(agg).forEach(a => allRuns.push({ algo:m.display_name, icon:m.icon, color:m.color,
        iter:a.iter, vehicle:a.vehicle, unit:a.unit, score:a.score, km:+a.km.toFixed(2),
        runtime_ms:+(a.rt/a.n).toFixed(1), feasible:(a.feas===a.n) }));
    } else {
      (m.per_iter_detail||[]).forEach(d => allRuns.push({ algo:m.display_name, icon:m.icon, color:m.color,
        iter:d.iter, vehicle:null, unit:null, score:d.coverage, km:d.distance_km, runtime_ms:d.runtime_ms, feasible:d.feasible_pct>=100 }));
    }
  });
  allRuns.sort((a,b) => (b.score - a.score) || (a.km - b.km));
  const top3 = allRuns.slice(0,3);
  const nextBest = allRuns.slice(3,10);
  const vehId = it => it.vehicle ? `${capV(it.vehicle)} ${it.unit}` : '';

  let html = `<div class="card">
    <h2>👑 Global Best Iteration Podium</h2>
    <div class="sub">Kendaraan-run paling optimal (gabungan 2 shift, no-overlap)</div>
    <div class="podium-wrapper">`;

  const podiumRanks = [
    {item: top3[1], rankClass:'rank-2', badge:'RUNNER UP'},
    {item: top3[0], rankClass:'rank-1', badge:'GLOBAL BEST'},
    {item: top3[2], rankClass:'rank-3', badge:'3RD PLACE'},
  ];
  podiumRanks.forEach(({item, rankClass, badge}) => {
    if (!item) return;
    html += `<div class="podium-box ${rankClass}">
      <span class="podium-badge">${badge}</span>
      <div class="podium-title">${item.icon} Iterasi #${item.iter}</div>
      <div class="podium-algo">${item.algo}</div>
      ${item.vehicle ? `<div class="podium-veh"><span class="veh-tag veh-${item.vehicle}">${vehId(item)}</span> · ${item.feasible?'✓ feasible':'⚠ belum'}</div>` : ''}
      <div class="podium-opt-metrics">${item.score} <span>Titik Dikunjungi</span></div>
      ${item.km > 0 ? `<div class="podium-opt-metrics" style="font-size:15px;margin-top:4px">${item.km} <span>Kilometer</span></div>` : ''}
      <div class="podium-footer">
        <div class="podium-stat"><span>Kendaraan</span>${vehId(item)||'—'}</div>
        <div class="podium-stat"><span>Runtime</span>${item.runtime_ms} ms</div>
      </div>
    </div>`;
  });
  html += `</div>`;

  if (nextBest.length) {
    html += `<div class="podium-other"><div class="podium-other-title">Top 4–10 Global Best</div><div class="podium-list">`;
    nextBest.forEach((item, idx) => {
      html += `<div class="podium-item">
        <div class="rank-chip">#${idx+4}</div>
        <div class="algo-name">${item.icon} ${item.algo}</div>
        <div class="algo-meta">Iterasi #${item.iter}${item.vehicle?' · '+vehId(item):''} · ${item.score} titik</div>
        ${item.km > 0 ? `<div class="mini-stat"><span>Km</span><span>${item.km}</span></div>` : ''}
        <div class="mini-stat"><span>Runtime</span><span>${item.runtime_ms} ms</span></div>
      </div>`;
    });
    html += `</div></div>`;
  }
  html += `</div>`;

  /* ── Leaderboard ── */
  html += `<div class="card">
    <h2>🏆 Leaderboard Komparasi Algoritma</h2>
    <div class="sub">Peringkat performa terintegrasi (55% Coverage · 20% Stabilitas/std · 15% Efisiensi · 10% Success)</div>
    <div class="lb">`;

  M.forEach((m, idx) => {
    const cm = m.convergence_meta || {};
    const bestIterNum = cm.iter || 1;
    const bestTotal = cm.vehicle_total != null ? cm.vehicle_total
      : ((m.per_iter_detail||[]).length ? Math.max(...m.per_iter_detail.map(d=>d.coverage)) : Math.max(...(m.per_iteration||[0])));
    const bestId = cm.vehicle ? `${capV(cm.vehicle)} unit ${cm.unit}, iterasi #${cm.iter}` : `iterasi #${bestIterNum}`;

    /* per-shift identity rows */
    const psd = m.per_shift_detail || [];
    let shiftRows = '';
    psd.forEach(d => {
      const isBest = cm.vehicle && d.iter===cm.iter && d.vehicle===cm.vehicle && d.unit===cm.unit;
      shiftRows += `<tr class="${isBest?'best-row':''}" data-veh="${d.vehicle}" data-shift="${d.shift}">
        <td>#${d.iter}${isBest?' ⭐':''}</td>
        <td><span class="veh-tag veh-${d.vehicle}">${capV(d.vehicle)} ${d.unit}</span></td>
        <td>Shift ${d.shift}</td>
        <td><b>${d.visited}</b></td>
        <td>${d.distance_km>0? d.distance_km+' km':'—'}</td>
        <td>${d.runtime_ms} ms</td>
        <td class="${d.feasible?'yes':'no'}">${d.feasible?'✓':'✗'}</td>
      </tr>`;
    });
    const shiftKeys = Object.keys(m.convergence_by_shift||{});

    html += `<div class="lb-card pos-${idx}">
      <div class="lb-rank">${m.medal || ('#'+m.rank)}</div>
      <div class="lb-avatar" style="background:${m.color}1f;color:${m.color}">${m.icon}</div>
      <div style="width:100%">
        <div class="lb-name">${m.display_name} <span class="tier tier-${m.tier}">TIER ${m.tier}</span></div>
        <div class="lb-sub">${m.tagline}</div>
        <div class="best-run-banner">🎯 Best Run: ${bestId} — <strong>${bestTotal} titik</strong></div>
        <div class="lb-section">
          <div class="bars-vertical">
            ${bar('Coverage vs Target', m.coverage_avg/target*100, m.coverage_avg+' / '+target, m.color)}
          </div>
          ${(()=>{const cs=(m.per_iter_detail||[]).map(d=>d.coverage);const mx=cs.length?Math.max(...cs):0,mn=cs.length?Math.min(...cs):0;return `<div class="lb-std">📊 Std antar ${D.n_iterations} iterasi: <b>${m.coverage_std} titik</b> · coverage/iterasi ${mn}–${mx} titik · makin kecil std makin stabil</div>`;})()}
          <div class="metrics-grid">
            <div class="metric-pill runtime" onclick="toggleSubcard(${idx},'runtime',this)"><span class="icon">⚡</span> Runtime: <b>${m.avg_runtime_ms} ms/shift</b><span class="chev">▼</span></div>
            <div class="metric-pill convergence" onclick="toggleSubcard(${idx},'convergence',this)"><span class="icon">🚀</span> Konvergensi: <b>${m.convergence_speed_pct}%</b><span class="chev">▼</span></div>
            <div class="metric-pill throughput" onclick="toggleSubcard(${idx},'throughput',this)"><span class="icon">🛞</span> Throughput: <b>${m.throughput} titik/jam</b><span class="chev">▼</span></div>
            <div class="metric-pill feasibility" onclick="toggleSubcard(${idx},'feasibility',this)"><span class="icon">✅</span> Feasible: <b>${m.feasible_pct}%</b><span class="chev">▼</span></div>
            <div class="metric-pill distance" onclick="toggleSubcard(${idx},'distance',this)"><span class="icon">🛣️</span> Jarak: <b>${m.avg_distance_km} km/run</b><span class="chev">▼</span></div>
            <div class="metric-pill speed" onclick="toggleSubcard(${idx},'speed',this)"><span class="icon">💨</span> Kecepatan: <b>${m.avg_speed_kph} km/jam</b><span class="chev">▼</span></div>
            <div class="metric-pill spatial" onclick="toggleSubcard(${idx},'spatial',this)"><span class="icon">🎯</span> Efisiensi: <b>${m.spatial_eff} titik/km</b><span class="chev">▼</span></div>
            <div class="metric-pill serviceratio" onclick="toggleSubcard(${idx},'serviceratio',this)"><span class="icon">🧰</span> Rasio Layanan: <b>${m.service_ratio_pct}%</b><span class="chev">▼</span></div>
            <div class="metric-pill stability" onclick="toggleSubcard(${idx},'stability',this)"><span class="icon">📊</span> Std: <b>${m.coverage_std} titik</b><span class="chev">▼</span></div>
          </div>
          <div class="subcard-panel" id="subcard-${idx}"><div class="subcard-inner" id="subcard-inner-${idx}"></div></div>
        </div>
        <div style="margin-top:14px">
          <button class="accordion-trigger" onclick="toggleAccordion(${idx},this)"><span class="caret">▼</span> Detail Per Shift (identitas kendaraan &amp; iterasi)</button>
          <div class="accordion-content" id="accordion-${idx}">
            <div class="acc-single">
              <div class="acc-head">
                <div class="dt-title" style="margin:0">📋 Detail per Shift — ${psd.length} baris · ${m.per_iter_detail.length} iterasi</div>
                <div class="acc-filters">
                  <label>Kendaraan</label>
                  <select id="fveh-${idx}" onchange="filterShiftTable(${idx})">
                    <option value="">Semua</option><option value="motor">Motor</option><option value="mobil">Mobil</option>
                  </select>
                  <label>Shift</label>
                  <select id="fshift-${idx}" onchange="filterShiftTable(${idx})">
                    <option value="">Semua</option>${shiftKeys.map(k=>`<option value="${k}">Shift ${k}</option>`).join('')}
                  </select>
                </div>
              </div>
              <div class="table-container">
                <table class="iter-table shift-table" id="shifttbl-${idx}">
                  <thead><tr>
                    <th>Iter</th><th>Kendaraan</th><th>Shift</th><th>Titik</th><th>Jarak</th><th>Runtime</th><th>Feasible</th>
                  </tr></thead>
                  <tbody>${shiftRows}</tbody>
                </table>
              </div>
              <div class="acc-note">⭐ = kendaraan-run terbaik (${bestId}, ${bestTotal} titik gabungan 2 shift). Kurva konvergensi ada di sub-card <b>🚀 Konvergensi</b> di atas.</div>
            </div>
          </div>
        </div>
      </div>
      <div style="padding-left:16px">
        <div class="lb-score">
          <div class="sn">${m.overall_score}</div>
          <div class="sl">OVERALL SCORE</div>
        </div>
      </div>
    </div>`;
  });
  html += `</div></div>`;

  /* ── Feasibility Roadmap (kondisi mencapai target) ── */
  const F = D.feasibility;
  let htmlFeas = '';
  if (F && F.vehicles && F.vehicles.length) {
    htmlFeas += `<div class="card">
      <h2>🧭 Peta Jalan Feasibility</h2>
      <div class="sub">Agar tiap kendaraan mencapai target <b>${F.target} titik</b>: rekomendasi dihitung dari <b>hasil model terbaik</b> (${D.n_iterations} iterasi) — bukan greedy.</div>
      <div class="feas-grid">`;
    F.vehicles.forEach(v => {
      const ok = v.status === 'TERCAPAI';
      const r = v.rec || {};
      const lever = (label, val) => `<div class="feas-lever ${val==null?'na':''}"><div class="lk">${label}</div><div class="lv">${val==null?'—':val}</div></div>`;

      let middle;
      if (ok) {
        middle = `<div class="feas-ok-note">✓ Target sudah dicapai model (${v.best_model} ≥ ${v.target}) — tak perlu perubahan.</div>`;
      } else {
        middle = `<div class="dt-title">Agar ${capV(v.vehicle)} feasible (≥ ${v.target} titik), pilih SALAH SATU <span class="feas-tag">estimasi dari hasil model</span></div>
          <div class="feas-levers">
            ${lever('➕ Tambah kendaraan', r.add_vehicles!=null ? '+'+r.add_vehicles+' unit' : null)}
            ${lever('⏱️ Perpanjang shift', r.shift_hours!=null ? '≥ '+r.shift_hours+' jam' : null)}
            ${lever('🧰 Waktu/titik', r.service_min!=null ? '≤ '+r.service_min+' mnt' : null)}
          </div>`;
      }

      htmlFeas += `<div class="feas-card ${ok?'ok':'bad'}">
        <div class="feas-head">
          <div class="name">${capV(v.vehicle)} — terbaik <b>${v.best_model}</b>/${v.target} titik</div>
          <span class="feas-badge ${ok?'ok':'bad'}">${ok?'Target tercapai':'Belum tercapai'}</span>
        </div>
        <div class="feas-narr">${v.narrative}</div>
        ${middle}
      </div>`;
    });
    htmlFeas += `</div></div>`;
  }

  /* ── Multi-metric charts ── */
  let htmlMulti = `<div class="card">
    <div style="margin-bottom:8px"><h2>📈 Analisis Multimetrik</h2></div>
    <div class="chart-selector-wrapper">
      ${Object.entries(METRICS).map(([k,mt]) => `<button class="chart-btn ${activeMetricView===k?'active':''}" data-metric="${k}" onclick="switchMetricView('${k}')">${mt.label}</button>`).join('')}
    </div>
    <div class="sub">Pilih metrik apa saja. Kiri: konvergensi best-so-far selama pencarian (hanya Coverage/Jarak/Total Waktu yang dioptimasi langsung). Kanan: tren metrik per iterasi (1..N), semua model — klik legenda untuk sembunyikan/tampilkan, arahkan kursor untuk nilai.</div>
    <div class="charts-layout">
      <div class="chart-box">
        <div class="chart-title" id="titleConv">📉 Konvergensi</div>
        <div id="chartConv"></div>
      </div>
      <div class="chart-box">
        <div class="chart-title" id="titleIter">📊 Metrik per Iterasi</div>
        <div id="chartIter"></div>
      </div>
    </div>
  </div>`;

  /* ── Comparison table (sortable) ── */
  /* Flatten jarak & waktu rata-rata per kendaraan (motor/mobil) ke properti model */
  M.forEach(m => {
    const bv = m.by_vehicle || {};
    m.dist_motor = bv.motor ? bv.motor.distance_km : null;
    m.dist_mobil = bv.mobil ? bv.mobil.distance_km : null;
    m.time_motor = bv.motor ? bv.motor.total_min  : null;
    m.time_mobil = bv.mobil ? bv.mobil.total_min  : null;
  });
  const cmpCols = [
    {label:'Model', key:'display_name', kind:'txt'},
    {label:'Coverage', key:'coverage_avg', kind:'max'},
    {label:'Rekor', key:'coverage_best', kind:'max'},
    {label:'Target %', key:'target_attainment_pct', kind:'plain', suf:'%'},
    {label:'Std (titik)', key:'coverage_std', kind:'min'},
    {label:'Success', key:'success_rate_pct', kind:'plain', suf:'%'},
    {label:'Titik/jam', key:'throughput', kind:'max'},
    {label:'Titik/km', key:'spatial_eff', kind:'max'},
    {label:'Runtime', key:'avg_runtime_ms', kind:'min', suf:' ms'},
    {label:'🛣️ Jarak Motor', key:'dist_motor', kind:'plain', suf:' km'},
    {label:'🛣️ Jarak Mobil', key:'dist_mobil', kind:'plain', suf:' km'},
    {label:'⏱️ Waktu Motor', key:'time_motor', kind:'plain', suf:' mnt'},
    {label:'⏱️ Waktu Mobil', key:'time_mobil', kind:'plain', suf:' mnt'},
    {label:'Feasible', key:'feasible_pct', kind:'plain', suf:'%'},
    {label:'Skor', key:'overall_score', kind:'max', bold:true},
  ];
  let htmlCmp = `<div class="card">
    <h2>📋 Tabel Perbandingan Lengkap</h2>
    <div class="sub">Semua indikator objektif per model (termasuk rata-rata jarak &amp; waktu per kendaraan motor/mobil) — klik header untuk mengurutkan</div>
    <div class="table-card" style="overflow-x:auto"><table class="cmp" id="cmpTable"><thead><tr>` +
    cmpCols.map((c,i) => `<th class="sortable" onclick="sortCmpTable(${i})" data-key="${c.key}" data-kind="${c.kind}">${c.label}<span class="arr">▼</span></th>`).join('') +
    `</tr></thead><tbody>`;
  const bestOf = k => Math.max(...M.map(m => m[k]));
  const minOf  = k => Math.min(...M.map(m => m[k]));
  const fmtCell = (m, c) => {
    const v = m[c.key];
    if (c.kind==='txt') return `<td><b>${m.icon} ${m.display_name}</b></td>`;
    if (v==null) return `<td data-v="0">—</td>`;
    let cls = '';
    if (c.kind==='max' && v===bestOf(c.key)) cls='best';
    if (c.kind==='min' && v===minOf(c.key)) cls='best';
    const inner = (c.bold? `<b>${v}${c.suf||''}</b>` : `${v}${c.suf||''}`);
    return `<td class="${cls}" data-v="${v}">${inner}</td>`;
  };
  M.forEach(m => { htmlCmp += '<tr>' + cmpCols.map(c => fmtCell(m,c)).join('') + '</tr>'; });
  htmlCmp += `</tbody></table></div></div>`;

  /* ── Susun urutan kartu: Leaderboard → Tabel Perbandingan → Multimetrik → Feasibility ── */
  html += htmlCmp + htmlMulti + htmlFeas;

  /* ── Insights (dynamic from D.insights) ── */
  const insClasses = ['active-accent','active-success','active-warning'];
  html += `<div class="card">
    <h2>💡 Operational Insight</h2>
    <div class="sub">Ringkasan evaluasi strategis berdasarkan hasil training</div>
    <div class="insights-vertical">
      ${(D.insights||[]).map((t, i) => `<div class="insight-row ${insClasses[i % insClasses.length]}">
        <div class="insight-desc">${t}</div>
      </div>`).join('')}
    </div>
  </div>`;

  /* ── Metric guide ── */
  html += `<div class="card"><h2>📖 Panduan Indikator</h2><div class="sub">Arti tiap metrik evaluasi</div>
    <div class="guide">${(D.metric_guide||[]).map(g => `<div class="g"><b>${g.label}</b><p>${g.desc}</p></div>`).join('')}</div></div>`;

  grid.innerHTML = html;
  drawSelectedCharts();
}

const hiddenModels = new Set();
function toggleSeries(name){
  if (hiddenModels.has(name)) hiddenModels.delete(name); else hiddenModels.add(name);
  drawSelectedCharts();
}

function drawSelectedCharts(){
  const D = DATA.dashboard; if (!D) return;
  const M = D.models;
  const hostConv  = document.getElementById('chartConv');
  const hostIter  = document.getElementById('chartIter');
  const titleConv = document.getElementById('titleConv');
  const titleIter = document.getElementById('titleIter');
  if (!hostConv || !hostIter) return;
  const ALL = M.map(m => ({name: m.display_name, color: m.color}));
  const LC = (host, series, opts) => lineChart(host,
    series.filter(s => !hiddenModels.has(s.name)),
    Object.assign({legendAll: ALL, hidden: hiddenModels, toggle: 'toggleSeries'}, opts));

  const meta = METRICS[activeMetricView] || METRICS.coverage;

  /* Kanan: tren metrik per iterasi (semua model) */
  if (titleIter) titleIter.innerHTML = `📊 ${meta.label} per Iterasi (${meta.unit})`;
  const iterSeries = M.map(m => ({ name: m.display_name, color: m.color,
    points: (m.per_iter_detail||[]).map(d => ({x: d.iter, y: d[meta.iter]||0})) }));
  LC(hostIter, iterSeries, {xmin:1, xmax: Math.max(2, D.n_iterations), xlabel:'Iterasi', xname:'iter', unit:meta.unit});

  /* Kiri: konvergensi best-so-far — hanya untuk metrik yang dioptimasi langsung */
  if (meta.conv) {
    const unit = meta.conv==='visited' ? 'titik' : 'mnt';
    const arah = meta.max ? 'maksimasi ↑' : 'minimasi ↓';
    if (titleConv) titleConv.innerHTML = `📉 Konvergensi ${meta.label} (best-so-far, ${arah})`;
    const fn = meta.max ? bestSoFar : bestSoFarMin;
    const convSeries = M.filter(m => (m.convergence||[]).length).map(m => ({
      name: m.display_name, color: m.color,
      points: fn(m.convergence.map(c => ({x: c.progress, y: c[meta.conv]||0}))) }));
    if (convSeries.length) {
      LC(hostConv, convSeries, {xmin:0, xmax:100, xlabel:'Progres Pencarian (%)', xname:'progres', unit:unit});
    } else {
      hostConv.innerHTML = '<div style="padding:30px;text-align:center;font-size:12px;color:var(--muted)">Data konvergensi tidak tersedia</div>';
    }
  } else {
    if (titleConv) titleConv.innerHTML = '📉 Konvergensi (tidak tersedia)';
    hostConv.innerHTML = `<div style="padding:26px;text-align:center;font-size:12px;color:var(--muted);line-height:1.7">
      Konvergensi per-generasi hanya untuk metrik yang <b>dioptimasi langsung</b> saat pencarian:
      <b>Coverage</b> (titik), <b>Jarak</b> (travel time) &amp; <b>Total Waktu</b>.<br>
      <b>${meta.label}</b> adalah metrik turunan — lihat trennya antar iterasi di panel kanan,
      atau buka sub-card <b>${meta.label}</b> di Leaderboard untuk timeline + ringkasan.</div>`;
  }
}

/* ── TAB PETA TITIK ── */
function renderPoints(){
  if (!document.getElementById('mapPoints')) return;
  mapPoints = L.map('mapPoints', {zoomControl:true}); baseTiles(mapPoints);
  const pool = DATA.pool || [];
  const counts = {}; pool.forEach(p => counts[p.cat] = (counts[p.cat]||0)+1);
  const grp = L.featureGroup().addTo(mapPoints);
  pool.forEach(p => L.circleMarker(p.coord, {radius:5, color:'#fff', weight:1.5,
    fillColor:CAT[p.cat]||'#888', fillOpacity:.9})
    .bindPopup(`<b>${p.label}</b><br><small>${p.cat}</small>`).addTo(grp));
  if (DATA.depot) L.marker(DATA.depot.coord, {icon:depotIcon(), zIndexOffset:1000})
    .bindPopup(`<b>DEPOT</b><br>${DATA.depot.label}`).addTo(mapPoints);
  if (pool.length) mapPoints.fitBounds(grp.getBounds().pad(.12));
  else if (DATA.depot) mapPoints.setView(DATA.depot.coord, 12);

  const side = document.getElementById('pointsSide'); if (!side) return;
  side.innerHTML = `<h2>📍 Titik Kandidat</h2>
    <div class="sub">Titik tetap tersebar di Surabaya — objektif dimaksimasi</div>
    <div class="pcount"><div class="ic" style="background:#fde8e8">🚨</div>
      <div><div class="n" style="color:${CAT.emergency}">${counts.emergency||0}</div><div class="l">Emergency (Polsek + Damkar)</div></div></div>
    <div class="pcount"><div class="ic" style="background:#e7efff">🚌</div>
      <div><div class="n" style="color:${CAT.transport}">${counts.transport||0}</div><div class="l">Transport (Terminal + Halte)</div></div></div>
    <div class="pcount"><div class="ic" style="background:#e7f8ee">★</div>
      <div><div class="n" style="color:${CAT.depot}">1</div><div class="l">Depot (${(DATA.depot||{}).label||''})</div></div></div>
    <div class="pcount"><div class="ic" style="background:#eef1f6">Σ</div>
      <div><div class="n">${pool.length}</div><div class="l">Total titik kandidat</div></div></div>
    <div class="legend" style="margin-top:10px">
      <span class="it"><span class="dot" style="background:var(--emergency)"></span>Emergency</span>
      <span class="it"><span class="dot" style="background:var(--transport)"></span>Transport</span>
      <span class="it"><span class="dot" style="background:var(--depot)"></span>Depot</span></div>`;
  setTimeout(() => mapPoints.invalidateSize(), 100);
}
</script>
</body>
</html>"""
