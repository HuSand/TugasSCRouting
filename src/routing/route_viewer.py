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
    log.info(f"   Route viewer -> route_viewer.html")
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
  .metrics-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:14px}
  .metric-pill{display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:14px;color:#fff;font-size:12.5px;font-weight:700;box-shadow:0 10px 24px rgba(15,23,42,.08)}
  .metric-pill span.icon{font-size:14px}
  .metric-pill.runtime{background:linear-gradient(135deg,#312e81,#4338ca)}
  .metric-pill.convergence{background:linear-gradient(135deg,#0f766e,#115e59)}
  .metric-pill.throughput{background:linear-gradient(135deg,#1d4ed8,#2563eb)}
  .metric-pill.feasibility{background:linear-gradient(135deg,#15803d,#166534)}
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

  /* Accordion */
  .accordion-trigger{background:#fff;border:1px solid var(--line);color:var(--ink);font-size:11.5px;font-weight:700;
    cursor:pointer;padding:8px 16px;border-radius:8px;display:inline-flex;align-items:center;gap:6px;outline:none;transition:.12s;box-shadow:0 1px 2px rgba(0,0,0,.03)}
  .accordion-trigger:hover{background:#f8fafc;border-color:#cbd5e1}
  .accordion-content{display:none;margin-top:14px;border-top:1.5px dashed var(--line);padding-top:14px}
  .accordion-content.open{display:block}
  .table-container{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.02)}
  .iter-table{width:100%;border-collapse:collapse;font-size:12px;text-align:center}
  .iter-table th,.iter-table td{padding:10px 12px;border-bottom:1px solid var(--line)}
  .iter-table th{background:#f8fafc;color:var(--muted);font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.5px}
  .iter-table tr:last-child td{border-bottom:none}
  .iter-table tr.best-row{background:#f0fdf4;font-weight:700;color:#15803d}
  .iter-table tr.best-row td{border-top:1px solid #bbf7d0;border-bottom:1px solid #bbf7d0}
  .acc-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .dt-title{font-size:12px;font-weight:700;margin-bottom:8px;color:var(--ink)}
  @media(max-width:700px){.acc-grid{grid-template-columns:1fr}}

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

  /* Chart selector */
  .chart-selector-wrapper{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;margin-bottom:10px}
  .chart-btn{background:#fff;border:1.5px solid var(--line);padding:8px 16px;border-radius:10px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;transition:.15s;min-width:170px;text-align:center}
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

    <section class="panel" id="p-dash">
      <div class="dash"><div class="dash-grid" id="dashGrid"></div></div>
    </section>

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
    `Target ${DATA.target} titik · ${DATA.n_iterations||'?'} iterasi`;
}

/* ── Tabs ── */
let mapRoute=null, mapPoints=null, dashDone=false, pointsDone=false;
let activeMetricView = 'coverage';

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
    vehicleSel.add(new Option(`${r.vehicle} #${r.vehicle_unit} — total ${r.vehicle_total} titik`, `${r.vehicle}#${r.vehicle_unit}`)));
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
  if (rStats) rStats.innerHTML = `
    <div class="stat good"><div class="k">Titik dikunjungi</div><div class="v">${sh.visited_count}</div></div>
    <div class="stat ${sh.feasible?'':'warn'}"><div class="k">Feasible</div><div class="v">${sh.feasible?'✓ Ya':'⚠ Tidak'}</div></div>
    <div class="stat"><div class="k">Travel</div><div class="v">${sh.travel_min}<small> mnt</small></div></div>
    <div class="stat"><div class="k">Total waktu</div><div class="v">${sh.total_min}<small> mnt</small></div></div>`;
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
  const bounds = []; let totalVisited=0, feasOk=0, feasTotal=0;
  allRuns.forEach(run => {
    const sh = run.shifts.find(s => String(s.shift) === shiftNum); if (!sh) return;
    const col = VEH_COLORS[run.vehicle] || '#64748b';
    const isMotor = run.vehicle === 'motor';
    totalVisited += sh.visited_count; feasTotal++;
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
  if (rStats) rStats.innerHTML = `
    <div class="stat good"><div class="k">Total dikunjungi</div><div class="v">${totalVisited}</div></div>
    <div class="stat ${feasOk===feasTotal?'good':'warn'}"><div class="k">Feasible</div><div class="v">${feasOk}/${feasTotal}</div></div>` +
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
function lineChart(host, series, opts){
  if (!host) return;
  opts = opts || {};
  const W = opts.w||460, H = opts.h||230, P = {l:42, r:14, t:14, b:34};
  const xs = series.flatMap(s => s.points.map(p => p.x));
  const ys = series.flatMap(s => s.points.map(p => p.y));
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
  series.forEach(se => {
    if (!se.points.length) return;
    const d = se.points.map((p,i) => (i?'L':'M') + sx(p.x).toFixed(1) + ' ' + sy(p.y).toFixed(1)).join(' ');
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2.4" stroke-linejoin="round"/>`;
    const last = se.points[se.points.length-1];
    s += `<circle cx="${sx(last.x).toFixed(1)}" cy="${sy(last.y).toFixed(1)}" r="3" fill="${se.color}"/>`;
  });
  s += '</svg>';
  s += '<div class="chart-legend">' + series.map(se => `<span><i style="background:${se.color}"></i>${se.name}</span>`).join('') + '</div>';
  host.innerHTML = s;
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

function toggleAccordion(idx){
  const content = document.getElementById(`accordion-${idx}`);
  if (!content) return;
  const wasOpen = content.classList.contains('open');
  content.classList.toggle('open');
  if (!wasOpen) {
    const m = DATA.dashboard && DATA.dashboard.models[idx];
    if (!m) return;
    const el = document.getElementById('conv-mini-' + m.name);
    if (el && !el.dataset.drawn) {
      el.dataset.drawn = '1';
      const pts = (m.convergence||[]).map(c => ({x: c.progress, y: c.visited}));
      if (pts.length) {
        lineChart(el, [{name: m.display_name, color: m.color, points: pts}],
          {xmin:0, xmax:100, xlabel:'Progres pencarian (%)', w:320, h:160});
      } else {
        el.innerHTML = '<div style="padding:20px;text-align:center;font-size:11px;color:var(--muted)">Tidak ada data konvergensi</div>';
      }
    }
  }
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

  /* ── Podium (real data from per_iter_detail) ── */
  let allRuns = [];
  M.forEach(m => {
    const perIter = m.per_iter_detail || [];
    if (perIter.length) {
      perIter.forEach(d => {
        allRuns.push({ algo: m.display_name, icon: m.icon, color: m.color,
          iter: d.iter, score: d.coverage, km: d.distance_km, runtime_ms: d.runtime_ms, feasible_pct: d.feasible_pct });
      });
    } else {
      m.per_iteration.forEach((val, i) => {
        allRuns.push({ algo: m.display_name, icon: m.icon, color: m.color,
          iter: i+1, score: val, km: 0, runtime_ms: parseFloat(m.avg_runtime_ms)||0, feasible_pct: m.feasible_pct });
      });
    }
  });
  allRuns.sort((a,b) => (b.score - a.score) || (a.km - b.km));
  const top3 = allRuns.slice(0,3);
  const nextBest = allRuns.slice(3,10);

  let html = `<div class="card">
    <h2>👑 Global Best Iteration Podium</h2>
    <div class="sub">Tiga eksekusi paling optimal — coverage tertinggi, jarak terpendek</div>
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
      <div class="podium-opt-metrics">${item.score} <span>Titik Dikunjungi</span></div>
      ${item.km > 0 ? `<div class="podium-opt-metrics" style="font-size:15px;margin-top:4px">${item.km} <span>Kilometer</span></div>` : ''}
      <div class="podium-footer">
        <div class="podium-stat"><span>Coverage</span>${item.score} titik</div>
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
        <div class="algo-meta">Iterasi #${item.iter} · ${item.score} titik</div>
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
    <div class="sub">Peringkat performa terintegrasi (55% Coverage · 20% Konsistensi · 15% Efisiensi · 10% Success)</div>
    <div class="lb">`;

  M.forEach((m, idx) => {
    const perIter = m.per_iter_detail || [];
    const maxVal = perIter.length ? Math.max(...perIter.map(d => d.coverage)) : Math.max(...m.per_iteration);
    const bestIterNum = perIter.length
      ? (perIter.find(d => d.coverage === maxVal) || {iter:1}).iter
      : (m.per_iteration.indexOf(maxVal) + 1);

    /* Accordion iter table rows using real per_iter_detail */
    let iterRows = '';
    if (perIter.length) {
      perIter.forEach(d => {
        const isBest = (d.coverage === maxVal);
        iterRows += `<tr class="${isBest ? 'best-row' : ''}">
          <td>Run #${d.iter}${isBest ? ' ⭐' : ''}</td>
          <td><b>${d.coverage}</b> / ${target}</td>
          <td>${d.distance_km > 0 ? d.distance_km + ' km' : '—'}</td>
          <td>${d.runtime_ms} ms</td>
          <td>${d.feasible_pct}%</td>
        </tr>`;
      });
    } else {
      m.per_iteration.forEach((val, i) => {
        const isBest = (val === maxVal);
        iterRows += `<tr class="${isBest ? 'best-row' : ''}">
          <td>Run #${i+1}${isBest ? ' ⭐' : ''}</td>
          <td><b>${val}</b> / ${target}</td>
          <td>—</td>
          <td>${m.avg_runtime_ms} ms</td>
          <td>${m.feasible_pct}%</td>
        </tr>`;
      });
    }

    html += `<div class="lb-card pos-${idx}">
      <div class="lb-rank">${m.medal || ('#'+m.rank)}</div>
      <div class="lb-avatar" style="background:${m.color}1f;color:${m.color}">${m.icon}</div>
      <div style="width:100%">
        <div class="lb-name">${m.display_name} <span class="tier tier-${m.tier}">TIER ${m.tier}</span></div>
        <div class="lb-sub">${m.tagline}</div>
        <div class="best-run-banner">🎯 Best Run: Iterasi #${bestIterNum} — <strong>${maxVal} titik</strong></div>
        <div class="lb-section">
          <div class="bars-vertical">
            ${bar('Coverage vs Target', m.coverage_avg/target*100, m.coverage_avg+' / '+target, m.color)}
            ${bar('Konsistensi', m.consistency_pct, m.consistency_pct+'%', '#10b981')}
            ${bar('Success Rate', m.success_rate_pct, m.success_rate_pct+'%', '#f59e0b')}
          </div>
          <div class="metrics-grid">
            <div class="metric-pill runtime"><span class="icon">⚡</span> Runtime: <b>${m.avg_runtime_ms} ms/shift</b></div>
            <div class="metric-pill convergence"><span class="icon">🚀</span> Konvergensi: <b>${m.convergence_speed_pct}%</b></div>
            <div class="metric-pill throughput"><span class="icon">🛞</span> Throughput: <b>${m.throughput} titik/jam</b></div>
            <div class="metric-pill feasibility"><span class="icon">✅</span> Feasible: <b>${m.feasible_pct}%</b></div>
          </div>
        </div>
        <div style="margin-top:14px">
          <button class="accordion-trigger" onclick="toggleAccordion(${idx})">▼ Detail Per Iterasi &amp; Konvergensi</button>
          <div class="accordion-content" id="accordion-${idx}">
            <div class="acc-grid">
              <div>
                <div class="dt-title">⏱ History Runtime (${perIter.length || m.per_iteration.length} Iterasi)</div>
                <div class="table-container">
                  <table class="iter-table">
                    <thead><tr>
                      <th>Run</th><th>Titik</th><th>Jarak</th><th>Runtime</th><th>Feasible</th>
                    </tr></thead>
                    <tbody>${iterRows}</tbody>
                  </table>
                </div>
              </div>
              <div>
                <div class="dt-title">📈 Kurva Konvergensi</div>
                <div id="conv-mini-${m.name}" style="min-height:160px"></div>
              </div>
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

  /* ── Multi-metric charts ── */
  html += `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:4px">
      <h2>📈 Analisis Multimetrik</h2>
      <div class="chart-selector-wrapper">
        <button class="chart-btn ${activeMetricView==='coverage'?'active':''}" data-metric="coverage" onclick="switchMetricView('coverage')">🎯 Coverage (Titik)</button>
        <button class="chart-btn ${activeMetricView==='distance'?'active':''}" data-metric="distance" onclick="switchMetricView('distance')">🛣️ Jarak (Kilometer)</button>
        <button class="chart-btn ${activeMetricView==='time'?'active':''}" data-metric="time" onclick="switchMetricView('time')">⏱️ Waktu (Menit)</button>
      </div>
    </div>
    <div class="sub">Kiri: kurva optimasi selama pencarian (%). Kanan: metrik per iterasi training.</div>
    <div class="charts-layout">
      <div class="chart-box">
        <div class="chart-title" id="titleConv">🎯 Kurva Konvergensi</div>
        <div id="chartConv"></div>
      </div>
      <div class="chart-box">
        <div class="chart-title" id="titleIter">📊 Metrik per Iterasi</div>
        <div id="chartIter"></div>
      </div>
    </div>
  </div>`;

  /* ── Comparison table ── */
  html += `<div class="card">
    <h2>📋 Tabel Perbandingan Lengkap</h2>
    <div class="sub">Semua indikator objektif per model</div>
    <div class="table-card"><table class="cmp"><thead><tr>
      <th>Model</th><th>Coverage</th><th>Rekor</th><th>Target %</th><th>Konsistensi</th>
      <th>Success</th><th>Titik/jam</th><th>Runtime</th><th>Feasible</th><th>Skor</th>
    </tr></thead><tbody>`;
  const bestOf = k => Math.max(...M.map(m => m[k]));
  const minOf  = k => Math.min(...M.map(m => m[k]));
  M.forEach(m => { html += `<tr>
    <td><b>${m.icon} ${m.display_name}</b></td>
    <td class="${m.coverage_avg===bestOf('coverage_avg')?'best':''}">${m.coverage_avg}</td>
    <td class="${m.coverage_best===bestOf('coverage_best')?'best':''}">${m.coverage_best}</td>
    <td>${m.target_attainment_pct}%</td>
    <td class="${m.consistency_pct===bestOf('consistency_pct')?'best':''}">${m.consistency_pct}%</td>
    <td>${m.success_rate_pct}%</td>
    <td class="${m.throughput===bestOf('throughput')?'best':''}">${m.throughput}</td>
    <td class="${m.avg_runtime_ms===minOf('avg_runtime_ms')?'best':''}">${m.avg_runtime_ms} ms</td>
    <td>${m.feasible_pct}%</td>
    <td class="${m.overall_score===bestOf('overall_score')?'best':''}"><b>${m.overall_score}</b></td>
  </tr>`; });
  html += `</tbody></table></div></div>`;

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

function drawSelectedCharts(){
  const D = DATA.dashboard; if (!D) return;
  const M = D.models;
  const hostConv  = document.getElementById('chartConv');
  const hostIter  = document.getElementById('chartIter');
  const titleConv = document.getElementById('titleConv');
  const titleIter = document.getElementById('titleIter');
  if (!hostConv || !hostIter) return;

  if (activeMetricView === 'distance') {
    if (titleConv) titleConv.innerHTML = '🛣️ Kurva Travel Time selama Pencarian (mnt)';
    if (titleIter) titleIter.innerHTML = '🛣️ Jarak Tempuh per Iterasi (km)';

    /* Convergence: travel_min from convergence frames (real data) */
    const convSeries = M.filter(m => (m.convergence||[]).some(c => c.travel_min != null && c.travel_min > 0))
      .map(m => ({ name: m.display_name, color: m.color,
        points: m.convergence.map(c => ({x: c.progress, y: c.travel_min||0})) }));
    if (convSeries.length) {
      lineChart(hostConv, convSeries, {xmin:0, xmax:100, xlabel:'Progres Pencarian (%)'});
    } else {
      hostConv.innerHTML = '<div style="padding:30px;text-align:center;font-size:12px;color:var(--muted)">Data travel_min tidak tersedia</div>';
    }

    /* Iter: distance_km from per_iter_detail (real data) */
    const iterSeries = M.filter(m => (m.per_iter_detail||[]).some(d => d.distance_km > 0))
      .map(m => ({ name: m.display_name, color: m.color,
        points: m.per_iter_detail.map(d => ({x: d.iter, y: d.distance_km})) }));
    if (iterSeries.length) {
      lineChart(hostIter, iterSeries, {xmin:1, xmax: Math.max(2, D.n_iterations), xlabel:'Iterasi'});
    } else {
      hostIter.innerHTML = '<div style="padding:30px;text-align:center;font-size:12px;color:var(--muted)">Data jarak tidak tersedia</div>';
    }

  } else if (activeMetricView === 'time') {
    if (titleConv) titleConv.innerHTML = '⏱️ Kurva Total Waktu selama Pencarian (mnt)';
    if (titleIter) titleIter.innerHTML = '⏱️ Runtime Komputasi per Iterasi (ms)';

    /* Convergence: total_min from convergence frames (real data) */
    const convSeries = M.filter(m => (m.convergence||[]).some(c => c.total_min != null && c.total_min > 0))
      .map(m => ({ name: m.display_name, color: m.color,
        points: m.convergence.map(c => ({x: c.progress, y: c.total_min||0})) }));
    if (convSeries.length) {
      lineChart(hostConv, convSeries, {xmin:0, xmax:100, xlabel:'Progres Pencarian (%)'});
    } else {
      hostConv.innerHTML = '<div style="padding:30px;text-align:center;font-size:12px;color:var(--muted)">Data total_min tidak tersedia</div>';
    }

    /* Iter: runtime_ms from per_iter_detail (real data) */
    const iterSeries = M.filter(m => (m.per_iter_detail||[]).length)
      .map(m => ({ name: m.display_name, color: m.color,
        points: m.per_iter_detail.map(d => ({x: d.iter, y: d.runtime_ms})) }));
    if (iterSeries.length) {
      lineChart(hostIter, iterSeries, {xmin:1, xmax: Math.max(2, D.n_iterations), xlabel:'Iterasi'});
    } else {
      lineChart(hostIter,
        M.map(m => ({ name: m.display_name, color: m.color,
          points: m.per_iteration.map((v,i) => ({x: i+1, y: parseFloat(m.avg_runtime_ms)||0})) })),
        {xmin:1, xmax: Math.max(2, D.n_iterations), xlabel:'Iterasi'});
    }

  } else {
    /* Coverage (default) — real data */
    if (titleConv) titleConv.innerHTML = '🎯 Kurva Konvergensi Coverage (%)';
    if (titleIter) titleIter.innerHTML = '🎯 Coverage per Iterasi (titik)';

    lineChart(hostConv,
      M.filter(m => m.convergence && m.convergence.length).map(m => ({
        name: m.display_name, color: m.color,
        points: m.convergence.map(c => ({x: c.progress, y: c.visited}))
      })), {xmin:0, xmax:100, xlabel:'Progres Pencarian (%)'});

    const iterSeries = M.filter(m => (m.per_iter_detail||[]).length)
      .map(m => ({ name: m.display_name, color: m.color,
        points: m.per_iter_detail.map(d => ({x: d.iter, y: d.coverage})) }));
    lineChart(hostIter,
      iterSeries.length ? iterSeries : M.map(m => ({ name: m.display_name, color: m.color,
        points: m.per_iteration.map((v,i) => ({x: i+1, y: v})) })),
      {xmin:1, xmax: Math.max(2, D.n_iterations), xlabel:'Iterasi'});
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
      <div><div class="n">${pool.length}</div><div class="l">Total Pos Terdaftar</div></div></div>
    <div class="legend" style="margin-top:10px">
      <span class="it"><span class="dot" style="background:var(--emergency)"></span>Emergency</span>
      <span class="it"><span class="dot" style="background:var(--transport)"></span>Transport</span>
      <span class="it"><span class="dot" style="background:var(--depot)"></span>Depot</span></div>`;
  setTimeout(() => mapPoints.invalidateSize(), 100);
}
</script>
</body>
</html>"""