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
    --bg:#eef2f7; --panel:#fff; --ink:#1f2933; --muted:#7b8794; --line:#e6eaf0;
    --accent:#2563eb; --accent-soft:#eaf1ff; --hi:#10b981;
    --emergency:#e5484d; --transport:#2563eb; --depot:#16a34a;
    --gold:#f6b73c; --silver:#b9c2cf; --bronze:#cd8a52;
    --shadow:0 6px 24px rgba(20,33,61,.10);
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{font-family:'Inter',system-ui,Arial,sans-serif;color:var(--ink);background:var(--bg);
       height:100vh;display:flex;flex-direction:column;overflow:hidden}

  /* ── Topbar / tabs ── */
  .topbar{display:flex;align-items:center;gap:18px;padding:0 20px;height:56px;flex:none;
          background:#fff;box-shadow:0 2px 12px rgba(20,33,61,.07);z-index:2000}
  .brand{font-weight:800;font-size:15px;letter-spacing:-.3px}
  .brand span{color:var(--accent)}
  .tabs{display:flex;gap:4px;background:#f1f4f9;padding:4px;border-radius:12px}
  .tab{border:0;background:transparent;color:var(--muted);font:inherit;font-weight:600;font-size:13px;
       padding:7px 16px;border-radius:9px;cursor:pointer;transition:.15s}
  .tab:hover{color:var(--ink)}
  .tab.active{background:#fff;color:var(--accent);box-shadow:0 2px 8px rgba(20,33,61,.10)}
  .meta-right{margin-left:auto;font-size:11.5px;color:var(--muted);text-align:right;line-height:1.5}

  main{flex:1;position:relative;overflow:hidden}
  .panel{position:absolute;inset:0;display:none}
  .panel.active{display:flex}

  /* ── shared widgets ── */
  .select{position:relative}
  .select select{width:100%;appearance:none;-webkit-appearance:none;padding:10px 34px 10px 12px;
    border:1.5px solid var(--line);border-radius:11px;background:#fbfcfe;color:var(--ink);
    font-size:13.5px;font-weight:500;font-family:inherit;cursor:pointer;transition:.15s}
  .select select:hover{border-color:#c7d2e0}
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
    font-size:11px;font-weight:700;border:2.5px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.3)}

  /* ── Tab Rute ── */
  .route-wrap{display:flex;width:100%;height:100%}
  .route-side{width:380px;flex:none;height:100%;display:flex;flex-direction:column;background:#fff;
    box-shadow:var(--shadow);z-index:1000}
  .route-side .head{padding:16px 18px 12px;background:linear-gradient(135deg,#2563eb,#4f8cff);color:#fff}
  .route-side .head .pill{display:inline-block;background:rgba(255,255,255,.18);padding:2px 9px;
    border-radius:999px;font-size:11px;margin-top:8px;font-weight:600}
  .route-side .body{flex:1;overflow-y:auto;padding:14px 16px 24px}
  .stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0 14px}
  .stat{background:#fbfcfe;border:1px solid var(--line);border-radius:12px;padding:10px 12px}
  .stat .k{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
  .stat .v{font-size:18px;font-weight:700;margin-top:2px}
  .stat .v small{font-size:11px;font-weight:500;color:var(--muted)}
  .stat.good .v{color:var(--hi)} .stat.warn .v{color:var(--emergency)}
  .sec-title{font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--muted);
    margin:6px 0 8px;display:flex;justify-content:space-between}
  .toggle{font-size:11px;color:var(--accent);cursor:pointer;font-weight:600;user-select:none}
  .timeline{position:relative}
  .leg{position:relative;padding:8px 10px 10px 30px;border-radius:10px;cursor:pointer;transition:.12s}
  .leg:hover{background:#f3f6fb} .leg.active{background:var(--accent-soft)}
  .leg::before{content:"";position:absolute;left:11px;top:26px;bottom:-6px;width:2px;background:var(--line)}
  .leg:last-child::before{display:none}
  .leg .num{position:absolute;left:2px;top:9px;width:20px;height:20px;border-radius:50%;color:#fff;
    font-size:10.5px;font-weight:700;display:flex;align-items:center;justify-content:center;
    border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.18)}
  .leg .route{font-size:13px;font-weight:600;line-height:1.35}
  .leg .route .arrow{color:var(--muted);font-weight:500;margin:0 3px}
  .leg .streets{font-size:11px;color:var(--muted);margin-top:3px;line-height:1.45}
  #mapRoute,#mapPoints{flex:1;height:100%}
  .leaflet-container{font-family:'Inter',system-ui,sans-serif}

  /* ── Tab Dashboard ── */
  .dash{width:100%;height:100%;overflow-y:auto;padding:22px 26px 40px}
  .dash-grid{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
  .card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:18px 20px}
  .card h2{margin:0 0 2px;font-size:15px;font-weight:700}
  .card .sub{font-size:12px;color:var(--muted);margin-bottom:14px}
  .insights{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .ins{display:flex;gap:10px;background:#fbfcff;border:1px solid var(--line);border-left:3px solid var(--accent);
    border-radius:10px;padding:10px 12px;font-size:12.5px;line-height:1.5}
  /* leaderboard */
  .lb{display:flex;flex-direction:column;gap:10px}
  .lb-card{display:grid;grid-template-columns:46px 52px 1fr 92px;gap:14px;align-items:center;
    background:#fff;border:1.5px solid var(--line);border-radius:14px;padding:14px 16px;transition:.15s}
  .lb-card:hover{box-shadow:0 8px 22px rgba(20,33,61,.1)}
  .lb-card.champ{border-color:var(--gold);background:linear-gradient(180deg,#fffdf5,#fff)}
  .lb-rank{font-size:22px;font-weight:800;text-align:center;color:var(--muted)}
  .lb-avatar{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:26px}
  .lb-name{font-size:15px;font-weight:700;display:flex;align-items:center;gap:8px}
  .lb-sub{font-size:11.5px;color:var(--muted);margin:1px 0 8px}
  .tier{font-size:10px;font-weight:800;padding:1px 8px;border-radius:6px;color:#fff;letter-spacing:.5px}
  .tier-S{background:#8b5cf6} .tier-A{background:#10b981} .tier-B{background:#2563eb} .tier-C{background:#94a3b8}
  .bars{display:flex;flex-direction:column;gap:5px}
  .bar .bh{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);margin-bottom:2px}
  .bar .bh b{color:var(--ink);font-weight:600}
  .track{height:7px;background:#eef1f6;border-radius:6px;overflow:hidden}
  .fill{height:100%;border-radius:6px;transition:width .5s}
  .lb-mini{font-size:11px;color:var(--muted);margin-top:8px;display:flex;flex-wrap:wrap;gap:10px}
  .lb-score{text-align:center}
  .lb-score .sn{font-size:30px;font-weight:800;line-height:1}
  .lb-score .sl{font-size:9px;font-weight:700;letter-spacing:1px;color:var(--muted)}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:22px}
  .chart-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:8px;justify-content:center}
  .chart-legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
  table.cmp{width:100%;border-collapse:collapse;font-size:12px}
  table.cmp th,table.cmp td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line)}
  table.cmp th:first-child,table.cmp td:first-child{text-align:left}
  table.cmp th{font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);font-weight:600}
  table.cmp tr:hover td{background:#fafbfe}
  table.cmp td.best{color:var(--hi);font-weight:700}
  .guide{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .guide .g{background:#fbfcff;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
  .guide .g b{font-size:12px} .guide .g p{margin:3px 0 0;font-size:11.5px;color:var(--muted);line-height:1.5}
  .empty{max-width:520px;margin:80px auto;text-align:center;color:var(--muted)}

  /* ── Tab Peta Titik ── */
  .points-wrap{display:flex;width:100%;height:100%}
  .points-side{width:300px;flex:none;background:#fff;box-shadow:var(--shadow);z-index:1000;padding:18px;overflow-y:auto}
  .points-side h2{font-size:15px;margin:0 0 4px}
  .points-side .sub{font-size:12px;color:var(--muted);margin-bottom:16px}
  .pcount{display:flex;align-items:center;gap:10px;background:#fbfcfe;border:1px solid var(--line);
    border-radius:12px;padding:12px;margin-bottom:10px}
  .pcount .ic{width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px}
  .pcount .n{font-size:22px;font-weight:800;line-height:1}
  .pcount .l{font-size:11px;color:var(--muted)}
  @media(max-width:900px){.insights,.charts,.guide{grid-template-columns:1fr}}
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
            <div class="field"><label>Kendaraan</label><div class="select"><select id="vehicleSel"></select></div></div>
            <div class="field"><label>Shift</label><div class="select"><select id="shiftSel"></select></div></div>
            <div class="stats" id="rStats"></div>
            <div class="legend" style="margin-bottom:14px">
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
const DATA = __ROUTE_DATA__;
const CAT = { emergency:'#e5484d', transport:'#2563eb', depot:'#16a34a', '':'#64748b' };

document.getElementById('metaRight').innerHTML =
  `Generated ${DATA.generated}<br>Target ${DATA.target} titik/kendaraan · shift ${DATA.shift_min}m · service ${DATA.service_min}m`;
document.getElementById('targetPill').textContent =
  `Target ${DATA.target} titik · ${DATA.n_iterations||'?'} iterasi`;

/* ───────── Tabs ───────── */
let mapRoute=null, mapPoints=null, dashDone=false, pointsDone=false;
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  b.classList.add('active');
  const t=b.dataset.tab; document.getElementById('p-'+t).classList.add('active');
  if(t==='route'&&mapRoute) setTimeout(()=>mapRoute.invalidateSize(),60);
  if(t==='dash'&&!dashDone){renderDashboard();dashDone=true;}
  if(t==='points'){ if(!pointsDone){renderPoints();pointsDone=true;}
                    else setTimeout(()=>mapPoints.invalidateSize(),60); }
}));

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

/* ───────── TAB RUTE ───────── */
mapRoute=L.map('mapRoute',{zoomControl:true}); baseTiles(mapRoute);
const modelSel=document.getElementById('modelSel'),vehicleSel=document.getElementById('vehicleSel'),
      shiftSel=document.getElementById('shiftSel'),legsDiv=document.getElementById('legs'),
      rStats=document.getElementById('rStats');
[...new Set(DATA.runs.map(r=>r.model))].forEach(m=>{
  const dn=(DATA.dashboard&&DATA.dashboard.models.find(x=>x.name===m));
  modelSel.add(new Option(dn?dn.display_name:m,m));
});
function runsFor(m){return DATA.runs.filter(r=>r.model===m);}
function curRun(){return runsFor(modelSel.value).find(r=>`${r.vehicle}#${r.vehicle_unit}`===vehicleSel.value);}
function fillVehicles(){vehicleSel.innerHTML='';runsFor(modelSel.value).forEach(r=>
  vehicleSel.add(new Option(`${r.vehicle} #${r.vehicle_unit} — total ${r.vehicle_total} titik`,`${r.vehicle}#${r.vehicle_unit}`)));}
function fillShifts(){shiftSel.innerHTML='';const r=curRun();if(!r)return;
  r.shifts.forEach(s=>shiftSel.add(new Option(`Shift ${s.shift} — ${s.visited_count} titik`,s.shift)));}

let rLayers=[],poolLayer=null,hiLayer=null,showPool=false;
function clearRoute(){rLayers.forEach(l=>mapRoute.removeLayer(l));rLayers=[];if(hiLayer){mapRoute.removeLayer(hiLayer);hiLayer=null;}}
function drawPool(){if(poolLayer){mapRoute.removeLayer(poolLayer);poolLayer=null;}
  if(!showPool||!DATA.pool)return;poolLayer=L.layerGroup();
  DATA.pool.forEach(p=>L.circleMarker(p.coord,{radius:3,color:CAT[p.cat]||'#888',weight:1,fillOpacity:.5,opacity:.5})
    .bindPopup(`${p.label} <small>(${p.cat})</small>`).addTo(poolLayer));
  poolLayer.addTo(mapRoute);}
function renderRoute(){
  clearRoute();const run=curRun();if(!run)return;
  const sh=run.shifts.find(s=>String(s.shift)===String(shiftSel.value));if(!sh)return;
  rLayers.push(L.marker(DATA.depot.coord,{icon:depotIcon(),zIndexOffset:1000}).addTo(mapRoute)
    .bindPopup(`<b>DEPOT</b><br>${DATA.depot.label}`));
  if(sh.route_coords&&sh.route_coords.length){
    rLayers.push(L.polyline(sh.route_coords,{color:'#2563eb',weight:4,opacity:.45,lineJoin:'round'}).addTo(mapRoute));
    mapRoute.fitBounds(L.latLngBounds(sh.route_coords).pad(.18));}
  const legs=sh.legs||[];
  legs.forEach((lg,i)=>{if(i<legs.length-1){const c=lg.coords[lg.coords.length-1];
    rLayers.push(L.marker(c,{icon:numIcon(i+1,CAT[lg.to_cat]||CAT['']),zIndexOffset:500}).addTo(mapRoute)
      .bindPopup(`<b>#${i+1} ${lg.to}</b><br><small>${lg.to_cat||'-'}</small>`));}});
  rStats.innerHTML=`
    <div class="stat good"><div class="k">Titik dikunjungi</div><div class="v">${sh.visited_count}</div></div>
    <div class="stat ${sh.feasible?'':'warn'}"><div class="k">Feasible</div><div class="v">${sh.feasible?'✓ Ya':'⚠ Tidak'}</div></div>
    <div class="stat"><div class="k">Travel</div><div class="v">${sh.travel_min}<small> mnt</small></div></div>
    <div class="stat"><div class="k">Total waktu</div><div class="v">${sh.total_min}<small> mnt</small></div></div>`;
  legsDiv.innerHTML='';
  legs.forEach((lg,i)=>{const isStop=i<legs.length-1;
    const col=isStop?(CAT[lg.to_cat]||CAT['']):CAT.depot;const num=isStop?(i+1):'★';
    const st=(lg.streets||[]).join(' → ')||'(jalan tak bernama)';
    const row=document.createElement('div');row.className='leg';
    row.innerHTML=`<span class="num" style="background:${col}">${num}</span>
      <div class="route">${lg.from}<span class="arrow">→</span>${lg.to}</div><div class="streets">${st}</div>`;
    row.onclick=()=>{document.querySelectorAll('.leg').forEach(r=>r.classList.remove('active'));row.classList.add('active');
      if(hiLayer)mapRoute.removeLayer(hiLayer);
      hiLayer=L.polyline(lg.coords,{color:'#10b981',weight:7,opacity:.95,lineJoin:'round'}).addTo(mapRoute);
      mapRoute.fitBounds(hiLayer.getBounds().pad(.35));};
    legsDiv.appendChild(row);});
}
document.getElementById('poolToggle').addEventListener('click',function(){showPool=!showPool;
  this.style.color=showPool?'#10b981':'';this.textContent=(showPool?'● ':'◌ ')+'Semua titik';drawPool();});
modelSel.onchange=()=>{fillVehicles();fillShifts();renderRoute();};
vehicleSel.onchange=()=>{fillShifts();renderRoute();};
shiftSel.onchange=renderRoute;
fillVehicles();fillShifts();renderRoute();

/* ───────── SVG line chart ───────── */
function lineChart(host,series,opts){
  opts=opts||{};const W=opts.w||460,H=opts.h||230,P={l:38,r:14,t:14,b:34};
  const xs=series.flatMap(s=>s.points.map(p=>p.x)),ys=series.flatMap(s=>s.points.map(p=>p.y));
  const xmin=opts.xmin!=null?opts.xmin:Math.min(0,...xs),xmax=opts.xmax!=null?opts.xmax:Math.max(1,...xs);
  const ymax=opts.ymax!=null?opts.ymax:Math.max(1,...ys)*1.12,ymin=0;
  const sx=x=>P.l+(x-xmin)/((xmax-xmin)||1)*(W-P.l-P.r);
  const sy=y=>H-P.b-(y-ymin)/((ymax-ymin)||1)*(H-P.t-P.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet">`;
  for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4,py=sy(v);
    s+=`<line x1="${P.l}" y1="${py}" x2="${W-P.r}" y2="${py}" stroke="#eef2f7"/>`;
    s+=`<text x="${P.l-6}" y="${py+3}" text-anchor="end" font-size="9" fill="#9aa3b8">${Math.round(v)}</text>`;}
  s+=`<text x="${(W)/2}" y="${H-6}" text-anchor="middle" font-size="10" fill="#7b8794">${opts.xlabel||''}</text>`;
  series.forEach(se=>{if(!se.points.length)return;
    const d=se.points.map((p,i)=>(i?'L':'M')+sx(p.x).toFixed(1)+' '+sy(p.y).toFixed(1)).join(' ');
    s+=`<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2.4" stroke-linejoin="round"/>`;
    const last=se.points[se.points.length-1];
    s+=`<circle cx="${sx(last.x).toFixed(1)}" cy="${sy(last.y).toFixed(1)}" r="3" fill="${se.color}"/>`;});
  s+='</svg>';
  s+='<div class="chart-legend">'+series.map(se=>`<span><i style="background:${se.color}"></i>${se.name}</span>`).join('')+'</div>';
  host.innerHTML=s;
}

/* ───────── TAB DASHBOARD ───────── */
function bar(label,pct,val,color){pct=Math.max(2,Math.min(100,pct));
  return `<div class="bar"><div class="bh"><span>${label}</span><b>${val}</b></div>
    <div class="track"><div class="fill" style="width:${pct}%;background:${color}"></div></div></div>`;}

function renderDashboard(){
  const grid=document.getElementById('dashGrid');
  const D=DATA.dashboard;
  if(!D||!D.models||!D.models.length){
    grid.innerHTML=`<div class="empty"><h2>Belum ada data dashboard</h2>
      <p>Jalankan ulang <code>python main.py train</code> dengan kode terbaru untuk menghasilkan metrik performa.</p></div>`;return;}
  const M=D.models,target=D.target;

  // Insights
  let html=`<div class="card"><h2>💡 Insight</h2><div class="sub">Ringkasan performa antar model & antar iterasi</div>
    <div class="insights">${D.insights.map(t=>`<div class="ins">${t}</div>`).join('')}</div></div>`;

  // Leaderboard
  html+=`<div class="card"><h2>🏆 Leaderboard Model</h2>
    <div class="sub">Peringkat skor keseluruhan (55% coverage · 20% konsistensi · 15% efisiensi · 10% success)</div><div class="lb">`;
  M.forEach(m=>{
    html+=`<div class="lb-card ${m.rank===1?'champ':''}">
      <div class="lb-rank">${m.medal||('#'+m.rank)}</div>
      <div class="lb-avatar" style="background:${m.color}1f;color:${m.color}">${m.icon}</div>
      <div>
        <div class="lb-name">${m.display_name}<span class="tier tier-${m.tier}">TIER ${m.tier}</span></div>
        <div class="lb-sub">${m.tagline} · rata-rata ${m.coverage_avg} titik · rekor ${m.coverage_best}</div>
        <div class="bars">
          ${bar('Coverage vs target',m.coverage_avg/target*100,m.coverage_avg+' / '+target,m.color)}
          ${bar('Konsistensi',m.consistency_pct,m.consistency_pct+'%','#10b981')}
          ${bar('Success rate',m.success_rate_pct,m.success_rate_pct+'%','#f59e0b')}
        </div>
        <div class="lb-mini"><span>⚡ ${m.avg_runtime_ms} ms/shift</span><span>🚀 konvergensi ${m.convergence_speed_pct}%</span>
          <span>🛞 ${m.throughput} titik/jam</span><span>✅ ${m.feasible_pct}% feasible</span>
          <span>⏱ utilisasi ${m.time_util_pct}%</span></div>
      </div>
      <div class="lb-score"><div class="sn" style="color:${m.color}">${m.overall_score}</div><div class="sl">SCORE</div></div>
    </div>`;});
  html+=`</div></div>`;

  // Charts
  html+=`<div class="card"><h2>📈 Analisis Konvergensi & Iterasi</h2>
    <div class="sub">Kiri: peningkatan coverage selama pencarian. Kanan: hasil tiap iterasi training.</div>
    <div class="charts"><div><div style="font-size:12px;font-weight:600;margin-bottom:4px">Kurva Konvergensi</div><div id="chartConv"></div></div>
      <div><div style="font-size:12px;font-weight:600;margin-bottom:4px">Coverage per Iterasi</div><div id="chartIter"></div></div></div></div>`;

  // Comparison table
  html+=`<div class="card"><h2>📋 Tabel Perbandingan Lengkap</h2><div class="sub">Semua indikator objektif per model</div>
    <table class="cmp"><thead><tr>
      <th>Model</th><th>Coverage</th><th>Rekor</th><th>Target %</th><th>Konsistensi</th>
      <th>Success</th><th>Titik/jam</th><th>Runtime</th><th>Feasible</th><th>Skor</th></tr></thead><tbody>`;
  const bestOf=k=>Math.max(...M.map(m=>m[k]));
  const minOf=k=>Math.min(...M.map(m=>m[k]));
  M.forEach(m=>{html+=`<tr>
    <td><b>${m.icon} ${m.display_name}</b></td>
    <td class="${m.coverage_avg===bestOf('coverage_avg')?'best':''}">${m.coverage_avg}</td>
    <td class="${m.coverage_best===bestOf('coverage_best')?'best':''}">${m.coverage_best}</td>
    <td>${m.target_attainment_pct}%</td>
    <td class="${m.consistency_pct===bestOf('consistency_pct')?'best':''}">${m.consistency_pct}%</td>
    <td>${m.success_rate_pct}%</td>
    <td class="${m.throughput===bestOf('throughput')?'best':''}">${m.throughput}</td>
    <td class="${m.avg_runtime_ms===minOf('avg_runtime_ms')?'best':''}">${m.avg_runtime_ms} ms</td>
    <td>${m.feasible_pct}%</td>
    <td class="${m.overall_score===bestOf('overall_score')?'best':''}"><b>${m.overall_score}</b></td></tr>`;});
  html+=`</tbody></table></div>`;

  // Metric guide
  html+=`<div class="card"><h2>📖 Panduan Indikator</h2><div class="sub">Arti tiap metrik</div>
    <div class="guide">${D.metric_guide.map(g=>`<div class="g"><b>${g.label}</b><p>${g.desc}</p></div>`).join('')}</div></div>`;

  grid.innerHTML=html;

  // draw charts
  lineChart(document.getElementById('chartConv'),
    M.filter(m=>m.convergence&&m.convergence.length).map(m=>({name:m.display_name,color:m.color,
      points:m.convergence.map(c=>({x:c.progress,y:c.visited}))})),
    {xmin:0,xmax:100,xlabel:'Progres pencarian (%)'});
  lineChart(document.getElementById('chartIter'),
    M.map(m=>({name:m.display_name,color:m.color,
      points:m.per_iteration.map((v,i)=>({x:i+1,y:v}))})),
    {xmin:1,xmax:Math.max(2,D.n_iterations),xlabel:'Iterasi'});
}

/* ───────── TAB PETA TITIK ───────── */
function renderPoints(){
  mapPoints=L.map('mapPoints',{zoomControl:true}); baseTiles(mapPoints);
  const pool=DATA.pool||[];
  const counts={};pool.forEach(p=>counts[p.cat]=(counts[p.cat]||0)+1);
  const grp=L.featureGroup().addTo(mapPoints);
  pool.forEach(p=>L.circleMarker(p.coord,{radius:5,color:'#fff',weight:1.5,
    fillColor:CAT[p.cat]||'#888',fillOpacity:.9}).bindPopup(`<b>${p.label}</b><br><small>${p.cat}</small>`).addTo(grp));
  L.marker(DATA.depot.coord,{icon:depotIcon(),zIndexOffset:1000}).bindPopup(`<b>DEPOT</b><br>${DATA.depot.label}`).addTo(mapPoints);
  if(pool.length) mapPoints.fitBounds(grp.getBounds().pad(.12)); else mapPoints.setView(DATA.depot.coord,12);

  const side=document.getElementById('pointsSide');
  side.innerHTML=`<h2>📍 Titik Kandidat</h2><div class="sub">Titik tetap yang tersebar merata di Surabaya — objektif dimaksimasi</div>
    <div class="pcount"><div class="ic" style="background:#fde8e8">🚨</div>
      <div><div class="n" style="color:${CAT.emergency}">${counts.emergency||0}</div><div class="l">Emergency (polisi + pemadam)</div></div></div>
    <div class="pcount"><div class="ic" style="background:#e7efff">🚌</div>
      <div><div class="n" style="color:${CAT.transport}">${counts.transport||0}</div><div class="l">Transport (halte + terminal)</div></div></div>
    <div class="pcount"><div class="ic" style="background:#e7f8ee">★</div>
      <div><div class="n" style="color:${CAT.depot}">1</div><div class="l">Depot (${DATA.depot.label})</div></div></div>
    <div class="pcount"><div class="ic" style="background:#eef1f6">Σ</div>
      <div><div class="n">${pool.length}</div><div class="l">Total titik kandidat</div></div></div>
    <div class="legend" style="margin-top:10px">
      <span class="it"><span class="dot" style="background:var(--emergency)"></span>Emergency</span>
      <span class="it"><span class="dot" style="background:var(--transport)"></span>Transport</span>
      <span class="it"><span class="dot" style="background:var(--depot)"></span>Depot</span></div>`;
  setTimeout(()=>mapPoints.invalidateSize(),60);
}
</script>
</body>
</html>"""
