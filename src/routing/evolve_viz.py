"""
GA Evolution Viewer
Generates a self-contained HTML that shows how each GA's best route
changes across generations — with a timeline slider, play/pause,
and an "All Algorithms" overlay mode that also renders Dijkstra/A*.
"""

import json
import logging
from pathlib import Path
from typing import List

import networkx as nx

from src.routing.base import RouteResult, Scenario

log = logging.getLogger(__name__)

ALL_COLORS = {
    "sandy_ga":          "#E63946",
    "burhan_ga":         "#2196F3",
    "bimo_ga":           "#4CAF50",
    "gerald_ga":         "#FF9800",
    "gerald_sa":         "#D81B60",
    "christofides":      "#2ECC71",
    "dijkstra_time":     "#9C27B0",
    "dijkstra_distance": "#795548",
    "astar_time":        "#00BCD4",
    "astar_distance":    "#607D8B",
}


def _node_coords(G: nx.MultiDiGraph, route: list) -> list:
    coords = []
    for n in route:
        nd = G.nodes.get(n)
        if nd:
            coords.append([round(float(nd["y"]), 5), round(float(nd["x"]), 5)])
    return coords


def build_evolution_viewer(
    G: nx.MultiDiGraph,
    scenarios: List[Scenario],
    results: List[RouteResult],
    output_dir: Path,
) -> "Path | None":
    """
    Build evolution_viewer.html from benchmark results.
    Includes per-generation GA history AND static Dijkstra/A* baselines.
    Returns output path or None if no GA data found.
    """
    ga_results       = [r for r in results if r.found and "gen_history" in r.metadata and r.metadata["gen_history"]]
    baseline_results = [r for r in results if r.found and "gen_history" not in r.metadata]

    if not ga_results:
        log.warning("No GA evolution history found — skipping evolution viewer.")
        return None

    scenarios_data = {
        s.name: {
            "source":       list(s.source_coords),
            "target":       list(s.target_coords),
            "source_label": s.source_label,
            "target_label": s.target_label,
            "stops": [
                {
                    "label": label,
                    "coords": list(coords),
                }
                for label, coords in zip(s.label_sequence, s.coord_sequence)
            ],
            "optimize_order": s.optimize_order,
            "round_trip":     s.round_trip,
        }
        for s in scenarios
    }

    # GA per-generation history
    algorithms_data: dict = {}
    for r in ga_results:
        algorithms_data.setdefault(r.algorithm_name, {})[r.scenario_name] = \
            r.metadata["gen_history"]

    # Baseline static routes (Dijkstra, A*)
    baselines_data: dict = {}
    for r in baseline_results:
        baselines_data.setdefault(r.algorithm_name, {})[r.scenario_name] = {
            "min":    round(r.total_time_s / 60, 3),
            "dist":   round(r.total_distance_m / 1000, 3),
            "coords": _node_coords(G, r.route),
        }

    payload = json.dumps(
        {
            "scenarios":  scenarios_data,
            "algorithms": algorithms_data,
            "baselines":  baselines_data,
            "colors":     ALL_COLORS,
        },
        separators=(",", ":"),
    )

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", payload)
    out  = output_dir / "evolution_viewer.html"
    out.write_text(html, encoding="utf-8")
    log.info(
        f"  Evolution viewer -> evolution_viewer.html  "
        f"({len(ga_results)} GA + {len(baseline_results)} baseline result(s) embedded)"
    )
    return out


# ──────────────────────────────────────────────────────────────
# HTML template  (single placeholder: __DATA_JSON__)
# ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GA Evolution Viewer — Surabaya Routing</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#111827;color:#e5e7eb;height:100vh;display:flex;flex-direction:column}
#topbar{background:#1f2937;padding:10px 16px;display:flex;align-items:center;
        gap:12px;border-bottom:2px solid #e63946;flex-wrap:wrap}
#topbar h1{font-size:15px;color:#e63946;white-space:nowrap;letter-spacing:.5px;margin-right:4px}
.ctrl{display:flex;align-items:center;gap:6px}
.ctrl label{font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
select{background:#374151;color:#e5e7eb;border:1px solid #4b5563;border-radius:5px;
       padding:5px 10px;font-size:13px;cursor:pointer;outline:none}
select:focus{border-color:#e63946}
#stats{display:flex;gap:16px;margin-left:auto;flex-wrap:wrap;align-items:center}
.stat{display:flex;flex-direction:column;align-items:center;min-width:56px}
.stat-val{font-size:17px;font-weight:700;color:#e63946;line-height:1.1}
.stat-lbl{font-size:10px;color:#6b7280;margin-top:2px;text-transform:uppercase;white-space:nowrap}
#map{flex:1;min-height:200px;position:relative}
#bottombar{background:#1f2937;padding:10px 16px;border-top:2px solid #374151;
           display:flex;align-items:center;gap:14px;flex-wrap:wrap}
#play-btn{background:#e63946;color:#fff;border:none;border-radius:5px;
          padding:7px 18px;font-size:13px;font-weight:600;cursor:pointer;
          white-space:nowrap;min-width:90px;transition:background .15s}
#play-btn:hover{background:#c0303e}
#gen-label{font-size:12px;color:#9ca3af;white-space:nowrap;min-width:90px}
#gen-slider{flex:1;min-width:150px;accent-color:#e63946;cursor:pointer;height:6px}
.speed-ctrl{display:flex;align-items:center;gap:6px}
.speed-ctrl label{font-size:11px;color:#9ca3af}
#speed-select{background:#374151;color:#e5e7eb;border:1px solid #4b5563;
              border-radius:5px;padding:4px 8px;font-size:12px}
#progress{height:3px;background:#e63946;width:0%;transition:width .1s linear}
/* animate toggle */
#anim-toggle{appearance:none;-webkit-appearance:none;width:36px;height:20px;
             background:#374151;border-radius:10px;cursor:pointer;position:relative;
             transition:background .2s;border:none;outline:none;vertical-align:middle}
#anim-toggle:checked{background:#e63946}
#anim-toggle::after{content:'';position:absolute;left:3px;top:3px;width:14px;height:14px;
                    background:#fff;border-radius:50%;transition:left .2s}
#anim-toggle:checked::after{left:19px}
/* map legend overlay */
#map-legend{position:absolute;bottom:24px;right:10px;z-index:999;
            background:rgba(31,41,55,.93);border:1px solid #374151;
            border-radius:8px;padding:10px 14px;font-size:12px;
            line-height:1.9;min-width:250px;display:none;
            box-shadow:0 4px 12px rgba(0,0,0,.5)}
#map-legend h3{font-size:11px;color:#9ca3af;text-transform:uppercase;
               letter-spacing:.6px;margin-bottom:4px}
.leg-row{display:flex;align-items:center;gap:8px}
.leg-swatch{width:28px;height:4px;border-radius:2px;flex-shrink:0}
.leg-dashed{background:repeating-linear-gradient(90deg,currentColor 0,currentColor 5px,transparent 5px,transparent 9px);height:3px}
.leg-name{color:#e5e7eb;flex:1}
.leg-val{color:#9ca3af;font-size:11px;white-space:nowrap}
</style>
</head>
<body>

<div id="topbar">
  <h1>&#9652; GA Evolution Viewer</h1>
  <div class="ctrl">
    <label>Algorithm</label>
    <select id="algo-select"></select>
  </div>
  <div class="ctrl">
    <label>Scenario</label>
    <select id="scenario-select"></select>
  </div>
  <div id="stats">
    <div class="stat"><span class="stat-val" id="s-dist">-</span><span class="stat-lbl">GA Dist (km)</span></div>
    <div class="stat"><span class="stat-val" id="s-gen">—</span><span class="stat-lbl">Gen</span></div>
    <div class="stat"><span class="stat-val" id="s-time">—</span><span class="stat-lbl">GA Best (min)</span></div>
    <div class="stat"><span class="stat-val" id="s-impr">—</span><span class="stat-lbl">Improved</span></div>
    <div class="stat" id="stat-ref-wrap"><span class="stat-val" id="s-ref">—</span><span class="stat-lbl">Dijkstra (min)</span></div>
    <div class="stat"><span class="stat-val" id="s-total">—</span><span class="stat-lbl">Total Gens</span></div>
  </div>
</div>

<div id="progress"></div>
<div id="map">
  <div id="map-legend"><h3>Legend</h3><div style="color:#9ca3af;font-size:11px;margin-bottom:4px">Numbered pins are all destinations. GA lines show explored candidates; stats show best-so-far.</div><div id="legend-rows"></div></div>
</div>

<div id="bottombar">
  <button id="play-btn">&#9654; Play</button>
  <span id="gen-label">Gen — / —</span>
  <input type="range" id="gen-slider" min="1" max="1" value="1">
  <div class="speed-ctrl">
    <label>Speed</label>
    <select id="speed-select">
      <option value="2500">Very Slow</option>
      <option value="1500" selected>Slow</option>
      <option value="700">Normal</option>
      <option value="220">Fast</option>
      <option value="50">Turbo</option>
    </select>
  </div>
  <div class="speed-ctrl">
    <label>Animate</label>
    <input type="checkbox" id="anim-toggle" checked title="Animate route drawing">
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

// ── Map ───────────────────────────────────────────────────────
const map = L.map('map');
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{
  attribution:'&copy; OpenStreetMap &copy; CARTO',maxZoom:19
}).addTo(map);

// layer stores
let gaLayers     = {};   // algo_name -> L.Polyline  (GA routes, update per gen)
let baseLayers   = {};   // algo_name -> L.Polyline  (static baselines)
let stopMarkers  = [];
let playTimer    = null;
let isPlaying    = false;

const algoSel    = document.getElementById('algo-select');
const scenSel    = document.getElementById('scenario-select');
const slider     = document.getElementById('gen-slider');
const genLabel   = document.getElementById('gen-label');
const playBtn    = document.getElementById('play-btn');
const speedSel   = document.getElementById('speed-select');
const animToggle = document.getElementById('anim-toggle');
const progress   = document.getElementById('progress');
const legend     = document.getElementById('map-legend');
const legRows    = document.getElementById('legend-rows');

// ── Path draw animation (SVG stroke-dashoffset trick) ─────────
function animatePath(polyline, durationMs){
  if(!animToggle.checked) return;
  setTimeout(()=>{
    const el=polyline.getElement();
    if(!el) return;
    try{
      const len=el.getTotalLength();
      el.style.strokeDasharray=len+' '+len;
      el.style.strokeDashoffset=len;
      el.style.transition='none';
      void el.getBoundingClientRect(); // force reflow so initial state sticks
      el.style.transition=`stroke-dashoffset ${durationMs}ms ease-out`;
      el.style.strokeDashoffset='0';
    }catch(e){}
  },0);
}

const IS_ALL = '__ALL__';

// ── Populate dropdowns ────────────────────────────────────────
// First option: All Algorithms
const allOpt = document.createElement('option');
allOpt.value = IS_ALL; allOpt.textContent = '— All Algorithms —';
algoSel.appendChild(allOpt);

const ALL_ALGO_NAMES = Array.from(new Set([
  ...Object.keys(DATA.algorithms||{}),
  ...Object.keys(DATA.baselines||{})
]));

ALL_ALGO_NAMES.forEach(a=>{
  const o=document.createElement('option'); o.value=a; o.textContent=a;
  algoSel.appendChild(o);
});
Object.keys(DATA.scenarios).forEach(s=>{
  const o=document.createElement('option'); o.value=s;
  o.textContent = DATA.scenarios[s].round_trip ? s+' ↩' : s;
  scenSel.appendChild(o);
});

// ── Helpers ───────────────────────────────────────────────────
function col(name){ return DATA.colors[name]||'#aaa'; }

function iconDiv(letter,bg){
  return L.divIcon({
    html:`<div style="background:${bg};color:#fff;border-radius:50%;width:30px;height:30px;
          display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;
          border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.45)">${letter}</div>`,
    iconSize:[30,30],iconAnchor:[15,15],className:''
  });
}

function stopIcon(label,bg){
  return L.divIcon({
    html:`<div style="background:${bg};color:#fff;border-radius:50%;width:30px;height:30px;
          display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;
          border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.45)">${label}</div>`,
    iconSize:[30,30],iconAnchor:[15,15],className:''
  });
}

function clearGALayers(){
  Object.values(gaLayers).forEach(l=>map.removeLayer(l));
  gaLayers={};
}
function clearBaseLayers(){
  Object.values(baseLayers).forEach(l=>map.removeLayer(l));
  baseLayers={};
}
function clearMarkers(){
  stopMarkers.forEach(m=>map.removeLayer(m));
  stopMarkers=[];
}

function getHistory(algoName){
  const sc=scenSel.value;
  return ((DATA.algorithms[algoName]||{})[sc]||[]);
}

function maxGenAll(){
  // slider max for "All" mode = max generations across all GA algos for this scenario
  let m=1;
  Object.keys(DATA.algorithms).forEach(a=>{
    const h=getHistory(a);
    if(h.length>m) m=h.length;
  });
  return m;
}

function isGAAlgo(algoName){
  return Object.prototype.hasOwnProperty.call(DATA.algorithms||{}, algoName);
}

// ── Render GA routes at generation idx ───────────────────────
function renderGARoute(algoName, genIdx, weight, opacity){
  const hist=getHistory(algoName);
  if(!hist.length) return;
  const frame=hist[Math.min(genIdx,hist.length-1)];
  const drawCoords=frame.candidate_coords||frame.coords;
  if(!drawCoords||drawCoords.length<2) return;
  if(gaLayers[algoName]) map.removeLayer(gaLayers[algoName]);
  const drawMin=frame.candidate_min!==undefined ? frame.candidate_min : frame.min;
  const drawDist=frame.candidate_dist!==undefined ? frame.candidate_dist : frame.dist;
  const distText = drawDist!==undefined ? ` | ${Number(drawDist).toFixed(2)} km` : '';
  const gaPl=L.polyline(drawCoords,{
    color:col(algoName),weight:weight||5,opacity:opacity||.92,
    interactive:true
  }).bindTooltip(`${algoName} | gen ${frame.gen} candidate | ${drawMin.toFixed(1)} min${distText}`).addTo(map);
  gaLayers[algoName]=gaPl;
  // animate line draw — duration scales with playback speed so fast/turbo still feel snappy
  const speedMs=parseInt(speedSel.value,10);
  animatePath(gaPl, Math.round(speedMs*0.80));
}

// ── Render static baseline route ─────────────────────────────
function renderBaseline(algoName){
  const sc=scenSel.value;
  const b=(DATA.baselines[algoName]||{})[sc];
  if(!b||!b.coords||b.coords.length<2) return;
  if(baseLayers[algoName]) map.removeLayer(baseLayers[algoName]);
  // Christofides baseline is intentionally solid to match GA-style line readability.
  const isSolid = algoName==='christofides';
  const baselineStyle = {
    color:col(algoName),weight:3,opacity:.65,dashArray:'8 5',
    interactive:true
  };
  if(isSolid){
    delete baselineStyle.dashArray;
    baselineStyle.opacity=.9;
    baselineStyle.weight=4;
  }
  const basePl=L.polyline(b.coords,baselineStyle)
    .bindTooltip(`${algoName} | ${b.min.toFixed(1)} min | ${b.dist.toFixed(2)} km (static)`).addTo(map);
  baseLayers[algoName]=basePl;
  animatePath(basePl, 1100);
}

// ── Update legend ─────────────────────────────────────────────
function updateLegend(genIdx){
  const isAll=algoSel.value===IS_ALL;
  legRows.innerHTML='';
  if(isAll){
    legend.style.display='block';
    // GA rows
    Object.keys(DATA.algorithms).forEach(a=>{
      const hist=getHistory(a);
      if(!hist.length) return;
      const frame=hist[Math.min(genIdx,hist.length-1)];
      const shownMin=frame.candidate_min!==undefined ? frame.candidate_min : frame.min;
      const shownDist=frame.candidate_dist!==undefined ? frame.candidate_dist : frame.dist;
      const gaDist=shownDist!==undefined ? ` | ${Number(shownDist).toFixed(2)} km` : '';
      const row=document.createElement('div'); row.className='leg-row';
      row.innerHTML=`<div class="leg-swatch" style="background:${col(a)}"></div>
        <span class="leg-name">${a}</span>
        <span class="leg-val">${shownMin.toFixed(1)} min${gaDist}</span>`;
      legRows.appendChild(row);
    });
    // Baseline rows
    const sc=scenSel.value;
    Object.keys(DATA.baselines).forEach(b=>{
      const data=(DATA.baselines[b]||{})[sc];
      if(!data) return;
      const swatchClass = b==='christofides' ? 'leg-swatch' : 'leg-swatch leg-dashed';
      const row=document.createElement('div'); row.className='leg-row';
      row.innerHTML=`<div class="${swatchClass}" style="color:${col(b)};background:${b==='christofides' ? col(b) : ''}"></div>
        <span class="leg-name">${b}</span>
        <span class="leg-val">${data.min.toFixed(1)} min | ${data.dist.toFixed(2)} km</span>`;
      legRows.appendChild(row);
    });
  } else {
    legend.style.display='none';
  }
}

// ── Main render for a generation idx ─────────────────────────
function renderGen(genIdx){
  const sc   = scenSel.value;
  const algo = algoSel.value;
  const isAll = algo===IS_ALL;

  clearGALayers();

  if(isAll){
    // Draw all GA routes at genIdx
    Object.keys(DATA.algorithms).forEach(a=>renderGARoute(a,genIdx,4,.85));

    // Stats: best GA time at this gen
    let bestMin=Infinity, bestDist=null, bestAlgo='';
    Object.keys(DATA.algorithms).forEach(a=>{
      const h=getHistory(a); if(!h.length) return;
      const f=h[Math.min(genIdx,h.length-1)];
      if(f.min<bestMin){bestMin=f.min;bestDist=f.dist;bestAlgo=a;}
    });
    const totalGens=maxGenAll();
    const refData=(DATA.baselines['dijkstra_time']||{})[sc];
    const firstMins=Object.keys(DATA.algorithms).map(a=>{
      const h=getHistory(a); return h.length?h[0].min:Infinity;
    });
    const firstBest=Math.min(...firstMins);
    const impr=firstBest>0?((firstBest-bestMin)/firstBest*100).toFixed(1):'0.0';

    document.getElementById('s-gen').textContent  = genIdx+1;
    document.getElementById('s-time').textContent = bestMin<Infinity?bestMin.toFixed(1):'—';
    document.getElementById('s-dist').textContent = bestDist!==null&&bestDist!==undefined?Number(bestDist).toFixed(2):'-';
    document.getElementById('s-impr').textContent = impr+'%';
    document.getElementById('s-ref').textContent  = refData?refData.min.toFixed(1):'—';
    document.getElementById('s-total').textContent= totalGens;
    genLabel.textContent=`Gen ${genIdx+1} / ${totalGens}`;
    progress.style.width=(((genIdx+1)/totalGens)*100)+'%';

  } else {
    // Single algorithm mode
    const refData=(DATA.baselines['dijkstra_time']||{})[sc];
    if(isGAAlgo(algo)){
      renderGARoute(algo,genIdx,5,.92);
      const hist=getHistory(algo);
      if(!hist.length) return;
      const frame=hist[Math.min(genIdx,hist.length-1)];
      const first=hist[0].min;
      const impr=first>0?((first-frame.min)/first*100).toFixed(1):'0.0';

      document.getElementById('s-gen').textContent  = frame.gen;
      document.getElementById('s-time').textContent = frame.min.toFixed(1);
      document.getElementById('s-dist').textContent = frame.dist!==undefined?Number(frame.dist).toFixed(2):'-';
      document.getElementById('s-impr').textContent = impr+'%';
      document.getElementById('s-ref').textContent  = refData?refData.min.toFixed(1):'—';
      document.getElementById('s-total').textContent= hist.length;
      genLabel.textContent=`Gen ${frame.gen} / ${hist.length}`;
      progress.style.width=((frame.gen/hist.length)*100)+'%';
    } else {
      renderBaseline(algo);
      const b=(DATA.baselines[algo]||{})[sc];
      document.getElementById('s-gen').textContent  = '1';
      document.getElementById('s-time').textContent = b?b.min.toFixed(1):'—';
      document.getElementById('s-dist').textContent = b?Number(b.dist).toFixed(2):'-';
      document.getElementById('s-impr').textContent = '0.0%';
      document.getElementById('s-ref').textContent  = refData?refData.min.toFixed(1):'—';
      document.getElementById('s-total').textContent= '1';
      genLabel.textContent='Gen 1 / 1';
      progress.style.width='100%';
    }
  }

  updateLegend(genIdx);
}

// ── Reset scene (algo or scenario changed) ────────────────────
function resetScene(){
  stopPlay();
  clearGALayers();
  clearBaseLayers();
  clearMarkers();

  const sc    = DATA.scenarios[scenSel.value];
  const algo  = algoSel.value;
  const isAll = algo===IS_ALL;

  if(sc){
    const stops=sc.stops&&sc.stops.length ? sc.stops : [
      {label:sc.source_label,coords:sc.source},
      {label:sc.target_label,coords:sc.target}
    ];
    stops.forEach((stop,idx)=>{
      const isStart = idx===0;
      const pinColor = isStart ? '#e63946' : '#4b5563';
      const marker=L.marker(stop.coords,{
        icon:stopIcon(String(idx+1), pinColor),
        zIndexOffset:1000
      }).addTo(map).bindTooltip(`${idx+1}. ${stop.label}${isStart&&sc.round_trip?' (Start / Return)':''}`);
      marker.bindPopup(`<b>Destination ${idx+1}${isStart&&sc.round_trip?' — Start &amp; Return':''}</b><br>${stop.label}`);
      stopMarkers.push(marker);
    });
    // For round-trip: show a faint dashed line hinting the return leg direction
    if(sc.round_trip && stops.length>=2){
      const returnHint=L.polyline([stops[stops.length-1].coords, stops[0].coords],{
        color:'#e63946',weight:2,opacity:.35,dashArray:'6 8',interactive:false
      }).addTo(map);
      stopMarkers.push(returnHint);
    }
    try{
      map.fitBounds(stops.map(s=>s.coords),{padding:[30,30]});
    }catch(e){
      map.setView(sc.source,13);
    }
  }

  // Draw baselines: all when in "All" mode, only the selected one when a baseline is picked,
  // nothing when a single GA/SA algo is selected (keeps the view uncluttered).
  if(isAll){
    Object.keys(DATA.baselines).forEach(b=>renderBaseline(b));
  } else if(!isGAAlgo(algo)){
    renderBaseline(algo);
  }

  const maxG = isAll ? maxGenAll() : (getHistory(algo).length||1);
  slider.min=1; slider.max=maxG; slider.value=1;
  progress.style.width='0%';
  renderGen(0);
}

// ── Playback ──────────────────────────────────────────────────
function stopPlay(){
  if(playTimer){clearInterval(playTimer);playTimer=null;}
  isPlaying=false;
  playBtn.innerHTML='&#9654; Play';
}
function getMaxGen(){
  const algo=algoSel.value;
  return algo===IS_ALL ? maxGenAll() : (getHistory(algo).length||1);
}
function startPlay(){
  isPlaying=true;
  playBtn.innerHTML='&#9646;&#9646; Pause';
  const ms=parseInt(speedSel.value,10);
  playTimer=setInterval(()=>{
    const cur=parseInt(slider.value,10);
    const next=cur+1;
    if(next>getMaxGen()){stopPlay();return;}
    slider.value=next;
    renderGen(next-1);
  },ms);
}

playBtn.addEventListener('click',()=>{
  if(isPlaying){stopPlay();}
  else{
    if(parseInt(slider.value)>=getMaxGen()){slider.value=1;renderGen(0);}
    startPlay();
  }
});
slider.addEventListener('input',()=>{stopPlay();renderGen(parseInt(slider.value)-1);});
algoSel.addEventListener('change',resetScene);
scenSel.addEventListener('change',resetScene);

resetScene();
</script>
</body>
</html>"""
