#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Chicago TD-CVRP v2/v3 visualizer: side-by-side, true OSM-edge roads.

Supports directly the `solutions.json` emitted by v2 upstream_library/compare,
upstream_library/ablations and base_10tasks. Uses v2 Road.exact's *oracle*
road-leg TDSP evaluation protocol for both static and oracle planners.

All data (roads, legs, stops) are embedded in ONE HTML, no CDN, online tiles,
GIS server, trained model or official ALNS package required.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / 'chicago_tdcvrp_solver_suite_v2'
if V2.is_dir() and str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
try:
    from road_eval import Road
    from models import verify
except ImportError as exc:
    raise SystemExit('Cannot import chicago_tdcvrp_solver_suite_v2/road_eval.py and models.py; '\
                     'extract this tool in the road_vrp repository root. '+str(exc))

COLORS=['#e44d45','#2474dd','#159c78','#9c60cd','#df920f','#b94a83','#6366f1','#04869a']


def normalized(value):
    return str(value)


def _parse_linestring(text):
    """Parse OSMnx GraphML edge `geometry` WKT (LINESTRING or MULTILINESTRING).

    For unusual geoms, use installed shapely if present. Without a recognized
    geometry the endpoint line is used; count and disclose every fallback.
    """
    if not text:
        return None
    if hasattr(text,'coords'):
        return [(float(x),float(y)) for x,y,*_ in text.coords]
    s=str(text).strip()
    if s.upper().startswith('LINESTRING'):
        try:
            z=s[s.index('(')+1:s.rindex(')')]
            return [tuple(map(float,p.strip().split()[:2])) for p in z.split(',')]
        except (ValueError,IndexError,TypeError):
            pass
    if s.upper().startswith('MULTILINESTRING'):
        try:
            from shapely import wkt
            g=wkt.loads(s)
            return [pt for piece in g.geoms for pt in piece.coords]
        except Exception:
            pass
    return None


def read_graph(graph_path):
    g=nx.read_graphml(graph_path)
    if not g.is_directed():
        raise ValueError('GraphML is not a directed road network')
    nodes={str(n):(float(v['x']),float(v['y'])) for n,v in g.nodes(data=True)}
    if not nodes:
        raise ValueError('Empty GraphML')
    lon0=(min(v[0] for v in nodes.values())+max(v[0] for v in nodes.values()))/2
    lat0=(min(v[1] for v in nodes.values())+max(v[1] for v in nodes.values()))/2
    # Local equirectangular projection in metres. Good for small downtown area.
    xscale=111320*math.cos(math.radians(lat0))
    yscale=111320
    def project(p):
        return [round((p[0]-lon0)*xscale,2),round((lat0-p[1])*yscale,2)]
    geom={}
    fallbacks=0
    edges=g.edges(keys=True,data=True) if g.is_multigraph() else ((u,v,0,a) for u,v,a in g.edges(data=True))
    for u,v,key,attrs in edges:
        u,v,key=normalized(u),normalized(v),normalized(key)
        points=_parse_linestring(attrs.get('geometry'))
        if not points or len(points)<2:
            points=[nodes[u],nodes[v]]
            fallbacks+=1
        else:
            du=(points[0][0]-nodes[u][0])**2+(points[0][1]-nodes[u][1])**2
            dv=(points[-1][0]-nodes[u][0])**2+(points[-1][1]-nodes[u][1])**2
            if dv<du:points=list(reversed(points))
        geom[(u,v,key)]=[project(p) for p in points]
    return nodes,geom,project,fallbacks


def route_signatures(routes):
    # Vehicle indices are arbitrary; order within a route matters.
    return tuple(sorted(tuple(int(c) for c in route) for route in routes))


def load_solutions(paths, task_id, solver_names, seed, planner, scenario):
    entries=[]; seen={}
    filters=set(solver_names)
    for path in paths:
        items=json.loads(path.read_text(encoding='utf8'))
        if not isinstance(items,list):raise ValueError(f'{path}: expected list of solutions')
        for r in items:
            if r.get('task_id')!=task_id or int(r.get('seed',-1))!=seed:continue
            if r.get('solver') not in filters:continue
            if planner!='all' and r.get('planner')!=planner:continue
            if scenario!='all' and r.get('scenario_id')!=scenario:continue
            identifier=(r['task_id'],r['scenario_id'],r['solver'],r['planner'],int(r['seed']))
            if identifier in seen:
                previous=seen[identifier]
                if route_signatures(previous['routes'])!=route_signatures(r['routes']):
                    print(f'WARNING: duplicate {identifier} has DIFFERENT routes in {path}; '
                          'keeping record from earlier --solutions input',file=sys.stderr)
                continue
            seen[identifier]=r
            entries.append(r)
    if not entries:
        raise ValueError(f'No matching results for task={task_id}, solvers={solver_names}, seed={seed}, '
                         f'planner={planner}, scenario={scenario}. Check --solutions paths.')
    return sorted(entries,key=lambda r:(r['scenario_id'],r['planner'],r['solver']))


def build_payload(args):
    costs=Road(args.cost_dir)
    node_geog,geoms,project,fallbacks=read_graph(args.graphml)
    # Edge-index correspondence must be exact: drawing the wrong graph is worse
    # than failing loudly.
    missing=[]
    for e in costs.edges.itertuples(index=False):
        ident=(normalized(e.osm_u),normalized(e.osm_v),normalized(e.osm_key))
        if ident not in geoms:missing.append(ident)
    if missing:
        raise ValueError(f'GraphML and edge_index.csv mismatch: {len(missing)} edges missing; '
                         f'e.g. {missing[:3]}')
    task=json.loads((args.task_dir/(args.task_id+'.json')).read_text(encoding='utf8'))
    customers={int(c['id']):c for c in task['customers']}
    depot=normalized(task['depot_node'])
    entries=load_solutions(args.solutions,args.task_id,args.solvers.split(','),args.seed,args.planner,args.scenario)
    variants=[]
    for rec in entries:
        task['require_all_vehicles']=rec.get('vehicle_rule','exactly')=='exactly'
        loads=verify(task,rec['routes'])
        departure=costs.to_sec(rec['departure_local'])
        total=0.0; fleet_elapsed=0.0; edge_count=0; observed_count=0
        vehicles=[]
        for vi,client_ids in enumerate(rec['routes']):
            t=departure
            here=depot
            legs=[]; stops=[]; car_sec=0.0
            if not client_ids:
                vehicles.append({'index':vi,'color':COLORS[vi%len(COLORS)],'load':loads[vi],
                                'clients':[],'travel_sec':0.,'legs':[],'stops':[],'elapsed_sec':0.})
                continue
            for order,cid in enumerate(list(client_ids)+[0]):
                there=depot if cid==0 else normalized(customers[int(cid)]['node'])
                depart_t=t
                drive_sec,edge_ids=costs.route(here,there,depart_t,mode='oracle')
                geometry=[]; replay_t=t
                leg_obs=0
                for edgeid in edge_ids:
                    row=costs.edges.iloc[int(edgeid)]
                    polyline=geoms[(normalized(row.osm_u),normalized(row.osm_v),normalized(row.osm_key))]
                    if geometry and geometry[-1]==polyline[0]:geometry.extend(polyline[1:])
                    else:geometry.extend(polyline)
                    slot=min(int(replay_t//costs.dt),len(costs.times)-1)
                    leg_obs+=int(costs.observed[slot,int(edgeid)])
                    replay_t+=costs.travel(int(edgeid),replay_t)
                if abs((replay_t-t)-drive_sec)>1e-4:
                    raise ValueError(f'TDSP leg replay mismatch: {here}->{there}')
                if not geometry:
                    geometry=[project(node_geog[here]),project(node_geog[there])]
                t+=drive_sec;car_sec+=drive_sec;total+=drive_sec
                edge_count+=len(edge_ids);observed_count+=leg_obs
                arrival=(costs.times[0]+pd.to_timedelta(t,unit='s')).strftime('%H:%M:%S')
                legs.append({'from':here,'to':there,'to_customer':int(cid),'line':geometry,
                             'travel_sec':round(drive_sec,4),'observed_count':leg_obs})
                if cid:
                    stops.append({'id':int(cid),'order':order+1,
                                  'demand':int(customers[int(cid)]['demand']),
                                  'arrival':arrival,'pt':project(node_geog[there])})
                    t+=float(task['service_seconds'])
                here=there
            fleet_elapsed+=t-departure
            vehicles.append({'index':vi,'color':COLORS[vi%len(COLORS)],'load':loads[vi],
                             'clients':list(map(int,client_ids)),'travel_sec':round(car_sec,4),
                             'elapsed_sec':round(t-departure,4),'legs':legs,'stops':stops})
        saved_cost=rec.get('recorded_cost_sec')
        if saved_cost is None:saved_cost=rec.get('road_exact_ex_post',{}).get('realized_travel_sec')
        if saved_cost is not None and abs(float(saved_cost)-total)>args.tolerance_sec:
            raise ValueError(f'Cost mismatch task {rec["task_id"]} {rec["scenario_id"]} '
                             f'{rec["solver"]} seed={rec["seed"]}: recomputed {total:.6f}s '
                             f'vs saved {float(saved_cost):.6f}s. Confirm same graph/task/cost tensor.')
        share=observed_count/max(1,edge_count)
        saved_share=rec.get('observed_edge_share')
        if saved_share is not None and abs(share-float(saved_share))>1e-5:
            raise ValueError(f'Observed-edge-share mismatch for {rec["solver"]} '+
                             f'{rec["scenario_id"]}: {share:.6f} vs saved {saved_share}')
        variants.append({'scenario':rec['scenario_id'],'solver':rec['solver'],'planner':rec['planner'],
                         'seed':int(rec['seed']),'departure':rec['departure_local'],
                         'travel_sec':round(total,5),'elapsed_sec':round(fleet_elapsed,5),
                         'observed_share':round(share,6),'vehicles':vehicles})
        print(f'  {rec["scenario_id"]} / {rec["solver"]}: {total:.2f}s '
              f'({share:.1%} monitored road-edge entries)')
    # Use only unique OSM geometry, avoiding redundant opposite-direction edges.
    background=[]; seen_lines=set()
    for poly in geoms.values():
        # Deduplicate identical reversible polylines, but preserve parallel lanes.
        key=min(tuple(map(tuple,poly)),tuple(map(tuple,reversed(poly))))
        if key not in seen_lines:
            seen_lines.add(key);background.append(poly)
    pts=[p for route in background for p in route]
    if not pts:raise ValueError('Graph has no coordinates')
    bounds=[min(x for x,y in pts),min(y for x,y in pts),
            max(x for x,y in pts),max(y for x,y in pts)]
    interest=[project(node_geog[depot])]+[project(node_geog[str(c['node'])]) for c in task['customers']]
    padding=180.0
    cbounds=[min(p[0] for p in interest)-padding,min(p[1] for p in interest)-padding,
             max(p[0] for p in interest)+padding,max(p[1] for p in interest)+padding]
    return {'task_id':args.task_id,'bounds':bounds,'customer_bounds':cbounds,
            'depot_pt':project(node_geog[depot]),'customer_locations':[
            {'id':int(c['id']),'pt':project(node_geog[str(c['node'])])} for c in task['customers']],
            'vehicles_capacity':int(task['capacity']),'colors':COLORS,
            'background':background,'variants':variants,
            'diagnostics':{'graph_nodes':len(node_geog),'graph_directed_edges':len(geoms),
                           'graph_geometry_fallback':fallbacks,
                           'road_observations_note':'Only monitored road-edge traversals are measured; other edges use static OSM estimates.',
                           'evaluation_note':'STATIC and ORACLE customer sequences both use historical time-dependent road-leg shortest paths in replay.'}}


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Chicago TD-CVRP | Open-source ALNS 车辆路线对比</title>
<style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei","Noto Sans CJK SC",sans-serif;color:#192638}
*{box-sizing:border-box}body{margin:0;background:#f1f5f9}
header{padding:16px 24px;background:#fff;border-bottom:1px solid #dae2ea;display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap}
h1{font-size:18px;letter-spacing:.2px;margin:0 0 5px}header p{margin:0;color:#65768c;font-size:12px}
.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap;background:white;padding:12px 22px;border-bottom:1px solid #e3e8f0}
.controls label,.controls .ctl{font-size:12px;font-weight:650;color:#4c5d73}
select{display:block;border-radius:7px;border:1px solid #c4cfdb;background:#fff;color:#23364d;font-size:13px;padding:8px;min-width:150px;max-width:230px;margin-top:5px}
button{cursor:pointer;background:white;border:1px solid #b9c6d7;border-radius:6px;color:#23364d;padding:8px 11px}
button:hover{background:#eaf2fc}.summary{display:flex;gap:15px;align-items:center;font-size:13px;flex-wrap:wrap}
.metric{color:#65768c}.metric strong{color:#26374c;font-variant-numeric:tabular-nums}
#difference{font-weight:750;font-size:14px}
.switches{display:flex;flex-wrap:wrap;align-items:center;gap:9px;font-size:12px;padding:8px 22px;background:#fff}
.switches label{display:flex;gap:4px;align-items:center}.vcolor{display:inline-block;width:10px;height:10px;border-radius:100%}
.comparison{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;padding:10px;min-height:calc(100vh - 190px)}
.panel{background:#fff;border:1px solid #d6e0eb;border-radius:11px;overflow:hidden;min-width:0;display:flex;flex-direction:column}
.panelhead{padding:10px 14px;border-bottom:1px solid #e7eaf0;display:flex;align-items:center;justify-content:space-between;gap:7px;flex-wrap:wrap}
.panelhead b{font-size:15px}.panelhead small{font-size:12px;color:#667589}.panelhead strong{font-size:19px;display:block;font-variant-numeric:tabular-nums}
.mapwrap{position:relative;min-height:380px;flex:1;background:#f7fafc}.mapwrap canvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab}.mapwrap canvas.dragging{cursor:grabbing}
.tooltip{display:none;position:absolute;pointer-events:none;z-index:3;background:#1c283a;color:#fff;font-size:12px;border-radius:6px;padding:8px 10px;max-width:240px;line-height:1.55;white-space:pre-line}
.travelinfo{font-size:12px;border-top:1px solid #e8edf4;padding:9px 13px;line-height:1.75;max-height:170px;overflow:auto}.travelinfo details{margin:3px 0}
.bottom{margin:0;padding:11px 22px;color:#64748b;font-size:12px;background:#fff;border-top:1px solid #e1e7ef;line-height:1.6}
@media(max-width:850px){.comparison{grid-template-columns:1fr}.mapwrap{min-height:440px}}
</style></head><body>
<header><div><h1>Chicago TD-CVRP30 · 开源ALNS车辆路线对比</h1><p>真实OSM路网几何 · 历史交通下时变最短路回放 · 完全离线交互地图</p></div>
<div class="summary"><span class="metric">任务 <strong id="taskname"></strong></span><span class="metric" id="meta"></span><span id="difference"></span></div></header>
<div class="controls"><label>交通日期/时段<select id="scen"></select></label><label>左侧求解器<select id="left"></select></label><label>右侧求解器<select id="right"></select></label>
<div class="ctl"><button id="focus">聚焦客户区域</button> <button id="reset">查看全路网</button> <button id="zoomIn">放大 +</button> <button id="zoomOut">缩小 −</button> <button id="save">保存双图PNG</button></div></div>
<div class="switches"><strong>车辆图层：</strong><span id="vehiclecontrols"></span><label><input id="roads" type="checkbox" checked>道路底图</label><label><input id="labels" type="checkbox" checked>客户编号</label><label><input id="arrows" type="checkbox" checked>方向箭头</label><span style="color:#64748b">滚轮缩放 · 拖动平移（双图同步）</span></div>
<main class="comparison"><section class="panel"><div id="lh" class="panelhead"></div><div class="mapwrap" id="w0"><canvas id="c0"></canvas><div id="tip0" class="tooltip"></div></div><div class="travelinfo" id="routes0"></div></section>
<section class="panel"><div id="rh" class="panelhead"></div><div class="mapwrap" id="w1"><canvas id="c1"></canvas><div id="tip1" class="tooltip"></div></div><div class="travelinfo" id="routes1"></div></section></main>
<p class="bottom"><b>方法说明：</b>每种颜色表示该方案中的一辆车，左右车辆编号不代表相同客户组。所有客户分配与访问顺序来自已保存的求解器方案；彩色路线是在相应日期用原始有向OSM道路图和历史时变边权逐段重建，不是实测GPS轨迹。静态规划和Oracle规划均按同一动态路网重回测；监测缺失边采用静态估计成本。路线重叠、交叉不自动等于非最优。</p>
<script>const DATA=__PAYLOAD__;
const sc=document.getElementById('scen'),L=document.getElementById('left'),R=document.getElementById('right');
const canvas=[document.getElementById('c0'),document.getElementById('c1')],contexts=canvas.map(c=>c.getContext('2d'));
const panels=[document.getElementById('w0'),document.getElementById('w1')];
const view={x:0,y:0,scale:1},sizes=[[100,100],[100,100]];
let displayed=[null,null],currentScenario='',hover=[[],[]],enabled=Array(DATA.colors.length).fill(true),showRoads=true,showLabels=true,showArrows=true,focused=true;
const scenes=[...new Set(DATA.variants.map(x=>x.scenario))];const methods=[...new Set(DATA.variants.map(x=>x.solver+' / '+x.planner))];
for(const date of scenes){let o=document.createElement('option');o.value=date;o.textContent=date;sc.append(o)}
for(const select of [L,R])for(const name of methods){let o=document.createElement('option');o.value=name;o.textContent=name;select.append(o)}
R.value=methods.find(x=>x.startsWith('upstream_alns / '))||methods[1]||methods[0];L.value=methods.find(x=>x==='alns / oracle')||methods[0];
function itemFor(scene,method){return DATA.variants.find(x=>x.scenario===scene&&x.solver+' / '+x.planner===method)||null}
function compare(){currentScenario=sc.value;displayed=[itemFor(sc.value,L.value),itemFor(sc.value,R.value)];for(const index of [0,1]){
 const x=displayed[index],h=document.getElementById(index?'rh':'lh'),r=document.getElementById('routes'+index);
 if(!x){h.textContent='此日期没有对应求解结果';r.textContent='';continue}
 const obs=(100*x.observed_share).toFixed(1),minutes=(x.travel_sec/60).toFixed(2);
 h.innerHTML='<div><b>'+x.solver+' · '+x.planner.toUpperCase()+'</b><br/><small>'+x.scenario+' ｜ seed '+x.seed+' ｜ 观测边 '+obs+'%</small></div><div style="text-align:right"><strong>'+x.travel_sec.toFixed(2)+' 秒</strong><small>'+minutes+' 分钟 · 含服务车时 '+(x.elapsed_sec/60).toFixed(1)+'分钟</small></div>';
 r.innerHTML=x.vehicles.map((v,j)=>'<details><summary><span class="vcolor" style="background:'+v.color+'"></span> 车辆'+(j+1)+'：载重 '+v.load+'/'+DATA.vehicles_capacity+' · 行驶 '+(v.travel_sec/60).toFixed(2)+'分钟 · 客户'+v.clients.length+'个</summary><div>仓库 → '+v.clients.join(' → ')+' → 仓库</div></details>').join('');
 }
 const [a,b]=displayed;if(a&&b){const gain=a.travel_sec-b.travel_sec;document.getElementById('difference').textContent='右侧比左侧 '+(gain>=0?'节省 ':'增加 ')+Math.abs(gain).toFixed(2)+' 秒（'+(Math.abs(gain)/a.travel_sec*100).toFixed(2)+'%）';}
 else document.getElementById('difference').textContent='';draw()}
function dims(){for(let i=0;i<2;i++){let bb=panels[i].getBoundingClientRect(),w=Math.max(100,bb.width),h=Math.max(100,bb.height),d=window.devicePixelRatio||1;sizes[i]=[w,h];canvas[i].width=Math.floor(w*d);canvas[i].height=Math.floor(h*d);contexts[i].setTransform(d,0,0,d,0,0)}}
function reset(whole=false){focused=!whole;const q=whole?DATA.bounds:DATA.customer_bounds;view.x=(q[0]+q[2])/2;view.y=(q[1]+q[3])/2;view.scale=.86*Math.min(...sizes.map(([w,h])=>Math.min(w/(q[2]-q[0]),h/(q[3]-q[1]))));draw()}
function mapPoint(p,i){const [w,h]=sizes[i];return [(p[0]-view.x)*view.scale+w/2,(p[1]-view.y)*view.scale+h/2]}
function line(ctx,p,i){if(p.length<2)return;const u=mapPoint(p[0],i);ctx.beginPath();ctx.moveTo(...u);for(let j=1;j<p.length;j++)ctx.lineTo(...mapPoint(p[j],i));ctx.stroke()}
function marker(ctx,p,i,color,txt,rad=10){const [x,y]=mapPoint(p,i);ctx.beginPath();ctx.arc(x,y,rad,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.lineWidth=2.5;ctx.strokeStyle=color;ctx.stroke();if(txt!==null){ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='bold 10px system-ui';ctx.fillStyle=color;ctx.fillText(String(txt),x,y)}}
function arrow(ctx,segment,i,color){const a=mapPoint(segment[0],i),b=mapPoint(segment[segment.length-1],i);let dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);if(length<5)return;let ux=dx/length,uy=dy/length,m=mapPoint(segment[Math.floor(segment.length/2)],i);ctx.beginPath();ctx.moveTo(m[0]+ux*7,m[1]+uy*7);ctx.lineTo(m[0]-ux*5-uy*4,m[1]-uy*5+ux*4);ctx.lineTo(m[0]-ux*5+uy*4,m[1]-uy*5-ux*4);ctx.closePath();ctx.fillStyle=color;ctx.fill()}
function draw(){for(const i of [0,1]){const ctx=contexts[i],[w,h]=sizes[i];ctx.clearRect(0,0,w,h);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,w,h);
 if(showRoads){ctx.strokeStyle='#d0d9e1';ctx.lineWidth=.85;ctx.globalAlpha=.7;for(const seg of DATA.background)line(ctx,seg,i);ctx.globalAlpha=1}
 const solution=displayed[i];hover[i]=[];if(!solution)continue;
 solution.vehicles.forEach((v,k)=>{if(!enabled[k])return;ctx.strokeStyle=v.color;ctx.lineWidth=3.3;ctx.lineJoin='round';ctx.lineCap='round';ctx.globalAlpha=.88;
 for(const leg of v.legs){line(ctx,leg.line,i);if(showArrows&&leg.line.length>1)arrow(ctx,leg.line,i,v.color)}ctx.globalAlpha=1});
 solution.vehicles.forEach((v,k)=>{if(!enabled[k])return;for(const stop of v.stops){marker(ctx,stop.pt,i,v.color,showLabels?stop.id:null,showLabels?10:6);const [x,y]=mapPoint(stop.pt,i);hover[i].push({x,y,stop,vehicle:k+1,load:v.load})}});
 marker(ctx,DATA.depot_pt,i,'#1a293b','D',13);
 const metres=100;let px=metres*view.scale;if(px>=30&&px<=w*.3){ctx.lineWidth=2;ctx.strokeStyle='#455569';ctx.beginPath();ctx.moveTo(20,h-28);ctx.lineTo(20+px,h-28);ctx.stroke();ctx.fillStyle='#455569';ctx.font='11px system-ui';ctx.fillText('100 m',20,h-34)}
 }}
function zoom(factor,pixel){const [w,h]=sizes[0],x=pixel?pixel[0]:w/2,y=pixel?pixel[1]:h/2;const old=view.scale;const mx=view.x+(x-w/2)/old,my=view.y+(y-h/2)/old;view.scale=Math.max(.005,Math.min(50,old*factor));view.x=mx-(x-w/2)/view.scale;view.y=my-(y-h/2)/view.scale;draw()}
let dragging=false,last=null;
canvas.forEach((c,i)=>{c.addEventListener('wheel',e=>{e.preventDefault();let b=c.getBoundingClientRect(),[w,h]=sizes[i];let p=[e.clientX-b.left,e.clientY-b.top];if(i===1){p[0]+=((sizes[0][0]-w)/2);p[1]+=((sizes[0][1]-h)/2)}zoom(Math.exp(-Math.sign(e.deltaY)*.16),p)},{passive:false});
c.addEventListener('pointerdown',e=>{dragging=true;last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId);c.classList.add('dragging')});
c.addEventListener('pointerup',()=>{dragging=false;c.classList.remove('dragging')});c.addEventListener('pointercancel',()=>{dragging=false;c.classList.remove('dragging')});
c.addEventListener('pointermove',e=>{if(dragging){view.x-=(e.clientX-last[0])/view.scale;view.y-=(e.clientY-last[1])/view.scale;last=[e.clientX,e.clientY];draw();return}
 const bb=c.getBoundingClientRect(),x=e.clientX-bb.left,y=e.clientY-bb.top,tip=document.getElementById('tip'+i);let best=null,dist=16;
 for(const v of hover[i]){const d=Math.hypot(x-v.x,y-v.y);if(d<dist){dist=d;best=v}}
 if(best){tip.style.display='block';tip.style.left=Math.max(8,Math.min(x+12,sizes[i][0]-220))+'px';tip.style.top=Math.max(8,y-64)+'px';tip.textContent='车辆 '+best.vehicle+'  |  客户 '+best.stop.id+'\n第 '+best.stop.order+' 站  ·  需求 '+best.stop.demand+'\n预计到达 '+best.stop.arrival+'  |  车载 '+best.load}else tip.style.display='none'});
});
const vc=document.getElementById('vehiclecontrols');vc.innerHTML=DATA.colors.slice(0,Math.max(...DATA.variants.map(v=>v.vehicles.length))).map((c,i)=>'<label><input type="checkbox" data-car="'+i+'" checked><span class="vcolor" style="background:'+c+'"></span>车'+(i+1)+'</label>').join('');
vc.querySelectorAll('input').forEach(el=>el.addEventListener('change',()=>{enabled[+el.dataset.car]=el.checked;draw()}));
document.getElementById('roads').onchange=e=>{showRoads=e.target.checked;draw()};document.getElementById('labels').onchange=e=>{showLabels=e.target.checked;draw()};document.getElementById('arrows').onchange=e=>{showArrows=e.target.checked;draw()};
for(const el of [sc,L,R])el.onchange=compare;
document.getElementById('focus').onclick=()=>reset(false);document.getElementById('reset').onclick=()=>reset(true);
document.getElementById('zoomIn').onclick=()=>zoom(1.3);document.getElementById('zoomOut').onclick=()=>zoom(1/1.3);
document.getElementById('save').onclick=()=>{const w=Math.floor(sizes[0][0]*2),h=Math.floor(Math.min(...sizes.map(x=>x[1]))),dst=document.createElement('canvas');dst.width=w*2;dst.height=h*2+90;let c=dst.getContext('2d');c.scale(2,2);c.fillStyle='white';c.fillRect(0,0,w,h+45);for(let i=0;i<2;i++){c.drawImage(canvas[i],i*sizes[0][0],45,sizes[0][0],h);c.fillStyle='#111827';c.font='bold 13px system-ui';const s=displayed[i];c.fillText(s?s.solver+' '+s.travel_sec.toFixed(1)+'秒':'无数据',i*sizes[0][0]+15,25)}const a=document.createElement('a');a.download='Chicago_'+DATA.task_id+'_'+sc.value+'_routes.png';a.href=dst.toDataURL('image/png');a.click()};
document.getElementById('taskname').textContent=DATA.task_id;document.getElementById('meta').textContent=DATA.diagnostics.graph_nodes+'个节点 / '+DATA.diagnostics.graph_directed_edges+'条有向边';
function onResize(){dims();reset(!focused)}window.addEventListener('resize',onResize);dims();sc.value=scenes[0];compare();reset(false);
</script></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solutions',type=Path,nargs='+',required=True,help='One or more v2 solutions.json files')
    parser.add_argument('--task-dir',type=Path,default=Path('datasets/processed/chicago/tdcvrp30/tasks_10'))
    parser.add_argument('--cost-dir',type=Path,default=Path('datasets/processed/chicago/dynamic_costs'))
    parser.add_argument('--graphml',type=Path,default=Path('datasets/raw/chicago/chicago_loop_drive.graphml'))
    parser.add_argument('--task-id',default='task_0000')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--planner',choices=['oracle','static','all'],default='oracle')
    parser.add_argument('--scenario',default='all',help='all or 2025-06-05_0800 etc')
    parser.add_argument('--solvers',default='alns,upstream_alns',help='Comma-separated solver names')
    parser.add_argument('--output',type=Path,default=Path('runs/chicago_solver_suite_v2/maps/task_0000_compare.html'))
    parser.add_argument('--tolerance-sec',type=float,default=.05,help='Required agreement with saved exact ex-post travel cost')
    args=parser.parse_args()
    if args.tolerance_sec<0:parser.error('--tolerance-sec must be nonnegative')
    data=build_payload(args)
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html=HTML.replace('__PAYLOAD__',payload)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html,encoding='utf8')
    print(f'HTML written: {args.output} ({args.output.stat().st_size/1024:.1f}KiB, '
          f'{len(data["variants"])} solver/date views, fully offline)')
    print('Open file:// in Chrome/Edge/Firefox or transfer HTML via SCP.')
    if data['diagnostics']['graph_geometry_fallback']:
        print('GraphML edges drawn as straight endpoints due to missing geometry:',
              data['diagnostics']['graph_geometry_fallback'])


if __name__=='__main__':main()
