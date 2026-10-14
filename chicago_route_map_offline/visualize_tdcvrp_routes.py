#!/usr/bin/env python3
"""Create a fully offline, interactive geographic road-route map for Chicago TD-CVRP.

Rebuilds each road-level shortest path with historical time-dependent edge costs.
Inputs must correspond to same dataset version as stored VRP solutions.
No network requests or web tile dependencies; background streets come from OSM GraphML.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
import networkx as nx

# When called as chicago_route_map_offline/visualize_tdcvrp_routes.py, make the
# sibling tdcvrp package in the project root importable without installation.
root = Path(__file__).resolve().parent.parent
if (root / 'tdcvrp' / 'road.py').exists() and str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from tdcvrp.road import RoadCosts
    from tdcvrp.solver import validate
except ImportError as exc:
    raise SystemExit("Cannot import tdcvrp. Run from road_vrp project root, with the original tdcvrp package available. "+str(exc))

COLORS = ['#e54e42','#2185d0','#00a58a','#a65adb','#dd8c18']

def norm_id(a):
    s=str(a)
    return s[:-2] if s.endswith('.0') and s[:-2].isdigit() else s


def parse_wkt_linestring(wkt):
    """Parse normal OSMnx LINESTRING WKT without needing shapely installed."""
    if not isinstance(wkt, str) or not wkt.startswith('LINESTRING'):
        return None
    try:
        body=wkt[wkt.index('(')+1:wkt.rindex(')')]
        pts=[]
        for pair in body.split(','):
            vals=pair.strip().split()
            pts.append((float(vals[0]),float(vals[1])))
        return pts if len(pts)>=2 else None
    except (ValueError,IndexError):
        return None


def load_geometry(graphml):
    G=nx.read_graphml(graphml)
    if not G.is_directed():
        raise ValueError('Expected directed GraphML from OSMnx')
    coords={}
    for node,d in G.nodes(data=True):
        if 'x' not in d or 'y' not in d:
            raise ValueError(f'Missing lat/lon fields x/y on GraphML node {node!r}')
        coords[norm_id(node)]=(float(d['x']),float(d['y']))
    if not coords:
        raise ValueError('Empty OSM graph')
    mids=sorted(coords.values())
    lon0=(min(p[0] for p in mids)+max(p[0] for p in mids))/2
    lat0=(min(p[1] for p in mids)+max(p[1] for p in mids))/2
    lat_scale=111320.
    lon_scale=111320.*math.cos(math.radians(lat0))
    def project(p):
        return [round((p[0]-lon0)*lon_scale,2),round((lat0-p[1])*lat_scale,2)]

    edges={}
    iterator=G.edges(keys=True,data=True) if G.is_multigraph() else ((u,v,0,d) for u,v,d in G.edges(data=True))
    fallback_geometry=0
    for u,v,k,d in iterator:
        u,v,k=norm_id(u),norm_id(v),norm_id(k)
        pts=parse_wkt_linestring(d.get('geometry'))
        if pts is None:
            pts=[coords[u], coords[v]]
            fallback_geometry+=1
        else:
            # OSMnx graph edge geometry may be stored in either direction.
            start_to_u=(pts[0][0]-coords[u][0])**2+(pts[0][1]-coords[u][1])**2
            end_to_u=(pts[-1][0]-coords[u][0])**2+(pts[-1][1]-coords[u][1])**2
            if end_to_u<start_to_u:
                pts=pts[::-1]
        edges[(u,v,k)]=[project(p) for p in pts]
    return coords, edges, project, fallback_geometry


def build_data(args):
    road=RoadCosts(args.cost_dir)
    coords,geom,project,n_fallback=load_geometry(args.graphml)
    sol_list=json.loads(Path(args.solutions).read_text(encoding='utf-8'))
    if not isinstance(sol_list,list):
        raise ValueError('solutions.json should be a list')
    task_dir=Path(args.task_dir)
    cache={}
    def get_task(tid):
        if tid not in cache:
            p=task_dir/(tid+'.json')
            cache[tid]=json.loads(p.read_text(encoding='utf-8'))
        return cache[tid]

    # Validate geometry identity with dynamic cost index BEFORE any rendering.
    absent=[]
    for r in road.edges.itertuples(index=False):
        triple=(norm_id(r.osm_u), norm_id(r.osm_v), norm_id(r.osm_key))
        if triple not in geom:
            absent.append(triple)
    if absent:
        raise ValueError(f'{len(absent)} edge IDs from edge_index.csv missing in supplied GraphML; check graph versions. Examples: {absent[:4]}')
    variants=[]
    tasks_seen=set()
    for entry in sol_list:
        if args.task_id and entry['task_id']!=args.task_id:
            continue
        if args.scenario and entry['scenario_id']!=args.scenario:
            continue
        if args.planner and entry['planner']!=args.planner:
            continue
        if len(variants)>=args.max_views:
            break
        task=get_task(entry['task_id'])
        validate(task,entry['routes'])
        tasks_seen.add(entry['task_id'])
        cdict={int(c['id']):c for c in task['customers']}
        depot=norm_id(task['depot_node'])
        if depot not in coords:
            raise ValueError('Depot not in provided GraphML')
        departing=road.to_seconds(entry['departure_local'])
        t_total=0.
        veh_out=[]
        for i,sequence in enumerate(entry['routes']):
            here=depot
            time=departing
            pts=[]
            legs=[]
            stop_info=[]
            for order,target_id in enumerate(list(sequence)+[0]):
                target=depot if target_id==0 else norm_id(cdict[int(target_id)]['node'])
                if target not in coords:
                    raise ValueError(f'Customer node {target} missing from supplied OSM GraphML')
                sec,path=road.shortest(here,target,time,mode='dynamic',path=True)
                if not math.isfinite(sec):
                    raise ValueError(f'Unreachable {here}->{target} from {entry["scenario_id"]}')
                line=[]
                for idx in path:
                    e=road.edges.iloc[int(idx)]
                    p=geom[(norm_id(e.osm_u),norm_id(e.osm_v),norm_id(e.osm_key))]
                    if line and line[-1]==p[0]:line.extend(p[1:])
                    else:line.extend(p)
                # No road travel for coincident nodes; show a point instead.
                if not line:line=[project(coords[here]),project(coords[target])]
                t_total+=sec
                time+=sec
                pnow=project(coords[target])
                if target_id:
                    stop_info.append({'id':int(target_id),'node':target,
                                      'visit_order':order+1,'demand':int(cdict[int(target_id)]['demand']),
                                      'arrival':road.timestamp(time)[11:19], 'pt':pnow})
                    time+=float(task['service_seconds'])
                legs.append({'to':int(target_id),'pts':line,'sec':round(sec,3),
                             'from_node':here,'to_node':target,
                             'observed_edges':road.measure_path(path,time-sec-(float(task['service_seconds']) if target_id else 0.0)) if args.include_road_observations else None})
                here=target
            veh_out.append({'vehicle':i+1,'color':COLORS[i%len(COLORS)],'load':sum(cdict[int(x)]['demand'] for x in sequence),
                            'customers':list(map(int,sequence)), 'road_travel_sec':round(sum(x['sec'] for x in legs),3),
                            'stops':stop_info,'legs':legs,'elapsed_sec':round(time-departing,2)})
        prev=entry.get('road_exact_ex_post',{}).get('realized_travel_sec')
        if prev is not None and abs(float(prev)-t_total)>0.05:
            raise ValueError(f'Road-level recomputation differs from solutions.json for {entry["scenario_id"]}/{entry["planner"]}: '
                             f'{t_total:.3f}s vs saved {prev:.3f}s. Verify cost tensors, graph and original solution.')
        variants.append({'key':f'{entry["task_id"]} | {entry["scenario_id"]} | {entry["planner"]}',
                         'task':entry['task_id'],'scenario':entry['scenario_id'],
                         'planner':entry['planner'],'departure':entry['departure_local'],
                         'depot':{'node':depot,'pt':project(coords[depot])},'vehicles':veh_out,
                         'road_travel_sec':round(t_total,3),
                         'total_elapsed_sec':round(sum(v['elapsed_sec'] for v in veh_out),3),
                         'observed_share':round(float(entry.get('road_exact_ex_post',{}).get('observed_entry_share',0)),4),
                         'customer_count':len(cdict),'vehicles_count':len(veh_out),'vehicle_capacity':int(task['capacity'])})
        print('Prepared',variants[-1]['key'],f'travel={t_total:.2f}s')
    if not variants:
        raise ValueError('No matching solutions. Check --task-id / --scenario / --planner')
    variants.sort(key=lambda v: (v['task'], v['scenario'], 0 if v['planner']=='oracle' else 1))
    if len(tasks_seen)>1:
        print('NOTE: multiple tasks included: variants may have different geographic customers',file=sys.stderr)
    roads=[]
    seen=set()
    for (u,v,k),line in geom.items():
        # Draw each two-way geometric street segment once, keep parallel roadways.
        signature=tuple(sorted([u,v]))+(k,)
        if signature in seen:continue
        seen.add(signature)
        roads.append(line)
    xs=[p[0] for a in roads for p in a]
    ys=[p[1] for a in roads for p in a]
    bounds=[round(min(xs),2),round(min(ys),2),round(max(xs),2),round(max(ys),2)]
    return {'roads':roads,'variants':variants,'bounds':bounds,
            'meta':{'graph_edges':len(geom),'graph_nodes':len(coords),'geometry_fallback_edges':n_fallback,
                    'data_warning':'Real road graph, historical corridor speeds + unobserved static cost fallback. '
                                   'Historical hindsight oracle; not forecast or GPS trace.'}}


HTML=r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Chicago TD-CVRP30 | 车辆路线地图（离线）</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#1f2937;font-family:system-ui,-apple-system,"Microsoft YaHei",Arial,sans-serif}
header{height:68px;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;display:flex;align-items:center;justify-content:space-between;gap:15px}
header b{font-size:18px}header small{font-size:12px;color:#667085}main{display:grid;grid-template-columns:310px 1fr;height:calc(100vh - 68px)}
aside{background:white;border-right:1px solid #ddd;overflow-y:auto;padding:16px;box-shadow:0 0 5px #ddd}
section{margin:12px 0 18px 0}select{width:100%;font-size:13px;padding:9px;border:1px solid #ced4da;border-radius:6px;background:white}
.summary{font-size:13px;line-height:1.8;padding:10px;background:#f7f8fa;border-radius:8px}
.vehicle{border-bottom:1px solid #eceef0;padding:10px 0;font-size:13px;line-height:1.7}
.vehicle label{display:flex;align-items:center;gap:8px;cursor:pointer;font-weight:650}
.swatch{width:12px;height:12px;border-radius:100%;display:inline-block}
.seq{padding-left:22px;word-break:break-word;color:#4b5563}small{color:#667085}
#canvasWrap{position:relative;min-width:0;background:#f8fafb}canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}canvas.dragging{cursor:grabbing}
#controls{position:absolute;right:16px;top:16px;display:flex;flex-direction:column;gap:7px}
#controls button{background:white;border:1px solid #cbd5e1;border-radius:7px;min-width:37px;height:35px;box-shadow:0 2px 6px #0002;cursor:pointer;font-size:18px}
#tip{position:absolute;display:none;pointer-events:none;background:#1f2937f2;color:white;padding:9px 12px;font-size:12px;border-radius:6px;max-width:280px;line-height:1.6}
#footer{position:absolute;bottom:12px;left:12px;background:#ffffffed;border:1px solid #ddd;padding:8px 12px;border-radius:6px;font-size:12px;color:#4b5563}
.warn{font-size:12px;color:#92400e;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:10px;margin-top:12px;line-height:1.6}
@media(max-width:750px){main{grid-template-columns:1fr;grid-template-rows:300px 1fr}aside{border-right:0;border-bottom:1px solid #ddd}header{height:auto;min-height:68px}main{height:calc(100vh - 68px)}}
</style></head><body>
<header><div><b>Chicago TD-CVRP30 · 5车配送路线</b><div><small>完整OSM路网底图｜历史交通回放｜离线可用，无需外网地图瓦片</small></div></div><div><small id="meta"></small></div></header>
<main><aside>
<section><label style="font-weight:650;font-size:13px">选择配送任务 / 情景 / 策略</label><select id="variant"></select></section>
<section class="summary" id="summary"></section>
<section><b style="font-size:14px">车辆图层（可单独显示/隐藏）</b><div id="vehicleLegend"></div></section>
<label style="font-size:12px"><input type="checkbox" id="roadsToggle" checked> 显示OSM道路底图</label>
<label style="font-size:12px"><input type="checkbox" id="labelsToggle" checked> 显示客户编号</label>
<div class="warn">说明：彩色线是按该日期历史交通成本重新计算的道路级最短路径，客户访问顺序来自已保存的VRP方案。不是GPS实测行驶轨迹；未观测道路使用静态估计成本。</div>
</aside><div id="canvasWrap"><canvas id="map"></canvas><div id="controls"><button title="放大" id="zoomIn">＋</button><button title="缩小" id="zoomOut">－</button><button title="复位" id="reset">⌂</button></div><div id="tip"></div><div id="footer">鼠标滚轮：缩放　|　按住拖动：平移　|　悬停客户点：查看信息</div></div></main>
<script>
const DATA=__PAYLOAD__;
const canvas=document.querySelector('#map'),ctx=canvas.getContext('2d');
const host=document.querySelector('#canvasWrap'), dropdown=document.querySelector('#variant');
const selected={i:0,vehicles:[],roads:true,labels:true}, view={cx:0,cy:0,scale:1};
const bb=DATA.bounds;let W=1,H=1,DPR=1,mouseDown=false,lastMouse=null,hoverMarker=null;
DATA.variants.forEach((v,i)=>{let op=document.createElement('option');op.value=i;op.textContent=v.key;dropdown.appendChild(op)});
function reset(){view.cx=(bb[0]+bb[2])/2;view.cy=(bb[1]+bb[3])/2;view.scale=Math.min(W/(bb[2]-bb[0]),H/(bb[3]-bb[1]))*.91;draw()}
function xy(p){return [(p[0]-view.cx)*view.scale+W/2,(p[1]-view.cy)*view.scale+H/2]}
function drawLine(line){if(line.length<2)return;let first=xy(line[0]);ctx.beginPath();ctx.moveTo(first[0],first[1]);for(let i=1;i<line.length;i++){let p=xy(line[i]);ctx.lineTo(p[0],p[1])}ctx.stroke()}
function drawArrow(line,color){if(line.length<2)return;let a=xy(line[Math.max(0,Math.floor(line.length/2)-1)]),b=xy(line[Math.min(line.length-1,Math.floor(line.length/2))]);let dx=b[0]-a[0],dy=b[1]-a[1];let norm=Math.hypot(dx,dy);if(norm<1){a=xy(line[0]);b=xy(line[line.length-1]);dx=b[0]-a[0];dy=b[1]-a[1];norm=Math.hypot(dx,dy)}if(norm<1)return;let x=b[0],y=b[1],u=dx/norm,v=dy/norm;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x+8*u,y+8*v);ctx.lineTo(x-6*u-5*v,y-6*v+5*u);ctx.lineTo(x-6*u+5*v,y-6*v-5*u);ctx.closePath();ctx.fill()}
function drawPin(p,color,text,size=7){let[x,y]=xy(p);ctx.beginPath();ctx.arc(x,y,size,0,2*Math.PI);ctx.fillStyle='white';ctx.fill();ctx.lineWidth=2.5;ctx.strokeStyle=color;ctx.stroke();ctx.fillStyle=color;ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='bold 10px system-ui';if(text!==null)ctx.fillText(String(text),x,y+0.6)}
function draw(){ctx.clearRect(0,0,W,H);ctx.fillStyle='#f9fafb';ctx.fillRect(0,0,W,H);
if(selected.roads){ctx.strokeStyle='#cbd5db';ctx.lineWidth=1;ctx.globalAlpha=.75;for(const l of DATA.roads)drawLine(l);ctx.globalAlpha=1}
const variant=DATA.variants[selected.i];hoverMarker=[];
variant.vehicles.forEach((v,i)=>{if(!selected.vehicles[i])return;ctx.strokeStyle=v.color;ctx.lineWidth=3.7;ctx.lineCap='round';ctx.lineJoin='round';ctx.globalAlpha=.87;for(const leg of v.legs){drawLine(leg.pts);drawArrow(leg.pts,v.color)}ctx.globalAlpha=1});
variant.vehicles.forEach((v,i)=>{if(!selected.vehicles[i])return;for(const stop of v.stops){drawPin(stop.pt,v.color,selected.labels?stop.id:null,selected.labels?10:6);const p=xy(stop.pt);hoverMarker.push({x:p[0],y:p[1],v:v,stop:stop})}});
let dp=variant.depot.pt;drawPin(dp,'#111827','D',13);
}
function resize(){let r=host.getBoundingClientRect();W=Math.max(100,r.width);H=Math.max(100,r.height);DPR=window.devicePixelRatio||1;canvas.width=Math.floor(W*DPR);canvas.height=Math.floor(H*DPR);ctx.setTransform(DPR,0,0,DPR,0,0);reset()}
function showVariant(i){selected.i=i;const v=DATA.variants[i];selected.vehicles=v.vehicles.map(()=>true);
let total=v.road_travel_sec.toFixed(1),h=(v.road_travel_sec/60).toFixed(1), e=(v.total_elapsed_sec/60).toFixed(1);
let other=DATA.variants.find(w=>w.task===v.task && w.scenario===v.scenario && w.planner!==(v.planner));
let comparison=other?'<br/>同情景静态/动态差异：'+(100*( (v.planner==='oracle'?other.road_travel_sec:v.road_travel_sec) - (v.planner==='oracle'?v.road_travel_sec:other.road_travel_sec) )/(v.planner==='oracle'?other.road_travel_sec:v.road_travel_sec)).toFixed(2)+'% (正数表示动态更快)':'';
document.querySelector('#summary').innerHTML='<b>'+v.scenario+' · '+v.planner.toUpperCase()+'</b><br/>客户：'+v.customer_count+'个　车辆：'+v.vehicles_count+'辆<br/>行驶总时间：<b>'+total+'秒 ('+h+'分钟)</b><br/>含服务的车时合计：'+e+'分钟<br/>道路观测进入比例：'+(v.observed_share*100).toFixed(1)+'%'+comparison;
let html='';for(let x=0;x<v.vehicles.length;x++){let car=v.vehicles[x];html+='<div class="vehicle"><label><input type="checkbox" data-vehicle="'+x+'" checked><span class="swatch" style="background:'+car.color+'"></span>车辆'+car.vehicle+'　载重'+car.load+' / '+v.vehicle_capacity+'</label><div class="seq">'+car.customers.join(' → ')+' → 仓库</div><small>行驶 '+(car.road_travel_sec/60).toFixed(1)+' 分钟</small></div>'}
document.querySelector('#vehicleLegend').innerHTML=html;document.querySelectorAll('[data-vehicle]').forEach(a=>a.addEventListener('change',e=>{selected.vehicles[+e.target.dataset.vehicle]=e.target.checked;draw()}));draw();}
dropdown.addEventListener('change',()=>showVariant(+dropdown.value));
document.querySelector('#roadsToggle').addEventListener('change',e=>{selected.roads=e.target.checked;draw()});
document.querySelector('#labelsToggle').addEventListener('change',e=>{selected.labels=e.target.checked;draw()});
function zoomAt(px,py,factor){let mx=(px-W/2)/view.scale+view.cx,my=(py-H/2)/view.scale+view.cy;view.scale*=factor;view.cx=mx-(px-W/2)/view.scale;view.cy=my-(py-H/2)/view.scale;draw()}
canvas.addEventListener('wheel',e=>{e.preventDefault();let r=canvas.getBoundingClientRect();zoomAt(e.clientX-r.left,e.clientY-r.top,Math.exp(-Math.sign(e.deltaY)*.16))},{passive:false});
canvas.addEventListener('pointerdown',e=>{mouseDown=true;lastMouse=[e.clientX,e.clientY];canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(mouseDown){let dx=e.clientX-lastMouse[0],dy=e.clientY-lastMouse[1];view.cx-=dx/view.scale;view.cy-=dy/view.scale;lastMouse=[e.clientX,e.clientY];draw();return}
let r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let closest=null,dist=13;for(const p of hoverMarker){let d=Math.hypot(x-p.x,y-p.y);if(d<dist){closest=p;dist=d}}let tip=document.querySelector('#tip');if(closest){tip.style.display='block';tip.style.left=Math.min(W-230,x+16)+'px';tip.style.top=Math.max(8,y-50)+'px';let v=closest.v,s=closest.stop;tip.textContent='车辆'+v.vehicle+' · 客户'+s.id+'（第'+s.visit_order+'站）· 需求'+s.demand+' · 到达'+s.arrival}else tip.style.display='none'});
canvas.addEventListener('pointerup',e=>{mouseDown=false;canvas.classList.remove('dragging')});canvas.addEventListener('pointercancel',()=>{mouseDown=false;canvas.classList.remove('dragging')});
document.querySelector('#zoomIn').addEventListener('click',()=>zoomAt(W/2,H/2,1.35));
document.querySelector('#zoomOut').addEventListener('click',()=>zoomAt(W/2,H/2,1/1.35));
document.querySelector('#reset').addEventListener('click',reset);
document.querySelector('#meta').textContent=DATA.meta.graph_nodes+' 路网节点 / '+DATA.meta.graph_edges+' 有向边';
window.addEventListener('resize',resize);resize();showVariant(0);
</script></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solutions',required=True,type=Path)
    p.add_argument('--task-dir',required=True,type=Path)
    p.add_argument('--graphml',required=True,type=Path)
    p.add_argument('--cost-dir',required=True,type=Path)
    p.add_argument('--output',default='runs/chicago_tdcvrp30/maps/vehicle_routes.html',type=Path)
    p.add_argument('--task-id',default='task_0000')
    p.add_argument('--scenario',default='')
    p.add_argument('--planner',choices=['','static','oracle'],default='')
    p.add_argument('--max-views',type=int,default=10)
    p.add_argument('--include-road-observations',action='store_true',help='Compute observed-edge statistics per leg (slower)')
    args=p.parse_args()
    content=build_data(args)
    # Never require Internet: inline canvas + all route and road geometry.
    payload=json.dumps(content,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html=HTML.replace('__PAYLOAD__',payload)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html,encoding='utf-8')
    print('SAVED:',args.output,'views:',len(content['variants']),'bytes:',args.output.stat().st_size)
    print('GraphML edges without WKT geometry (drawn using node-to-node segment):',content['meta']['geometry_fallback_edges'])
    print('Open the HTML locally in a web browser. It uses no external tiles, scripts or network requests.')

if __name__=='__main__':main()
