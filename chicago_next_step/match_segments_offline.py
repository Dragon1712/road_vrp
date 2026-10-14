"""Offline PRELIMINARY matching of Chicago monitored corridors to directed OSM paths.

Inputs: segment_coverage.csv (user's QC file) and a locally downloaded OSMnx GraphML.
This is an approximate endpoint+street-name+length matching; manually inspect the
matches before using them as ground-truth dynamic road costs.

Only requires pandas/numpy/networkx; no internet, geopandas or OSMnx at runtime.
"""
import argparse
from pathlib import Path
import math
import re
import ast
import numpy as np
import pandas as pd
import networkx as nx

REMOVALS={'n','s','e','w','north','south','east','west','street','st','avenue','ave',
          'road','rd','drive','dr','boulevard','blvd','parkway','pkwy','expressway','expy',
          'highway','hwy','pl','place'}

def road_key(s):
    if s is None: return ''
    if isinstance(s,(list,tuple)):s=' '.join(str(x) for x in s)
    val=str(s).lower()
    val=re.sub(r'[^a-z0-9]+',' ',val)
    return ' '.join(x for x in val.split() if x not in REMOVALS)

def similar_street(monitored,road):
    x=road_key(monitored);y=road_key(road)
    if not x or not y: return False
    # Work for e.g. "Madison" and "West Madison Street", "79th" and "E 79th St".
    return x==y or (' '+x+' ') in (' '+y+' ') or (' '+y+' ') in (' '+x+' ')

def edges_dict(raw, multi):
    if multi: return list(raw.items())
    return [(0,raw)]

def select_edge(raw,multi,street,weight_mismatch):
    candidates=[]
    for key,attr in edges_dict(raw,multi):
        try:length=float(attr.get('length','nan'))
        except (TypeError,ValueError):continue
        if not math.isfinite(length) or length<=0:continue
        name_ok=similar_street(street,attr.get('name',''))
        weighted=length*(1 if name_ok else weight_mismatch)
        candidates.append((weighted,length,key,name_ok))
    return min(candidates) if candidates else (math.inf,math.inf,None,False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--segments',required=True,help='path to segment_coverage.csv')
    ap.add_argument('--graphml',required=True,help='downloaded locally with download_osm_locally.py')
    ap.add_argument('--output-dir',default='datasets/processed/chicago/osm_match')
    ap.add_argument('--snap-max-m',type=float,default=175)
    ap.add_argument('--length-ratio-min',type=float,default=.65)
    ap.add_argument('--length-ratio-max',type=float,default=1.85)
    ap.add_argument('--street-fraction-min',type=float,default=.50)
    ap.add_argument('--mismatch-penalty',type=float,default=5.0)
    ap.add_argument('--min-coverage',type=float,default=0.0,help='filter output ONLY, NOT graph topology')
    a=ap.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    seg=pd.read_csv(a.segments)
    required=['segment_id','street','valid_speed_ratio','length_m','start_latitude','start_longitude','end_latitude','end_longitude']
    missing=[x for x in required if x not in seg.columns]
    if missing:raise ValueError('Missing columns '+str(missing))
    seg=seg[seg.valid_speed_ratio>=a.min_coverage].copy()
    G=nx.read_graphml(a.graphml)
    if not G.is_directed():raise ValueError('Expected an OSMnx directed graphml. Got undirected graph')
    multi=G.is_multigraph()
    nodelist=[];coords=[]
    for nid,attr in G.nodes(data=True):
        try:lat=float(attr['y']);lon=float(attr['x'])
        except (KeyError,ValueError,TypeError):continue
        nodelist.append(nid);coords.append((lat,lon))
    coords=np.asarray(coords,dtype=float)
    if len(coords)==0:raise ValueError('Graph nodes lack OSMnx x,y coordinates')
    ref_lat=float(np.mean(coords[:,0]))
    xsc=111320*math.cos(math.radians(ref_lat));ysc=110950
    xy=coords * np.array([ysc,xsc])

    def nearest(lat,lon,k=3):
        pt=np.array([lat*ysc,lon*xsc]);d=np.sum((xy-pt)**2,axis=1)
        indices=np.argpartition(d,min(k,len(d))-1)[:k]
        return sorted([(nodelist[i],float(math.sqrt(d[i]))) for i in indices],key=lambda v:v[1])

    matched=[];edge_matches=[]
    print('Graph:',len(G.nodes),'nodes',len(G.edges),'edges; monitored segments',len(seg))
    for ix,row in seg.iterrows():
        sid=int(row.segment_id)
        length=float(row.length_m)
        starts=nearest(float(row.start_latitude),float(row.start_longitude))
        ends=nearest(float(row.end_latitude),float(row.end_longitude))
        starts=[z for z in starts if z[1]<=a.snap_max_m]
        ends=[z for z in ends if z[1]<=a.snap_max_m]
        baseline=dict(segment_id=sid,street=row.street,valid_speed_ratio=float(row.valid_speed_ratio),
                      observed_length_m=length,osm_route_length_m=np.nan,length_ratio=np.nan,
                      snap_start_m=np.nan,snap_end_m=np.nan,street_fraction=np.nan,
                      osm_start_node='',osm_end_node='',match_status='no_close_endpoint',path_edge_count=0)
        if not starts or not ends:
            matched.append(baseline);continue
        alternatives=sorted([(s,sd,t,td) for s,sd in starts for t,td in ends if s!=t],
                            key=lambda x:x[1]+x[3])
        if not alternatives:
            baseline['match_status']='same_endpoint';matched.append(baseline);continue
        choice=None
        for st,sd,en,ed in alternatives[:5]:
            def cost_fn(u,v,edata):
                return select_edge(edata,multi,row.street,a.mismatch_penalty)[0]
            try:
                route=nx.shortest_path(G,st,en,weight=cost_fn)
            except (nx.NetworkXNoPath,nx.NodeNotFound):continue
            length_route=0.;name_route=0.;selected=[]
            for u,v in zip(route,route[1:]):
                _,dist,key,name_ok=select_edge(G.get_edge_data(u,v),multi,row.street,a.mismatch_penalty)
                if not math.isfinite(dist):break
                length_route+=dist
                if name_ok:name_route+=dist
                selected.append((u,v,key,dist,name_ok))
            if not selected or length_route<=0:continue
            ratio=length_route/length if length>0 else math.nan
            fraction=name_route/length_route
            good=(a.length_ratio_min<=ratio<=a.length_ratio_max and fraction>=a.street_fraction_min)
            score=abs(math.log(ratio))+3*(1-fraction)+.001*(sd+ed)
            candidate=(0 if good else 1,score,st,sd,en,ed,length_route,ratio,fraction,selected)
            if choice is None or candidate[:2]<choice[:2]:choice=candidate
            if good:break
        if choice is None:
            baseline['match_status']='no_directed_path';matched.append(baseline);continue
        _,score,st,sd,en,ed,dist,ratio,fraction,selected=choice
        status='accepted' if (a.length_ratio_min<=ratio<=a.length_ratio_max and fraction>=a.street_fraction_min) else 'poor_match_review'
        baseline.update(osm_route_length_m=round(dist,2),length_ratio=round(ratio,4),
                        snap_start_m=round(sd,2),snap_end_m=round(ed,2),street_fraction=round(fraction,4),
                        osm_start_node=str(st),osm_end_node=str(en),match_status=status,path_edge_count=len(selected))
        matched.append(baseline)
        if status=='accepted':
            for u,v,key,d,name_ok in selected:
                edge_matches.append(dict(segment_id=sid,osm_u=str(u),osm_v=str(v),osm_key=str(key),
                    osm_length_m=round(d,2),osm_street_match=name_ok,
                    segment_valid_speed_ratio=float(row.valid_speed_ratio)))
    m=pd.DataFrame(matched).sort_values(['match_status','segment_id'])
    e=pd.DataFrame(edge_matches,columns=['segment_id','osm_u','osm_v','osm_key','osm_length_m','osm_street_match','segment_valid_speed_ratio'])
    m.to_csv(out/'segment_matches.csv',index=False)
    e.to_csv(out/'accepted_osm_edge_mapping.csv',index=False)
    accepted=m.match_status.eq('accepted')
    eligible=m.valid_speed_ratio.ge(.5)
    print('\nMatch status:\n',m.match_status.value_counts(dropna=False).to_string())
    print('Accepted / all:',int(accepted.sum()),'/',len(m))
    print('Accepted among >=50% coverage:',int((accepted&eligible).sum()),'/',int(eligible.sum()))
    print('Unique OSM directed edges linked:',len(e[['osm_u','osm_v','osm_key']].drop_duplicates()))
    print('Files saved to',out)
    print('\nIMPORTANT: This is a preliminary candidate match. Review against a map, validate directions,')
    print('street names, endpoint snaps and route length ratios before transferring dynamic speeds to OSM edges.')

if __name__=='__main__':main()
