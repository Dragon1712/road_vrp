#!/usr/bin/env python3
"""Convert new suite solutions to original visualize_tdcvrp_routes.py input.

Visualize STATIC/ORACLE solutions of a chosen solver and seed on actual OSM roads.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solutions',required=True)
    p.add_argument('--solver',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    items=json.loads(Path(args.solutions).read_text(encoding='utf-8'))
    out=[]
    for r in items:
        if r['solver']!=args.solver or int(r['seed'])!=args.seed: continue
        x=dict(r)
        x['road_exact_ex_post']={'realized_travel_sec':float(r['recorded_cost_sec']),
                                 'observed_entry_share':float(r.get('observed_edge_share',0))}
        x['search_seed']=r['seed']
        out.append(x)
    if not out:raise ValueError('No solutions matching solver and seed')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Wrote',len(out),'solutions for original offline route-map visualizer:',target)

if __name__=='__main__':main()
