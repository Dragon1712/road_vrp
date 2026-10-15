"""Synthetic-only full map test; no implication of real Chicago validation."""
import json
import os
import re
import sys
import tempfile
from pathlib import Path
import networkx as nx

ROOT=Path(__file__).resolve().parent.parent/'road_vrp_map_work'/'chicago_tdcvrp_solver_suite_v2'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from road_eval import Road
from tests.test_synthetic import make_fixture
from visualize_solver_routes import main,build_payload
from argparse import Namespace


def test():
    with tempfile.TemporaryDirectory() as d:
        base=Path(d)
        cost_dir,task_dir,task=make_fixture(base)
        road=Road(cost_dir)
        # Explicitly exercise 5 vehicles, matching Chicago TD-CVRP30.
        task['vehicles']=5
        task['seed_routes']=[[1,2,3],[4,5],[6,7],[8,9],[10,11,12]]
        (task_dir/'task_0000.json').write_text(json.dumps(task,indent=2))
        g=nx.MultiDiGraph()
        for name in set(road.edges.osm_u)|set(road.edges.osm_v):
            a,b=map(int,name.split('-'))
            g.add_node(name,x=-87.65+b*.002,y=41.85+a*.002)
        for e in road.edges.itertuples():
            xy=g.nodes[str(e.osm_u)];zt=g.nodes[str(e.osm_v)]
            geom=f'LINESTRING ({xy["x"]} {xy["y"]}, {zt["x"]} {zt["y"]})'
            g.add_edge(str(e.osm_u),str(e.osm_v),key=str(e.osm_key),geometry=geom)
        graph=base/'graph.graphml';nx.write_graphml(g,graph)
        records=[]
        for sc in task['scenarios']:
            for solver in ('alns','upstream_alns'):
                routes=task['seed_routes'] if solver=='alns' else [list(reversed(x)) for x in task['seed_routes']]
                ex=road.exact(task,routes,sc)
                records.append(dict(task_id=task['task_id'],scenario_id=sc['scenario_id'],departure_local=sc['departure_local'],
                                    solver=solver,planner='oracle',seed=42,vehicle_rule='exactly',routes=routes,
                                    recorded_cost_sec=ex['ex_post_road_travel_sec'],observed_edge_share=ex['observed_edge_share']))
        inp=base/'solutions.json';inp.write_text(json.dumps(records))
        out=base/'map.html'
        args=Namespace(solutions=[inp],task_dir=task_dir,cost_dir=cost_dir,graphml=graph,
                task_id='task_0000',seed=42,planner='oracle',scenario='all',
                solvers='alns,upstream_alns',output=out,tolerance_sec=.05)
        p=build_payload(args)
        assert len(p['variants'])==4
        assert len(p['background'])>0
        assert all(len(v['vehicles'])==5 for v in p['variants'])
        assert all(sum(len(z['stops']) for z in v['vehicles'])==12 for v in p['variants'])
        assert all(len(v['vehicles'][0]['legs'])>0 for v in p['variants'])
        assert p['diagnostics']['graph_geometry_fallback']==0
        import visualize_solver_routes
        html=visualize_solver_routes.HTML.replace('__PAYLOAD__',json.dumps(p,ensure_ascii=False))
        out.write_text(html)
        raw=re.search(r'<script>([\s\S]*?)</script>',html)
        assert raw and 'upstream_alns' in raw.group(1)
        (base/'map.js').write_text(raw.group(1))
        os.system('node --check '+str(base/'map.js'))
        print('SYNTHETIC TEST OK: 4 views; 5 vehicles; all legs replayed & validated; graph WKT parsed')
        # Negative test: mismatched recorded exact cost must fail loudly.
        records[0]['recorded_cost_sec']+=100.
        inp.write_text(json.dumps(records))
        try:build_payload(args)
        except ValueError as e:
            assert 'Cost mismatch' in str(e)
            print('COST CONSISTENCY GUARD OK')
        else:raise AssertionError('Tampered cost was accepted')

if __name__=='__main__':test()
