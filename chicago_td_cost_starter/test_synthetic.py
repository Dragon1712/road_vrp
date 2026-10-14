#!/usr/bin/env python3
"""Small fully-offline integration and mathematical consistency test."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import networkx as nx

from build_dynamic_edge_costs import build
from tdsp_probe import probe, cost_at


def test():
    with TemporaryDirectory() as td:
        p = Path(td)
        G=nx.MultiDiGraph()
        for n, x in [("A",-87.64),("B",-87.635),("C",-87.63),("D",-87.625)]:
            G.add_node(n, x=x, y=41.88)
        for u, v, k, dist, road in [
            ("A","B",0,100,"primary"),
            ("B","C",0,200,"primary"),
            ("C","D",0,150,"residential"),
            ("D","A",0,170,"residential"),
            ("A","D",0,200,"service"),
            ("C","A",0,190,"primary")]:
            G.add_edge(u,v,key=k,length=dist,highway=road)
        graphml = p/"graph.graphml"
        nx.write_graphml(G, graphml)
        mapping = pd.DataFrame([
            dict(segment_id=127,osm_u="A",osm_v="B",osm_key="0",osm_length_m=100,mapped_path_length_m=300,path_fraction=1/3),
            dict(segment_id=127,osm_u="B",osm_v="C",osm_key="0",osm_length_m=200,mapped_path_length_m=300,path_fraction=2/3),
            dict(segment_id=129,osm_u="A",osm_v="D",osm_key="0",osm_length_m=200,mapped_path_length_m=200,path_fraction=1.0),
        ])
        mapping.to_csv(p/"mapping.csv",index=False)
        times = pd.date_range("2025-06-02", periods=4, freq="10min")
        pd.DataFrame({"time_local":times.strftime("%Y-%m-%d %H:%M:%S")}).to_csv(p/"time.csv",index=False)
        traffic = []
        for i,t in enumerate(times):
            a=[1200.,60.,np.nan,120.][i]
            b=[80.,np.nan,55.,40.][i]
            for s,v in [(127,a),(129,b)]:
                traffic.append({"slot":t, "segment_id":s,
                               "segment_travel_sec":v,"observed":bool(np.isfinite(v))})
        pd.DataFrame(traffic).to_csv(p/"traffic.csv",index=False)
        output=p/"out"
        report=build(SimpleNamespace(graphml=graphml,mapping=p/"mapping.csv",
                                     traffic_long=p/"traffic.csv",time_index=p/"time.csv",
                                     output_dir=output))
        edges=pd.read_csv(output/"edge_index.csv",dtype={"osm_u":str,"osm_v":str,"osm_key":str})
        z=np.load(output/"edge_costs.npz")
        raw=z["travel_sec_raw"];fifo=z["travel_sec_fifo"]
        mask=z["chicago_observed_mask"]
        idx={tuple(x):i for i,x in enumerate(zip(edges.osm_u,edges.osm_v,edges.osm_key))}
        ab=idx["A","B","0"];bc=idx["B","C","0"];ad=idx["A","D","0"]
        cd=idx["C","D","0"]
        assert len(edges)==6 and mask.shape==(4,6)
        assert np.isclose(raw[0,ab],400) and np.isclose(raw[0,bc],800)
        assert np.isclose(raw[0,ab]+raw[0,bc],1200)
        assert np.isclose(raw[1,ab]+raw[1,bc],60)
        assert mask[0,ab] and mask[0,bc] and not mask[2,ab]
        assert np.isclose(raw[2,ab],edges.loc[ab,"static_travel_sec"])
        assert not mask[:,cd].any()
        assert np.isclose(raw[0,ad],80) and not mask[1,ad]
        assert report["candidate_mapped_directed_edges"]==3
        assert np.all(fifo>=raw-1e-4)
        assert report["fifo_grid_values_adjusted"]>0
        arrival=fifo.astype(float)+np.arange(4)[:,None]*600
        assert np.all(np.diff(arrival,axis=0)>=-1e-3)
        assert cost_at(fifo,bc,300,600)>0
        res=probe(SimpleNamespace(cost_dir=output,departure="2025-06-02 00:00:00",
                                  num_pairs=20,seed=7,output_csv=p/"probe.csv"))
        assert len(res)==20
        assert np.all(res["oracle_improvement_sec"]>=-1e-4)
        assert (p/"probe.csv").exists()
        print("ALL SYNTHETIC TESTS PASS")


if __name__=="__main__":
    test()
