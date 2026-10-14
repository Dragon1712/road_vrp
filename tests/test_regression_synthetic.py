"""Generate synthetic 100-OD reference CSVs then confirm regression flags no drift."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .test_synthetic import make_synthetic
from tdcvrp.road import RoadCosts
from tdcvrp.cli import do_regress
from types import SimpleNamespace
import tempfile


def run():
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp)
        road=RoadCosts(make_synthetic(base/'road'))
        rng=np.random.default_rng(2026)
        nodes=road.largest_scc
        pairs=[rng.choice(nodes,2,replace=False) for _ in range(100)]
        for tag,clock in [('morning','2025-06-05 08:00:00'),('evening','2025-06-05 17:00:00')]:
            at=road.to_seconds(clock)
            rows=[]
            for a,b in pairs:
                stat,st_path=road.shortest(a,b,at,mode='static',path=True)
                oracle,dyn_path=road.shortest(a,b,at,mode='dynamic',path=True)
                static_real=road.evaluate_fixed_edge_path(st_path,at)
                obs=road.measure_path(dyn_path,at)
                rows.append({'origin':a,'destination':b,
                             'static_planned_sec':round(stat,3),
                             'static_route_realized_sec':round(static_real,3),
                             'oracle_td_route_sec':round(oracle,3),
                             'oracle_improvement_sec':round(max(static_real-oracle,0),3),
                             'oracle_td_edges':len(dyn_path),
                             'oracle_td_observed_edges_at_entry_slot':obs,
                             'oracle_td_observed_edge_share_at_entry_slot':round(obs/max(len(dyn_path),1),4)})
            pd.DataFrame(rows).to_csv(base/(tag+'.csv'),index=False)
        args=SimpleNamespace(cost_dir=base/'road',morning=base/'morning.csv',
                             evening=base/'evening.csv',output_dir=base/'regression',tolerance_sec=.01)
        do_regress(args)
        report=json.loads((base/'regression'/'tdsp_regression_report.json').read_text())
        assert report['pairs_checked']==200
        assert report['time_cost_all_pass']
        # Intentionally corrupt the reference: regression must fail loudly.
        m=pd.read_csv(base/'morning.csv')
        m.loc[0,'oracle_td_route_sec']+=10
        m.to_csv(base/'morning.csv',index=False)
        try:
            do_regress(args)
        except SystemExit:
            pass
        else:raise AssertionError('Regression failed to catch intentional data corruption')
        print('200-OD SYNTHETIC REGRESSION TEST PASS; tampering detected')

if __name__=='__main__': run()
