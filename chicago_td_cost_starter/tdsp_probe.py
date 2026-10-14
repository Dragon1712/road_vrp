#!/usr/bin/env python3
"""Offline time-dependent Dijkstra smoke test; never uses future traffic as a deployable predictor.

Uses FIFO arrival-time knots and linearly interpolates the arrival function.
Sample OD pairs from the largest strongly connected OSM component and compare:
  (a) static-shortest path evaluated ex post on same time-dependent traffic,
  (b) oracle time-dependent-shortest path evaluated on that traffic.

The latter is a hindsight oracle upper bound (lower bound on travel time), not a fair
real-time predictor, and does not constitute a complete TD-CVRP solver.
"""
import argparse
import heapq
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def cost_at(costs, edge_id, t, dt):
    """Arrival linear interpolation: preserves FIFO for every interior departure."""
    if t < 0 or t >= len(costs) * dt:
        raise ValueError("Traversal time outside 7-day measured window; choose earlier departure")
    i = min(int(t // dt), len(costs)-1)
    if i == len(costs)-1:
        return float(costs[i, edge_id])
    alpha = (t-i*dt)/dt
    a0 = i*dt + float(costs[i, edge_id])
    a1 = (i+1)*dt + float(costs[i+1, edge_id])
    a = (1-alpha)*a0 + alpha*a1
    return max(0., a-t)


def shortest_path(adj, origin, target, depart_sec, cost_func):
    d = {origin: float(depart_sec)}
    back = {}
    h = [(float(depart_sec), origin)]
    while h:
        t, node = heapq.heappop(h)
        if t > d[node] + 1e-8:
            continue
        if node == target:
            path = []
            while node != origin:
                prev, edge_idx = back[node]
                path.append(edge_idx)
                node = prev
            return t-depart_sec, list(reversed(path))
        for nxt, idx in adj.get(node, ()):
            w = cost_func(idx, t)
            arrive = t+w
            if arrive < d.get(nxt, float("inf")) - 1e-9:
                d[nxt] = arrive
                back[nxt] = (node, idx)
                heapq.heappush(h, (arrive, nxt))
    return float("inf"), []


def evaluate_route(path, depart_sec, cost_func):
    t = float(depart_sec)
    for e in path:
        t += cost_func(e, t)
    return t-depart_sec


def probe(args):
    base = Path(args.cost_dir)
    edge = pd.read_csv(base/"edge_index.csv", dtype={"osm_u":str, "osm_v":str})
    time = pd.read_csv(base/"time_index.csv")
    times = pd.DatetimeIndex(pd.to_datetime(time["time_local"]))
    depart = pd.Timestamp(args.departure)
    depart_sec = (depart-times[0]).total_seconds()
    z = np.load(base/"edge_costs.npz")
    raw = z["travel_sec_raw"]
    fifo = z["travel_sec_fifo"]
    observed = z["chicago_observed_mask"]
    dt = float(z["slot_seconds"])
    if not (0 <= depart_sec < len(times)*dt):
        raise ValueError(f"Departure {depart} outside {times[0]}--{times[-1]+pd.Timedelta(seconds=dt)}")
    if fifo.shape != (len(times), len(edge)) or raw.shape != fifo.shape:
        raise ValueError("Shape misalignment; files may originate from different runs")
    f_cost = lambda i,t: cost_at(fifo, i, t, dt)
    base_cost = edge.static_travel_sec.to_numpy(float)
    s_cost = lambda i,t: float(base_cost[i])

    adj = {}
    H = nx.DiGraph()
    for row in edge.itertuples(index=False):
        adj.setdefault(row.osm_u, []).append((row.osm_v, int(row.edge_idx)))
        H.add_edge(row.osm_u, row.osm_v)
    largest = max(nx.strongly_connected_components(H), key=len)
    nodes = np.array(sorted(largest), dtype=str)
    if len(nodes) < 2:
        raise ValueError("No strongly connected subnetwork of at least two nodes")
    rng = np.random.default_rng(args.seed)
    output = []
    skipped = 0
    for trial in range(args.num_pairs):
        a,b = rng.choice(nodes, 2, replace=False)
        try:
            static_sec, static_edges = shortest_path(adj, a,b,depart_sec,s_cost)
            oracle_sec, dynamic_edges = shortest_path(adj,a,b,depart_sec,f_cost)
            ex_post_static_sec = evaluate_route(static_edges,depart_sec,f_cost)
            # Exact oracle property within floating tolerances.
            if ex_post_static_sec + 1e-5 < oracle_sec:
                raise AssertionError("Ex-post static beats time-dependent Dijkstra: interpolation/FIFO bug")
            t = depart_sec
            monitored_traversals = 0
            for idx in dynamic_edges:
                cell = min(int(t // dt),len(times)-1)
                monitored_traversals += int(observed[cell,idx])
                t += f_cost(idx,t)
            output.append({
                "origin":a,"destination":b,
                "static_planned_sec":round(static_sec,3),
                "static_route_realized_sec":round(ex_post_static_sec,3),
                "oracle_td_route_sec":round(oracle_sec,3),
                "oracle_improvement_sec":round(max(ex_post_static_sec-oracle_sec,0),3),
                "oracle_td_edges":len(dynamic_edges),
                "oracle_td_observed_edges_at_entry_slot":monitored_traversals,
                "oracle_td_observed_edge_share_at_entry_slot":round(monitored_traversals/max(len(dynamic_edges),1),4)
            })
        except ValueError as ex:
            skipped += 1
            print("Skip pair crossing time-window boundary:",str(ex))
    res = pd.DataFrame(output)
    if not len(res):
        raise ValueError("No OD pairs successfully evaluated. Try an earlier departure")
    args.output_csv.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(args.output_csv,index=False)
    summary = {
        "departure_local_naive":str(depart),
        "requested_pairs": args.num_pairs,
        "successful_pairs":len(res),
        "skipped_pairs":skipped,
        "largest_scc_nodes":len(largest),
        "mean_oracle_improvement_sec":round(float(res.oracle_improvement_sec.mean()),3),
        "fraction_pairs_oracle_route_encounters_measured_edges":float((res.oracle_td_observed_edges_at_entry_slot > 0).mean()),
        "mean_oracle_path_edge_observation_share":float(res.oracle_td_observed_edge_share_at_entry_slot.mean()),
        "WARNING":"Oracle uses historical ex-post traffic: pilot analysis only. Traffic data is not known in advance in real applications."
    }
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("Saved OD probes:",args.output_csv)
    return res


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cost-dir",type=Path,required=True)
    p.add_argument("--departure",default="2025-06-05 08:00:00")
    p.add_argument("--num-pairs",type=int,default=30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output-csv",type=Path,default=Path("tdsp_od_probe.csv"))
    probe(p.parse_args())


if __name__ == "__main__":
    main()
