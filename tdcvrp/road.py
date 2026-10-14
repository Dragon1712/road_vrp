"""Read exact edge_index/edge_costs artifacts from chicago_td_cost_starter.

Historic traffic = EX-POST cost labels, never a prediction available at dispatch.
Continuous-time edge travel uses linear interpolation of sampled ARRIVAL functions.
"""
from __future__ import annotations
import heapq
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx


class RoadCosts:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.edges = pd.read_csv(self.folder / 'edge_index.csv', dtype={'osm_u':str, 'osm_v':str, 'osm_key':str})
        self.times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(self.folder/'time_index.csv')['time_local']))
        with np.load(self.folder/'edge_costs.npz', allow_pickle=False) as z:
            self.fifo = z['travel_sec_fifo'].astype(np.float64)
            self.raw = z['travel_sec_raw'].astype(np.float64)
            self.observed = z['chicago_observed_mask'].astype(bool)
            self.mapped = z['is_mapped_edge'].astype(bool)
            self.dt = float(z['slot_seconds'])
        self.static = self.edges['static_travel_sec'].to_numpy(np.float64)
        assert len(self.edges) == self.fifo.shape[1] == len(self.static)
        assert len(self.times) == self.fifo.shape[0]
        assert self.observed.shape == self.fifo.shape
        assert np.array_equal(self.edges.edge_idx.to_numpy(int), np.arange(len(self.edges))), 'edge_idx must be consecutive and in row order'
        self.adj = {}
        self.H = nx.DiGraph()
        for e in self.edges.itertuples(index=False):
            self.adj.setdefault(str(e.osm_u), []).append((str(e.osm_v), int(e.edge_idx)))
            self.H.add_edge(str(e.osm_u), str(e.osm_v))
        self.nodes = set(self.H.nodes)
        sccs = nx.strongly_connected_components(self.H)
        self.largest_scc = sorted(max(sccs, key=len))
        if not np.isfinite(self.fifo).all() or (self.fifo <= 0).any():
            raise ValueError('Invalid FIFO costs')

    def to_seconds(self, t):
        secs = (pd.Timestamp(t) - self.times[0]).total_seconds()
        if not (0 <= secs < len(self.times)*self.dt):
            raise ValueError('Departure outside traffic interval')
        return float(secs)

    def timestamp(self, secs):
        return str(self.times[0] + pd.Timedelta(seconds=float(secs)))

    def cost(self, idx, t):
        """Travel time on directed edge idx at absolute-relative seconds t."""
        if not (0 <= t < self.fifo.shape[0]*self.dt):
            raise ValueError(f'Road traversal t={t:.1f} outside measured traffic range')
        j = int(t // self.dt)
        if j == self.fifo.shape[0]-1:
            return float(self.fifo[j, idx])
        alpha = (t-j*self.dt)/self.dt
        arrival0 = j*self.dt + self.fifo[j, idx]
        arrival1 = (j+1)*self.dt + self.fifo[j+1, idx]
        return max(0., (1-alpha)*arrival0 + alpha*arrival1 - t)

    def shortest(self, source, target, depart, mode='dynamic', path=False):
        """FIFO TD-Dijkstra, or static shortest-path. Path is list of edge_idx."""
        source, target = str(source), str(target)
        if source == target:
            return 0.0, []
        d = {source:float(depart)}
        prev = {} if path else None
        h = [(float(depart), source)]
        while h:
            t,u = heapq.heappop(h)
            if t > d[u]+1e-8:
                continue
            if u == target:
                if not path:
                    return t-depart, []
                edges = []
                while u != source:
                    u0,edge = prev[u]
                    edges.append(edge)
                    u = u0
                return t-depart, edges[::-1]
            for v,eidx in self.adj.get(u, ()):
                cost = self.cost(eidx,t) if mode=='dynamic' else self.static[eidx]
                arr=t+cost
                if arr < d.get(v, float('inf'))-1e-9:
                    d[v]=arr
                    if path: prev[v]=(u,eidx)
                    heapq.heappush(h,(arr,v))
        return float('inf'), []

    def one_to_many(self, source, targets, depart, mode='dynamic'):
        """One Dijkstra finds all target times: key to efficient OD precomputation."""
        source = str(source)
        remaining=set(map(str,targets))
        d={source:float(depart)}
        h=[(float(depart),source)]
        found={}
        while h and remaining:
            t,u=heapq.heappop(h)
            if t > d[u]+1e-8:
                continue
            if u in remaining:
                found[u]=t-depart
                remaining.remove(u)
            for v,eidx in self.adj.get(u,()):
                cost=self.cost(eidx,t) if mode=='dynamic' else self.static[eidx]
                arr=t+cost
                if arr < d.get(v,float('inf'))-1e-9:
                    d[v]=arr
                    heapq.heappush(h,(arr,v))
        return found

    def evaluate_fixed_edge_path(self, edges, depart):
        """Use for regression: static chosen ROAD path evaluated on historical traffic."""
        t=float(depart)
        for e in edges: t += self.cost(e,t)
        return t-depart

    def measure_path(self, edges, depart):
        """How many traversed edges have source-data observation at their entry slot?"""
        t=float(depart)
        count=0
        for e in edges:
            slot=min(int(t//self.dt),len(self.times)-1)
            count+=int(self.observed[slot,e])
            t+=self.cost(e,t)
        return count
