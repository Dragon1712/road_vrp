"""Offline time-dependent road router for Chicago `dynamic_costs` files.

FIFO arrival-function interpolation is consistent with the earlier TDSP implementation.
Historical costs are EX-POST labels, not future information available to a planner.
"""
from __future__ import annotations
import heapq
import numpy as np
import pandas as pd
from pathlib import Path


class Road:
    def __init__(self, cost_dir):
        d = Path(cost_dir)
        self.edges = pd.read_csv(d/'edge_index.csv', dtype={'osm_u':str,'osm_v':str,'osm_key':str})
        self.times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(d/'time_index.csv')['time_local']))
        with np.load(d/'edge_costs.npz',allow_pickle=False) as z:
            self.fifo = np.asarray(z['travel_sec_fifo'],dtype=np.float64)
            self.observed = np.asarray(z['chicago_observed_mask'],dtype=bool)
            self.is_mapped = np.asarray(z['is_mapped_edge'],dtype=bool)
            self.dt = float(z['slot_seconds'])
        self.static = self.edges.static_travel_sec.to_numpy(float)
        if self.fifo.shape != (len(self.times),len(self.edges)) or self.observed.shape!=self.fifo.shape:
            raise ValueError('Traffic arrays do not match edge_index/time_index')
        if not np.array_equal(self.edges.edge_idx.to_numpy(),np.arange(len(self.edges))):
            raise ValueError('edge_idx is not aligned to edge_index row order')
        self.adj={}
        for r in self.edges.itertuples():
            self.adj.setdefault(str(r.osm_u),[]).append((str(r.osm_v),int(r.edge_idx)))
        if not np.isfinite(self.fifo).all() or (self.fifo<=0).any():
            raise ValueError('Invalid time-dependent road travel times')
        self.end_sec=len(self.times)*self.dt

    def to_sec(self,clock):
        sec=(pd.Timestamp(clock)-self.times[0]).total_seconds()
        if not 0 <= sec < self.end_sec:
            raise ValueError(f'{clock} outside traffic series')
        return float(sec)

    def travel(self,e,t):
        if not 0 <= t < self.end_sec:
            raise ValueError(f'Traffic time {t:.3f}s beyond available historic window')
        k=int(t//self.dt)
        if k==len(self.times)-1:
            return float(self.fifo[k,e])
        f=(t-k*self.dt)/self.dt
        arrival=(1-f)*(k*self.dt+self.fifo[k,e])+f*((k+1)*self.dt+self.fifo[k+1,e])
        return max(0.,float(arrival-t))

    def dijkstra(self,source,targets,departure,mode='oracle',keep_paths=False):
        source=str(source)
        needed=set(map(str,targets))
        dist={source:float(departure)}
        heap=[(float(departure),source)]
        prev={} if keep_paths else None
        found={}
        while heap and needed:
            t,u=heapq.heappop(heap)
            if t>dist[u]+1e-8:continue
            if u in needed:
                found[u]=(t-departure,self._extract(source,u,prev) if keep_paths else [])
                needed.remove(u)
            for v,e in self.adj.get(u,[]):
                w=self.static[e] if mode=='static' else self.travel(e,t)
                alt=t+w
                if alt<dist.get(v,float('inf'))-1e-9:
                    dist[v]=alt
                    if keep_paths:prev[v]=(u,e)
                    heapq.heappush(heap,(alt,v))
        return found

    @staticmethod
    def _extract(source,dest,prev):
        edges=[]
        while dest!=source:
            u,e=prev[dest]
            edges.append(e)
            dest=u
        return edges[::-1]

    def route(self,a,b,depart,mode='oracle'):
        x=self.dijkstra(a,[b],depart,mode,True).get(str(b))
        if x is None: raise ValueError(f'Unreachable directed OD {a}->{b}')
        return x

    def largest_scc_nodes(self):
        """Tarjan SCC from NetworkX optional; tasks from earlier package already SCC-safe."""
        import networkx as nx
        graph=nx.DiGraph()
        for r in self.edges.itertuples():graph.add_edge(str(r.osm_u),str(r.osm_v))
        return sorted(max(nx.strongly_connected_components(graph),key=len))

    def exact(self,task,routes,scenario,collect_legs=False):
        from models import verify
        loads=verify(task,routes)
        node_of={int(c['id']):str(c['node']) for c in task['customers']}
        depot=str(task['depot_node']); start=self.to_sec(scenario['departure_local'])
        total=0.; elapsed=0.; nobs=0; nedges=0; leglist=[]
        for vehicle,seq in enumerate(routes):
            if not seq:
                continue
            t=start; here=depot;legs=[]
            for customer in tuple(seq)+(0,):
                dest=depot if customer==0 else node_of[int(customer)]
                sec,edges=self.route(here,dest,t)
                if not np.isfinite(sec):raise ValueError('Nonfinite route leg')
                nobs_here=0
                te=t
                for edge in edges:
                    slot=min(int(te//self.dt),len(self.times)-1)
                    nobs_here+=int(self.observed[slot,edge])
                    te+=self.travel(edge,te)
                # same continuous-time arrival function, tolerating floating error
                if abs((te-t)-sec)>1e-4:
                    raise AssertionError(f'TDSP leg replay mismatch {(te-t)-sec}')
                nobs+=nobs_here;nedges+=len(edges)
                total+=sec;t+=sec
                if customer:t+=float(task['service_seconds'])
                if collect_legs:
                    legs.append({'customer_id':int(customer),'from_node':here,'to_node':dest,
                                 'travel_sec':sec,'observed_edge_count':nobs_here,
                                 'edge_count':len(edges)})
                here=dest
            elapsed+=t-start
            if collect_legs:leglist.append({'vehicle':vehicle,'load':loads[vehicle],'customers':list(seq),'legs':legs})
        return dict(ex_post_road_travel_sec=total,
                    ex_post_total_vehicle_elapsed_sec=elapsed,
                    observed_edge_share=nobs/max(nedges,1),
                    observed_edge_count=nobs,edge_traversals=nedges,
                    num_vehicles_used=sum(bool(x) for x in routes),
                    vehicle_loads=loads,route_details=leglist)


class ODCost:
    """Search-stage approximation: sample all-pairs TDSP at knots and interpolate arrivals.

    Exact final evaluations are always re-run on original directed OSM edges.
    """
    def __init__(self,road,task,scenario,step_minutes=20,hours=4,quiet=True):
        import time
        tic=time.perf_counter()
        self.nodes=[str(task['depot_node'])]+[str(c['node']) for c in task['customers']]
        if len(set(self.nodes))!=len(self.nodes):
            raise ValueError('Depot/customer nodes must be distinct')
        self.index={v:i for i,v in enumerate(self.nodes)}
        self.start=road.to_sec(scenario['departure_local'])
        self.step=step_minutes*60.
        if hours<=0 or self.step<=0:raise ValueError('Positive OD horizon and step required')
        self.knots=self.start+np.arange(int(np.ceil(hours*3600/self.step))+1)*self.step
        if self.knots[-1]+3600>road.end_sec:
            raise ValueError('OD horizon plus 1h exceeds available traffic data')
        n=len(self.nodes)
        self.static=np.zeros((n,n),dtype=float)
        targets=self.nodes
        for i,node in enumerate(targets):
            found=road.dijkstra(node,targets,0,mode='static')
            self.static[i]=[found.get(v,(float('inf'),[]))[0] for v in targets]
        self.dynamic=np.empty((len(self.knots),n,n),dtype=float)
        for k,t in enumerate(self.knots):
            for i,node in enumerate(targets):
                found=road.dijkstra(node,targets,t,mode='oracle')
                self.dynamic[k,i]=[found.get(v,(float('inf'),[]))[0] for v in targets]
            if not quiet and ((k+1)%5==0 or k==0):print('  OD knot',k+1,'/',len(self.knots),flush=True)
        if not np.isfinite(self.static).all() or not np.isfinite(self.dynamic).all():
            raise ValueError('Unreachable OD pair in task; sample SCC nodes')
        arrivals=self.knots[:,None,None]+self.dynamic
        if np.min(np.diff(arrivals,axis=0)) < -1e-5:
            raise ValueError('OD FIFO consistency error')
        self.preprocessing_wall_sec=time.perf_counter()-tic

    def cost(self,a,b,t,mode):
        if a==b:return 0.
        if mode=='static':return float(self.static[a,b])
        if mode!='oracle':raise ValueError('planner must be static or oracle')
        position=(t-self.start)/self.step
        if position < -1e-7 or position > len(self.knots)-1+1e-7:
            raise ValueError('Route time exceeds sampled OD horizon; increase --od-hours')
        k=max(0,min(int(np.floor(position)),len(self.knots)-2))
        alpha=float(np.clip(position-k,0,1))
        arrival=(1-alpha)*(self.knots[k]+self.dynamic[k,a,b])+alpha*(self.knots[k+1]+self.dynamic[k+1,a,b])
        return max(0.,float(arrival-t))
