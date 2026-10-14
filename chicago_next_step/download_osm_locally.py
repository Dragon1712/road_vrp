"""Only run on a computer WITH internet (OSMnx >= 2.0)."""
from pathlib import Path
import argparse
import osmnx as ox

p=argparse.ArgumentParser(description='Download Chicago candidate directed drive graph LOCALLY')
p.add_argument('--output',default='chicago_loop_drive.graphml')
p.add_argument('--west',type=float,default=-87.667)
p.add_argument('--south',type=float,default=41.851)
p.add_argument('--east',type=float,default=-87.592)
p.add_argument('--north',type=float,default=41.906)
a=p.parse_args()

print('OSMnx version:',ox.__version__)
print('BBox (west,south,east,north):',a.west,a.south,a.east,a.north)
# OSMnx 2.x bbox is (left, bottom, right, top) = (west,south,east,north)
G=ox.graph_from_bbox((a.west,a.south,a.east,a.north),network_type='drive',simplify=True,retain_all=False)
if G.number_of_nodes()==0 or G.number_of_edges()==0:
    raise RuntimeError('Empty graph: inspect OSM network connectivity and bbox')
output=Path(a.output)
output.parent.mkdir(parents=True,exist_ok=True)
ox.save_graphml(G,filepath=output)
print(f'Saved {output} : {len(G.nodes)} nodes, {len(G.edges)} directed edges')
