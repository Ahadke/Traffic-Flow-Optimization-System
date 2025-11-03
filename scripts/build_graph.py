"""
Build road network graph from road shapefile geometry.

This module creates a NetworkX directed graph representing the road network
from shapefile data, suitable for use in traffic simulation and routing.
"""

import os
import geopandas as gpd
import networkx as nx
import numpy as np


def build_road_graph(shapefile_path=None, verbose=True):
    """
    Build a directed graph from road shapefile data.
    
    Parameters:
    -----------
    shapefile_path : str, optional
        Path to the shapefile. If None, uses default location.
    verbose : bool, default=True
        If True, print graph statistics.
    
    Returns:
    --------
    networkx.DiGraph
        Directed graph with nodes as coordinate tuples and edges with
        'length' and 'road_id' attributes.
    """
    # Default path: assume script is in scripts/ directory
    if shapefile_path is None:
        # Try relative path first (when called from scripts/)
        default_path = os.path.join('data', 'MRDB_2024_published.shp')
        if not os.path.exists(default_path):
            # Try parent directory path
            default_path = os.path.join('..', 'data', 'MRDB_2024_published.shp')
        shapefile_path = default_path
    
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found at: {shapefile_path}")
    
    roads = gpd.read_file(shapefile_path)
    G = nx.DiGraph()
    
    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        
        for i in range(len(coords) - 1):
            start = tuple(coords[i])
            end = tuple(coords[i + 1])
            length = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            road_id = row.get('fid', idx)
            G.add_edge(start, end, length=length, road_id=road_id)
    
    if verbose:
        print("Graph created with:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        if G.number_of_nodes() > 0:
            print(f"  Density: {nx.density(G):.6f}")
    
    return G


# Example usage / test
if __name__ == "__main__":
    try:
        G = build_road_graph(verbose=True)
        print("\n[SUCCESS] Road network graph built successfully")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
