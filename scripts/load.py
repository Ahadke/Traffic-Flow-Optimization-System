"""
Load and inspect traffic CSV data and shapefiles.

This module provides functions to load traffic data and road network shapefiles
with automatic path resolution.
"""

import os
import pandas as pd
import geopandas as gpd


def find_data_file(filename, data_dir='data'):
    """Find data file with automatic path resolution."""
    paths_to_try = [
        os.path.join(data_dir, filename),
        os.path.join('..', data_dir, filename),
        os.path.join('.', data_dir, filename),
        filename  # Try direct path
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find {filename}. Tried: {paths_to_try}")


def load_traffic_data(verbose=True):
    """
    Load all traffic count datasets.
    
    Returns:
    --------
    dict
        Dictionary with keys: 'aadf', 'aadf_by_direction', 'raw_counts'
    """
    data = {}
    
    files = {
        'aadf': 'dft_traffic_counts_aadf.csv',
        'aadf_by_direction': 'dft_traffic_counts_aadf_by_direction.csv',
        'raw_counts': 'dft_traffic_counts_raw_counts.csv'
    }
    
    for key, filename in files.items():
        try:
            path = find_data_file(filename)
            data[key] = pd.read_csv(path, low_memory=False)
            if verbose:
                print(f"Loaded {key}: {data[key].shape}")
        except FileNotFoundError as e:
            if verbose:
                print(f"Warning: Could not load {filename}: {e}")
            data[key] = None
    
    return data


def load_road_network(verbose=True):
    """
    Load road network shapefile.
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Road network data
    """
    try:
        path = find_data_file('MRDB_2024_published.shp')
        roads = gpd.read_file(path)
        if verbose:
            print(f"Loaded road network: {roads.shape}")
        return roads
    except FileNotFoundError as e:
        if verbose:
            print(f"Warning: Could not load shapefile: {e}")
        return None


# Example usage / test
if __name__ == "__main__":
    print("=" * 60)
    print("Loading Traffic Data")
    print("=" * 60)
    print()
    
    # Load traffic data
    data = load_traffic_data(verbose=True)
    
    if data.get('aadf') is not None:
        print("\nAADF preview:")
        print(data['aadf'].head())
        print(f"\nAADF columns: {list(data['aadf'].columns[:10])}...")
    
    print()
    
    # Load road network
    print("=" * 60)
    print("Loading Road Network")
    print("=" * 60)
    roads = load_road_network(verbose=True)
    
    if roads is not None:
        print("\nRoad network preview:")
        print(roads.head())
        print(f"\nRoad network columns: {list(roads.columns)}")
    
    print("\n[SUCCESS] Data loading complete!")
