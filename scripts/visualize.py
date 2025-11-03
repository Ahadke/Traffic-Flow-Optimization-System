# Visualize the road network and sample traffic data points.

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

roads = gpd.read_file('data/MRDB_2024_published.shp')
aadf = pd.read_csv('data/dft_traffic_counts_aadf.csv', low_memory=False)

plt.figure(figsize=(12,12))
roads.plot(ax=plt.gca(), linewidth=0.5, edgecolor='gray', alpha=0.5)

# Plot small sample (1000) of traffic points to avoid memory issues
N = 1000
if 'latitude' in aadf.columns and 'longitude' in aadf.columns:
    plt.scatter(aadf['longitude'].iloc[:N], aadf['latitude'].iloc[:N], color='blue', s=2, label='AADF Sample Points')

plt.title("Major Road Network with Traffic Count Points (Sample)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()
