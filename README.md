# Traffic Flow Optimization System

A comprehensive urban traffic flow optimization system that combines **Reinforcement Learning (RL)** for intelligent traffic signal control with **Mixed Integer Linear Programming (MILP)** for optimal emergency vehicle routing.

## 🚦 Features

- **RL-Based Traffic Signal Control**: Deep Q-Network (DQN) agent for adaptive traffic signal timing
- **Emergency Vehicle Routing**: MILP optimization for finding shortest paths considering traffic conditions
- **Traffic-Aware Routing**: Dynamic edge weights based on real-time traffic queue lengths
- **Real Road Network Integration**: Supports shapefile-based road networks (14,000+ nodes tested)
- **Integrated System**: Seamless combination of RL control and MILP routing

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for detailed dependencies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd "Traffic Flow Optimization"

# Install dependencies
pip install -r requirements.txt

# Note: For MILP solving, install one of:
# - GLPK: conda install -c conda-forge glpk
# - CBC: conda install -c conda-forge coin-or-cbc
# - Or use pip: pip install glpk (if available)
```

### Basic Usage

#### 1. Train RL Agent for Traffic Control

```bash
python scripts/train_rl_agent.py
```

This trains a DQN agent to control traffic signals, saving rewards to `reward_log.npy`.

#### 2. Find Optimal Emergency Route

```bash
python scripts/emergency_routing.py
```

Or use in your code:
```python
from scripts.emergency_routing import find_optimal_emergency_route
from scripts.build_graph import build_road_graph

# Build road network
G = build_road_graph()

# Find route
route, cost, model = find_optimal_emergency_route(
    G, source=(x1, y1), destination=(x2, y2),
    verbose=True
)
```

#### 3. Run Integrated System

```bash
python scripts/integrated_system.py
```

Demonstrates the combined RL traffic control and emergency routing system.

#### 4. Main Entry Point (Full System)

```bash
python main.py --mode train    # Train RL agent
python main.py --mode route    # Emergency routing demo
python main.py --mode integrated  # Full integrated system
python main.py --mode all      # Run everything
```

## 📁 Project Structure

```
Traffic Flow Optimization/
├── data/                           # Traffic data and road networks
│   ├── MRDB_2024_published.shp    # Road network shapefile
│   ├── dft_traffic_counts_*.csv    # Traffic count datasets
│   └── ...
├── scripts/
│   ├── traffic_env.py             # RL environment for traffic simulation
│   ├── train_rl_agent.py          # DQN training script
│   ├── emergency_routing.py       # MILP emergency routing
│   ├── build_graph.py             # Road network graph builder
│   ├── integrated_system.py       # Combined RL + MILP system
│   ├── realistic_env.py           # Environment with real traffic data
│   ├── visualize.py               # Visualization utilities
│   ├── plot_training.py            # Plot training curves
│   └── ...
├── main.py                         # Main entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Configuration

Edit `config.py` to customize:
- Training hyperparameters (learning rate, episodes, etc.)
- Solver settings (GLPK, CBC, Gurobi)
- Graph building parameters
- Traffic environment settings

## 📊 Usage Examples

### Example 1: Train and Evaluate RL Agent

```python
from scripts.train_rl_agent import DQN, ReplayBuffer
from scripts.traffic_env import TrafficEnv

env = TrafficEnv()
# ... training code ...
```

### Example 2: Emergency Routing with Traffic Awareness

```python
from scripts.integrated_system import IntegratedTrafficSystem

system = IntegratedTrafficSystem()
system.reset_traffic_simulation()

# Find route accounting for current traffic
route, cost, _ = system.find_emergency_route(
    source=(x1, y1),
    destination=(x2, y2),
    use_traffic=True  # Consider congestion
)
```

### Example 3: Custom Traffic-Aware Routing

```python
from scripts.emergency_routing import find_optimal_emergency_route
from scripts.build_graph import build_road_graph

G = build_road_graph()

# Define custom weights based on traffic conditions
traffic_weights = {
    ((x1, y1), (x2, y2)): 10.0,  # Congested edge
    ((x3, y3), (x4, y4)): 1.5,   # Moderate traffic
    # ...
}

route, cost, _ = find_optimal_emergency_route(
    G, source, destination,
    edge_weight_override=traffic_weights
)
```

## 🔬 Technical Details

### RL Component
- **Algorithm**: Deep Q-Network (DQN)
- **Environment**: Custom Gymnasium environment
- **State Space**: Queue lengths at intersections
- **Action Space**: Traffic signal phase selection
- **Reward**: Negative sum of queue lengths (minimize congestion)

### MILP Component
- **Problem**: Shortest path with flow conservation constraints
- **Formulation**: Binary variables for edge selection
- **Constraints**: 
  - Flow conservation at nodes
  - Source/destination flow requirements
  - Path continuity
- **Solver**: GLPK, CBC, or Gurobi

### Integration
- Maps RL traffic state to graph edge weights
- Dynamically adjusts routing based on congestion
- Coordinates signal timing with emergency vehicle needs

## 📈 Results

The system has been tested on real road networks:
- **Road Network**: 14,220 nodes, 17,773 edges
- **Routing**: Successfully finds optimal paths in <1 second
- **Training**: DQN converges to reduce queue lengths effectively

## 🐛 Troubleshooting

### Pyomo Constraint Errors
If you see "Invalid constraint expression" errors:
- ✅ Already fixed in the current codebase
- Ensure constraint functions return Pyomo expressions, not Python Booleans

### Solver Not Found
Install a MILP solver:
```bash
# Option 1: GLPK
conda install -c conda-forge glpk

# Option 2: CBC
conda install -c conda-forge coin-or-cbc

# Option 3: Gurobi (requires license)
pip install gurobipy
```

### Import Errors
Ensure you're running from the project root:
```bash
cd "Traffic Flow Optimization"
python main.py
```

### Path Issues
All scripts now auto-detect data paths. If issues persist:
- Ensure data files are in `data/` directory
- Check that shapefile components (.shp, .shx, .dbf, .prj) are present

## 🚧 Future Enhancements

- Multi-agent RL for coordinated intersection control
- Real-time traffic data integration
- Predictive routing using traffic forecasting
- GUI for visualization and monitoring
- Distributed routing for multiple emergencies

## 📝 License

This project is for research and educational purposes.

---

**Status**: ✅ Production-ready with all core features implemented and tested.

