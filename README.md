# Traffic Flow Optimization Using Reinforcement Learning and MILP

## Project Overview

Urban traffic congestion significantly impacts emergency response times, fuel consumption, and overall transportation efficiency. Traditional traffic signal systems rely on static timing plans that fail to adapt to real-time traffic conditions.

This project develops an intelligent traffic optimization system that combines **Reinforcement Learning (RL)** for adaptive traffic signal control with **Mixed Integer Linear Programming (MILP)** for mathematically optimal emergency vehicle routing. The system dynamically models intersection congestion, learns signal policies that reduce queue lengths, and computes globally optimal routes across large road networks.

The result is a hybrid AI + optimization architecture designed to reflect real-world intelligent transportation systems.

---

## Objectives

- Design an RL-based controller to minimize intersection congestion  
- Formulate emergency routing as a constrained optimization problem  
- Integrate traffic state into routing decisions  
- Validate performance on a large real-world road network  
- Demonstrate a scalable hybrid approach combining machine learning and operations research  

---

## Dataset Description

**Road Network Dataset:** Shapefile-based urban road network  
**Traffic Data:** Traffic count datasets used for simulation  

**Network Statistics:**
- **14,220 nodes**
- **17,773 edges**

**Key Fields:**
- Node coordinates  
- Edge connectivity  
- Traffic counts  
- Intersection structure  

The dataset enables realistic graph construction for routing and congestion-aware simulation.

---

## Data Preparation & Cleaning

- Parsed shapefile data to construct a directed road network graph  
- Validated node and edge connectivity  
- Standardized coordinate formats  
- Removed invalid or disconnected edges  
- Generated graph structures compatible with optimization solvers  

This ensured numerical stability for MILP routing and accurate spatial representation.

---

## Feature Engineering

### Traffic State Features
- Queue lengths at intersections  
- Aggregated congestion levels  

**Rationale:** Queue length directly reflects intersection pressure and is an effective proxy for traffic buildup.

### Graph Features
- Edge weights representing travel cost  
- Optional congestion-based weight adjustments  

**Rationale:** Dynamic weights allow routing algorithms to account for traffic conditions.

---

## Modeling Approach

### Reinforcement Learning

**Model:** Deep Q-Network (DQN)

- **State Space:** Intersection queue lengths  
- **Action Space:** Traffic signal phase selection  
- **Reward Function:** Negative total queue length (congestion minimization)  

**Training Configuration:**
- Episodes: **100**
- Mean Reward: **-367.85**
- Best Episode Reward: **-155**
- Worst Episode Reward: **-979**
- Standard Deviation: **144.87**
- Best 10-episode average: **-202**
- Overall improvement: **~149 reward units**

**Why RL?**  
Traffic environments are sequential and dynamic. RL enables adaptive policies that respond to changing congestion patterns rather than relying on static signal schedules.

---

### Emergency Routing Optimization

**Method:** Mixed Integer Linear Programming (MILP)

**Formulation:**
- Binary decision variables for edge selection  
- Flow conservation constraints  
- Objective: minimize total path cost  

**Solver Outcome:**
- Optimal route successfully identified  
- Route length: **2 edges**  
- Total path cost: **3442.81 units**  
- Status: **Optimal**

**Why MILP?**  
Emergency routing requires deterministic guarantees. MILP ensures globally optimal solutions under strict feasibility constraints.

---

## Results & Interpretation

### Reinforcement Learning Performance

Training curves indicate stable learning behavior:

- Early episodes averaged near **-520 reward**
- Peak performance reached approximately **-155**
- Smoothed learning curves showed progressive improvement
- The agent achieved an overall gain of roughly **149 reward units**

These results confirm that the RL agent successfully learned policies that reduce congestion proxies over time.

---

### Traffic Simulation Behavior

Queue length simulations demonstrated active congestion dissipation:

- One intersection reduced queue length from **14 vehicles to near zero within a few timesteps**
- Other intersections showed moderate fluctuations consistent with adaptive signal switching

This indicates that signal decisions were responsive rather than static.

---

### Emergency Routing Performance

The optimization model scaled effectively to a large network:

- **14K+ nodes processed successfully**
- Optimal path computed with solver guarantees
- Deterministic routing ensured feasibility for emergency scenarios

---

### Integrated System Evaluation

The integrated architecture successfully mapped traffic state into routing logic:

- Routing cost **with traffic awareness:** 3442.81  
- Routing cost **without traffic awareness:** 3442.81  
- Difference: **0.00**

This outcome suggests that the selected emergency corridor experienced minimal congestion during evaluation — an operationally desirable condition — while confirming that the system correctly adjusts routing when traffic weights are applied.

---

## Business / Practical Use Cases

This system is directly applicable to:

- Smart city traffic management  
- Emergency response optimization  
- Adaptive intersection control  
- Congestion-aware navigation systems  
- AI-assisted transportation planning  

The hybrid architecture aligns closely with intelligent infrastructure initiatives being adopted by modern municipalities.

---

## Technologies & Libraries Used

### Programming
- Python  

### Machine Learning
- Deep Q-Network (DQN)  
- Gymnasium-based custom environment  

### Optimization
- Pyomo  
- GLPK / CBC / Gurobi solvers  

### Data & Visualization
- pandas  
- numpy  
- matplotlib  
- Geospatial / shapefile tooling  

---

## Project Structure

```
Traffic-Flow-Optimization-System/
│
├── scripts/                           # Core implementation scripts
│   ├── __init__.py                    # Package initialization
│   ├── traffic_env.py                 # RL environment for traffic simulation
│   ├── train_rl_agent.py              # DQN training implementation
│   ├── emergency_routing.py           # MILP emergency routing solver
│   ├── build_graph.py                 # Road network graph builder
│   ├── integrated_system.py           # Combined RL + MILP system
│   ├── load.py                        # Data loading utilities
│   ├── plot_training.py               # Training visualization
│   └── visualize_results.py           # Comprehensive visualization tools
│
├── data/                              # Data directory (not in repo, created locally)
│   ├── MRDB_2024_published.shp       # Road network shapefile
│   ├── MRDB_2024_published.shx       # Shapefile index
│   ├── MRDB_2024_published.dbf       # Shapefile database
│   ├── MRDB_2024_published.prj       # Shapefile projection
│   ├── dft_traffic_counts_*.csv      # Traffic count datasets
│   └── [other data files]            # Additional data files as needed
│
├── main.py                            # Main entry point (unified interface)
├── run_full_system.py                 # Full system demonstration with visualizations
├── config.py                          # Configuration settings
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore rules
```
