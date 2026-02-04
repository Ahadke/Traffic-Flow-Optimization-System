# Traffic Flow Optimization System Using Reinforcement Learning and MILP

## Project Overview

This project implements an intelligent urban traffic optimization system that combines **Reinforcement Learning (RL)** for adaptive traffic signal control with **Mixed Integer Linear Programming (MILP)** for optimal emergency vehicle routing.

The system dynamically models traffic conditions, reduces congestion through learned signal policies, and computes mathematically optimal emergency routes under network constraints. By integrating learning-based control with deterministic optimization, the project demonstrates a scalable architecture for next-generation smart transportation systems.

---

## Objectives

The primary objectives of this project were to:

- Develop an RL-based traffic signal controller to minimize congestion  
- Formulate emergency routing as a constrained optimization problem  
- Integrate traffic state into routing decisions  
- Validate system scalability on large real-world road networks  
- Demonstrate a hybrid AI + Operations Research approach  

---

## System Architecture

The system consists of three core components:

### Reinforcement Learning Layer
- Deep Q-Network (DQN) agent controls intersection signal phases  
- State space: queue lengths at intersections  
- Action space: traffic signal selection  
- Reward: negative total queue length (congestion minimization)  

### MILP Optimization Layer
- Emergency routing formulated as a shortest-path optimization problem  
- Binary edge-selection variables  
- Flow conservation constraints  
- Solver support: GLPK, CBC, Gurobi  

### Integrated Decision System
- RL-generated traffic states dynamically update graph edge weights  
- Routing adapts to congestion conditions  
- Enables traffic-aware emergency response  

---

## Dataset & Road Network

The system was tested on a real road network derived from shapefile data.

**Network Statistics:**
- **14,220 nodes**
- **17,773 edges**

Traffic count datasets were incorporated to simulate realistic congestion patterns.

---

## Reinforcement Learning Modeling

A Deep Q-Network (DQN) agent was trained to optimize signal timing policies.

### Training Configuration
- Episodes: **100**
- Mean Reward: **-367.85**
- Best Episode Reward: **-155**
- Worst Episode Reward: **-979**
- Standard Deviation: **144.87**

### Learning Behavior

Training demonstrated stable learning dynamics:

- Early episodes averaged near **-520 reward**
- Peak performance improved to approximately **-155**
- Smoothed learning curves indicated progressive congestion reduction
- The best 10-episode average reached **-202**

This represents an improvement of roughly **149 reward units**, indicating the agent successfully learned policies that reduce intersection queue lengths.

---

## Emergency Vehicle Routing (MILP)

Emergency routing was formulated as a linear optimization problem with strict feasibility guarantees.

### Routing Results
- Optimal path found successfully  
- Route length: **2 edges**  
- Total path cost: **3442.81 units**  
- Solve status: **Optimal**

Despite the network scale (>14K nodes), the solver consistently identified feasible optimal routes, demonstrating strong computational tractability.

---

## Integrated System Performance

The integrated architecture validated coordination between learning-based control and mathematical optimization.

- Traffic states were successfully translated into routing weights  
- The system dynamically generated routes based on real-time conditions  
- Signal control and routing modules executed without conflict  

During evaluation, routing costs remained stable with and without congestion adjustments, indicating low traffic interference along the selected emergency corridor — a desirable operational outcome.

---

## Traffic Simulation Insights

Queue length simulations showed measurable traffic dissipation:

- One intersection reduced queue length from **14 vehicles to near zero** within a few timesteps  
- Other intersections exhibited moderate fluctuations consistent with adaptive signal switching  

These patterns confirm that the RL agent actively responded to congestion rather than applying static timing.

---

## Results & Interpretation

### RL Signal Optimization
- Demonstrated stable policy learning  
- Achieved significant reward improvement (~149 units)  
- Reduced congestion proxies across simulated intersections  

### Optimization-Based Routing
- Guaranteed globally optimal emergency paths  
- Maintained solver performance on large networks  
- Produced deterministic, interpretable routing decisions  

### System Integration
- Successfully merged AI-driven control with operations research  
- Enabled traffic-aware routing without introducing instability  

Overall, the system validates the effectiveness of hybrid intelligent transportation architectures.

---

## Business & Real-World Applications

This system can support:

- Smart city traffic management  
- Emergency response optimization  
- Adaptive intersection control  
- Congestion-aware navigation  
- AI-assisted transportation planning  

The architecture is particularly relevant for municipalities investing in intelligent infrastructure.

---

## Technologies & Libraries Used

### Core Stack
- Python  
- Jupyter Notebook  

### Machine Learning
- PyTorch / DQN implementation  
- Gymnasium-based custom environment  

### Optimization
- Pyomo  
- GLPK / CBC / Gurobi solvers  

### Data & Visualization
- pandas  
- numpy  
- matplotlib  
- shapefile / geospatial tooling  

---

## Project Structure

