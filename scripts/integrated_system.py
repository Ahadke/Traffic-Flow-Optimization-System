"""
Integrated Traffic Flow Optimization System.

This script demonstrates the combined use of:
1. Reinforcement Learning (RL) for traffic signal control
2. MILP optimization for emergency vehicle routing

The system can route emergency vehicles while accounting for current traffic conditions
controlled by the RL agent.
"""

import os
import sys
import numpy as np
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.build_graph import build_road_graph
    from scripts.emergency_routing import find_optimal_emergency_route
    from scripts.traffic_env import TrafficEnv
except ImportError:
    # Try direct imports if running from scripts/ directory
    from build_graph import build_road_graph
    from emergency_routing import find_optimal_emergency_route
    from traffic_env import TrafficEnv


class IntegratedTrafficSystem:
    """
    Integrated system combining RL traffic control and emergency routing.
    """
    
    def __init__(self, road_graph=None, traffic_env=None, congestion_factor=1.5):
        """
        Initialize the integrated traffic system.
        
        Parameters:
        -----------
        road_graph : networkx.DiGraph, optional
            Road network graph. If None, will attempt to build from shapefile.
        traffic_env : gym.Env, optional
            Traffic environment for RL control. If None, uses default TrafficEnv.
        congestion_factor : float, default=1.5
            Multiplier for edge weights when routing through congested areas.
            Higher values penalize congested routes more.
        """
        if road_graph is None:
            try:
                self.road_graph = build_road_graph(verbose=False)
            except Exception as e:
                print(f"Warning: Could not load road graph from shapefile: {e}")
                print("Using example graph instead.")
                self.road_graph = self._create_example_graph()
        else:
            self.road_graph = road_graph
        
        self.traffic_env = traffic_env if traffic_env else TrafficEnv()
        self.congestion_factor = congestion_factor
        
        # Map traffic state indices to graph edges (simplified mapping)
        # In a real system, this would map specific intersections/signals to graph edges
        self.traffic_state_map = self._create_traffic_state_mapping()
    
    def _create_example_graph(self):
        """Create a simple example graph if shapefile is unavailable."""
        G = nx.DiGraph()
        # Create a small grid graph
        for i in range(3):
            for j in range(3):
                node = (i, j)
                if i < 2:
                    G.add_edge(node, (i+1, j), length=1.0)
                if j < 2:
                    G.add_edge(node, (i, j+1), length=1.0)
        return G
    
    def _create_traffic_state_mapping(self):
        """
        Map traffic environment state to graph edges.
        
        In a real implementation, this would map:
        - Traffic signals (in state) -> Graph edges they control
        - Queue lengths -> Edge congestion weights
        
        For now, uses a simplified mapping.
        """
        # Get first few edges from graph
        num_state_dims = 10
        if hasattr(self.traffic_env.observation_space, 'shape'):
            num_state_dims = self.traffic_env.observation_space.shape[0] if isinstance(self.traffic_env.observation_space.shape, tuple) else 10
        edges = list(self.road_graph.edges())[:num_state_dims]
        return {i: edge for i, edge in enumerate(edges)}
    
    def compute_traffic_aware_weights(self, traffic_state):
        """
        Compute edge weights based on current traffic state.
        
        Parameters:
        -----------
        traffic_state : numpy.ndarray
            Current state from traffic environment (e.g., queue lengths).
        
        Returns:
        --------
        dict
            Dictionary mapping edge tuples to traffic-adjusted weights.
        """
        edge_weights = {}
        
        # Base weights from graph
        for edge in self.road_graph.edges():
            base_weight = self.road_graph.edges[edge].get('length', 1.0)
            edge_weights[edge] = base_weight
        
        # Apply traffic congestion adjustments
        # Map traffic state to edges and increase weights for congested areas
        for idx, edge in self.traffic_state_map.items():
            if idx < len(traffic_state):
                # Queue length indicates congestion
                queue_length = traffic_state[idx]
                # Normalize queue length (assuming max queue ~50 from env)
                congestion_ratio = min(queue_length / 50.0, 1.0)
                # Increase weight for congested edges
                edge_weights[edge] *= (1.0 + congestion_ratio * (self.congestion_factor - 1.0))
        
        return edge_weights
    
    def find_emergency_route(self, source, destination, use_traffic=True, verbose=False):
        """
        Find optimal emergency route, optionally accounting for current traffic.
        
        Parameters:
        -----------
        source : hashable
            Source node in the road graph.
        destination : hashable
            Destination node in the road graph.
        use_traffic : bool, default=True
            If True, route considers current traffic state.
        verbose : bool, default=False
            If True, print detailed routing information.
        
        Returns:
        --------
        tuple
            (route_edges, total_cost, model) from find_optimal_emergency_route
        """
        edge_weight_override = None
        
        if use_traffic:
            # Get current traffic state
            traffic_state = self.traffic_env.state
            edge_weight_override = self.compute_traffic_aware_weights(traffic_state)
        
        return find_optimal_emergency_route(
            self.road_graph,
            source,
            destination,
            edge_weight_override=edge_weight_override,
            verbose=verbose
        )
    
    def step_traffic_simulation(self, action):
        """
        Step the traffic environment forward.
        
        Parameters:
        -----------
        action : int
            Action for traffic signal control.
        
        Returns:
        --------
        tuple
            Standard gym environment step return: (state, reward, done, truncated, info)
        """
        return self.traffic_env.step(action)
    
    def reset_traffic_simulation(self, seed=None):
        """Reset the traffic environment."""
        return self.traffic_env.reset(seed=seed)


def demo_integrated_system():
    """
    Demonstration of the integrated traffic optimization system.
    """
    print("=" * 60)
    print("Integrated Traffic Flow Optimization System Demo")
    print("=" * 60)
    print()
    
    # Initialize system
    print("[1/4] Initializing integrated system...")
    system = IntegratedTrafficSystem()
    print(f"    Road graph: {system.road_graph.number_of_nodes()} nodes, "
          f"{system.road_graph.number_of_edges()} edges")
    print(f"    Traffic environment initialized")
    print()
    
    # Reset traffic environment
    print("[2/4] Resetting traffic simulation...")
    traffic_state, _ = system.reset_traffic_simulation()
    print(f"    Initial traffic state (queue lengths): {traffic_state}")
    print()
    
    # Simulate some traffic evolution
    print("[3/4] Simulating traffic for a few steps...")
    for step in range(3):
        action = system.traffic_env.action_space.sample()
        traffic_state, reward, done, _, _ = system.step_traffic_simulation(action)
        print(f"    Step {step+1}: action={action}, state={traffic_state}, reward={reward:.2f}")
    print()
    
    # Find emergency route
    print("[4/4] Finding optimal emergency route...")
    # Get source and destination from graph
    nodes = list(system.road_graph.nodes())
    if len(nodes) >= 2:
        source = nodes[0]
        destination = nodes[-1]
        
        print(f"    Source: {source}")
        print(f"    Destination: {destination}")
        print()
        
        # Route without traffic awareness
        try:
            route_edges_no_traffic, cost_no_traffic, _ = system.find_emergency_route(
                source, destination, use_traffic=False, verbose=False
            )
            print(f"    Route (ignoring traffic): {len(route_edges_no_traffic)} edges, cost={cost_no_traffic:.2f}")
        except Exception as e:
            print(f"    Route (ignoring traffic): Failed - {e}")
        
        # Route with traffic awareness
        try:
            route_edges_traffic, cost_traffic, _ = system.find_emergency_route(
                source, destination, use_traffic=True, verbose=False
            )
            print(f"    Route (traffic-aware): {len(route_edges_traffic)} edges, cost={cost_traffic:.2f}")
            if cost_traffic > cost_no_traffic:
                print(f"    -> Traffic-aware routing found alternative route due to congestion")
        except Exception as e:
            print(f"    Route (traffic-aware): Failed - {e}")
    else:
        print("    Warning: Graph too small for routing demonstration")
    
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_integrated_system()

