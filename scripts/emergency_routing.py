"""
Emergency Vehicle Routing using MILP (Mixed Integer Linear Programming).

This module solves the shortest path problem for emergency vehicles on a road network
graph using Pyomo optimization.

HOW MILP ROUTING WORKS:
======================
1. PROBLEM: Find shortest path from source to destination in a graph

2. DECISION VARIABLES:
   - x[e] = 1 if edge e is used in path, 0 otherwise (binary variables)

3. OBJECTIVE: Minimize sum of edge weights along the path
   minimize: Σ (weight[e] * x[e]) for all edges e

4. CONSTRAINTS:
   a) Flow conservation at source: outflow - inflow = 1
      (exactly 1 more edge leaving than entering)
   
   b) Flow conservation at destination: inflow - outflow = 1
      (exactly 1 more edge entering than leaving)
   
   c) Flow conservation at intermediate nodes: outflow = inflow
      (same number of edges entering and leaving)
   
   d) Source constraint: exactly one edge leaves source
      (ensures path starts properly)

5. SOLVER: Uses MILP solver (GLPK/CBC/Gurobi) to find optimal solution
   The solution gives us which edges to use (x[e] = 1)

6. RESULT: Sequence of edges forming optimal path from source to destination
"""

import networkx as nx
from pyomo.environ import *


def find_optimal_emergency_route(G, source, destination, edge_weight_attr='length', 
                                  edge_weight_override=None, solver_name='glpk', verbose=False):
    """
    Find the optimal route for an emergency vehicle from source to destination.
    
    Uses MILP to solve the shortest path problem on a directed graph with traffic-aware edge weights.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Directed graph representing the road network. Edges should have weight attributes.
    source : hashable
        Source node identifier.
    destination : hashable
        Destination node identifier.
    edge_weight_attr : str, default='length'
        Edge attribute name to use as weight/cost (e.g., 'length', 'travel_time').
        This is used as fallback if edge_weight_override doesn't provide a value.
    edge_weight_override : dict, optional
        Dictionary mapping edge tuples (u, v) to custom weights.
        Allows dynamic traffic-aware routing by overriding edge weights based on current traffic.
        If provided, these weights take precedence over edge_weight_attr.
    solver_name : str, default='glpk'
        Name of the Pyomo solver to use (e.g., 'glpk', 'cbc', 'gurobi').
    verbose : bool, default=False
        If True, print solver output and detailed route information.
    
    Returns:
    --------
    tuple : (route_edges, total_cost, model)
        route_edges: List of edge tuples in the optimal path.
        total_cost: Total cost/weight of the optimal path.
        model: The solved Pyomo model (for inspection/debugging).
    
    Raises:
    -------
    ValueError
        If source or destination nodes are not in the graph.
    RuntimeError
        If the optimization fails or no solution is found.
    """
    # ============================================================================
    # INPUT VALIDATION
    # ============================================================================
    # Check that source and destination nodes exist in the graph
    if source not in G:
        raise ValueError(f"Source node {source} not in graph")
    if destination not in G:
        raise ValueError(f"Destination node {destination} not in graph")
    
    # ============================================================================
    # PREPARE EDGE WEIGHTS
    # ============================================================================
    # Ensure all edges have positive weights (required for shortest path)
    # If weight is missing or zero, set to default value of 1.0
    for u, v, d in G.edges(data=True):
        if edge_weight_attr not in d or d[edge_weight_attr] <= 0:
            d[edge_weight_attr] = 1.0
    
    # Get lists of nodes and edges for the optimization model
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    if len(edges) == 0:
        raise ValueError("Graph has no edges")
    
    # ============================================================================
    # CREATE PYOMO OPTIMIZATION MODEL
    # ============================================================================
    # Pyomo uses a ConcreteModel for deterministic problems (we know the data)
    model = ConcreteModel()
    
    # ============================================================================
    # DECISION VARIABLES
    # ============================================================================
    # x[e] = Binary variable for each edge:
    #   x[e] = 1 if edge e is used in the optimal path
    #   x[e] = 0 if edge e is NOT used
    # These binary variables will be determined by the solver
    model.x = Var(edges, domain=Binary)
    
    # ============================================================================
    # HELPER FUNCTION: GET EDGE WEIGHT
    # ============================================================================
    # This function supports traffic-aware routing:
    # - If edge_weight_override is provided, use custom weights (for traffic)
    # - Otherwise, use the standard edge attribute (e.g., 'length')
    def get_edge_weight(edge):
        if edge_weight_override is not None and edge in edge_weight_override:
            return edge_weight_override[edge]  # Use traffic-adjusted weight
        return G.edges[edge].get(edge_weight_attr, 1.0)  # Use standard weight
    
    # ============================================================================
    # OBJECTIVE FUNCTION
    # ============================================================================
    # Minimize total path cost = sum of weights for all edges used in path
    # Since x[e] = 1 only for edges in path, this sums only those edges
    model.obj = Objective(
        expr=sum(get_edge_weight(e) * model.x[e] for e in edges),
        sense=minimize  # We want the minimum cost path
    )
    
    # ============================================================================
    # FLOW CONSERVATION CONSTRAINTS
    # ============================================================================
    # These constraints ensure we get a valid path from source to destination
    # 
    # IMPORTANT PYOMO NOTE: Constraint rules must return Pyomo expressions
    # (equality/inequality) or Constraint.Feasible, NEVER Python True/False
    def flow_rule(model, node, *args):
        """
        Flow conservation rule for each node.
        
        This ensures:
        - Source: net outflow = 1 (path starts here)
        - Destination: net inflow = 1 (path ends here)
        - Intermediate: outflow = inflow (continuous path)
        """
        # Find all edges connected to this node
        outgoing = [e for e in edges if e[0] == node]  # Edges leaving node
        incoming = [e for e in edges if e[1] == node]  # Edges entering node
        
        # SOURCE NODE: Must have net outflow of 1
        # This means: (edges leaving) - (edges entering) = 1
        # Ensures the path starts at source
        if node == source:
            out_sum = sum(model.x[e] for e in outgoing)  # Sum of edges leaving
            in_sum = sum(model.x[e] for e in incoming)   # Sum of edges entering
            return (out_sum - in_sum) == 1  # Net outflow = 1
        
        # DESTINATION NODE: Must have net inflow of 1
        # This means: (edges entering) - (edges leaving) = 1
        # Ensures the path ends at destination
        elif node == destination:
            out_sum = sum(model.x[e] for e in outgoing)
            in_sum = sum(model.x[e] for e in incoming)
            return (in_sum - out_sum) == 1  # Net inflow = 1
        
        # INTERMEDIATE NODES: Flow conservation
        # This means: (edges leaving) = (edges entering)
        # Ensures path is continuous (no dead ends)
        else:
            out_sum = sum(model.x[e] for e in outgoing)
            in_sum = sum(model.x[e] for e in incoming)
            # If node has no edges, constraint is automatically satisfied
            if len(outgoing) == 0 and len(incoming) == 0:
                return Constraint.Feasible  # No constraint needed
            # Otherwise, enforce flow conservation
            return (out_sum - in_sum) == 0  # Outflow = inflow
    
    # Apply flow conservation constraint to all nodes
    model.flow_constr = Constraint(nodes, rule=flow_rule)
    
    # ============================================================================
    # ADDITIONAL CONSTRAINTS FOR ROBUSTNESS
    # ============================================================================
    # Explicit constraint: exactly one edge leaves the source
    # This reinforces the source flow constraint and ensures path starts properly
    source_outgoing = [e for e in edges if e[0] == source]
    if len(source_outgoing) > 0:
        model.source_outflow = Constraint(expr=sum(model.x[e] for e in source_outgoing) == 1)
    
    # Ensure at least one edge is selected (prevents empty solution)
    model.min_path_length = Constraint(expr=sum(model.x[e] for e in edges) >= 1)
    
    # Ensure destination is reachable (at least one edge enters destination)
    model.destination_inflow = Constraint(expr=sum(model.x[e] for e in edges if e[1] == destination) >= 1)
    
    # ============================================================================
    # SOLVE THE OPTIMIZATION PROBLEM
    # ============================================================================
    # Create solver instance (GLPK, CBC, or Gurobi)
    solver = SolverFactory(solver_name)
    if solver is None:
        raise RuntimeError(f"Solver '{solver_name}' not available. Install it or use another solver (e.g., 'cbc', 'glpk')")
    
    # Solve the MILP model
    # tee=verbose: if True, prints solver progress
    result = solver.solve(model, tee=verbose)
    
    # ============================================================================
    # CHECK SOLUTION STATUS
    # ============================================================================
    # Verify that solver found an optimal solution
    if result.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(f"Optimization failed: {result.solver.termination_condition}")
    
    # ============================================================================
    # EXTRACT THE SOLUTION
    # ============================================================================
    # Find all edges where x[e] = 1 (used in optimal path)
    # Value > 0.5 handles floating-point precision issues with binary variables
    route_edges = [e for e in edges if model.x[e].value is not None and model.x[e].value > 0.5]
    
    # Get total cost of the optimal path
    total_cost = value(model.obj)
    
    if verbose:
        print(f"Optimal emergency route: {route_edges}")
        print(f"Total cost: {total_cost}")
        print(f"Path length (number of edges): {len(route_edges)}")
    
    return route_edges, total_cost, model


# Example usage / test
if __name__ == "__main__":
    # Example directed graph (replace with your actual road network)
    G = nx.DiGraph()
    G.add_edge((0, 0), (1, 0), length=1)
    G.add_edge((1, 0), (1, 1), length=1)
    G.add_edge((0, 0), (0, 1), length=2)
    G.add_edge((0, 1), (1, 1), length=1)
    
    source = (0, 0)
    destination = (1, 1)
    
    print("Graph nodes:", list(G.nodes))
    print("Graph edges with attributes:", list(G.edges(data=True)))
    print("Source:", source)
    print("Destination:", destination)
    print()
    
    try:
        route_edges, total_cost, model = find_optimal_emergency_route(
            G, source, destination, verbose=True
        )
        print(f"\n[SUCCESS] Found optimal route with cost {total_cost}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
