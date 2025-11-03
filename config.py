"""
Configuration settings for Traffic Flow Optimization System.
"""

# ============================================================================
# RL Training Configuration
# ============================================================================
RL_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'batch_size': 32,
    'buffer_size': 10000,
    'n_episodes': 200,
    'max_steps_per_episode': 50,
    'hidden_layers': [64, 64],
}

# ============================================================================
# MILP Solver Configuration
# ============================================================================
SOLVER_CONFIG = {
    'solver_name': 'glpk',  # Options: 'glpk', 'cbc', 'gurobi'
    'verbose': False,
    'tee': False,  # Print solver output
}

# ============================================================================
# Graph Building Configuration
# ============================================================================
GRAPH_CONFIG = {
    'shapefile_path': None,  # None = auto-detect
    'default_length': 1.0,   # Default edge length if missing
    'verbose': True,
}

# ============================================================================
# Emergency Routing Configuration
# ============================================================================
ROUTING_CONFIG = {
    'edge_weight_attr': 'length',
    'congestion_factor': 1.5,  # Multiplier for congested edges
    'use_traffic_aware': True,
}

# ============================================================================
# Traffic Environment Configuration
# ============================================================================
TRAFFIC_ENV_CONFIG = {
    'observation_high': 50,
    'observation_shape': (2,),  # Number of intersections
    'n_actions': 2,
    'max_steps': 50,
}

# ============================================================================
# Path Configuration
# ============================================================================
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'reward_log_file': 'reward_log.npy',
}

# ============================================================================
# Integrated System Configuration
# ============================================================================
INTEGRATED_CONFIG = {
    'enable_traffic_aware_routing': True,
    'traffic_state_update_frequency': 1,  # Update every N steps
}

