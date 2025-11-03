"""
Main entry point for Traffic Flow Optimization System.

This script provides a unified interface to all system components:
- RL training for traffic signal control
- Emergency vehicle routing (MILP)
- Integrated system demonstration
"""

import argparse
import os
import sys
import numpy as np

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.traffic_env import TrafficEnv
from scripts.train_rl_agent import DQN, ReplayBuffer
from scripts.build_graph import build_road_graph
from scripts.emergency_routing import find_optimal_emergency_route
from scripts.integrated_system import IntegratedTrafficSystem
from scripts.plot_training import plot_training_curve
try:
    import config
except ImportError:
    # Create minimal config if not available
    class Config:
        RL_CONFIG = {
            'n_episodes': 200, 'learning_rate': 0.001, 'gamma': 0.99,
            'epsilon_start': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.01,
            'buffer_size': 10000, 'batch_size': 32
        }
        SOLVER_CONFIG = {'solver_name': 'glpk', 'verbose': False, 'tee': False}
        GRAPH_CONFIG = {'shapefile_path': None, 'default_length': 1.0, 'verbose': True}
        ROUTING_CONFIG = {'edge_weight_attr': 'length', 'congestion_factor': 1.5, 'use_traffic_aware': True}
        TRAFFIC_ENV_CONFIG = {'observation_high': 50, 'observation_shape': (2,), 'n_actions': 2, 'max_steps': 50}
        PATHS = {'data_dir': 'data', 'models_dir': 'models', 'logs_dir': 'logs', 'reward_log_file': 'reward_log.npy'}
        INTEGRATED_CONFIG = {'enable_traffic_aware_routing': True, 'traffic_state_update_frequency': 1}
    config = Config()


def train_rl_agent(episodes=None, verbose=True):
    """Train RL agent for traffic signal control."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    print("=" * 60)
    print("Training RL Agent for Traffic Signal Control")
    print("=" * 60)
    
    episodes = episodes or config.RL_CONFIG['n_episodes']
    
    env = TrafficEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    model = DQN(obs_dim, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=config.RL_CONFIG['learning_rate'])
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(max_size=config.RL_CONFIG['buffer_size'])
    
    epsilon = config.RL_CONFIG['epsilon_start']
    epsilon_decay = config.RL_CONFIG['epsilon_decay']
    epsilon_min = config.RL_CONFIG['epsilon_min']
    gamma = config.RL_CONFIG['gamma']
    batch_size = config.RL_CONFIG['batch_size']
    
    all_rewards = []
    
    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(model(torch.FloatTensor(state))).item()
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # Training step
            if len(buffer.buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_vals = model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_vals * (1 - dones)
                
                loss = loss_fn(q_vals, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        all_rewards.append(total_reward)
        
        if verbose and (ep % 20 == 0 or ep == episodes - 1):
            avg_reward = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else total_reward
            print(f"Episode {ep:4d}/{episodes} | Reward: {total_reward:7.2f} | "
                  f"Avg (last 20): {avg_reward:7.2f} | Epsilon: {epsilon:.3f}")
    
    # Save rewards
    os.makedirs(os.path.dirname(config.PATHS['reward_log_file']) or '.', exist_ok=True)
    np.save(config.PATHS['reward_log_file'], all_rewards)
    print(f"\nTraining complete! Rewards saved to {config.PATHS['reward_log_file']}")
    
    # Plot training curve if available
    try:
        # Create plot directly
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(all_rewards, alpha=0.6, label='Episode Reward')
        if len(all_rewards) > 20:
            window = min(20, len(all_rewards) // 10)
            moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(all_rewards)), moving_avg, 
                    label=f'Moving Average (window={window})', linewidth=2)
        plt.title('RL Training: Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=150)
        plt.close()
        print("Training curve saved to training_curve.png")
    except Exception as e:
        if verbose:
            print(f"Could not save plot: {e}")
    
    return model, all_rewards


def demo_emergency_routing(verbose=True):
    """Demonstrate emergency vehicle routing."""
    print("=" * 60)
    print("Emergency Vehicle Routing Demo")
    print("=" * 60)
    
    try:
        # Try to build graph from shapefile
        G = build_road_graph(verbose=verbose)
        print(f"\nLoaded road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"Warning: Could not load shapefile ({e}). Using example graph.")
        G = create_example_graph()
    
    # Select source and destination
    nodes = list(G.nodes())
    if len(nodes) < 2:
        print("Error: Graph too small for routing demonstration")
        return
    
    source = nodes[0]
    destination = nodes[-1]
    
    print(f"\nSource: {source}")
    print(f"Destination: {destination}")
    print(f"Distance: {np.sqrt((destination[0]-source[0])**2 + (destination[1]-source[1])**2):.2f} units")
    print()
    
    try:
        route_edges, total_cost, model = find_optimal_emergency_route(
            G, source, destination,
            solver_name=config.SOLVER_CONFIG['solver_name'],
            verbose=verbose
        )
        
        print(f"\n[SUCCESS] Optimal route found!")
        print(f"  Path length: {len(route_edges)} edges")
        print(f"  Total cost: {total_cost:.2f}")
        print(f"  Route: {route_edges[:3]}..." if len(route_edges) > 3 else f"  Route: {route_edges}")
        
    except Exception as e:
        print(f"\n[ERROR] Routing failed: {e}")
        import traceback
        traceback.print_exc()


def demo_integrated_system(verbose=True):
    """Demonstrate integrated RL + MILP system."""
    print("=" * 60)
    print("Integrated Traffic Optimization System")
    print("=" * 60)
    print()
    
    try:
        system = IntegratedTrafficSystem()
        print(f"System initialized with {system.road_graph.number_of_nodes()} nodes")
        
        # Reset traffic
        system.reset_traffic_simulation()
        print("Traffic simulation reset")
        
        # Simulate some traffic
        print("\nSimulating traffic for 5 steps...")
        for step in range(5):
            action = system.traffic_env.action_space.sample()
            state, reward, done, _, _ = system.step_traffic_simulation(action)
            if verbose:
                print(f"  Step {step+1}: state={state}, reward={reward:.2f}")
        
        # Find emergency route
        nodes = list(system.road_graph.nodes())
        if len(nodes) >= 2:
            source = nodes[0]
            destination = nodes[-1]
            
            print(f"\nFinding emergency route from {source} to {destination}...")
            route, cost, _ = system.find_emergency_route(
                source, destination,
                use_traffic=config.INTEGRATED_CONFIG['enable_traffic_aware_routing'],
                verbose=False
            )
            
            print(f"[SUCCESS] Route found: {len(route)} edges, cost={cost:.2f}")
        else:
            print("Warning: Graph too small")
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


def create_example_graph():
    """Create a simple example graph for testing."""
    import networkx as nx
    
    G = nx.DiGraph()
    # Create a small grid
    for i in range(5):
        for j in range(5):
            node = (i, j)
            if i < 4:
                G.add_edge(node, (i+1, j), length=1.0)
            if j < 4:
                G.add_edge(node, (i, j+1), length=1.0)
    return G




def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Traffic Flow Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train              # Train RL agent
  python main.py --mode route              # Emergency routing demo
  python main.py --mode integrated         # Full integrated system
  python main.py --mode all                # Run everything
  python main.py --mode train --episodes 100  # Custom training episodes
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'route', 'integrated', 'all'],
        default='all',
        help='Operation mode (default: all)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (uses config if not specified)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    print("\n" + "=" * 60)
    print("Traffic Flow Optimization System")
    print("=" * 60)
    print()
    
    try:
        if args.mode == 'train' or args.mode == 'all':
            train_rl_agent(episodes=args.episodes, verbose=verbose)
            print()
        
        if args.mode == 'route' or args.mode == 'all':
            demo_emergency_routing(verbose=verbose)
            print()
        
        if args.mode == 'integrated' or args.mode == 'all':
            demo_integrated_system(verbose=verbose)
            print()
        
        print("=" * 60)
        print("All operations complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

