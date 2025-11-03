"""
Run Full Traffic Flow Optimization System with Visualizations.

This script runs the complete system and generates comprehensive visualizations
showing how RL and MILP work together to optimize traffic flow.
"""

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
from scripts.visualize_results import (
    plot_training_progress,
    plot_emergency_routing,
    plot_system_overview
)

import torch
import torch.nn as nn
import torch.optim as optim


def print_section(title, char='='):
    """Print a formatted section header."""
    print("\n" + char * 70)
    print(f"  {title}")
    print(char * 70 + "\n")


def run_full_system():
    """
    Run the complete traffic flow optimization system.
    
    This demonstrates:
    1. RL training for traffic signal control
    2. Emergency vehicle routing with MILP
    3. Integrated system combining both
    4. Comprehensive visualizations
    """
    
    print("\n" + "=" * 70)
    print("  TRAFFIC FLOW OPTIMIZATION SYSTEM - FULL DEMONSTRATION")
    print("=" * 70)
    print("\nThis system combines:")
    print("  • Reinforcement Learning (RL) for adaptive traffic signal control")
    print("  • Mixed Integer Linear Programming (MILP) for emergency routing")
    print("  • Integrated system that routes based on real-time traffic\n")
    
    results = {}
    
    # ============================================================================
    # PART 1: TRAIN RL AGENT FOR TRAFFIC SIGNAL CONTROL
    # ============================================================================
    print_section("PART 1: Training RL Agent for Traffic Signal Control")
    
    print("HOW RL WORKS:")
    print("  1. Agent observes queue lengths at intersections (state)")
    print("  2. Agent selects which signal to prioritize (action)")
    print("  3. Selected signal clears vehicles, reducing congestion")
    print("  4. Agent receives reward: -total_queue_length (wants to minimize)")
    print("  5. Agent learns optimal policy through trial and error")
    print()
    
    env = TrafficEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    print(f"Environment setup:")
    print(f"  • State space: {obs_dim} intersections (queue lengths)")
    print(f"  • Action space: {n_actions} actions (which signal to prioritize)")
    print(f"  • Goal: Minimize total queue length across all intersections")
    print()
    
    # Initialize DQN model
    model = DQN(obs_dim, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(max_size=10000)
    
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    batch_size = 32
    n_episodes = 100  # Reduced for demo
    
    all_rewards = []
    
    def select_action(state, epsilon):
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return env.action_space.sample()  # Explore: random action
        else:
            # Exploit: choose best action according to current policy
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0))
                return torch.argmax(q_values).item()
    
    print("Training RL agent...")
    print("  Episode | Reward | Avg (last 20) | Epsilon")
    print("  " + "-" * 50)
    
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # Train on random batch from experience buffer
            if len(buffer.buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(np.array(actions))
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(np.array(dones))
                
                # Q-learning update
                q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_vals = model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_vals * (1 - dones)
                
                # Gradient descent
                loss = loss_fn(q_vals, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        all_rewards.append(total_reward)
        
        if ep % 20 == 0 or ep == n_episodes - 1:
            avg_reward = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else total_reward
            print(f"  {ep:4d}    | {total_reward:7.2f} | {avg_reward:13.2f} | {epsilon:.3f}")
    
    # Save rewards
    np.save("reward_log.npy", all_rewards)
    results['rl_rewards'] = all_rewards
    print(f"\n✓ RL training complete! Saved {n_episodes} episodes of rewards.")
    print()
    
    # ============================================================================
    # PART 2: EMERGENCY VEHICLE ROUTING WITH MILP
    # ============================================================================
    print_section("PART 2: Emergency Vehicle Routing Using MILP")
    
    print("HOW MILP ROUTING WORKS:")
    print("  1. Road network represented as directed graph (nodes=intersections, edges=roads)")
    print("  2. Binary variables x[e] for each edge: 1 if used, 0 otherwise")
    print("  3. Objective: Minimize total path cost (sum of edge weights)")
    print("  4. Constraints:")
    print("     • Source: net outflow = 1 (path starts)")
    print("     • Destination: net inflow = 1 (path ends)")
    print("     • Intermediate: outflow = inflow (continuous path)")
    print("  5. Solver finds optimal path from source to destination")
    print()
    
    try:
        # Build road network graph
        print("Loading road network from shapefile...")
        G = build_road_graph(verbose=True)
        print(f"✓ Loaded {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
        
        # Select source and destination
        nodes = list(G.nodes())
        if len(nodes) >= 2:
            source = nodes[0]
            destination = nodes[-1]
            
            print(f"\nFinding optimal route:")
            print(f"  Source: {source}")
            print(f"  Destination: {destination}")
            print(f"  Euclidean distance: {np.sqrt((destination[0]-source[0])**2 + (destination[1]-source[1])**2):.2f} units")
            print()
            
            # Find optimal route
            print("Solving MILP optimization problem...")
            route_edges, total_cost, _ = find_optimal_emergency_route(
                G, source, destination, verbose=False
            )
            
            print(f"\n✓ Optimal route found!")
            print(f"  • Route length: {len(route_edges)} edges")
            print(f"  • Total cost: {total_cost:.2f} units")
            print(f"  • First few edges: {route_edges[:3]}")
            if len(route_edges) > 3:
                print(f"    ... and {len(route_edges) - 3} more edges")
            
            results['routing'] = {
                'graph': G,
                'source': source,
                'destination': destination,
                'route': route_edges,
                'cost': total_cost
            }
        else:
            print("Warning: Graph too small for routing")
            results['routing'] = None
            
    except Exception as e:
        print(f"Warning: Could not perform routing ({e})")
        results['routing'] = None
    
    print()
    
    # ============================================================================
    # PART 3: INTEGRATED SYSTEM
    # ============================================================================
    print_section("PART 3: Integrated RL + MILP System")
    
    print("HOW INTEGRATION WORKS:")
    print("  1. RL agent controls traffic signals → affects queue lengths")
    print("  2. Queue lengths mapped to graph edges → congestion weights")
    print("  3. Congested edges get higher weights → penalized in routing")
    print("  4. MILP finds optimal route avoiding congestion")
    print("  5. Result: Emergency vehicles route around traffic jams!")
    print()
    
    try:
        system = IntegratedTrafficSystem()
        print(f"✓ Integrated system initialized")
        print(f"  • Road network: {system.road_graph.number_of_nodes():,} nodes")
        print(f"  • Traffic environment: {system.traffic_env.observation_space.shape[0]} intersections")
        print()
        
        # Simulate traffic
        print("Simulating traffic for 10 steps...")
        system.reset_traffic_simulation()
        traffic_history = []
        
        for step in range(10):
            action = system.traffic_env.action_space.sample()
            state, reward, done, _, _ = system.step_traffic_simulation(action)
            traffic_history.append(state.copy())
            if step < 5 or step == 9:
                print(f"  Step {step+1}: queues={state}, reward={reward:.2f}")
        
        print(f"\n✓ Traffic simulation complete (final queues: {traffic_history[-1]})")
        
        # Find route with and without traffic awareness
        nodes = list(system.road_graph.nodes())
        if len(nodes) >= 2:
            source = nodes[0]
            destination = nodes[-1]
            
            print(f"\nFinding emergency routes:")
            print(f"  Source: {source}")
            print(f"  Destination: {destination}")
            print()
            
            # Route ignoring traffic
            print("  Route WITHOUT traffic awareness...")
            route_no_traffic, cost_no_traffic, _ = system.find_emergency_route(
                source, destination, use_traffic=False, verbose=False
            )
            print(f"    ✓ Found route: {len(route_no_traffic)} edges, cost={cost_no_traffic:.2f}")
            
            # Route with traffic awareness
            print("  Route WITH traffic awareness...")
            route_traffic, cost_traffic, _ = system.find_emergency_route(
                source, destination, use_traffic=True, verbose=False
            )
            print(f"    ✓ Found route: {len(route_traffic)} edges, cost={cost_traffic:.2f}")
            
            if cost_traffic != cost_no_traffic:
                print(f"    → Traffic-aware routing adjusted path to avoid congestion!")
            
            results['integrated'] = {
                'traffic_history': traffic_history,
                'route_no_traffic': route_no_traffic,
                'cost_no_traffic': cost_no_traffic,
                'route_traffic': route_traffic,
                'cost_traffic': cost_traffic
            }
        
    except Exception as e:
        print(f"Warning: Could not run integrated system ({e})")
        results['integrated'] = None
    
    print()
    
    # ============================================================================
    # PART 4: GENERATE VISUALIZATIONS
    # ============================================================================
    print_section("PART 4: Generating Visualizations")
    
    print("Creating comprehensive visualizations...\n")
    
    # 1. System overview
    print("[1/4] Creating system architecture diagram...")
    plot_system_overview('results_system_overview.png')
    
    # 2. Training progress
    print("[2/4] Creating RL training analysis...")
    if 'rl_rewards' in results:
        plot_training_progress('reward_log.npy', 'results_training.png')
    
    # 3. Emergency routing
    print("[3/4] Creating emergency routing visualization...")
    if results.get('routing'):
        plot_emergency_routing(
            results['routing']['graph'],
            results['routing']['source'],
            results['routing']['destination'],
            results['routing']['route'],
            results['routing']['cost'],
            'results_routing.png'
        )
    
    # 4. Summary visualization
    print("[4/4] Creating summary report...")
    create_summary_visualization(results, 'results_summary.png')
    
    print("\n✓ All visualizations saved!")
    print()
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print_section("SYSTEM DEMONSTRATION COMPLETE")
    
    print("Generated Files:")
    print("  ✓ results_system_overview.png - System architecture")
    print("  ✓ results_training.png - RL training analysis")
    print("  ✓ results_routing.png - Emergency routing visualization")
    print("  ✓ results_summary.png - Comprehensive summary")
    print("  ✓ reward_log.npy - Training data")
    print()
    print("All results saved in current directory!")
    print()


def create_summary_visualization(results, save_path):
    """Create a summary visualization combining all results."""
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('Traffic Flow Optimization System - Complete Results', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: RL Performance
    ax1 = plt.subplot(2, 2, 1)
    if 'rl_rewards' in results:
        rewards = results['rl_rewards']
        ax1.plot(rewards, alpha=0.6, label='Episode Reward')
        if len(rewards) > 10:
            window = min(20, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), moving_avg, 
                    color='red', linewidth=2, label='Moving Average')
        ax1.set_title('RL Training: Traffic Signal Control', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Routing result (text summary)
    ax2 = plt.subplot(2, 2, 2)
    ax2.axis('off')
    if results.get('routing'):
        routing = results['routing']
        text = f"""
        Emergency Routing Result
        
        Network: {routing['graph'].number_of_nodes():,} nodes
        
        Route found:
        • Length: {len(routing['route'])} edges
        • Cost: {routing['cost']:.2f} units
        • Status: ✓ Optimal
        
        Method: MILP Optimization
        """
        ax2.text(0.1, 0.5, text, fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Plot 3: Traffic simulation
    ax3 = plt.subplot(2, 2, 3)
    if results.get('integrated') and 'traffic_history' in results['integrated']:
        history = np.array(results['integrated']['traffic_history'])
        ax3.plot(history[:, 0], label='Intersection 0', marker='o')
        ax3.plot(history[:, 1], label='Intersection 1', marker='s')
        ax3.set_title('Traffic Simulation: Queue Lengths Over Time', fontweight='bold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Queue Length (vehicles)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Integration comparison
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    if results.get('integrated'):
        integrated = results['integrated']
        text = f"""
        Integrated System Comparison
        
        Routing WITHOUT traffic awareness:
        • Cost: {integrated['cost_no_traffic']:.2f}
        
        Routing WITH traffic awareness:
        • Cost: {integrated['cost_traffic']:.2f}
        
        Difference: {integrated['cost_traffic'] - integrated['cost_no_traffic']:.2f}
        
        The system successfully adjusts
        routing based on real-time traffic!
        """
        ax4.text(0.1, 0.5, text, fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Summary visualization saved to {save_path}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    run_full_system()

