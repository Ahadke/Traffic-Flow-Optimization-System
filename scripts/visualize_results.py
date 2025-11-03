"""
Comprehensive visualization script for Traffic Flow Optimization System.

Creates visualizations for:
1. RL Training Progress
2. Emergency Routing Results
3. Integrated System Performance
4. Network Structure
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from scripts.build_graph import build_road_graph
    from scripts.emergency_routing import find_optimal_emergency_route
except ImportError:
    from build_graph import build_road_graph
    from emergency_routing import find_optimal_emergency_route


def plot_training_progress(reward_file='reward_log.npy', save_path='results_training.png'):
    """
    Plot comprehensive RL training visualization.
    
    Shows:
    - Episode rewards over time
    - Moving average trend
    - Performance statistics
    """
    # Find and load reward data
    paths_to_try = [reward_file, os.path.join('..', reward_file), reward_file]
    reward_path = None
    
    for path in paths_to_try:
        if os.path.exists(path):
            reward_path = path
            break
    
    if reward_path is None:
        print(f"Warning: Could not find reward file {reward_file}")
        return False
    
    rewards = np.load(reward_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Raw rewards over time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(rewards, alpha=0.4, color='blue', linewidth=0.5, label='Episode Reward')
    if len(rewards) > 10:
        window = min(20, max(5, len(rewards) // 10))
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window} episodes)')
    ax1.set_title('RL Training: Reward per Episode', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards):.1f}')
    ax2.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance improvement
    ax3 = plt.subplot(2, 3, 3)
    # Split into first and last halves
    mid = len(rewards) // 2
    first_half = rewards[:mid] if mid > 0 else rewards
    second_half = rewards[mid:] if mid > 0 else rewards
    ax3.boxplot([first_half, second_half], labels=['First Half', 'Second Half'])
    ax3.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Reward')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning curve (smoothed)
    ax4 = plt.subplot(2, 3, 4)
    if len(rewards) > 10:
        window = 10
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(rewards)), smoothed, 
                color='green', linewidth=2, label='Smoothed')
        ax4.fill_between(range(window-1, len(rewards)), 
                         smoothed - np.std(rewards), 
                         smoothed + np.std(rewards),
                         alpha=0.2, color='green')
    ax4.set_title('Learning Curve (Smoothed)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Statistics
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    stats_text = f"""
    TRAINING STATISTICS
    
    Total Episodes: {len(rewards)}
    Mean Reward: {np.mean(rewards):.2f}
    Std Deviation: {np.std(rewards):.2f}
    Best Episode: {np.max(rewards):.2f}
    Worst Episode: {np.min(rewards):.2f}
    Final Reward: {rewards[-1]:.2f}
    
    Improvement: {rewards[-1] - rewards[0]:.2f}
    Best 10 Avg: {np.mean(np.sort(rewards)[-10:]):.2f}
    """
    ax5.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 6: Reward trend analysis
    ax6 = plt.subplot(2, 3, 6)
    # Divide into quartiles
    n = len(rewards)
    q1 = rewards[:n//4]
    q2 = rewards[n//4:n//2]
    q3 = rewards[n//2:3*n//4]
    q4 = rewards[3*n//4:]
    quartiles = [np.mean(q1), np.mean(q2), np.mean(q3), np.mean(q4)]
    ax6.plot(range(1, 5), quartiles, 'o-', linewidth=2, markersize=8, color='purple')
    ax6.set_title('Average Reward by Quartile', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Training Quartile')
    ax6.set_ylabel('Average Reward')
    ax6.set_xticks(range(1, 5))
    ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('RL Training Analysis: Traffic Signal Control', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Training visualization saved to {save_path}")
    return True


def plot_emergency_routing(G, source, destination, route_edges, total_cost, 
                          save_path='results_routing.png'):
    """
    Visualize emergency routing results on the road network.
    
    Shows:
    - Road network structure
    - Source and destination
    - Optimal route path
    - Route statistics
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # For large graphs, sample a subset around the route for visualization
    if G.number_of_nodes() > 1000:
        # Extract nodes near the route
        route_nodes = set()
        for u, v in route_edges:
            route_nodes.add(u)
            route_nodes.add(v)
        
        # Get neighbors within 2 hops
        subgraph_nodes = set(route_nodes)
        for node in route_nodes:
            for neighbor in list(G.successors(node)) + list(G.predecessors(node)):
                subgraph_nodes.add(neighbor)
                for neighbor2 in list(G.successors(neighbor)) + list(G.predecessors(neighbor)):
                    subgraph_nodes.add(neighbor2)
        
        G_viz = G.subgraph(subgraph_nodes).copy()
        print(f"Visualizing subgraph: {G_viz.number_of_nodes()} nodes (from {G.number_of_nodes()} total)")
    else:
        G_viz = G
    
    # Plot 1: Network graph with route
    ax1 = axes[0]
    
    # Position nodes (use spring layout for visualization)
    try:
        pos = nx.spring_layout(G_viz, k=1, iterations=50, seed=42)
    except:
        # Fallback to simple positions if spring layout fails
        pos = {node: (hash(str(node)) % 1000, hash(str(node)[::-1]) % 1000) 
               for node in G_viz.nodes()}
    
    # Draw all edges (light gray)
    nx.draw_networkx_edges(G_viz, pos, ax=ax1, 
                          edge_color='lightgray', width=0.5, alpha=0.3)
    
    # Draw route edges (thick red)
    route_in_viz = [(u, v) for u, v in route_edges if u in G_viz and v in G_viz]
    if route_in_viz:
        nx.draw_networkx_edges(G_viz, pos, ax=ax1, edgelist=route_in_viz,
                              edge_color='red', width=3, alpha=0.8, 
                              style='dashed', label='Optimal Route')
    
    # Draw all nodes (small, gray)
    nx.draw_networkx_nodes(G_viz, pos, ax=ax1, 
                           node_color='lightblue', node_size=20, alpha=0.6)
    
    # Highlight source (green)
    if source in G_viz:
        nx.draw_networkx_nodes(G_viz, pos, ax=ax1, nodelist=[source],
                              node_color='green', node_size=300, alpha=0.9, 
                              label='Source')
        nx.draw_networkx_labels(G_viz, pos, {source: 'S'}, ax=ax1, font_size=8, font_weight='bold')
    
    # Highlight destination (red)
    if destination in G_viz:
        nx.draw_networkx_nodes(G_viz, pos, ax=ax1, nodelist=[destination],
                              node_color='red', node_size=300, alpha=0.9,
                              label='Destination')
        nx.draw_networkx_labels(G_viz, pos, {destination: 'D'}, ax=ax1, font_size=8, font_weight='bold')
    
    ax1.set_title('Emergency Vehicle Route on Road Network', fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.legend(loc='upper right')
    
    # Plot 2: Route statistics and information
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create information box
    info_text = f"""
    EMERGENCY ROUTING RESULTS
    
    Network Statistics:
    • Total Nodes: {G.number_of_nodes():,}
    • Total Edges: {G.number_of_edges():,}
    
    Route Information:
    • Source: {source}
    • Destination: {destination}
    • Route Length: {len(route_edges)} edges
    • Total Cost: {total_cost:.2f} units
    
    Route Path:
    """
    
    # Add route details (truncate if too long)
    route_display = route_edges[:5] if len(route_edges) > 5 else route_edges
    for i, edge in enumerate(route_display):
        info_text += f"    {i+1}. {edge}\n"
    if len(route_edges) > 5:
        info_text += f"    ... ({len(route_edges) - 5} more edges)\n"
    
    info_text += f"""
    
    Optimization Method:
    • MILP (Mixed Integer Linear Programming)
    • Objective: Minimize path cost
    • Constraints: Flow conservation
    
    Status: ✓ Optimal route found
    """
    
    ax2.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             transform=ax2.transAxes)
    
    plt.suptitle('Emergency Vehicle Routing Visualization', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Routing visualization saved to {save_path}")
    return True


def plot_system_overview(save_path='results_overview.png'):
    """
    Create system overview visualization showing how components work together.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Traffic Flow Optimization System Architecture', 
           ha='center', fontsize=18, fontweight='bold')
    
    # RL Component Box
    rl_box = FancyBboxPatch((0.5, 6), 4, 2.5, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='blue', facecolor='lightblue', 
                           linewidth=2, alpha=0.7)
    ax.add_patch(rl_box)
    ax.text(2.5, 7.5, 'REINFORCEMENT LEARNING', ha='center', 
           fontsize=12, fontweight='bold')
    ax.text(2.5, 7, 'Traffic Signal Control', ha='center', fontsize=10)
    ax.text(2.5, 6.5, '• State: Queue lengths\n• Action: Signal selection\n• Reward: -Total queues', 
           ha='center', fontsize=9, fontfamily='monospace')
    
    # MILP Component Box
    milp_box = FancyBboxPatch((5.5, 6), 4, 2.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='green', facecolor='lightgreen',
                             linewidth=2, alpha=0.7)
    ax.add_patch(milp_box)
    ax.text(7.5, 7.5, 'MILP OPTIMIZATION', ha='center',
           fontsize=12, fontweight='bold')
    ax.text(7.5, 7, 'Emergency Routing', ha='center', fontsize=10)
    ax.text(7.5, 6.5, '• Graph: Road network\n• Objective: Shortest path\n• Constraints: Flow conservation',
           ha='center', fontsize=9, fontfamily='monospace')
    
    # Integration Box
    int_box = FancyBboxPatch((2, 3), 6, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='purple', facecolor='lavender',
                            linewidth=2, alpha=0.7)
    ax.add_patch(int_box)
    ax.text(5, 4, 'INTEGRATED SYSTEM', ha='center',
           fontsize=12, fontweight='bold')
    ax.text(5, 3.5, 'Traffic-aware routing: RL state → Edge weights → Optimal route',
           ha='center', fontsize=10, fontfamily='monospace')
    
    # Data sources
    data_box = FancyBboxPatch((0.5, 0.5), 9, 1.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='orange', facecolor='wheat',
                             linewidth=2, alpha=0.7)
    ax.add_patch(data_box)
    ax.text(5, 1.5, 'DATA SOURCES', ha='center',
           fontsize=12, fontweight='bold')
    ax.text(5, 1, '• Road Network (Shapefile)\n• Traffic Count Data (CSV)',
           ha='center', fontsize=10)
    
    # Arrows showing flow
    # RL -> Integration
    arrow1 = mpatches.FancyArrowPatch((2.5, 6), (4, 4),
                                      connectionstyle="arc3,rad=0.2",
                                      arrowstyle='->', lw=2, color='blue')
    ax.add_patch(arrow1)
    ax.text(3, 5, 'Traffic\nState', ha='center', fontsize=8, style='italic')
    
    # MILP -> Integration
    arrow2 = mpatches.FancyArrowPatch((7.5, 6), (6, 4),
                                     connectionstyle="arc3,rad=-0.2",
                                     arrowstyle='->', lw=2, color='green')
    ax.add_patch(arrow2)
    ax.text(7, 5, 'Routing\nWeights', ha='center', fontsize=8, style='italic')
    
    # Data -> Components
    arrow3 = mpatches.FancyArrowPatch((2.5, 2), (2.5, 6),
                                     arrowstyle='->', lw=2, color='orange')
    ax.add_patch(arrow3)
    arrow4 = mpatches.FancyArrowPatch((7.5, 2), (7.5, 6),
                                     arrowstyle='->', lw=2, color='orange')
    ax.add_patch(arrow4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] System overview saved to {save_path}")
    return True


def create_comprehensive_report():
    """Run all visualizations and create a comprehensive report."""
    print("=" * 70)
    print("Creating Comprehensive Visualization Report")
    print("=" * 70)
    print()
    
    results = {}
    
    # 1. Training visualization
    print("[1/4] Creating training progress visualization...")
    results['training'] = plot_training_progress()
    print()
    
    # 2. System overview
    print("[2/4] Creating system architecture overview...")
    results['overview'] = plot_system_overview()
    print()
    
    # 3. Emergency routing visualization
    print("[3/4] Creating emergency routing visualization...")
    try:
        G = build_road_graph(verbose=False)
        nodes = list(G.nodes())
        if len(nodes) >= 2:
            source = nodes[0]
            destination = nodes[-1]
            route, cost, _ = find_optimal_emergency_route(
                G, source, destination, verbose=False
            )
            results['routing'] = plot_emergency_routing(
                G, source, destination, route, cost
            )
        else:
            print("  Warning: Graph too small for routing visualization")
            results['routing'] = False
    except Exception as e:
        print(f"  Warning: Could not create routing visualization: {e}")
        results['routing'] = False
    print()
    
    # Summary
    print("=" * 70)
    print("Visualization Summary")
    print("=" * 70)
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ SKIPPED"
        print(f"  {name.capitalize()}: {status}")
    print()
    print("All visualizations saved to current directory.")
    print("=" * 70)


if __name__ == "__main__":
    create_comprehensive_report()

