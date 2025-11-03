"""
Plot training curves for RL agent.

Visualizes reward progression during training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(reward_file='reward_log.npy', save_path=None, show=True):
    """
    Plot RL training reward curve.
    
    Parameters:
    -----------
    reward_file : str
        Path to numpy file containing rewards
    save_path : str, optional
        Path to save the plot. If None, uses 'training_curve.png'
    show : bool
        If True, display the plot
    """
    # Find reward file
    paths_to_try = [reward_file, os.path.join('..', reward_file), reward_file]
    reward_path = None
    
    for path in paths_to_try:
        if os.path.exists(path):
            reward_path = path
            break
    
    if reward_path is None:
        raise FileNotFoundError(f"Could not find reward file: {reward_file}")
    
    rewards = np.load(reward_path)
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, color='blue', label='Episode Reward')
    plt.title('RL Training: Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot with moving average
    plt.subplot(1, 2, 2)
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    if len(rewards) > 10:
        window = min(20, max(5, len(rewards) // 10))
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Average (window={window})')
    
    plt.title('RL Training with Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    elif save_path is None:
        default_path = 'training_curve.png'
        plt.savefig(default_path, dpi=150)
        print(f"Plot saved to {default_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    try:
        plot_training_curve(show=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
