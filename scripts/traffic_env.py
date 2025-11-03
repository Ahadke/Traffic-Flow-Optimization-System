"""
Custom Gymnasium environment for traffic signal control simulation.

This environment simulates traffic queues at intersections and allows an RL agent
to control traffic signals to minimize congestion.

HOW IT WORKS:
- State: Queue lengths (number of waiting vehicles) at each intersection
- Actions: Select which traffic signal to activate (reduces queue at that intersection)
- Reward: Negative sum of queue lengths (agent wants to minimize total waiting)
- Dynamics: New vehicles arrive (Poisson process), selected signal clears vehicles
"""

import gymnasium as gym
import numpy as np


class TrafficEnv(gym.Env):
    """
    Traffic signal control environment.
    
    Simulates queue dynamics at traffic intersections where an RL agent
    controls which signal gets priority to reduce congestion.
    """
    
    def __init__(self):
        """Initialize the traffic environment."""
        super().__init__()
        
        # Observation space: queue lengths at intersections (0 to 50 vehicles)
        # Shape (2,) means we're simulating 2 intersections
        self.observation_space = gym.spaces.Box(low=0, high=50, shape=(2,))
        
        # Action space: 2 discrete actions (select which intersection to prioritize)
        # Action 0: prioritize intersection 0, Action 1: prioritize intersection 1
        self.action_space = gym.spaces.Discrete(2)
        
        # Initial state: random queue lengths at each intersection (0-20 vehicles)
        self.state = np.random.randint(0, 20, size=2)
        
        # Episode tracking
        self.steps = 0  # Current step in episode
        self.max_steps = 50  # Maximum steps per episode

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Called at start of each training episode.
        Returns: (observation, info_dict)
        """
        # Reset queue lengths to random initial values
        self.state = np.random.randint(0, 20, size=2)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Which intersection to prioritize (0 or 1)
        
        Returns:
            observation: New queue lengths
            reward: Negative sum of queues (we want to minimize this)
            done: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional info dict
        """
        # NEW VEHICLES ARRIVE: Add random vehicles to all queues (Poisson process)
        # This simulates ongoing traffic arriving at intersections
        self.state += np.random.poisson(2, size=2)
        
        # SIGNAL ACTION: The selected action clears vehicles at that intersection
        # Randomly clears 6-16 vehicles from the selected intersection's queue
        # (simulates traffic light allowing vehicles to pass)
        self.state[action] = max(self.state[action] - np.random.randint(6, 16), 0)
        
        # REWARD: Negative total queue length (agent wants to minimize waiting)
        # Lower total queues = higher (less negative) reward
        reward = -np.sum(self.state)
        
        # Update episode step counter
        self.steps += 1
        
        # Episode ends after max_steps
        done = self.steps >= self.max_steps
        
        return self.state, reward, done, False, {}

    def render(self):
        """Display current state (queue lengths)."""
        print(f"Queue lengths: {self.state}")
