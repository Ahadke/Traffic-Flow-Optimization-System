# scripts/6_realistic_env_init.py

import numpy as np
import pandas as pd
from scripts.traffic_env import TrafficEnv

import os

# Load traffic counts dataset - try multiple path options
data_paths = [
    os.path.join('data', 'dft_traffic_counts_aadf.csv'),
    os.path.join('..', 'data', 'dft_traffic_counts_aadf.csv')
]
aadf = None
for path in data_paths:
    if os.path.exists(path):
        aadf = pd.read_csv(path, low_memory=False)
        break
if aadf is None:
    raise FileNotFoundError(f"Could not find traffic counts CSV. Tried: {data_paths}")

# Function to scale traffic volumes to a queue length range, e.g., 0-20 cars
def scale_to_queue(count_series, max_queue=20):
    max_count = count_series.max()
    scaled = (count_series / max_count) * max_queue
    return scaled.astype(int)

# Sample a few count points to initialize queues
sample_size = 2  # As environment state size is 2 signals
counts_sample = aadf['all_motor_vehicles'].sample(sample_size, random_state=42).reset_index(drop=True)

# Scale counts to queue lengths
initial_queues = scale_to_queue(counts_sample, max_queue=20)

print(f"Initializing environment queue lengths from real counts: {list(initial_queues)}")

# Custom env class allowing custom initial state
class RealisticTrafficEnv(TrafficEnv):
    def __init__(self, initial_state=None):
        super().__init__()
        self.custom_initial_state = initial_state

    def reset(self, seed=None, options=None):
        if self.custom_initial_state is not None:
            self.state = np.array(self.custom_initial_state)
        else:
            self.state = np.random.randint(0, 20, size=2)
        self.step_count = 0
        return self.state, {}

# Create environment with realistic initial queues
env = RealisticTrafficEnv(initial_state=initial_queues.values)

state, _ = env.reset()
print("Environment state after reset:", state)

# Run one simulation episode to test
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
print("Total reward for one episode:", total_reward)
