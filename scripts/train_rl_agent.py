# scripts/train_rl_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.traffic_env import TrafficEnv

# ---- DQN Model ----
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.model(x)

# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

# ---- Training Setup ----
env = TrafficEnv()
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
model = DQN(obs_dim, n_actions)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
buffer = ReplayBuffer()
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32
n_episodes = 200

# ---- Action Selection ----
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return torch.argmax(model(torch.FloatTensor(state))).item()

# ---- Main Training Loop ----
all_rewards = []

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

            # Training step
        if len(buffer.buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            # Convert to numpy arrays first for better performance
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))
            
            q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_vals = model(next_states).max(1)[0]
            targets = rewards + gamma * next_q_vals * (1 - dones)
            
            loss = loss_fn(q_vals, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    all_rewards.append(total_reward)
    if ep % 20 == 0:
        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# ---- Save rewards for later visualization ----
np.save("reward_log.npy", all_rewards)
print("Training complete!")
