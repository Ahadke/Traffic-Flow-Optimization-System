from scripts.traffic_env import TrafficEnv

env = TrafficEnv()
state, _ = env.reset()
print("Initial state:", state)

for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    print("Step state:", state, "Reward:", reward)
