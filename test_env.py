import numpy as np
import sys
sys.path.append('src')

from workload_generator import WorkloadGenerator
from cloud_env import CloudEnvironment

# Load traffic data
gen = WorkloadGenerator()
workload = gen.generate_spike()

# Create environment
env = CloudEnvironment(workload)
state, _ = env.reset()

print("✅ Environment created successfully")
print(f"Initial state: {state}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Run 100 random steps
for i in range(100):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)

print("✅ 100 steps completed without crashing")
print(f"Metrics after 100 steps: {env.get_metrics()}")
print("✅ ALL GOOD - Cloud environment works!")