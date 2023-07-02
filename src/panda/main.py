import panda_gym
import gymnasium as gym
from controller import Controller

# params
n_episodes = 100
max_episode_length = 1000

# env
env = gym.make("PandaReach-v3", render_mode="human")
observation, info = env.reset()
# controller
ctrl = Controller()

for _ in range(n_episodes):
  observation, info = env.reset()
  for _ in range(max_episode_length):
    # update controller
    x0 = observation["observation"][0:3]
    xd = observation["desired_goal"][0:3]
    ctrl.set_x0(x0)
    ctrl.set_xd(xd)
    # solve
    action = ctrl.step()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()