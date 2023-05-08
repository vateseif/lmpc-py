import torch
import numpy as np
from agent import LMPCAgent, NNAgent
from pettingzoo.mpe import simple_tag_v2

torch.manual_seed(2023)

# params
n_episodes = 5
n_frames_per_episode = 500
n_eval_episodes = 5
num_adversaries = 1
num_obstacles = 2
input_size = 4+2*num_obstacles+2*num_adversaries
output_size = 4


# init env
env = simple_tag_v2.parallel_env(num_adversaries=num_adversaries, num_obstacles=num_obstacles, max_cycles=n_frames_per_episode, continuous_actions=True, render_mode='human')
env.reset()

# init LMPC agents (adversaries) and learning-based agent
nn_agent = NNAgent(input_size, output_size, checkpoint="models/tag/best_model.pth")
adversary_agents = LMPCAgent(env.agents[:-1], "tag")


for i in range(n_episodes):
  obs = env.reset()
  while env.agents:
    actions = adversary_agents.act(obs)
    action, log_prob = nn_agent(obs["agent_0"])
    actions["agent_0"] = np.concatenate(([1e-5], action), dtype=np.float32)
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
