import numpy as np
from agent import Agent
from pettingzoo.mpe import simple_tag_v2


env = simple_tag_v2.parallel_env(num_adversaries=3, num_obstacles=3, max_cycles=200, continuous_actions=True, render_mode='human')
obs = env.reset()

agent = Agent(1, "tag")

while env.agents:
  actions = {agent_name: agent.act(obs[agent_name]) if agent_name!="agent_0" else env.action_space(agent_name).sample()  for agent_name in env.agents }  # this is where you would insert your policy
  obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()