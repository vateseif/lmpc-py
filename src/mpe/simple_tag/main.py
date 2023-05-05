import numpy as np
from mpe.agent import Agent
from pettingzoo.mpe import simple_tag_v2


env = simple_tag_v2.env(num_adversaries=1, num_obstacles=0, max_cycles=100, continuous_actions=True, render_mode='human')
env.reset()

agent = Agent(1, "tag")

for i, agent_name in enumerate(env.agent_iter()):
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        break
    else:
      if agent_name == "agent_0":
        action = env.action_space(agent_name).sample()
      else:
        action = agent.act(observation)
    
    env.step(action)

env.close()