import numpy as np
from pettingzoo.mpe import simple_tag_v2


env = simple_tag_v2.env(num_adversaries=1, num_obstacles=0, max_cycles=25, continuous_actions=True, render_mode='human')

env.reset()
for i, agent in enumerate(env.agent_iter()):
    observation, reward, termination, truncation, info = env.last()
    print(agent, i)
    print(env.env.world.agents[int(i%2)].state.p_pos)
    print(env.env.world.agents[int(i%2)].state.p_vel)
    print(observation)
    if termination or truncation:
        break
    else:
        action = env.action_space(agent).sample()
    env.step(np.ones((5,), dtype=np.float32) * 1e-3)

    print("----------------------------------------------")
env.close()