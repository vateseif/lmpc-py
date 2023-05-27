import numpy as np

from agent import getAgent
from pettingzoo.mpe import simple_v2

"""
Simulation in the simple_v2 environment from pettingzo
"""

# init env
env = simple_v2.env(max_cycles=100, continuous_actions=True, render_mode="human")
env.reset()

# init agent
agent = getAgent()
# initial condition
p = np.expand_dims(env.env.world.policy_agents[0].state.p_pos.copy(), -1)
dp = np.expand_dims(env.env.world.policy_agents[0].state.p_vel.copy(), -1)

for i, ag in enumerate(env.agent_iter()):
    observation, reward, termination, truncation, info = env.last()
    # update terminal constraint
    xT = np.array([[observation[2]], [observation[3]]])
    agent.constraints[-1].xTd.value = xT  
    # initial state
    x0 = np.concatenate((np.ones((2,1))*1e-3, np.array([[observation[0]], [observation[1]]])))

    if termination or truncation:
        break
    else:
        #action = env.action_space(ag).sample()
        u, _, Phi = agent.solve(x0, solver="SCS")
        action = np.concatenate(([1e-3], u.squeeze()), dtype=np.float32)
    
    env.step(action)


env.close()