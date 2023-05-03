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
agent.model.setInitialState(np.concatenate((p, dp)))

for i, ag in enumerate(env.agent_iter()):
    observation, reward, termination, truncation, info = env.last()
    # update terminal constraint
    xT = np.array([[observation[2]], [observation[3]]]) + agent.model._x[:2]
    agent.constraints[-1].xTd.value = xT  
    
    if termination or truncation:
        break
    else:
        #action = env.action_space(ag).sample()
        u, _, Phi = agent.solve(agent.model._x, solver="MOSEK")
        agent.model.updateState(agent.model.step(u))
        action = np.concatenate(([0], u.squeeze()), dtype=np.float32)
    
    env.step(action)


env.close()