from envs import sequential_reference

env = sequential_reference.parallel_env(num_agents=2, max_cycles=1000)
observations = env.reset()

while env.agents:
    env.render()
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  
    
    observations, rewards, terminations, infos = env.step(actions)
env.close()