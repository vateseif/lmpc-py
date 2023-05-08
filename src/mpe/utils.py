import torch
import numpy as np

@torch.no_grad()
def evaluate_policy(agent, adversary, env, eval_episodes = 10):
  agent.policy.eval()
  for _ in range(eval_episodes):
    obs = env.reset()
    avg_reward = 0
    while env.agents:
      actions = adversary.act(obs)
      action, _ = agent(obs["agent_0"])
      actions["agent_0"] = np.concatenate(([1e-5], action), dtype=np.float32)
      next_state, rewards, terminations, truncations, infos = env.step(actions)

      obs = next_state
      avg_reward += rewards["agent_0"]
  
  agent.policy.train()
  avg_reward /= eval_episodes
  return avg_reward
  