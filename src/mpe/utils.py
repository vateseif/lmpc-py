import torch
import numpy as np
from datetime import datetime


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

class SaveBestModel:
  """
  Class to save the best model while training. If the current epoch's 
  average reward is more than the previous highest, then save the
  model state.
  """
  def __init__(self, best_avg_reward=-float('inf')):
    self.best_avg_reward = best_avg_reward
      
  def __call__(self, current_avg_reward:float, epoch:int, model:torch.nn.Module, optimizer):
    if current_avg_reward > self.best_avg_reward:
      self.best_avg_reward = current_avg_reward
      print(f"\nBest validation loss: {self.best_avg_reward}")
      print(f"\nSaving best model for epoch: {epoch+1}\n")
      torch.save({
          'epoch': epoch+1,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          #'loss': criterion,
          }, f'models/tag/best_{datetime.now()}.pth')
  