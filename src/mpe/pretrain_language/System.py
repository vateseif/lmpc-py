import os

import torch
import numpy as np

from Agent import Agent, device


class System:
  def __init__(self, dims) -> None:
    # init agent
    self.agents = {agent_id: Agent(dims[agent_id]) for agent_id in list(dims.keys())}

  def select_action(self, obs: np.ndarray):
    actions = {}
    for agent_id, o in obs.items():
      o = torch.from_numpy(o).unsqueeze(0).float()
      actions[agent_id] = self.agents[agent_id].action(o)
    return actions

  @classmethod
  def load(cls, dims, file):
    """init maddpg using the model saved in `file`"""
    instance = cls(dims)
    data = torch.load(file, map_location=torch.device(device))
    for _, agent in instance.agents.items():
      agent.speaker.load_state_dict(data['speaker'])
      agent.listener.load_state_dict(data['listener'])
    return instance