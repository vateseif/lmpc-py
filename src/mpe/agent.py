import lmpc
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from typing import Optional, List, Dict

envs = ["simple", "tag"]

class LMPCAgent:

  def __init__(self, names: List[str], env: str, locality: Optional[str]=None) -> None:
    assert env in envs, f"env {env} not in {envs}"
    self.N = len(names)
    self.names = names
    self.env = env
    self.locality = locality
    
    self.Ns = 4
    self.Na = 4
    self.T = 5
    self.dt = 0.1              # TODO get val from env
    self.tau = 0.25            # TODO get val from env
    self.eps = 1e-3
    self.size = 0.075
    self.max_speed = 1. if env=="tag" else None
    self.sensitivity = 3. if env=="tag" else 5.

    # system model
    self.sys = self._init_model()
    
    # controller
    self.controller = self._init_controller()

  def _init_model(self):
    # dynamics of 1 single agent
    A1 = np.array([[0, 0, 1, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, -self.tau/self.dt, 0], 
                  [0, 0, 0, -self.tau/self.dt]])
    B1 = np.array([[0, 0, 0, 0], 
                  [0, 0, 0, 0], 
                  [1, -1, 0, 0], 
                  [0, 0, 1, -1]]) 
    # dynamics of n agents
    A = np.kron(np.eye(self.N), A1)
    B = np.kron(np.eye(self.N), B1)
    # discrete dynamics
    Ad = np.eye(A.shape[0]) + A*self.dt
    Bd = B*self.dt*self.sensitivity
    
    # init sys
    sys = lmpc.DistributedLTI(self.N, self.Ns, self.Na)
    sys.loadAB(Ad, Bd)

    # locality model
    if self.locality != None:
      locality = None
      sys << locality
      pass
    return sys

  def _init_controller(self):
    # controller
    controller = lmpc.LMPC(self.T)
    controller << self.sys
    
    # box constraints control inputs
    controller.addConstraint(lmpc.BoundConstraint('u', 'upper', (1-self.eps)*np.ones((self.sys.Nu,1))))
    controller.addConstraint(lmpc.BoundConstraint('u', 'lower', self.eps * np.ones((self.sys.Nu,1))))

    if self.env == "simple":
      # objective
      controller.addObjectiveFun(lmpc.objectives.QuadForm(np.eye(self.sys.Nx)*0, np.eye(self.sys.Nu)))
      # terminal constraint
      xT = np.zeros((2,1))
      G = np.concatenate((np.eye(2), np.zeros((2,2))), axis=1)
      G = np.kron(np.eye(self.N), G)
      controller.addConstraint(lmpc.TerminalConstraint(xT, G))
    elif self.env == "tag":
      # objective
      G = np.concatenate((np.eye(2), np.zeros((2,2))), axis=1)
      G = np.kron(np.eye(self.N), G)
      Q = np.eye(self.N * 2) # position x and y for each agent
      controller.addObjectiveFun(lmpc.objectives.TerminalQuadForm(Q, np.zeros((self.N*2,1)), G))
    else:
      raise(Exception)

    controller._setupSolver(np.zeros((self.sys.Nx, 1)))

    return controller


  def act(self, obs):

    if self.env == "simple":
      # TODO
      pass
    else:
      x = np.zeros((0,1))
      xTd = np.zeros((0,1))
      for agent_name in self.names:
        p = np.array([[obs[agent_name][2]], [obs[agent_name][3]]])
        dp = np.array([[obs[agent_name][0]], [obs[agent_name][1]]])
        x = np.concatenate((x, p, dp))
        xTd = np.concatenate((xTd, p + np.array([[obs[agent_name][-4]], [obs[agent_name][-3]]])))
      
      self.controller.objectives[0].xTd.value = xTd
      u, _, _ = self.controller.solve(x, "SCS")
      action = {name: np.concatenate(([0], u.squeeze()[i*self.Na:(i+1)*self.Na]), dtype=np.float32) for i, name in enumerate(self.names)}
      
    return action

class GaussianPolicy(nn.Module):

  def __init__(self, in_size: int, hidden_size:int, out_size: int) -> None:
    super().__init__()
    self.linear = nn.Sequential(*[
      nn.Linear(in_size, 256),
      nn.ReLU(),
      nn.Linear(256, hidden_size),
      nn.ReLU(),
    ])
    self.mean = nn.Linear(hidden_size, out_size)
    self.log_std = nn.Linear(hidden_size, out_size)
    self.value = nn.Linear(hidden_size, 1)

  def forward(self, inputs):
    # forward pass of NN
    x = inputs
    x = self.linear(x)
    mean = self.mean(x)
    log_std = self.log_std(x) # if more than one action this will give you the diagonal elements of a diagonal covariance matrix
    log_std = torch.clamp(log_std, min=-2, max=20) # We limit the variance by forcing within a range of -0.2,0.2
    std = log_std.exp()
    value = self.value(x)
    return mean, std, value


class NNAgent(nn.Module):
  def __init__(self, input_size: int, output_size: int, gamma=0.99, lr_pi=3e-4, checkpoint:Optional[str]=None) -> None:
    super().__init__()

    self.gamma = gamma
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.policy = GaussianPolicy(input_size, 1024, output_size).to(self.device)
    self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr_pi)

    if checkpoint is not None:
      self.policy.load_state_dict(torch.load(checkpoint)["model_state_dict"])

    
    
  def forward(self, obs: np.ndarray):
    obs = torch.from_numpy(obs).float().to(self.device)
    mu, std, value = self.policy(obs) 
    # init normal distribution
    normal = Normal(mu, std)
    # sample action and compute log probability
    sample = normal.sample()
    log_prob = normal.log_prob(sample).sum()
    action = torch.sigmoid(sample) # bound actions from 0 to 1

    return action.cpu().numpy(), log_prob.sum(), value


  def train(self, trajectory: List[Dict[str,np.ndarray]]):

    states = [i["state"] for i in trajectory]
    values = [i["value"] for i in trajectory]
    actions = [i["action"] for i in trajectory]
    rewards = [i["reward"] for i in trajectory]
    log_probs = [i["log_prob"] for i in trajectory]
    
    #calculate rewards to go
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + self.gamma * R
        returns.insert(0, R)

    eps = np.finfo(np.float32).eps.item()
    returns = torch.tensor(returns).to(self.device)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    policy_loss = []
    value_loss = []
    for log_prob, value, R in zip(log_probs, values, returns):
      policy_loss.append( - log_prob * R)

      # calculate critic (value) loss using L1 smooth loss
      value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # A2C loss: sum of policy and value loss
    loss = torch.stack(policy_loss).sum().to(self.device) + torch.stack(value_loss).sum().to(self.device)
    # update policy weights
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return loss