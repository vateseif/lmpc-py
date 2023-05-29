from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpc import mpc
from mpc.mpc import QuadCost, LinDx


device = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
  def __init__(self, name, dims) -> None:
    self.name = name
    self.num_landmarks = dims['num_landmarks']
    if name.startswith('speaker') or name.startswith('agent'):
      self.speaker: nn.Module = MLPNetwork(dims["speaker_in"], dims["speaker_out"])
    if name.startswith('listener') or name.startswith('agent'):
      self.listener: nn.Module = MLPNetwork(dims["listener_in"], dims["listener_out"])
      self.controller: nn.Module = MPCLayer()
    
    #self.planner: nn.Module = Planner()
    
    pass

  def action(self, obs: torch.Tensor) -> Tuple[torch.Tensor]:
    batch_size, _ = obs.shape
    
    #speaker_in, xd, x_init = self.planner(obs, listener_out)
    action = torch.empty((batch_size, 0))
    if self.name.startswith('speaker') or self.name.startswith('agent'):
      i1 = (1+self.num_landmarks)*2 # start index of goal_i
      speaker_in = obs[:, i1:i1+3] if self.name.startswith('agent') else obs
      speaker_out = F.gumbel_softmax(self.speaker(speaker_in), hard=True)
      action = torch.cat((action, speaker_out), -1)
    if self.name.startswith('listener') or self.name.startswith('agent'):
      listener_out = self.listener(obs)
      xd = torch.cat((listener_out, torch.zeros(batch_size, 2)), -1)
      x_init = torch.cat((torch.zeros(batch_size, 2).to(device), obs[:, :2]), -1)
      controller_out = self.controller(x_init, xd)
      action = torch.cat((torch.zeros((batch_size, 1)), controller_out, action), 1)
    
    return action.squeeze(0).cpu().detach().numpy()


class MLPNetwork(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
    super(MLPNetwork, self).__init__()

    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        non_linear,
        nn.Linear(hidden_dim, hidden_dim),
        non_linear,
        nn.Linear(hidden_dim, out_dim),
    ).apply(self.init)

  @staticmethod
  def init(m):
    """init parameter of the module"""
    gain = nn.init.calculate_gain('relu')
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.01)

  def forward(self, x):
    return self.net(x)


class Planner(nn.Module):
  def __init__(self) -> None:
    super(Planner, self).__init__()

  def forward(self, obs: torch.Tensor, listener_out: torch.Tensor):
    batch_size, _ = obs.shape
    xd = torch.cat((listener_out, torch.zeros(batch_size, 2)), -1)
    x_init = torch.cat((torch.zeros(batch_size, 2).to(device), obs[:, :2]), -1)
    return obs[:, 8:11], xd, x_init

class MPCLayer(nn.Module):
  def __init__(self) -> None:
    super(MPCLayer, self).__init__()
    self.N = 1    
    self.Ns = 4
    self.Na = 4
    self.T = 5
    self.dt = 0.1              # TODO get val from env
    self.tau = 0.25            # TODO get val from env
    self.eps = 1e-3
    self.size = 0.075
    self.max_speed = None
    self.sensitivity = 5.
    self.LQR_ITER = 100
    self.u_upper = 1.
    self.u_lower = 0.
    self.u_init = None
    self.batch_size = None
    # model dy namics
    self.Dx = None
    
    # cost
    self.cost = None
    # controller
    self.controller = None
  
  def _init_model(self):
    # dynamics of 1 single agent
    A1 = torch.tensor([[0, 0, 1, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, -self.tau/self.dt, 0], 
                  [0, 0, 0, -self.tau/self.dt]])
    B1 = torch.tensor([[0, 0, 0, 0], 
                  [0, 0, 0, 0], 
                  [1, -1, 0, 0], 
                  [0, 0, 1, -1]]) 
    # discrete dynamics
    Ad = torch.eye(A1.shape[0]) + A1*self.dt
    Bd = B1*self.dt*self.sensitivity
    # extend dynamics over horizon and batch size
    A = Ad.repeat(self.T, self.batch_size, 1, 1)
    B = Bd.repeat(self.T, self.batch_size, 1, 1)
    F = torch.cat((A, B), dim=3).to(device)
    return LinDx(F)
  
  def _init_cost(self, xd: torch.Tensor):
    # Quadratic cost
    xd = xd                                                                       # (B, Ns) desired states
    w = torch.tensor([1., 1., 1e-1, 1e-1]).to(device)                             # (Ns,) state weights
    q = torch.cat((w, 1e-1*torch.ones(self.Na).to(device)))                     # (Ns+Na,) state-action weights
    Q = torch.diag(q).repeat(self.T, self.batch_size, 1, 1)                       # (T, B, Ns+Na, Ns+Na) weight matrix
    px = -torch.sqrt(w) * xd                                                                  # (B, Ns) linear cost vector
    p = torch.cat((px, 1e-2*torch.ones((self.batch_size, self.Na)).to(device)), 1)   # (T, B, Ns+Na) linear cost vector for state-action
    p = p.repeat(self.T, 1, 1)
    cost = QuadCost(Q, p)
    return cost
  
  def forward(self, x_init: torch.Tensor, xd: torch.Tensor) -> torch.Tensor:
    batch_size, _ = xd.shape
    # Update dynamics if batch_size changes
    if batch_size != self.batch_size:
      self.batch_size = batch_size
      self.Dx = self._init_model()
      self.u_init = None
    # Init cost wrt to desired states
    self.cost = self._init_cost(xd)
    # recreate controller using updated u_init (kind of wasteful right?)
    ctrl = mpc.MPC(self.Ns, self.Na, self.T, u_lower=self.u_lower, u_upper=self.u_upper, 
                  lqr_iter=self.LQR_ITER, exit_unconverged=False, eps=1e-2,
                  n_batch=self.batch_size, backprop=True, verbose=0, u_init=self.u_init,
                  grad_method=mpc.GradMethods.AUTO_DIFF)
    # solve mpc problem
    _, nominal_actions, _ = ctrl(x_init, self.cost, self.Dx)
    # update u_init for warming starting at next step
    #self.u_init = torch.cat((nominal_actions[1:], torch.zeros(1, self.batch_size, self.Na).to(device)), dim=0)
    return nominal_actions[0]   # (B, Na)
