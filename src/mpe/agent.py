import lmpc
import numpy as np

from typing import Optional

envs = ["simple", "tag"]

class Agent:

  def __init__(self, N: int, env: str, locality: Optional[str]=None) -> None:
    assert env in envs, f"env {env} not in {envs}"
    self.N = N
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
    Ad = np.eye(self.Ns) + A*self.dt
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
      p = np.array([[obs[2]], [obs[3]]])
      dp = np.array([[obs[0]], [obs[1]]])
      x = np.concatenate((p, dp))
      xTd = np.array([[obs[4]], [obs[5]]])
      self.controller.objectives[0].xTd.value = xTd
      u, _, _ = self.controller.solve(x, "SCS")
      action = np.concatenate(([0], u.squeeze()), dtype=np.float32)
    
    return action


