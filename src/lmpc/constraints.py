import numpy as np
import cvxpy as cp
from typing import List

from lmpc.core import Constraint, LocalityModel

class LMPCConstraint(Constraint):

  def compute(self, phi: cp.Variable):
    pass


class SLSConstraint(LMPCConstraint):
  def __init__(self, ZAB: np.ndarray, Nx: int, T: int) -> None:
    self.ZAB = ZAB
    self.rhs = np.eye(Nx*(T+1))

  def compute(self, phi: cp.Variable):
    return [self.ZAB @ phi == self.rhs]


class LocalityConstraint(LMPCConstraint):
  def __init__(self, 
              T: int,
              N: int,
              Nx: int,
              Nu: int,
              Ns: List[int],
              Na: List[int],
              id_x: List[List[int]],
              id_u: List[List[int]],
              locality: LocalityModel) -> None:
    # TODO: define a typevar Node instead of using int for states
    self.T = T
    self.N = N
    self.Nx = Nx
    self.Nu = Nu
    self.Ns = Ns
    self.Na = Na
    self.locality_model = locality
    # rows that correspond to each subsystem of Phi for states and actions
    self.rx = [ sum([[Nx*t + x for x in ix] for t in range(T+1)], []) for ix in id_x ]
    self.ru = [ sum([[Nu*t + u for u in iu] for t in range(T)], []) for iu in id_u ]
    pass

  def compute(self, phi: cp.Variable):
    # dimensions
    T, N, Nx, Ns, Na = self.T, self.N, self.Nx, self.Ns, self.Na
    # out-going sets
    self.locality_model._updateOutgoingSets()
    out_x = self.locality_model.out_x
    out_u = self.locality_model.out_u
    # separate x and u
    phi_x = phi[:Nx*(T+1)]
    phi_u = phi[Nx*(T+1):]
    # locality constraints
    constraints = []
    for i in range(N):
      for j in range(N):
        if i not in out_x[j]:
          constraints += [phi_x[self.rx[i]][:, self.rx[j]] == np.zeros((Ns[i]*(T+1), Ns[j]*(T+1)))]
        if i not in out_u[j]:
          constraints += [phi_u[self.ru[i]][:, self.rx[j]] == np.zeros((Na[i]*T, Ns[j]*(T+1)))]
    return constraints