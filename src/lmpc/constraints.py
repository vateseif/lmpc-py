import numpy as np
import cvxpy as cp
from typing import List, Optional

from src.lmpc.core import Constraint, LocalityModel

class LMPCConstraint(Constraint):

  def compute(self, T:int, x0:np.ndarray, phi: cp.Variable):
    pass


class SLSConstraint(LMPCConstraint):
  def __init__(self, ZAB: np.ndarray, Nx: int) -> None:
    self.Nx = Nx
    self.ZAB = ZAB

  def compute(self, T:int, x0:np.ndarray, phi: cp.Variable):
    Nx = self.Nx
    # check if w or only x0 was passed
    if x0.shape[0] == Nx*(T+1):
      rhs = np.eye(Nx*(T+1))
    else:
      rhs = np.concatenate((np.eye(Nx), np.zeros((Nx*T, Nx))), axis=0)
    return [self.ZAB @ phi == rhs]


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
    self.rx = id_x
    self.ru = [[Nx*(T+1)+i for i in iu] for iu in id_u]
    # rows that correspond to each subsystem of Phi for states and actions along horizon
    self.rxT = [ sum([[Nx*t + x for x in ix] for t in range(T+1)], []) for ix in id_x ]
    self.ruT = [ sum([[Nx*(T+1)+Nu*t + u for u in iu] for t in range(T)], []) for iu in id_u ]
    
    pass

  def compute(self, T:int, x0:np.ndarray, phi: cp.Variable):
    # dimensions
    T, N, Nx, Ns, Na = self.T, self.N, self.Nx, self.Ns, self.Na
    # out-going sets
    self.locality_model._updateOutgoingSets() # in this case it does nothing
    out_x = self.locality_model.out_x
    out_u = self.locality_model.out_u
    # index of states per subsytem depending on x0 or w
    if x0.shape[0] == Nx:
      rx = self.rx
      ru = self.ru
      Tc = 1
    else:
      rx = self.rxT
      ru = self.ruT
      Tc = T+1
    # locality constraints
    constraints = []
    for i in range(N):
      for j in range(N):
        if i not in out_x[j]:
          constraints += [phi[np.ix_(rx[i], rx[j])] == np.zeros((Ns[i]*Tc, Ns[j]*Tc))]
        if i not in out_u[j]:
          constraints += [phi[np.ix_(ru[i], rx[j])] == np.zeros((Na[i]*Tc, Ns[j]*Tc))]
    return constraints

class TerminalConstraint(LMPCConstraint):
  def __init__(self, xT: Optional[np.ndarray] = None) -> None:
    self.xT = xT

  def compute(self, T:int, x0:np.ndarray, phi: cp.Variable):
    Nx = x0.shape[0]
    if self.xT is None:
      self.xT = np.zeros((Nx, 1))
    return [phi[Nx*T:Nx*(T+1)] @ x0 == self.xT]

class LowerTriangulatConstraint(LMPCConstraint):
  # TODO: passing Nx and Nu each time is not efficient
  def __init__(self, Nx:int, Nu:int) -> None:
    self.Nx = Nx
    self.Nu = Nu
    pass

  def compute(self, T: int, x0: np.ndarray, phi: cp.Variable) -> List:
    ''' Enforce phi_x, phi_u lower triangular if w is given and not only x0'''
    Nx, Nu = self.Nx, self.Nu
    constraints = []
    # if x0 only given then skip
    if x0.shape[0] == Nx*(T+1):
      phi_x = phi[:Nx*(T+1)]
      phi_u = phi[Nx*(T+1):]
      for i in range(T-1):
        constraints += [phi_x[Nx*i:Nx*(i+1), Nx*(i+1):] == np.zeros((Nx, Nx*(T-i)))]
        if i < T-2: # u has 1 less time step
          constraints += [phi_u[Nu*i:Nu*(i+1), Nx*(i+1):] == np.zeros((Nu, Nx*(T-i)))]
    return constraints