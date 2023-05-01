import numpy as np
import cvxpy as cp
from typing import List, Optional

from src.lmpc.core import Constraint, LocalityModel, SystemModel

class LMPCConstraint(Constraint):

  def __init__(self, T:int, system) -> None:
    self.T = T
    self.N = system.N
    self.Nx = system.Nx
    self.Nu = system.Nu
    self.Ns = system.Ns
    self.Na = system.Na
    # rows that correspond to each subsystem of Phi for states and actions 
    self.rx = system._idx
    self.ru = [[self.Nx*(T+1)+i for i in iu] for iu in system._idu]
    # rows that correspond to each subsystem of Phi for states and actions along horizon
    self.rxT = [ sum([[self.Nx*t + x for x in ix] for t in range(T+1)], []) for ix in system._idx ]
    self.ruT = [ sum([[self.Nx*(T+1)+self.Nu*t + u for u in iu] for t in range(T)], []) for iu in system._idu ]
    
  def _initFromParent(self, parent):
    vars(self).update(vars(parent))

  def compute(self, x0:np.ndarray, phi: cp.Variable):
    pass


class SLSConstraint(LMPCConstraint):
  def __init__(self, ZAB: np.ndarray) -> None:
    self.ZAB = ZAB

  def compute(self, x0:np.ndarray, phi: cp.Variable):
    T, Nx = self.T, self.Nx
    # check if w or only x0 was passed
    if x0.shape[0] == Nx*(T+1):
      rhs = np.eye(Nx*(T+1))
    else:
      rhs = np.concatenate((np.eye(Nx), np.zeros((Nx*T, Nx))), axis=0)
    return [self.ZAB @ phi == rhs]


class LocalityConstraint(LMPCConstraint):

  def __init__(self, locality_model: LocalityModel):
    self.locality_model = locality_model

  def compute(self, x0:np.ndarray, phi: cp.Variable):
    # dimensions
    T, N, Nx, Ns, Na = self.T, self.N, self.Nx, self.Ns, self.Na
    # out-going sets
    self.locality_model._updateOutgoingSets() # in this case it does nothing
    out_x = self.locality_model.out_x
    out_u = self.locality_model.out_u
    # index of states per subsytem depending on x0 or w
    rx = self.rx if x0.shape[0] == Nx else self.rxT
    ru = self.ru if x0.shape[0] == Nx else self.ruT
    Tx = 1 if x0.shape[0] == Nx else T+1
    Tu = 1 if x0.shape[0] == Nx else T
    # locality constraints
    constraints = []
    for i in range(N):
      for j in range(N):
        if i not in out_x[j]:
          constraints += [phi[np.ix_(rx[i], rx[j])] == np.zeros((Ns[i]*Tx, Ns[j]*Tx))]
        if i not in out_u[j]:
          constraints += [phi[np.ix_(ru[i], rx[j])] == np.zeros((Na[i]*Tu, Ns[j]*Tx))]
    return constraints

class TerminalConstraint(LMPCConstraint):
  def __init__(self, xT: Optional[np.ndarray] = None) -> None:
    self.xT = xT

  def compute(self, x0:np.ndarray, phi: cp.Variable):
    T, Nx = self.T, self.Nx
    if self.xT is None:
      self.xT = np.zeros((Nx, 1))
    return [phi[Nx*T:Nx*(T+1),:Nx] @ x0[:Nx] == self.xT]

class LowerTriangulatConstraint(LMPCConstraint):

  def __init__(self) -> None:
    return

  def compute(self, x0: np.ndarray, phi: cp.Variable) -> List:
    ''' Enforce phi_x, phi_u lower triangular if w is given and not only x0'''
    T, Nx, Nu = self.T, self.Nx, self.Nu
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

class BoundConstraint(LMPCConstraint):
  def __init__(self, xu: str, lu: str, b: np.ndarray,
              sigma:float=0, p:int = 2) -> None:
    assert xu in ["x", "u"], "'x' or 'u' bound not specified"
    assert lu in ["lower", "upper"], "'lower' or 'upper' bound not specified"
    self.p = p    # disturbance norm
    self.xu = xu  # 'x' or 'u'
    self.lu = lu  # 'lower' or 'upper'
    self.sigma = sigma  # disturbance bound (None if no disturbance)
    self.b = b if lu == "upper" else -b    # bound value

  def _initFromParent(self, parent):
    super()._initFromParent(parent)
    T, Nx, Nu = self.T, self.Nx, self.Nu
    if self.xu == "x":
      I = np.eye((T+1)*Nx)
      if self.lu == "lower": I = -I
      self.H = np.concatenate((I, np.zeros((Nx*(T+1), Nu*T))), axis=1)
      self.b = np.concatenate([self.b]*(T+1), axis=0)
    else:
      I = np.eye(Nu*T)
      if self.lu == "lower": I = -I
      self.H = np.concatenate((np.zeros((Nu*T, Nx*(T+1))), I), axis=1)
      self.b = np.concatenate([self.b]*T, axis=0)

  def compute(self, x0: np.ndarray, phi: cp.Variable):
    T, Nx = self.T, self.Nx
    if self.sigma == 0:
      return [self.H @ phi[:, :Nx] @ x0[:Nx] <= self.b]
    
    assert phi.shape[1]==Nx*(T+1), "Dimension of Phi not correct"
    assert self.b.shape[0]==self.H.shape[0], "Dimensions of b dont match"
    constraints = []
    x = self.H @ phi[:, :Nx] @ x0[:Nx]
    for i in range(self.b.shape[0]):
      ej = np.zeros((self.b.shape[0], 1))
      ej[i][0] = 1
      constraints += [x[i] + self.sigma*cp.atoms.norm(ej.T @ self.H @ phi[:,Nx:], 'nuc') <= self.b[i]]

    return constraints
    