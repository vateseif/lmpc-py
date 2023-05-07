import numpy as np
import cvxpy as cp
from typing import Optional
from scipy.linalg import block_diag

from lmpc.core import ObjectiveFunc

class LMPCObjectiveFun(ObjectiveFunc):

  def _initFromParent(self, parent):
    vars(self).update(vars(parent))

  def compute(self, 
              T: int,
              x0: np.ndarray,
              phi: cp.Variable):
    pass


class QuadForm(LMPCObjectiveFun):
  def __init__(self, 
              Q: np.ndarray,
              R: np.ndarray,
              xud: Optional[np.ndarray] = None) -> None:
    
    self.Q = Q
    self.R = R
    #if xud
    self.xud = xud  # reference states and inputs


  def compute(self, 
              T: int,
              x0: np.ndarray, 
              phi: cp.Variable) -> cp.atom.Atom:

    if self.xud is None:
      self.xud = np.zeros((phi.shape[0], 1))

    # compute system response
    Nx = self.Nx
    xu = phi[:, :Nx] @ x0[:Nx]  # (Nx*(T+1)+Nu*T, 1)
    
    QT = np.kron(np.eye(T+1), self.Q)
    RT = np.kron(np.eye(T), self.R)
    QR = block_diag(QT, RT)
    
    return cp.quad_form(xu-self.xud, QR)

class TerminalQuadForm(LMPCObjectiveFun):
  
  def __init__(self, Q:np.ndarray, xTd: np.ndarray, G:Optional[np.ndarray]=None) -> None:
    ''' (G @ xT - xTd).T @ Q @ (G @ xT - xTd) '''
    self.Q = Q
    self.G = G      # basically the selction matrix
    self.xTd = xTd

  def compute(self, T: int, x0: np.ndarray, phi: cp.Variable):
    Nx = self.Nx
    if self.G is None:
      self.G = np.eye(Nx)

    assert self.G.shape[0] == self.xTd.shape[0], "dim 0 of G doesnt match xTd"
    assert self.G.shape[1] == Nx
    assert self.G.shape[1] == Nx, "dim 1 of G is not Nx"
    assert self.Q.shape[0] == self.G.shape[0]

    self.xTd = cp.Parameter(self.xTd.shape, value=self.xTd)
    xT = phi[Nx*T:Nx*(T+1), :Nx] @ x0[:Nx]

    return cp.QuadForm(self.G @ xT - self.xTd, self.Q)


class XQuadForm(LMPCObjectiveFun):
  def __init__(self, Q:np.ndarray, xd: np.ndarray, G:Optional[np.ndarray]=None) -> None:
    self.Q = Q
    self.G = G      # basically the selction matrix
    self.xd = xd


  def compute(self, T: int, x0: np.ndarray, phi: cp.Variable):
    Nx = self.Nx
    if self.G is None:
      self.G = np.eye(Nx)

    #assert self.G.shape[0] == self.xTd.shape[0], "dim 0 of G doesnt match xTd"
    #assert self.G.shape[1] == Nx
    #assert self.G.shape[1] == Nx, "dim 1 of G is not Nx"
    #assert self.Q.shape[0] == self.G.shape[0]

    self.Q = np.kron(np.eye(T+1), self.G @ self.Q @ self.G.T)
    self.G = np.kron(np.eye(T+1), self.G)
    self.xd = cp.Parameter(self.xd.shape, value=self.xd)
    x = phi[:Nx*(T+1), :Nx] @ x0[:Nx]

    return cp.QuadForm(self.G @ x - self.xd, self.Q)
    