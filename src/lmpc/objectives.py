import numpy as np
import cvxpy as cp
from typing import Optional
from scipy.linalg import block_diag

from src.lmpc.core import ObjectiveFunc

class LMPCObjectiveFun(ObjectiveFunc):

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
    self.xud = xud  # reference states and inputs


  def compute(self, 
              T: int,
              x0: np.ndarray, 
              phi: cp.Variable) -> cp.atom.Atom:

    if self.xud is None:
      self.xud = np.zeros((phi.shape[0], 1))

    # compute system response
    Nx = self.Q.shape[0]
    Nu = self.R.shape[0]
    off = Nx*(T+1) # offset of states indices
    xu = phi[:, :Nx] @ x0[:Nx]  # (Nx*(T+1)+Nu*T, 1)
    
    QT = np.kron(np.eye(T+1), self.Q)
    RT = np.kron(np.eye(T), self.R)
    QR = block_diag(QT, RT)
    
    return cp.quad_form(xu, QR)