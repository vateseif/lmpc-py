import numpy as np
import cvxpy as cp
from typing import List

from lmpc.constraints import LMPCConstraint
from lmpc.objectives import LMPCObjectiveFun
from lmpc.system_models import DistributedLTI
from lmpc.core import ControllerModel, ObjectiveFunc, Constraint


class LMPC(ControllerModel):

  def __init__(self, T:int) -> None:
    super().__init__()
    self.T = T
    self.model: DistributedLTI = None
    self.constraints: List[Constraint] = [] 
    self.objectives: List[ObjectiveFunc] = [] 

  def __lshift__(self, model: DistributedLTI):
    ''' Overload lshift to augment the SystemModel with the LocalityModel'''
    assert isinstance(model, DistributedLTI), f"{model} isnt of type DistributedLTI"
    assert model.locality_model != None, f"{model} doesnt have any locality_model"
    self.model = model
    self.addConstraint(self.model.getSLSConstraint(self.T))
    self.addConstraint(self.model.getLocalityConstraint(self.T))

  def addObjectiveFun(self, obj_fun: LMPCObjectiveFun):
    assert isinstance(obj_fun, LMPCObjectiveFun), "objective function not of type LMPCObjectiveFun"
    self.objectives.append(obj_fun)

  def addConstraint(self, con: LMPCConstraint):
    assert isinstance(con, LMPCConstraint), "constraint not of type LMPCConstraint"
    self.constraints.append(con)

  def _applyConstraints(self, x0: np.ndarray, phi: cp.Variable) -> List:  
    ''' Returns list of cvxpy constraints'''
    Nx, Nu, T = self.model.Nx, self.model.Nu, self.T
    # apply constraints stored in self.constraints
    constraints = []
    for con in self.constraints:
      constraints += con.compute(phi)
    # block lower triangular constraints
    phi_x = phi[:Nx*(T+1)]
    phi_u = phi[Nx*(T+1):]
    for i in range(T-1):
      constraints += [phi_x[Nx*i:Nx*(i+1), Nx*(i+1):] == np.zeros((Nx, Nx*(T-i)))]
      if i < T-2: # u has 1 less time step
        constraints += [phi_u[Nu*i:Nu*(i+1), Nx*(i+1):] == np.zeros((Nu, Nx*(T-i)))]
    return constraints

  def _applyObjective(self, x0: np.ndarray, phi: cp.Variable):
    # compute system solution
    objective = sum([obj.compute(self.T, x0, phi) for obj in self.objectives])
    return cp.Minimize(objective)

  def solve(self, x0: np.ndarray) -> np.ndarray:
    ''' Solve the MPC problem and return u0 if solution is optimal else raise'''
    Nx, Nu, T = self.model.Nx, self.model.Nu, self.T
    # define optim variables
    phi = cp.Variable((Nx*(T+1)+Nu*T, Nx*(T+1)))
    # solve
    prob = cp.Problem(
      self._applyObjective(x0, phi),
      self._applyConstraints(x0, phi)
      )
    prob.solve()
    if st := (prob.status != "optimal"): raise(f"Solution not found. status: {st}")
    # store results
    u0 = phi.value[Nx*(T+1):Nx*(T+1)+Nu, :Nx] @ x0 # (Nu, 1)
    return u0