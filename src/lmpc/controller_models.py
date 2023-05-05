import numpy as np
import cvxpy as cp
from typing import List, Optional, Tuple

from src.lmpc.constraints import BoundConstraint, LMPCConstraint
from src.lmpc.objectives import LMPCObjectiveFun
from src.lmpc.system_models import DistributedLTI
from src.lmpc.core import ControllerModel, ObjectiveFunc, Constraint


class LMPC(ControllerModel):

  def __init__(self, T:int) -> None:
    super().__init__()
    self.T = T
    self.model: DistributedLTI = None
    self.parentConstraint: LMPCConstraint = None
    self.constraints: List[Constraint] = [] 
    self.objectives: List[ObjectiveFunc] = [] 
    self.prob : Optional[cp.Problem] = None
    self.useCodeGenSolver = False
    self.solverPath = None

  def __lshift__(self, model: DistributedLTI):
    ''' Overload lshift to augment the ControllerModel with the DistributedLTI'''
    assert isinstance(model, DistributedLTI), f"{model} isnt of type DistributedLTI"
    #assert model.locality_model != None, f"{model} doesnt have any locality_model"
    self.model = model
    self.parentConstraint = LMPCConstraint(self.T, self.model)
    self.addConstraint(self.model.getSLSConstraint(self.T))
    self.addConstraint(self.model.getLowerTriangularConstraint())
    if model.locality_model!=None: self.addConstraint(self.model.getLocalityConstraint())

  def addObjectiveFun(self, obj_fun: LMPCObjectiveFun):
    #assert isinstance(obj_fun, LMPCObjectiveFun), "objective function not of type LMPCObjectiveFun"
    obj_fun._initFromParent(self.parentConstraint)
    self.objectives.append(obj_fun)

  def addConstraint(self, con: LMPCConstraint):
    assert isinstance(con, LMPCConstraint), "constraint not of type LMPCConstraint"
    con._initFromParent(self.parentConstraint)
    self.constraints.append(con)

  def removeConstraintOfType(self, con_type: LMPCConstraint):
    ''' Remove constraints of con_type '''
    self.constraints = [c for c in self.constraints if not isinstance(c, con_type)]
    return

  def _applyConstraints(self, x0: np.ndarray, phi: cp.Variable) -> List:  
    ''' Returns list of cvxpy constraints'''
    # apply constraints stored in self.constraints
    constraints = []
    for con in self.constraints:
      constraints += con.compute(x0, phi)
    return constraints

  def _applyObjective(self, x0: np.ndarray, phi: cp.Variable):
    # compute system solution
    objective = sum([obj.compute(self.T, x0, phi) for obj in self.objectives])
    return cp.Minimize(objective)

  def _checkInitialCondition(self, x0: np.ndarray):
    """ Check if x0 or w is passed and if LMPC is robust then concatenates x0 and 0 """
    Nx, T = self.model.Nx, self.T
    if x0.shape[0] == Nx:
      for c in self.constraints:
        if isinstance(c, BoundConstraint):
          if c.sigma != 0:
            x0 = np.concatenate((x0, np.zeros((Nx*T,1))),axis=0)
            break
    return x0

  def _setupSolver(self, x0: np.ndarray):
    Nx, Nu, T = self.model.Nx, self.model.Nu, self.T
    assert x0.shape[0]==Nx or x0.shape[0]==Nx*(T+1), "x0 dim neither Nx not Nx*(T+1)"
    # define optim variables
    self.phi = cp.Variable((Nx*(T+1)+Nu*T, x0.shape[0]))
    # define param
    self.x0 = cp.Parameter(x0.shape, name="x0")
    # solve
    self.prob = cp.Problem(
      self._applyObjective(self.x0, self.phi),
      self._applyConstraints(self.x0, self.phi)
    )
    return

  def solve(self, x0: np.ndarray, solver="MOSEK") -> Tuple[np.ndarray, float]:
    ''' Solve the MPC problem and return u0 if solution is optimal else raise'''
    Nx, Nu, T = self.model.Nx, self.model.Nu, self.T
    # Concatenate x0 with 0 if LMPC is to be robust
    x0 = self._checkInitialCondition(x0)
    # init cvxpy solver if first call
    if self.prob == None:
      self._setupSolver(x0)
    # update cvxpy parameter
    self.x0.value = x0
    # TODO define solver as property of class
    self.prob.solve(solver, verbose=False, warm_start=True)
    assert self.prob.status == "optimal", f"Solution not found. status: {self.prob.status}"
    # store results
    u0 = self.phi.value[Nx*(T+1):Nx*(T+1)+Nu, :Nx] @ x0[:Nx] # (Nu, 1)
    return u0, self.prob.value, self.phi.value

  def codeGen(self, x0: Optional[np.ndarray] = None, solverPath="solvers"):
    # TODO: finish setting up codegen (it seems that using Mosek is faster than codegen...)
    if self.prob == None:
      assert x0 is not None, "self.prob and x0 not defined, define one of them"
      self._setupSolver(x0)
    
    from cvxpygen import cpg
    cpg.generate_code(self.prob, code_dir=solverPath)
    self.useCodeGenSolver = True
    self.solverPath = solverPath
    return