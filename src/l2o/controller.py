import torch
import cvxpy as cp
import numpy as np
from typing import Tuple
from core import AbstractController
from llm import Objective, Optimization
from config.config import BaseControllerConfig


class BaseController(AbstractController):

  def __init__(self, cfg=BaseControllerConfig()) -> None:
    super().__init__(cfg)
    # init linear dynamics
    self.init_dynamics()
    # init CVXPY problem
    self.init_problem()

  def init_dynamics(self):
    # dynamics
    self.A = np.zeros((self.cfg.nx, self.cfg.nx))
    self.B = np.eye(self.cfg.nx)
    self.Ad = np.eye(self.cfg.nx) + self.A * self.cfg.dt
    self.Bd = self.B * self.cfg.dt

  def init_problem(self):
    # variables
    self.x = cp.Variable((self.cfg.T+1, self.cfg.nx), name='x')
    self.u = cp.Variable((self.cfg.T, self.cfg.nu), name='u')
    # parameters
    self.x0 = cp.Parameter(self.cfg.nx, name="x0")
    self.xd = cp.Parameter(self.cfg.nx, name="xd")
    self.x0.value = np.zeros((self.cfg.nx,))
    self.xd.value = np.zeros((self.cfg.nx,))
    # cost
    self.xd_cost_T = sum([cp.norm(xt - self.xd) for xt in self.x])
    self.obj = cp.Minimize(self.xd_cost_T)
    # constraints
    self.cvx_constraints = self.init_cvx_constraints()
    # put toghether nominal MPC problem
    self.prob = cp.Problem(self.obj, self.cvx_constraints)

  def init_cvx_constraints(self):
    constraints = []
    # upper and lower bounds
    constraints += [self.u <= self.cfg.hu*np.ones((self.cfg.T, self.cfg.nu))]
    constraints += [self.u >= self.cfg.lu*np.ones((self.cfg.T, self.cfg.nu))]
    # initial cond
    constraints += [self.x[0] == self.x0]
    # dynamics
    for t in range(self.cfg.T):
      constraints += [self.x[t+1] == self.Ad @ self.x[t] + self.Bd @ self.u[t]]
    # bouns on state (gripper always above table)
    for t in range(self.cfg.T+1):
      constraints += [self.x[t][2] >= 0]
    return constraints

  def reset(self, x0: np.ndarray) -> None:
    self.init_problem()
    self.x0.value = x0
    self.xd.value = x0
    return

  def _eval(self, code_str: str, x_cubes: Tuple[np.ndarray]):
    #TODO this is hard coded for when there are 4 cubes
    cube_1, cube_2, cube_3, cube_4 = x_cubes
    evaluated_code = eval(code_str, {
      "cp": cp,
      "np": np,
      "self": self,
      "cube_1": cube_1,
      "cube_2": cube_2,
      "cube_3": cube_3,
      "cube_4": cube_4
    })
    return evaluated_code

  def _solve(self):
    # solve for either uncostrained problem or for initial guess
    self.prob.solve(solver='MOSEK')
    return self.u.value[0]
  
  def step(self):
    return self._solve()



class ParametrizedRewardController(BaseController):

  def apply_gpt_message(self, gpt_message:str, x_cubes: Tuple[np.ndarray]):
    cube_1, cube_2, cube_3, cube_4 = x_cubes
    self.xd.value = eval(gpt_message)


class ObjectiveController(BaseController):

  def apply_gpt_message(self, objective: Objective, x_cubes: Tuple[np.ndarray]) -> None:
    # apply objective function
    obj = self._eval(objective.objective, x_cubes)
    self.obj = cp.Minimize(obj)
    # create new MPC problem
    self.prob = cp.Problem(self.obj, self.cvx_constraints)
    return 

class OptimizationController(BaseController):

  def apply_gpt_message(self, optimization: Optimization, x_cubes: Tuple[np.ndarray]) -> None:    
    # apply objective function
    obj = self._eval(optimization.objective, x_cubes)
    self.obj = cp.Minimize(obj)
    # apply constraints
    constraints = self.cvx_constraints
    for constraint in optimization.constraints:
      constraints += self._eval(constraint, x_cubes)
    # create new MPC problem
    self.prob = cp.Problem(self.obj, self.cvx_constraints)
    return 

class Controller:
  def __init__(self) -> None:

    # params
    self.nx, self.nu = 3, 3
    self.T = 15
    self.dt = 0.05
    self.lu = -0.5 # lower bound on u
    self.hu = 0.5  # higher bound on u
    self.gripper = 1. # 1. means the gripper is open
    self.collision_sensitivity = 0.0 # additional buffer to avoid collision

    # optimization params
    self.l = 10             # lambda coefficient to go in cost
    self.tolerance = 0.1        # SCP cost function improvement tolerance
    self.max_iterations = 20  # max number of SCP iterations
    
    self.solving_with = "nominal"

    # dynamics
    self.A = np.zeros((self.nx, self.nx))
    self.B = np.eye(self.nx)
    self.Ad = np.eye(self.nx) + self.A * self.dt
    self.Bd = self.B * self.dt
    self.E = np.eye(self.nx) # dynamics matrix for slack variable

    # variables
    self.x = cp.Variable((self.T+1, self.nx), name='x')
    self.u = cp.Variable((self.T, self.nu), name='u')
    # virtual controls
    self.v_dyn = cp.Variable((self.T+1, self.nx), name='v_dyn') # slack for dynamics
    self.v_const = cp.Variable((self.T+1, 1), name='v_const')   # slack for constraints

    # parameters
    self.x0 = cp.Parameter(self.nx, name="x0")
    self.xd = cp.Parameter(self.nx, name="xd")
    self.x0.value = np.zeros((self.nx,))
    self.xd.value = np.zeros((self.nx,))

    # cost functions
    self.xd_cost = cp.norm(self.x[-1] - self.xd)
    self.xd_cost_T = sum([cp.norm(xt - self.xd) for xt in self.x])
    self.scp_dyn_cost = self.l * sum([cp.norm(self.E @ vt, 1) for vt in self.v_dyn])  # cost on slack variables  
    self.scp_const_cost = self.l * cp.norm(self.v_const, 1)                           # cost on slack variables  

    # objective functions
    self.obj = cp.Minimize(self.xd_cost_T)
    self.scp_obj = cp.Minimize(self.xd_cost_T + self.scp_dyn_cost + self.scp_const_cost)

    # constraints
    self.cvx_constraints = self.init_cvx_constraints()
    self.scp_constraints = self.init_scp_constraints()
    self.collision_constraints = []

    # put toghether nominal problem
    self.prob = cp.Problem(self.obj, self.cvx_constraints)

  def init_cvx_constraints(self):
    constraints = []
    # upper and lower bounds
    constraints += [self.u <= self.hu*np.ones((self.T, self.nu))]
    constraints += [self.u >= self.lu*np.ones((self.T, self.nu))]
    # initial cond
    constraints += [self.x[0] == self.x0]
    # dynamics
    for t in range(self.T):
      constraints += [self.x[t+1] == self.Ad @ self.x[t] + self.Bd @ self.u[t]]
    # bouns on state (gripper always above table)
    for t in range(self.T+1):
      constraints += [self.x[t][2] >= 0]
    return constraints

  def init_scp_constraints(self):
    constraints = []
    # upper and lower bounds
    constraints += [self.u <= self.hu*np.ones((self.T, self.nu))]
    constraints += [self.u >= self.lu*np.ones((self.T, self.nu))]
    # initial cond
    constraints += [self.x[0] == self.x0]
    # dynamics
    for t in range(self.T):
      constraints += [self.x[t+1] == self.Ad @ self.x[t] + self.Bd @ self.u[t]+ self.E @ self.v_dyn[t]]
    # bouns on state (gripper always above table)
    for t in range(self.T+1):
      constraints += [self.x[t][2] >= 0]
    return constraints


  def add_constraint(self, constraint):
    if constraint.is_dcp():
      self.cvx_constraints += constraint
      self.prob = cp.Problem(self.obj, self.cvx_constraints)
    else:
      raise ValueError("Constraint has to be DCP")

  def set_x0(self, x0: np.ndarray, offset: np.ndarray = np.zeros(3)):
    self.x0.value = x0 + offset

  def set_xd(self, xd: np.ndarray, offset: np.ndarray = np.zeros(3)):
    self.xd.value = xd + offset
    
  def open_gripper(self):
    self.gripper = 1.

  def close_gripper(self):
    self.gripper = -1.

  def step(self):
    return self._solve()

  def add_collision_constraint(self, x_obs:np.ndarray, r_obs:float):
    self.collision_constraints += [(x_obs, r_obs)]
    pass

  def collision_constraint_satisfied(self, x:np.ndarray):
    '''Checks that no collisions occurs given a trajectory x'''
    for (x_obs, r_obs) in self.collision_constraints:
      # if 1 collision than return false
      if min(np.linalg.norm(x - x_obs, axis=1)) < (r_obs+self.collision_sensitivity):
        return False
    return True

  def compute_collision_constraint(self, x_bar: np.ndarray):
    scp_const = []
    x_bar_torch = torch.tensor(x_bar, requires_grad=True)

    for (x_obs, r_obs) in self.collision_constraints:
      x_obs_torch = torch.tensor(x_obs, requires_grad=False)
      r_obs_torch = torch.tensor([r_obs], requires_grad=False)

      distance_norm_torch = (r_obs_torch+self.collision_sensitivity)-torch.linalg.norm(x_bar_torch-x_obs_torch, dim=1)
      distance_norm_torch.backward(torch.ones((x_bar_torch.shape[0])))

      for t in range(self.T+1):
        xk = x_bar_torch[t]
        Ck = x_bar_torch.grad[t]
        r_p = distance_norm_torch[t] - Ck @ xk
        scp_const += [Ck.detach().numpy() @ self.x[t] + r_p.detach().numpy() <= self.v_const[t]]

    return scp_const

  def _solve(self):
    # solve for either uncostrained problem or for initial guess
    self.prob.solve(solver='MOSEK')
    # if problem is convex then return
    if len(self.collision_constraints)==0 or self.collision_constraint_satisfied(self.x.value):
      self.solving_with = "nominal" 
      return self.u.value[0]
    # if you're here problem is not convex so we need to use SCP
    i = 0
    cost_new = 0
    cost_old = float('inf')
    constraint_respected = False
    # SCP iterations. Stopping criteria:
    # - max number of iterations reached
    # - cost update smaller than tolerance and collision constraints satisfied
    while i<self.max_iterations and (
      (i==0 or abs(cost_old - cost_new)>self.tolerance) 
      or not constraint_respected):
      # candidate trajectory
      x_bar = self.x.value if self.solving_with=="nominal" else self.x_bar
      # scp constraints for each collision avoidance
      collision_const = self.compute_collision_constraint(x_bar)
      # solve problem
      scp_prob = cp.Problem(self.scp_obj, self.scp_constraints + collision_const)
      cost_new = scp_prob.solve(solver='MOSEK')
      constraint_respected = self.collision_constraint_satisfied(self.x.value)
      # update cost
      cost_old = cost_new
      #print(f"cost = {cost_new}")
      i+=1
    
    self.solving_with = 'SPC'
    self.x_bar = self.x.value
    return self.u.value[0]



ControllerOptions = {
  "parametrized": ParametrizedRewardController,
  "objective": ObjectiveController,
  "optimization": OptimizationController
}