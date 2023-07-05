import cvxpy as cp
import numpy as np

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
    self.l = 1000             # lambda coefficient to go in cost
    self.tolerance = 0.1      # SCP cost function improvement tolerance
    self.max_iterations = 10  # max number of SCP iterations
    


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

    # scp variable
    self.x_scp = cp.Variable((self.T+1, self.nx), name='x_scp')
    self.u_scp = cp.Variable((self.T, self.nu), name='u_scp')
    self.v_dyn = cp.Variable((self.T+1, self.nx), name='v_dyn') # slack for dynamocs
    self.v_const = cp.Variable((self.T+1, 1), name='v_const')   # slack for constraints

    # parameters
    self.x0 = cp.Parameter(self.nx, name="x0")
    self.xd = cp.Parameter(self.nx, name="xd")
    self.x0.value = np.zeros((self.nx,))
    self.xd.value = np.zeros((self.nx,))


    # cost functions
    self.xd_cost = cp.norm(self.x[-1] - self.xd)
    self.xd_cost_T = sum([cp.norm(xt - self.xd) for xt in self.x])
    # scp cost functions
    self.scp_xd_cost = cp.norm(self.x_scp[-1] - self.xd)
    self.scp_xd_cost_T = sum([cp.norm(xt - self.xd) for xt in self.x_scp])
    self.scp_dyn_cost = self.l * sum([cp.norm(self.E @ vt, 1) for vt in self.v_dyn])  # cost on slack variables  
    self.scp_const_cost = self.l * cp.norm(self.v_const, 1)                           # cost on slack variables  
    
    # objective functions
    self.obj = cp.Minimize(self.xd_cost_T)
    self.scp_obj = cp.Minimize(self.scp_xd_cost_T + self.scp_dyn_cost + self.scp_const_cost)

    # constraints
    self.constraints = self.init_constraints()
    self.scp_constraints = self.init_scp_constraints()
    self.collision_constraints = [] # non-convex constraints
    
    # put toghether nominal problem
    self.prob = cp.Problem(self.obj, self.constraints)

    # GPT functions
    self.functions = {
      "set_xd": self.set_xd,
      "open_gripper" : self.open_gripper,
      "close_gripper" : self.close_gripper,
      "add_constraint" : self.add_constraint
    }


  def init_constraints(self):
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
    constraints += [self.u_scp <= self.hu*np.ones((self.T, self.nu))]
    constraints += [self.u_scp >= self.lu*np.ones((self.T, self.nu))]
    # initial cond
    constraints += [self.x_scp[0] == self.x0]
    # dynamics
    for t in range(self.T):
      constraints += [self.x_scp[t+1] == self.Ad @ self.x_scp[t] + self.Bd @ self.u_scp[t] + self.E @ self.v_dyn[t]]
    # bouns on state (gripper always above table)
    for t in range(self.T+1):
      constraints += [self.x_scp[t][2] >= 0]
    return constraints

  def add_constraint(self, constraint):
    if constraint.is_dcp():
      self.constraints += constraint
      self.prob = cp.Problem(self.obj, self.constraints)
    else:
      raise ValueError("Constraint has to be DCP")

  def add_collision_constraint(self, x_obs:np.ndarray, r_obs:float):
    self.collision_constraints += [(x_obs, r_obs)]
    pass

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

  def compute_collision_constraint(self, x_bar:np.ndarray):
    ''' Computes the SCP constraints related to the collision avoidance constraints'''
    scp_const = []
    for (x_obs, r_obs) in self.collision_constraints:
        distance_norm = np.linalg.norm(x_bar - x_obs, axis=1)
        for t in range(self.T+1):
          nt = distance_norm[t]
          scp_const += [(x_bar[t] - x_obs).T/nt @ (x_bar[t]-self.x_scp[t]) + (r_obs+self.collision_sensitivity) - nt <= self.v_const[t]]
    return scp_const

  def collision_contraint_satisfied(self, x:np.ndarray):
    '''Checks that no collisions occurs given a trajectory x'''
    for (x_obs, r_obs) in self.collision_constraints:
      # if 1 collision than return false
      if min(np.linalg.norm(x - x_obs, axis=1)) < (r_obs+self.collision_sensitivity):
        return False
    return True

  def _solve(self):
    # solve for either uncostrained problem or for initial guess
    self.prob.solve(solver='MOSEK')
    # if problem is convex then return
    if len(self.collision_constraints)==0 or self.collision_contraint_satisfied(self.x.value):
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
      x_bar = self.x.value if i==0 else self.x_scp.value
      # scp constraints for each collision avoidance
      collision_const = self.compute_collision_constraint(x_bar)
      # solve problem
      scp_prob = cp.Problem(self.scp_obj, self.scp_constraints + collision_const)
      cost_new = scp_prob.solve(solver='MOSEK')
      constraint_respected = self.collision_contraint_satisfied(self.x_scp.value)
      # update cost
      cost_old = cost_new
      #print(f"cost = {cost_new}")
      i+=1
    
    self.solving_with = 'SPC'
    return self.u_scp.value[0]

