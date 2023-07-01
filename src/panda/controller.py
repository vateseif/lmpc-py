import cvxpy as cp
import numpy as np

class Controller:
  def __init__(self) -> None:

    # params
    self.nx, self.nu = 3, 3
    self.T = 5
    self.dt = 0.05
    self.lu = -1. # lower bound on u
    self.hu = 1.  # higher bound on u
    
    # dynamics
    self.A = np.zeros((self.nx, self.nx))
    self.B = np.eye(self.nx)
    self.Ad = np.eye(self.nx) + self.A * self.dt
    self.Bd = self.B * self.dt

    # variables
    self.x = cp.Variable((self.T+1, self.nx), name='x')
    self.u = cp.Variable((self.T, self.nu), name='u')

    # parameters
    self.x0 = cp.Parameter(self.nx, name="x0")
    self.xd = cp.Parameter(self.nx, name="xd")
    self.x0.value = np.zeros((self.nx,))
    self.xd.value = np.zeros((self.nx,))

    # put toghether problem
    self.obj = cp.Minimize(cp.norm(self.x[-1] - self.xd))
    self.constraints = self.init_constraints()
    self.prob = cp.Problem(self.obj, self.constraints)


  def init_constraints(self,):
    constraints = []
    # upper and lower bounds
    constraints += [self.u <= self.hu*np.ones((self.T, self.nu))]
    constraints += [self.u >= self.lu*np.ones((self.T, self.nu))]
    # initial cond
    constraints += [self.x[0] == self.x0]
    # dynamics
    for t in range(self.T):
      constraints += [self.x[t+1] == self.Ad @ self.x[t] + self.Bd @ self.u[t]]
    
    return constraints

  def set_x0(self, x0: np.ndarray):
    self.x0.value = x0

  def set_xd(self, xd: np.ndarray):
    self.xd.value = xd
    

  def step(self):
    self.prob.solve(solver='MOSEK')
    return self.u.value[0]

  


#ctrl = Controller()
#
#ctrl.set_xd(np.ones(ctrl.nx))
#
#ctrl.step()
#
#print(ctrl.u.value)
#print(ctrl.x.value)