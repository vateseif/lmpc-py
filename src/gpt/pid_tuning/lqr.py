import numpy as np
import scipy.linalg as linalg
from scipy.linalg import solve_discrete_are as are


class LQR:
  def __init__(self) -> None:

    self.name = 'lqr'

    # model parameters
    self.gravity = 9.8
    self.masscart = 1.0
    self.masspole = 0.1
    self.total_mass = (self.masspole + self.masscart)
    self.length = 0.5 # actually half the pole's length
    self.polemass_length = (self.masspole * self.length)
    self.force_mag = 10.0
    self.tau = 0.02 

    self.H = np.array([
      [1,0,0,0],
      [0,self.total_mass, 0, - self.polemass_length],
      [0,0,1,0],
      [0,- self.polemass_length, 0, (2 * self.length)**2 * self.masspole /3]
    ])
    self.Hinv = np.linalg.inv(self.H)

    # dynamics matrices A, B
    self.A = self.Hinv @ np.array([
        [0,1,0,0],
        [0,0,0,0],
        [0,0,0,1],
        [0,0, - self.polemass_length * self.gravity, 0]
      ])
    self.B = self.Hinv @ np.array([0,1.0,0,0]).reshape((4,1))
    # discretize dynamics
    self.Ad = np.eye(4) + self.A * self.tau
    self.Bd = self.B * self.tau

    # cost matrices
    #self.Q = np.diag([0.1, 1.0, 100.0, 5.])
    #self.R = np.array([[0.1]])
    self.Q = np.diag([1.0, 1.0, 1.0, 1.0])
    self.R = np.array([[0.1]])
    # desired state
    self.desired_state = np.zeros((4,))

    # params that can be changed by gpt
    self.parameters = {"x":0, "dx":1, "theta":2, "dtheta":3}

    # controller gain
    self.P = are(self.Ad, self.Bd, self.Q, self. R)
    self.K = np.linalg.inv(self.R + self.Bd.T @ self.P @ self.Bd) @ self.Bd.T @ self.P @ self.Ad 

  def act(self, x: np.ndarray):

    error = self.desired_state - x
    action = np.dot(-self.K, error)[0]
    action = min(action, 10)
    action = max(action, -10)
    return action

  def reset(self):
    pass
  
  def update(self, parametername, parametervalue):
    # update matrix gain
    pi = self.parameters[parametername]
    self.Q[pi][pi] = parametervalue

    # controller gain
    self.P = are(self.Ad, self.Bd, self.Q, self. R)
    self.K = np.linalg.inv(self.R + self.Bd.T @ self.P @ self.Bd) @ self.Bd.T @ self.P @ self.Ad 
    