import numpy as np
import matplotlib.pyplot as plt

from math import floor
from copy import deepcopy


class GlucoseSimulation:

  def __init__(self) -> None:
    # time step
    self.dt = 2

    # initial states
    self.G0 = 270 # initial glucose
    self.X0 = 0 # intial idk what
    
    # model params
    self.k1 = 0.02
    self.k2 = 0.02
    self.k3 = 1.5e-05
    self.Gb = 92 # baseline glucose
    self.Pb = 11 # baseline proteins (responsible for the consumption of glucose)
    self.Pmin = 7
    self.Pmax = 130
    

  def reset(self):
    self.G = self.G0
    self.X = self.X0


  def step(self, P):

    dG = -self.k1 * (self.G - self.Gb) - self.X * self.G
    dX = self.k3 * (P - self.Pb) - self.k2 * self.X

    self.G += dG * self.dt
    self.X += dX * self.dt

  def observation(self):
    done = False
    reward = 0.
    return [deepcopy(self.G), deepcopy(self.X)], reward, done 

  def simulate(self, policy, T=1000, plot=True):

    self.reset()

    x_history = []
    u_history = []
    t = np.linspace(0, T, floor(T/self.dt))
    for i in t:
      # observe
      x, r, done = self.observation()
      # compute action
      P = policy.act(x, i, T)
      # step
      Pn = np.random.randint(-5, 5) # disturbance
      self.step(P + Pn)
      # store
      x_history.append(x)
      u_history.append(P + Pn)

    x_history = np.vstack(x_history)

    # plot
    _, axs = plt.subplots(nrows=3)
    axs[0].plot(t, x_history[:, 0], label="G")
    axs[1].plot(t, x_history[:, 1], label="X")
    axs[2].plot(t, u_history, label="P")
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    axs[2].legend(loc='best')