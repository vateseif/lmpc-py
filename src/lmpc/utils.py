import numpy as np
import networkx as nx
from typing import Tuple
import matplotlib.pyplot as plt

from src.lmpc.core import SystemModel
from src.lmpc.system_models import DistributedLTI
from src.lmpc.locality_models import dLocality


def generateRandomSquareMeshLTI(n: int, p: float=0.4, seed:int=None) -> Tuple[np.ndarray]:

  # set np seed
  np.random.seed(seed)
  # construct random mesh connection between subsystemswith prob p
  G = nx.grid_2d_graph(n, n).to_directed()
  G = nx.convert_node_labels_to_integers(G)
  edges = list(G.edges())
  random_mesh = np.random.choice([0, 1], size=(G.number_of_edges()), p=[1-p, p])
  edges = [e for i, e in enumerate(list(G.edges())) if random_mesh[i]]
  G.clear_edges()
  G.add_edges_from(edges)
  locality = dLocality(d=1)
  locality.computeOutgoingSets(G)
  out_x = locality.out_x
  # system parameters
  N = n*n
  dt = 0.1 # time discretization
  imi = np.random.uniform(0, 2, size=(N))     # inertia inverse
  di = np.random.uniform(0.5, 1, size=(N))    # damping coeff
  kij = np.random.uniform(1, 1.5, size=(N,N)) # coupling
  ki = np.zeros((N))
  for i in range(N):
    for j in range(N):
      if i in out_x[j]:
        ki[i] += kij[i][j]
  
  # init DistributedLTI sys
  sys = DistributedLTI(N, Ns=2, Na=1)
  # compute A
  A = np.zeros((sys.Nx, sys.Nx))
  for i in range(N):
    for j in range(N):
      if i in out_x[j]:
        if i==j:
          Aij = np.array([[1, dt], [-ki[i]*imi[i]*dt, 1-di[i]*imi[i]*dt]])
        else:
          Aij = np.array([[0, 0], [kij[i][j]*imi[i]*dt, 0]])
        A[np.ix_(sys._idx[i], sys._idx[j])] = Aij
  # compute B
  Bi = np.array([[0], [1]])
  B = np.kron(np.eye(sys.N), Bi)

  return A, B

def generateCoupledPendulumsLTI() -> Tuple[np.ndarray]:
  # State dimensions
  N = 4         # number of pendulums
  NS = 2        # number of states for each pendulum (theta, dtheta)
  NA = 1        # number of actuators for each pendulum (dtheta)
  NX = NS*N     # number of total states
  NU = NA*N     # number of total actuators
  # System params
  m = 1         # [kg] mass of the pendulum
  k = 1         # [N/m] spring constant
  dg = 3        # [Ns/m] damping constant
  g = 10        # [m/s^2] gravity
  l = 1         # [m] length of pendulum
  TS = .1       # [s] sampling time
  # Construct dynamic system matrices: dx = Ax + Bu
  block_off_diag = np.array([[0,0], [k*l/m, dg/m/l]])
  block_diag_extra = np.array([[0, 1], [-g-k*l/m, -dg/(m*l)]])
  block_diag = np.array([[0, 1], [-g-2*k*l/m, -2*dg/(m*l)]])

  Ac = np.zeros((NX, NX))
  Bc = np.zeros((NX, NU))
  for j, i in enumerate(range(0, NX, 2)):
    Ac[i:i+2,i:i+2] = block_diag
    if j!=N-1:
      Ac[i:i+2,i+2:i+4] = block_off_diag
    if j!=0:
      Ac[i:i+2,i-2:i] = block_off_diag

    Bc[i:i+2, j] = np.array([0, 1])

  # Discretize system
  A = np.eye(NX) + TS * Ac
  B = TS * Bc
  return A, B


def plot_topology_locality(sys:SystemModel, n:int, d:int):
  # Visualize mesh graph and dLocality graph
  fig, ax = plt.subplots(1, 3, figsize=(10,3))
  pos = [(j, i) for i in range(n) for j in range(n)]
  pos = dict((i, pos[i]) for i in range(n*n))
  nx.draw(sys._topology, pos, with_labels=True, ax=ax[0])
  ax[0].set_title(f"Interconnection topology")
  cG_x, cG_u = sys.locality_model.computeOutgoingSets(sys._topology, True)
  nx.draw(cG_x, pos, with_labels=True, ax=ax[1])
  ax[1].set_title(f"d-locality with d={d}")
  nx.draw(cG_u, pos, with_labels=True, ax=ax[2])
  ax[2].set_title(f"d-locality with d={d+1}")