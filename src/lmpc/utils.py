from threading import local
import numpy as np
import networkx as nx
from typing import Tuple

from lmpc.system_models import DistributedLTI
from lmpc.locality_models import dLocality


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