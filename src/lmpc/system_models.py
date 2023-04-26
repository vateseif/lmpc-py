import numpy as np
import networkx as nx
from typing import List, Union

from .core import SystemModel, LocalityModel


class DistributedLTI(SystemModel):

  def __init__(self, N: int,
                    Ns: Union[int, List[int]],
                    Na: Union[int, List[int]]):

    # all subsystems have same number of states
    if isinstance(Ns, int):
      Ns = [Ns for _ in range(N)]
    # all subsystems have same number of actuators  
    if isinstance(Na, int):
      Na = [Na for _ in range(N)]

    # System dimensions
    self.N = N                # number of subsystems
    self.Ns = Ns              # states of each subsystem
    self.Na = Na              # actuators of each subsystem
    self.Nx = sum(self.Ns)    # total states
    self.Nu = sum(self.Na)    # total actuators

    # Compute indices of states that belong to each subsystem
    Ns_cum = np.cumsum(Ns).tolist()
    self._idx = list(map(lambda s1, s2: [*range(s1, s2)], [0]+Ns_cum[:-1], Ns_cum))

    # Compute indices of control actions that belong to each subsystem
    Na_cum = np.cumsum(Na).tolist()
    self._idu = list(map(lambda s1, s2: [*range(s1, s2)], [0]+Na_cum[:-1], Na_cum))


  def __lshift__(self, locality_model: LocalityModel):
    ''' Overload lshift to augment the SystemModel with the LocalityModel'''
    self.locality_model = locality_model
    self._outgoing_sets, self._cG = self.locality_model.computeOutgoingSets(self._topology)


  def loadAB(self, A: np.ndarray, B: np.ndarray) -> None:
    ''' Load A, B matrices for LTI dynamics x(t+1)=Ax(t)+Bu(t) '''
    assert (Adim := A.ndim) == 2, f"A has dimension {Adim} and not 2."
    assert (Bdim := A.ndim) == 2, f"B has dimension {Bdim} and not 2."
    assert A.shape[0] == A.shape[1] == self.Nx, f"Number of states in A doesnt match Nx"
    assert B.shape[0] == self.Nx, f"Number of states in B doesnt match Nx"
    assert B.shape[1] == self.Nu, f"Number of actions in B doesnt match Nu"

    # store matrices
    self.A = A
    self.B = B

    # compute interconnection topology
    self._topology = self._computeInterTopology()


  def _computeInterTopology(self) -> nx.DiGraph:
    ''' 
    Computes the interconnection topology (i.e. the dynamics coupling)
    as an directed graph. There exist an edge i->j if A_ij!=0 or B_ij!=0
    '''
    _topology = nx.DiGraph()
    # compute directed edges
    edges = []
    A0 = self.A != 0 # (NX, NX)
    B0 = self.B != 0 # (NX, NU)
    for i in range(self.N):
      for j in range(self.N):
        ix_x = np.ix_(self._idx[i], self._idx[j])
        ix_u = np.ix_(self._idx[i], self._idu[j])
        if A0[ix_x].any() or B0[ix_u].any():
          edges.append((j, i))
    # return topology graph
    _topology.add_edges_from(edges)
    return _topology


  def sanityCheck(self):

    if len(self.Ns) != self.N: self.errorMessage("len(self.Ns) != self.N")
    if len(self.Na) != self.N: self.errorMessage("len(self.Na) != self.N")
    
    
