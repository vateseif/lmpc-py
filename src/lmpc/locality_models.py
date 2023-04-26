import networkx as nx
from itertools import product
from typing import Optional, List, Tuple

from src.lmpc.core import LocalityModel

class dLocality(LocalityModel):
  def __init__(self, d:Optional[int] = None) -> None:
    self.d = d
    self._topology = None

  def computeOutgoingSets(self, 
                        _topology: nx.DiGraph,
                        return_cG: bool = False) -> Tuple[List[List[int]], nx.DiGraph]:
    '''
    Compute the set of outgoing nodes for each subsystem based on d-locality.
    If there's a path i->j <= d then j belongs to the outgoing set of i.
    '''
    if self.d is None:
      self.d = self._optimalLocalitySelection(_topology)
    
    self._topology = _topology
    self.N = self._topology.number_of_nodes()
    self.out_x = self._compute_out(self.d, self._topology)
    self.out_u = self._compute_out(self.d+1, self._topology)   

    if not return_cG:
      return

    # communication graph
    cG_x, cG_u = nx.DiGraph(), nx.DiGraph()
    cG_x.add_edges_from(sum([[*product([i], self.out_x[i])] for i in range(self.N)],[]))
    cG_u.add_edges_from(sum([[*product([i], self.out_u[i])] for i in range(self.N)],[]))
    return cG_x, cG_u

  def _updateOutgoingSets(self):
    ''' dLocality is fixed if the system is LTI '''
    return

  def _compute_out(self, d: int, _G: nx.DiGraph()) -> List[List[int]]:
    """ Returns matrix where matrix[i] = out_i(d) is the d-outgoing set for i """
    out = [[] for _ in range(self.N)]
    # lengths of shortest path for each node
    lengths = dict(nx.all_pairs_shortest_path_length(_G))
    # outgoing sets
    out = [[] for _ in range(self.N)]
    for i in range(self.N):
      for j in range(self.N):
        try: 
          if lengths[i][j] <= d: out[i].append(j)
        except: 
          continue
    return out

  def _optimalLocalitySelection(self, _G: nx.DiGraph) -> int:
    '''
    Computes the optimal (smallest) d without degradation of MPC performance.
    https://arxiv.org/pdf/2303.11264.pdf
    '''
    # TODO
    return 1
