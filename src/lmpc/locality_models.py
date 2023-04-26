import networkx as nx
from typing import Optional, List

from .core import LocalityModel

class dLocality(LocalityModel):
  def __init__(self, d:Optional[int] = None) -> None:
    self.d = d

  def computeOutgoingSets(self, 
                        _G: nx.DiGraph) -> List[List[int]]:
    '''
    Compute the set of outgoing nodes for each subsystem based on d-locality.
    If there's a path i->j <= d then j belongs to the outgoing set of i.
    '''
    if self.d is None:
      self.d = self._optimalLocalitySelection(_G)
    
    N = _G.number_of_nodes()
    # lengths of shortest path for each node
    lengths = dict(nx.all_pairs_shortest_path_length(_G))
    
    outgoing = [[] for _ in range(N)]
    for i in range(N):
      for j in range(N):
        try: 
          if lengths[i][j] <= self.d: outgoing[i].append(j)
        except: 
          continue
            
    return outgoing

  def _optimalLocalitySelection(self, _G: nx.DiGraph) -> int:
    '''
    Computes the optimal (smallest) d without degradation of MPC performance.
    https://arxiv.org/pdf/2303.11264.pdf
    '''
    # TODO
    return 1
