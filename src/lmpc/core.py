import inspect
import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod

class ObjectiveFunc(ABC):
  
  @abstractmethod
  def compute():
    pass


class Constraint(ABC):
  
  @abstractmethod
  def compute():
    pass

class ObjBase:
  '''
  The object base that defines debugging tools
  '''
  def initialize (self, **kwargs):
    pass

  def sanityCheck (self):
    # check the system parameters are coherent
    return True

  def errorMessage (self,msg):
    print(self.__class__.__name__+'-'+inspect.stack()[1][3]+': [ERROR] '+msg+'\n')
    return False

  def warningMessage (self,msg):
    print(self.__class__.__name__+'-'+inspect.stack()[1][3]+': [WARNING] '+msg+'\n')
    return False


class LocalityModel(ObjBase):
  ''' 
  Abstract class that governs the information flow between subsystems
  '''
  def __init__(self) -> None:
    self.out_x : List[List[int]]
    self.out_u : List[List[int]]
    pass

  @abstractmethod
  def computeOutgoingSets(self):
    # TODO: change name of function?
    pass
  
  @abstractmethod
  def _updateOutgoingSets(self):
    # in case locality is time-varying
    return


class SystemModel(ObjBase):
  ''' 
  Abstract class that models the system 
  '''
  def __init__(self) -> None:
    self._x: Optional[np.ndarray] = None
    self.locality_model: Optional[LocalityModel] = None
    pass

  def getState(self) -> np.ndarray:
    return self._x.copy()

  def setInitialState(self, x: np.ndarray) -> None:
    self._x = x

  def updateState(self, x: np.ndarray) -> None:
    assert self._x.shape == x.shape, "x doesnt match shape of self._x"
    self._x = x

  

class ControllerModel(ObjBase):
  ''' 
  Abstract class that computes the control actions
  '''
  def __init__(self) -> None:
    self.model: Optional[SystemModel] = None
    pass

  @abstractmethod
  def solve(self, x0: np.ndarray) -> np.ndarray:
    pass





  