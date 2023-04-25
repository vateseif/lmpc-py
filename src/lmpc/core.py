import inspect
import numpy as np

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
    pass

  def computeCommunicationGraph():
    return


class SystemModel(ObjBase):
  ''' 
  Abstract class that models the system 
  '''
  def __init__(self) -> None:
    self._x = np.empty([0])

  # overload the lshift operator to enhance model with LocalityModel
  def __lshift__ (self, locality_model: LocalityModel):
    assert isinstance(locality_model, LocalityModel), "locality_model is not of type LocalityModel"
    self.locality_model = locality_model
    return self

  def getState(self) -> np.ndarray:
    return self._x.copy()

  

class ControllerModel(ObjBase):
  ''' 
  Abstract class that computes the control actions
  '''
  def __init__(self) -> None:
    return

  def solve(x0: np.ndarray) -> np.ndarray:
    return None





  