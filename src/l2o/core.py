import os
import inspect
import numpy as np
from abc import abstractmethod
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union


class AbstractLLMConfig:
  prompt: str
  model_name: str
  temperature: float

class AbstractControllerConfig:
  T: int 
  nx: int  
  nu: int  
  dt: float 
  lu: float # lower bound on u
  hu: float # higher bound on u

class AbstractRobotConfig:
  name: str



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



class AbstractController(ObjBase):

  def __init__(self, cfg: AbstractControllerConfig) -> None:
    self.cfg = cfg

  @abstractmethod
  def reset(self, x0:np.ndarray) -> None:
    return

  @abstractmethod
  def apply_gpt_message(self, gpt_message:str) -> None:
    return
  
  @abstractmethod
  def step(self, obs:np.ndarray) -> np.ndarray:
    return


class AbstractLLM(ObjBase):

  def __init__(self, cfg:AbstractLLMConfig) -> None:
    self.cfg = cfg

  @abstractmethod
  def run(self):
    return



class AbstractRobot(ObjBase):

  def __init__(self, cfg:AbstractRobotConfig) -> None:
    self.cfg = cfg

    # components
    self.TP: AbstractLLM          # Task planner
    self.OD: AbstractLLM          # Optimization Designer
    self.MPC: AbstractController  # Controller