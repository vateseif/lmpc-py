import numpy as np
from typing import Tuple
from llm import BaseLLM
from core import AbstractRobot
from config.config import BaseRobotConfig, BaseLLMConfigs
from controller import BaseController, ControllerOptions



class BaseRobot(AbstractRobot):

  def __init__(self, cfg=BaseRobotConfig()) -> None:
    self.cfg = cfg

    self.TP = BaseLLM(BaseLLMConfigs["plan"]())
    self.OD = BaseLLM(BaseLLMConfigs["objective"]())
    self.MPC: BaseController = ControllerOptions[self.cfg.controller_type]()

  def set_x0(self, x0: np.ndarray):
    self.MPC.x0.value = x0
    return

  def create_plan(self):
    plan = self.TP.run("")
    return plan.tasks # TODO: plan.tasks is hardcoded here

  def next_plan(self, plan:str, x_cubes: Tuple[np.ndarray]):
    # if custom function is called apply that
    if "open" in plan and "gripper" in plan:
      self.MPC.open_gripper()
      return
    elif "close" in plan and "gripper" in plan:
      self.MPC.close_gripper()
      return
    # 
    optimization = self.OD.run(plan)

    self.MPC.apply_gpt_message(optimization.objective, x_cubes) # TODO: optimization.objective is hardcoded here
    return


  def step(self):
    action = self.MPC.step() 
    return np.hstack((action, self.MPC.gripper))