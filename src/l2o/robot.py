import numpy as np
from typing import Tuple
from core import AbstractRobot
from config.config import BaseRobotConfig
from controller import BaseController, ControllerOptions
from llm import TaskPlanner, OptimizationDesignerOptions, Optimization


class BaseRobot(AbstractRobot):

  def __init__(self, cfg=BaseRobotConfig()) -> None:
    self.cfg = cfg

    self.TP = TaskPlanner()
    self.OD = OptimizationDesignerOptions[self.cfg.od_type]()
    self.MPC: BaseController = ControllerOptions[self.cfg.controller_type]()

  def set_x0(self, x0: np.ndarray):
    self.MPC.x0.value = x0
    return

  def create_plan(self):
    plan = self.TP.run()
    return plan

  def next_plan(self, plan:str, x_cubes: Tuple[np.ndarray]):
    # if custom function is called apply that
    if "open" in plan and "gripper" in plan:
      self.MPC.open_gripper()
      return
    elif "close" in plan and "gripper" in plan:
      self.MPC.close_gripper()
      return
    # 
    optimization: Optimization = self.OD.run(plan)
    #optimization = Optimization(objective="sum([cp.norm(self.x[t] - cube_1 + np.array([-0.06, 0, 0])) for t in range(self.cfg.T)])")
    #optimization = Optimization(objective="cp.norm(self.x[-1] - cube_1 + np.array([-0.06, 0, 0]))") 
 
    self.MPC.apply_gpt_message(optimization.objective, x_cubes)
    return


  def step(self):
    action = self.MPC.step() 
    return np.hstack((action, self.MPC.gripper))