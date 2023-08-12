# NOTE: use conda activate safepanda for this env
import gym
import threading
import panda_gym
import numpy as np
#import gymnasium as gym
#from gpt import GPTAgent
from robot import BaseRobot
from controller import Controller

from time import sleep
from typing import Tuple


class Sim:
  def __init__(self) -> None:
    # params
    self.n_episodes = 100
    # robot
    self.robot = BaseRobot()
    # env
    self.env = gym.make("PandaBuildL-v2", render=True)
    self.observation = self.env.reset()

  def reset(self):
    # reset pand env
    observation = self.env.reset()
    # store observation
    self.x0 = observation[0:3]
    # reset controller
    self.robot.reset(self.x0)

  def step(self, action: np.ndarray):
    self.observation, _, done, _ = self.env.step(action)
    # store observation
    self.x0 = self.observation[0:3]

    self.x_cube1 = self.observation[-15:-12]
    self.x_cube2 = self.observation[-18:-15]
    self.x_cube3 = self.observation[-21:-18]
    self.x_cube4 = self.observation[-24:-21]
    return done

  def get_x_cubes(self) -> Tuple[np.ndarray]:
    return (self.x_cube1, self.x_cube2, self.x_cube3, self.x_cube4)

  def next_plan(self, plan:str):
    self.robot.next_plan(plan, self.get_x_cubes())
    return

  def run(self):
    for _ in range(self.n_episodes):
      self.reset()
      while True:
        # update controller
        self.robot.set_x0(self.x0)
        # compute action
        action = self.robot.step()
        # step env
        done = self.step(action)
        
        sleep(0.1)
        if done:
            break

    self.env.close()

if __name__ == "__main__":

  # simulator
  sim = Sim()

  thread = threading.Thread(target=sim.run)
  thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
  thread.start()

  #sim.next_plan("go to cube_1")
