# NOTE: use conda activate safepanda for this env
import gym
import threading
import panda_gym
import numpy as np
#import gymnasium as gym
#from gpt import GPTAgent
from controller import Controller

from time import sleep


class Sim:
  def __init__(self) -> None:
    # params
    self.n_episodes = 100
    # env
    self.env = gym.make("PandaBuildL-v2", render=True)
    self.observation = self.env.reset()

  def reset(self):
    # reset pand env
    observation = self.env.reset()
    # store observation
    self.x0 = observation[0:3]
    # reset controller
    robot.reset()

  def step(self, action: np.ndarray):
    self.observation, _, done, _ = self.env.step(action)
    # store observation
    self.x0 = self.observation[0:3]

    self.x_cube1 = self.observation[-15:-12]
    self.x_cube2 = self.observation[-18:-15]
    self.x_cube3 = self.observation[-21:-18]
    self.x_cube4 = self.observation[-24:-21]
    return done

  def run(self):
    for _ in range(self.n_episodes):
      self.reset()
      while True:
        # update controller
        robot.x0.value = self.x0
        # compute action
        action = robot.step()
        action = np.hstack((action, robot.gripper))
        # step env
        done = self.step(action)
        
        sleep(0.1)
        if done:
            break

    self.env.close()

if __name__ == "__main__":
  # controller
  robot = Controller()
  # simulator
  sim = Sim()  

  # agent
  #agent = GPTAgent(robot, sim)

  thread = threading.Thread(target=sim.run)
  thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
  thread.start()


