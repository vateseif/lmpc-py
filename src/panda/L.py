import gym
import threading
import panda_gym
import numpy as np
#import gymnasium as gym
from gpt import GPTAgent
from controller import Controller

from time import sleep


class Sim:
  def __init__(self) -> None:
    # params
    self.n_episodes = 100
    self.max_episode_length = 100000

    # env
    self.env = gym.make("PandaBuildL-v2", render=True)
    self.observation = self.env.reset()


    self.functions = {
      'x_cube1': self._x_cube1,
      'x_cube2': self._x_cube2,
      'x_cube3': self._x_cube3,
      'x_cube4': self._x_cube4,
    }

  def reset(self):
    # reset pand env
    observation = self.env.reset()
    # store observation
    self.x0 = observation[0:3]
    #self.x_cube = observation["achieved_goal"][0:3]
    #self.x_cube_target = observation["desired_goal"][0:3]
    # reset controller
    robot.set_x0(self.x0)
    robot.set_xd(self.x0)
    robot.open_gripper()

  def step(self, action: np.ndarray):
    self.observation, _, done, _ = self.env.step(action)
    # store observation
    self.x0 = self.observation[0:3]

    self.x_cube1 = self.observation[-15:-12]
    self.x_cube2 = self.observation[-18:-15]
    self.x_cube3 = self.observation[-21:-18]
    self.x_cube4 = self.observation[-24:-21]
    return done

  def _x0(self):
    return self.x0

  def _x_cube1(self):
    return self.x_cube1

  def _x_cube2(self):
    return self.x_cube2
  
  def _x_cube3(self):
    return self.x_cube3

  def _x_cube4(self):
    return self.x_cube4
  
  def run(self):
    for _ in range(self.n_episodes):
      self.reset()
      while True:
        # update controller
        robot.set_x0(self.x0)
        # compute action
        action = robot.step()
        action = np.hstack((action, robot.gripper))
        # step env
        done = self.step(action)
        
        sleep(0.1)
        if done:
            break

    self.env.close()


# controller
robot = Controller()
# simulator
sim = Sim()  

# agent
agent = GPTAgent(robot, sim)

thread = threading.Thread(target=sim.run)
thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
thread.start()


starts_msg = """
There are 4 cubes on the table and their position is defined as `x_cube1`, `x_cube2`, `x_cube3` and `x_cube4`. 
The cubes have size 0.04m, keep this into account to avoid collisions. When you grab a cube you can add constraints with some buffer distance to avoid hitting the other cubes.
Always raise the end effector when grabbing a cube to avoid collisions while moving it.

You must relocate the cubes such that their arrangement forms the letter 'L'. You have to figure out where to place the cubes to achieve this.
For example you can keep `cube_1` at its position and move the other cubes arund it to achieve the letter `L`.
"""

#agent.next_action(starts_msg, role="system")