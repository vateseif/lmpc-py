import threading
import numpy as np
import gymnasium as gym
from time import sleep

from pid import PID
from lqr import LQR
from gpt import GPTTuner

env = gym.make('CartPole-v1', render_mode='human')

#ctrl = PID()
ctrl = LQR()
tuner = GPTTuner(ctrl)

def run():
  while True:
    state = env.reset()[0]
    ctrl.reset()
    for t in range(500):
        env.render()
        # compute action
        action = ctrl.act(state)
        # step
        state, reward, termination, done, info = env.step(action)
        if done or termination:
            break
env.close()

# The pole starts oscillating slowly and the osccilations keep increasing until it goes unstable

thread = threading.Thread(target=run)
thread.daemon = True  # Set the thread as a daemon (will exit when the main program ends)
thread.start()

sleep(5)

feedback_i = 0
while True:
  feedback_i += 1
  user_msg = input("How is the system behaving? \n")
  try:
    #tuner.apply_action(user_msg)
    tuner.next_action(feedback_i, user_msg)
  except:
    print("failed to apply action")
  #print(ctrl.P, ctrl.I, ctrl.D)
  print(ctrl.Q)
