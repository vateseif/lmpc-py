import numpy as np
import gymnasium as gym

from pid import PID
from gpt import GPTTuner

env = gym.make('CartPole-v1', render_mode='human')

ctrl = PID(4)
tuner = GPTTuner(function_call=ctrl.update)
# The pole starts oscillating slowly and the osccilations keep increasing until it goes unstable
for i_episode in range(20):
    state = env.reset()[0]
    ctrl.reset()

    if i_episode > 0:
      user_msg = input("How is the system behaving? >>")
      tuner.next_action(i_episode, user_msg)
      print(ctrl.parameters)

    for t in range(500):
        env.render()

        # compute action
        action = ctrl.act(state)
        # step
        state, reward, termination, done, info = env.step(action)
        if done or termination:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()