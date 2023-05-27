import os
import numpy as np
from PIL import Image

from pettingzoo.mpe import simple_reference_v2

from System import System

ep_len = 50
discrete = False
n_tokens = 10
n_episodes = 10


def get_env():
  # init env
  env = simple_reference_v2.parallel_env(max_cycles=ep_len, continuous_actions = not discrete)
  env.reset()
  # get agents dimensions
  dims = {}
  for agent_id in env.agents:
    agent_dims = {'listener_out': 2, 'speaker_in': 3, 'speaker_out': 10}
    agent_dims['listener_in'] = env.observation_space(agent_id).shape[0]
    dims[agent_id] = agent_dims
  
  return env, dims


if __name__ == '__main__':
  # env
  env, dims = get_env()
  # directory with pretrained models and to save gifs
  model_dir = os.path.join(os.path.abspath(''), 'results/', 'simple_reference_v2', '1')
  assert os.path.exists(model_dir), f'{model_dir} doesnt exist'
  gif_dir = os.path.join(model_dir, 'gif')
  if not os.path.exists(gif_dir):
      os.makedirs(gif_dir)
  gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif
  # init agents
  system = System.load(dims, os.path.join(model_dir, 'model.pt'))
  episode_rewards = {agent: np.zeros(n_episodes) for agent in env.agents}
  
  for episode in range(n_episodes):
    states = env.reset()
    frame_list = []  # used to save gif
    agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
    while env.agents:  # interact with the env for an episode
      actions = system.select_action(states)
      states, rewards, dones, infos = env.step(actions)
      env.render(mode='rgb_array')
      # store frame and sum rewards
      frame_list.append(Image.fromarray(env.render(mode='rgb_array')))
      for agent_id, reward in rewards.items():  # update reward
          agent_reward[agent_id] += reward

    env.close()
    message = f'episode {episode + 1}, '
    # episode finishes, record reward
    for agent_id, reward in agent_reward.items():
      episode_rewards[agent_id][episode] = reward
      message += f'{agent_id}: {reward:>4f}; '
    print(message)
    # save gif
    frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                       save_all=True, append_images=frame_list[1:], duration=1, loop=0)