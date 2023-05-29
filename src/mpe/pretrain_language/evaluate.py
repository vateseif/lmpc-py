import os
import json
import argparse
import numpy as np
from PIL import Image

from System import System
from pretrain_language import name2env



parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='simple_reference_v2', help='name of the env',
                    choices=list(name2env.keys()))
parser.add_argument('--num_episodes', type=int, default=10, help='number of epochs')
parser.add_argument('--model_dir', type=str, default='1', help='numbder of directory to get model from')

ep_len = 50
discrete = False


def get_env(env_name, cfg):
  # init env
  env = name2env[env_name](num_agents=cfg['num_agents'], num_landmarks=cfg['num_landmarks'], max_cycles=ep_len, continuous_actions = not discrete)
  obs = env.reset()
  world = env.aec_env.env.env.world
  # get agents dimensions
  dims = {}
  for agent_id in env.agents:
    agent_dims = {'num_landmarks': cfg['num_landmarks']}
    if agent_id.startswith('speaker') or agent_id.startswith('agent'): 
      agent_dims['speaker_in'] = 3
      agent_dims['speaker_out'] = world.dim_c
    if agent_id.startswith('listener') or agent_id.startswith('agent'): 
      agent_dims['listener_out'] =  2
      agent_dims['listener_in'] = env.observation_space(agent_id).shape[0]
    dims[agent_id] = agent_dims
  return env, dims


if __name__ == '__main__':
  # args
  args = parser.parse_args()
  # directory with pretrained models and to save gifs
  model_dir = os.path.join(os.path.abspath(''), 'results/', args.env_name, args.model_dir)
  assert os.path.exists(model_dir), f'{model_dir} doesnt exist'
  gif_dir = os.path.join(model_dir, 'gif')
  if not os.path.exists(gif_dir):
      os.makedirs(gif_dir)
  gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif
  # read config file with env params
  with open(os.path.join(model_dir, 'config.json'), 'r') as f:
    cfg = json.load(f)
  # env
  env, dims = get_env(args.env_name, cfg)
  # init agents
  system = System.load(dims, os.path.join(model_dir, 'model.pt'))
  episode_rewards = {agent: np.zeros(args.num_episodes) for agent in env.agents}
  
  for episode in range(args.num_episodes):
    states = env.reset()
    frame_list = []  # used to save gif
    agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
    while env.agents:  # interact with the env for an episode
      actions = system.select_action(states)
      #actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
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