import os
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn

from random import randint

from Agent import MLPNetwork
from pettingzoo.mpe import simple_reference_v2, simple_speaker_listener_v3

name2env = {
  'simple_reference_v2' :           simple_reference_v2.parallel_env,
  'simple_speaker_listener_v3':     simple_speaker_listener_v3.parallel_env
}

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# number of tokens in alphabet                    
n_tokens = len(alphabet)

# dimension of goal id (RGB color of landmarks)
n_goal_id = 3

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='simple_reference_v2', help='name of the env',
                    choices=list(name2env.keys()))
parser.add_argument('--num_agents', type=int, default=2, help='number of agents')
parser.add_argument('--num_landmarks', type=int, default=3, help='number of landmarks')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--num_episodes', type=int, default=20000, help='number of epochs')



if __name__ == '__main__':
  args = parser.parse_args()
  # dimensions of speaker-listener models
  dims = {
    'simple_reference_v2':        {'speaker_in':n_goal_id, 
                                  'speaker_out':n_tokens,  
                                  'listener_in':2 + 2*args.num_landmarks + n_goal_id + n_tokens, 
                                  'listener_out': 2
                                  },
    'simple_speaker_listener_v3': { 'speaker_in':n_goal_id, 
                                    'speaker_out':n_tokens,  
                                    'listener_in':2 + 2*args.num_landmarks + n_tokens, 
                                    'listener_out': 2
                                  },
  }
  # init env
  env = name2env[args.env_name](num_agents=args.num_agents, num_landmarks=args.num_landmarks)
  obs = env.reset()

  # landmark characteristics (in this case using colot)
  landmarks_c = env.aec_env.env.env.world.colors
  landmarks_c = [torch.tensor(l).unsqueeze(0) for l in landmarks_c]

  # init speaker and listener
  speaker = MLPNetwork(dims[args.env_name]['speaker_in'], dims[args.env_name]['speaker_out'])
  listener = MLPNetwork(dims[args.env_name]['listener_in'], dims[args.env_name]['listener_out'])
  params = list(speaker.parameters()) + list(listener.parameters())
  optimizer = torch.optim.Adam(params, lr=1e-4)

  loss_history = [] # store loss during training
  # training
  for i in range(args.num_episodes):
    # relative position of landmarks wrt to listener
    landmarks_p = (torch.rand((args.batch_size, 2*args.num_landmarks)) - 0.5) * 2
    # velocity of listener
    vel = torch.rand((args.batch_size, 2))
    # sample target landmark
    ix = randint(0, args.num_landmarks-1)
    # pass through observer
    msg = F.gumbel_softmax(speaker(landmarks_c[ix].repeat(args.batch_size, 1)), hard=True)
    # goal id (kinda useless to have it)
    goal_id = torch.cat(list(landmarks_c[randint(0, args.num_landmarks-1)] for _ in range(args.batch_size)), 0)
    # listener observation
    if args.env_name=='simple_reference_v2':
      obs = torch.cat((vel, landmarks_p, goal_id, msg), 1) 
    else:
      obs = torch.cat((vel, landmarks_p, msg), 1) 
    # predict landmark pos
    pred = listener(obs)
    # backprop
    optimizer.zero_grad()
    target = landmarks_p[:,ix*2:(ix+1)*2]
    #lossfun = nn.L1Loss()
    lossfun = nn.MSELoss()
    loss = lossfun(pred, target)
    loss.backward()
    optimizer.step()

    if i%100==0:
      loss_history.append(loss.item())


  # create folder to save result
  env_dir = os.path.join(os.path.abspath(''), 'results/', args.env_name)
  if not os.path.exists(env_dir):
    os.makedirs(env_dir)
  total_files = len([file for file in os.listdir(env_dir)])
  result_dir = os.path.join(env_dir, f'{total_files + 1}')
  os.makedirs(result_dir)

  # save loss plot
  plt.plot(loss_history)
  plt.xlabel('episode')
  plt.ylabel('l1 loss')
  title = f'pretraining language model {args.env_name}'
  plt.title(title)
  plt.savefig(os.path.join(result_dir, title))

  # dict with models
  models = {'speaker': speaker, 'listener': listener}

  # store models
  torch.save(
      {name: model.state_dict() for name, model in models.items()},  # actor parameter
      os.path.join(result_dir, 'model.pt')
  )

  # store args
  config = {"num_agents": args.num_agents, "num_landmarks": args.num_landmarks}

  with open(os.path.join(result_dir, 'config.json'), 'w') as f:
      json.dump(config, f)