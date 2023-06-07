import os
import torch
import torch.nn as nn
import seaborn as sns
import colorcet as cc
from tqdm import tqdm
from copy import deepcopy
from random import randint
import matplotlib.pyplot as plt

from Agent import Listener, GumbelSpeaker, VQSpeaker

envs = ['simple_speaker_listener_v3']
speaker_types = ['vq', 'gumbel']

class Trainer:

  def __init__(self, num_agents=2, num_landmarks=3, lr=1e-2, alpha=10, s_type='vq',
              frozen_speaker=True, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> None:
    assert s_type in speaker_types, f"{s_type} is not in {speaker_types}"
    # Env dims
    self.num_agents = num_agents
    self.num_landmarks = num_landmarks

    # Agents can communicate using 1 char from the alphabet
    self.alphabet = alphabet
    # number of tokens in alphabet                    
    self.n_tokens = len(self.alphabet)
    # dimension of goal id (RGB color of landmarks)
    self.n_goal_id = 3

    # I/O of speaker and listener
    self.speaker_in = self.n_goal_id
    self.speaker_out = self.n_tokens
    self.listener_in = 2 + 2*num_landmarks + self.n_tokens + self.n_goal_id
    self.listener_out = 2

    # landamrk colors
    self.landmarks_c = sns.color_palette(cc.glasbey, n_colors=num_landmarks)
    self.landmarks_c = torch.tensor(self.landmarks_c).unsqueeze(1)

    # init speaker and listeners
    if s_type == 'vq':
      # VQ kwargs
      vq_kwargs = {'dim':self.n_tokens, 'codebook_size':self.n_tokens, 'decay':0.95, 'threshold_ema_dead_code':2}
      self.speaker = VQSpeaker(self.speaker_in, self.speaker_out, vq_kwargs, frozen=frozen_speaker)
    else:
      self.speaker = GumbelSpeaker(self.speaker_in, self.speaker_out)
    
    self.listener = Listener(self.listener_in, self.listener_out)

    # optimizer
    self.optimizer = torch.optim.Adam(list(self.speaker.parameters())+list(self.listener.parameters()), lr=lr)

    # VQ alpha param
    self.alpha = alpha

  def plot_color_palette(self):
    plt.scatter(list(range(self.num_landmarks)), [0 for _ in range(self.num_landmarks)], marker='o', c=self.landmarks_c.squeeze())
    plt.title('Color palette')


  def train(self, loss_fun=nn.MSELoss(), epochs=2000, batch_size=1024, save_plot=False):
    loss_history = []
    for i in tqdm(range(epochs)):
      # relative position of landmarks wrt to listener
      landmarks_p = (torch.rand((batch_size, 2*self.num_landmarks)) - 0.5) * 2
      landmarks_xy = landmarks_p.reshape(batch_size, self.num_landmarks, 2)
      # velocity of listener
      vel = torch.rand((batch_size, 2))
      # sample target landmark indices
      ids = torch.randint(self.num_landmarks, (batch_size,))
      # speaker input
      goal_landmarks = (self.landmarks_c.repeat(batch_size, 1, 1)[ids]).squeeze(1)
      # pass through observer
      msg, _, cmt_loss = self.speaker(goal_landmarks)
      # goal id (kinda useless to have it)
      goal_id = torch.cat(list(self.landmarks_c[randint(0, self.num_landmarks-1)] for _ in range(batch_size)), 0)
      # listener obesrvation
      obs = torch.cat((vel, landmarks_p, goal_id, msg), 1)
      # predict landmark pos
      pred = self.listener(obs)
      # backprop
      self.optimizer.zero_grad()
      target = landmarks_xy[torch.arange(batch_size), ids]
      loss = loss_fun(pred, target) + self.alpha * cmt_loss
      loss.backward()
      self.optimizer.step()

      if i%100==0:
        loss_history.append(loss.item())

    # plot loss
    plt.plot(loss_history)
    plt.title("Training loss $||p_{listener} - p_{landmark}||$")
    plt.ylabel(f"MSE loss")
    plt.xlabel("Iterations")

    # if save_dir given then save the plot
    if save_plot:
      env_dir = os.path.join(os.path.abspath(''), 'results/')
      if not os.path.exists(env_dir):
        os.makedirs(env_dir)
      total_files = len([file for file in os.listdir(env_dir)])
      result_dir = os.path.join(env_dir, f'{total_files + 1}')
      os.makedirs(result_dir)
      plt.savefig(os.path.join(result_dir, "loss"))    


  def export_models(self):
    speaker = deepcopy(self.speaker)
    listener = deepcopy(self.listener)
    return speaker.eval(), listener.eval()

  def computeCentroid(self, msg, l_xy):
    """Given a message (the argmax) it checks which other landmarks use the same msg and computes the centroid and the mean color"""
    xy = torch.mean(l_xy[:,self.msgLandmarkMap[msg]], dim=1)
    c = torch.mean(self.landmarks_c[self.msgLandmarkMap[msg]], dim=0)
    return xy, c

  def compute_msgLandmarkMap(self, speaker):
    self.msgLandmarkMap = {i: [] for  i in range(self.n_tokens)}
    for i in range(self.num_landmarks):
      _, msg_ix, _ = speaker(self.landmarks_c[i])
      self.msgLandmarkMap[msg_ix.item()].append(i)

  def evaluate(self):
    # get models in eval mode
    speaker, listener = self.export_models()
    # compute msgLandmarkMap
    self.compute_msgLandmarkMap(speaker)

    # create subplots
    nrows = -(self.num_landmarks // -2) # ceil division
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, int(2.5*self.num_landmarks)), dpi=200)

    # rows and columns indices
    r = list(range(nrows)) * 2
    c = [0]*nrows + [1]*nrows
    # init random landmarks pos
    landmarks_p_eval = (torch.rand((1, 2*self.num_landmarks)) - 0.5) * 2
    landmarks_xy_eval = landmarks_p_eval.reshape(1, self.num_landmarks, 2)
    for ix in range(self.num_landmarks):
      vel = torch.rand((1, 2))
      # pass through observer
      msg, msg_ix, _ = speaker(self.landmarks_c[ix].repeat(1, 1))
      # listener observation
      obs = torch.cat((vel, landmarks_p_eval, self.landmarks_c[randint(0, self.num_landmarks-1)], msg), 1)  
      # predict landmark pos
      action = listener(obs)
      # compute centroid of chosen message
      centroid_xy, centroid_c = self.computeCentroid(msg_ix.item(), landmarks_xy_eval)

      axs[r[ix]][c[ix]].scatter([l for i, l in enumerate(landmarks_p_eval[0]) if i%2==0], [l for i, l in enumerate(landmarks_p_eval[0]) if i%2==1], marker='o', c=self.landmarks_c.squeeze())
      axs[r[ix]][c[ix]].scatter(centroid_xy[0, 0], centroid_xy[0, 1], marker='v', c=centroid_c, label='centroid')
      axs[r[ix]][c[ix]].scatter(action[0,0].detach().numpy(), action[0,1].detach().numpy(), marker='x', c=self.landmarks_c[ix], label='listener')
      axs[r[ix]][c[ix]].legend(loc='best')
      axs[r[ix]][c[ix]].set_title(f"message: {self.alphabet[msg_ix.item()]}")

    # access each axes object via ax.flat
    for ax in axs.flat:
      ## check if something was plotted 
      if not bool(ax.has_data()):
          fig.delaxes(ax) # delete if nothing is plotted in the axes obj
    