import os
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import colorcet as cc
from tqdm import tqdm
from copy import deepcopy
from random import randint
import matplotlib.pyplot as plt

from Agent import Listener, GumbelSpeaker, VQSpeaker, ContinuousSpeaker

speaker_types = ['VQ', 'Gumbel', 'Continuous']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:

  def __init__(self, num_agents=2, num_landmarks=3, lr=1e-2, alpha=10, VQ_decay=0.99, s_type='VQ',
              frozen_speaker=True, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", save_results=False) -> None:
    assert s_type in speaker_types, f"{s_type} is not in {speaker_types}"
    # speaker type
    self.s_type = s_type
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

    # speaker vocabulary
    self.vocabulary_drift = 0
    self.vocabulary = -1*np.ones(self.num_landmarks)

    # landamrk colors
    self.landmarks_c = sns.color_palette(cc.glasbey, n_colors=num_landmarks)
    self.landmarks_c = torch.tensor(self.landmarks_c).unsqueeze(1).to(device)

    # init speaker and listeners
    self.listener = Listener(self.listener_in, self.listener_out)
    if s_type == 'VQ':
      # VQ kwargs
      vq_kwargs = {'dim':self.n_tokens, 'codebook_size':self.n_tokens, 'decay':VQ_decay, 'threshold_ema_dead_code':2}
      self.speaker = VQSpeaker(self.speaker_in, self.speaker_out, vq_kwargs, frozen=frozen_speaker)
    elif s_type == 'Gumbel':
      self.speaker = GumbelSpeaker(self.speaker_in, self.speaker_out)
    else:
      self.speaker = ContinuousSpeaker(self.speaker_in, self.speaker_out)
    
    # load models to device
    self.speaker.to(device)
    self.listener.to(device)

    # optimizer
    self.optimizer = torch.optim.Adam(list(self.speaker.parameters())+list(self.listener.parameters()), lr=lr)

    # VQ alpha param
    self.alpha = alpha

    # create dir where to save results
    self.save_results = save_results
    self.env_dir = os.path.join(os.path.abspath(''), 'results/')
    if not os.path.exists(self.env_dir):
        os.makedirs(self.env_dir)
    if self.save_results:
      total_files = len([file for file in os.listdir(self.env_dir)])
      self.result_dir = os.path.join(self.env_dir, f'{total_files + 1}_{s_type}_{num_landmarks}')
      self.speaker_vocabulary_dir = os.path.join(self.result_dir, 'speaker_vocabulary/')
      # create dir  for this training
      os.makedirs(self.result_dir)
      # create dir for speaker_vocabulary
      os.makedirs(self.speaker_vocabulary_dir)


  def plot_color_palette(self):
    plt.scatter(list(range(self.num_landmarks)), [0 for _ in range(self.num_landmarks)], marker='o', c=self.landmarks_c.squeeze())
    plt.title('Color palette')


  def train(self, loss_fun=nn.MSELoss(), epochs=2000, batch_size=1024, show=False):
    loss_history = []
    for i in tqdm(range(epochs)):
      # relative position of landmarks wrt to listener
      landmarks_p = ((torch.rand((batch_size, 2*self.num_landmarks)) - 0.5) * 2).to(device)
      landmarks_xy = landmarks_p.reshape(batch_size, self.num_landmarks, 2)
      # velocity of listener
      vel = torch.rand((batch_size, 2)).to(device)
      # sample target landmark indices
      ids = torch.randint(self.num_landmarks, (batch_size,)).to(device)
      # speaker input
      goal_landmarks = ((self.landmarks_c.repeat(batch_size, 1, 1)[ids]).squeeze(1)).to(device)
      # pass through observer
      msg, _, cmt_loss = self.speaker(goal_landmarks)
      # goal id (kinda useless to have it)
      goal_id = torch.cat(list(self.landmarks_c[randint(0, self.num_landmarks-1)] for _ in range(batch_size)), 0).to(device)
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
      loss_history.append(loss.item())
      # save speaker vocabulary every 10% of training
      if i%(epochs/10)==0:
        it = (i * 100) // epochs        
        if self.save_results and self.s_type != "Continuous":
          self.save_speaker_vocabulary(it)
        if self.s_type == "VQ":
          self.update_vocabulary()

    # plot loss
    if self.save_results or show:
      self.plot_loss(loss_history, show)    


  def export_models(self):
    speaker = deepcopy(self.speaker)
    listener = deepcopy(self.listener)
    return speaker.eval(), listener.eval()

  def computeCentroid(self, msg, l_xy):
    """Given a message (the argmax) it checks which other landmarks use the same msg and computes the centroid and the mean color"""
    xy = torch.mean(l_xy[:,self.msgLandmarkMap[msg]], dim=1)
    c = torch.mean(self.landmarks_c[self.msgLandmarkMap[msg]], dim=0)
    return xy.cpu(), c.cpu()

  def compute_msgLandmarkMap(self, speaker):
    self.msgLandmarkMap = {i: [] for  i in range(self.n_tokens)}
    for i in range(self.num_landmarks):
      _, msg_ix, _ = speaker(self.landmarks_c[i])
      self.msgLandmarkMap[msg_ix.item()].append(i)

  def update_vocabulary(self):
    # get speaker in eval mode
    speaker, _ = self.export_models()
    vocabulary = np.zeros(self.num_landmarks)
    for i in range(self.num_landmarks):
        _, msg_ix, _ = speaker(self.landmarks_c[i])
        vocabulary[i] = msg_ix
        plt.text(i-0.06, 0.01, self.alphabet[msg_ix])

    self.vocabulary_drift += np.sum(vocabulary!=self.vocabulary)
    self.vocabulary = vocabulary

  def save_speaker_vocabulary(self, it):
    # get speaker in eval mode
    speaker, _ = self.export_models()
    
    plt.figure(figsize=(12,1))
    plt.scatter(list(range(self.num_landmarks)), [0 for _ in range(self.num_landmarks)], marker='o', c=self.landmarks_c.squeeze().cpu())
    for i in range(self.num_landmarks):
      _, msg_ix, _ = speaker(self.landmarks_c[i])
      plt.text(i-0.06, 0.01, self.alphabet[msg_ix])
    plt.title(f'Speaker vocabulary at {it}% training')
    plt.axis('off')
    plt.savefig(os.path.join(self.speaker_vocabulary_dir, f"vocabulary_{it}%"), bbox_inches='tight')
    plt.close()
    

  def plot_loss(self, loss_history, show):
    # Smoothing parameters
    window_size = 10
    # Calculate the moving average
    smoothed_loss = np.convolve(loss_history, np.ones(window_size) / window_size, mode='valid')
    plt.plot(loss_history, 'b', alpha=0.5)
    plt.plot(smoothed_loss, 'b')
    plt.title(f"{self.s_type} training loss: {self.num_landmarks} landmarks ")
    plt.ylabel("MSE loss $||p_{listener} - p_{landmark}||$")
    plt.xlabel(f"Iterations")
    # if save_dir given then save the plot
    if self.save_results:
      plt.savefig(os.path.join(self.result_dir, f"loss_{self.num_landmarks}"))

    if not show:
      plt.close()


  def evaluate(self, show=False):
    # get models in eval mode
    speaker, listener = self.export_models()
    # compute msgLandmarkMap
    if self.s_type != "Continuous":
      self.compute_msgLandmarkMap(speaker)

    # create subplots
    nrows = -(self.num_landmarks // -2) # ceil division
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, int(2.5*self.num_landmarks)), dpi=200)

    # rows and columns indices
    r = list(range(nrows)) * 2
    c = [0]*nrows + [1]*nrows
    # init random landmarks pos
    landmarks_p_eval = ((torch.rand((1, 2*self.num_landmarks)) - 0.5) * 2).to(device)
    landmarks_xy_eval = landmarks_p_eval.reshape(1, self.num_landmarks, 2)
    # store final distance between listener and target landmark
    average_distance = []
    for ix in range(self.num_landmarks):
      vel = torch.rand((1, 2)).to(device)
      # pass through observer
      msg, msg_ix, _ = speaker(self.landmarks_c[ix].repeat(1, 1))
      # listener observation
      obs = torch.cat((vel, landmarks_p_eval, self.landmarks_c[randint(0, self.num_landmarks-1)], msg), 1)  
      # predict landmark pos
      action = listener(obs)
      # compute dist
      average_distance.append(np.linalg.norm(action[0].cpu().detach().numpy() - landmarks_xy_eval.cpu().detach().numpy()[0][ix]))
      # plot
      if self.save_results or show:
        axs[r[ix]][c[ix]].scatter([l for i, l in enumerate(landmarks_p_eval.cpu()[0]) if i%2==0], [l for i, l in enumerate(landmarks_p_eval.cpu()[0]) if i%2==1], marker='o', c=self.landmarks_c.squeeze().cpu())
        axs[r[ix]][c[ix]].scatter(action[0,0].cpu().detach().numpy(), action[0,1].cpu().detach().numpy(), marker='x', c=self.landmarks_c.cpu()[ix], label='listener')
        # compute centroid of chosen message
        if self.s_type != "Continuous":
          centroid_xy, centroid_c = self.computeCentroid(msg_ix.item(), landmarks_xy_eval)
          axs[r[ix]][c[ix]].scatter(centroid_xy[0, 0], centroid_xy[0, 1], marker='v', c=centroid_c, label='centroid')
          axs[r[ix]][c[ix]].set_title(f"message: {self.alphabet[msg_ix.cpu().item()]}")
        else:
          axs[r[ix]][c[ix]].set_title(f"message: [{np.round(msg.cpu().detach().numpy()[0], 1)}]")
        axs[r[ix]][c[ix]].legend(loc='best')

    if self.save_results or show:
      # access each axes object via ax.flat
      for ax in axs.flat:
        # check if something was plotted 
        if not bool(ax.has_data()):
            fig.delaxes(ax) # delete if nothing is plotted in the axes obj

    if self.save_results:
      # Save figure
      fig.savefig(os.path.join(self.result_dir, f'evaluation_{self.num_landmarks}'), bbox_inches='tight')

    if not show:
      plt.close()

    return np.mean(average_distance)
    