import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize



class ContinuousSpeaker(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super(ContinuousSpeaker, self).__init__()
    # init simple mlp
    self.mlp = MLPNetwork(in_dim, out_dim)

  def forward(self, obs):
    # apply simple forward through mlp
    msg = self.mlp(obs)
    # return one-hot msg, msg index, 0=cmt_loss
    return msg, msg.argmax(1), 0

class GumbelSpeaker(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super(GumbelSpeaker, self).__init__()
    # init simple mlp
    self.mlp = MLPNetwork(in_dim, out_dim)

  def forward(self, obs):
    # apply gumbel softmax to get discrete message (one hot encoded)
    msg = F.gumbel_softmax(self.mlp(obs), hard=True)
    # return one-hot msg, msg index, 0=cmt_loss
    return msg, msg.argmax(1), 0

class VQSpeaker(nn.Module):
  def __init__(self, in_dim, out_dim, vq_kwargs, frozen=True) -> None:
    super(VQSpeaker, self).__init__()

    # Init MLP (in general it's a frozen pretrained model)
    self.mlp = MLPNetwork(in_dim, out_dim)
    if frozen:
      self.mlp.eval()
    # Init a VQ with EMA keys
    self.vq = VectorQuantize(**vq_kwargs)

  def forward(self, obs: torch.Tensor):
    # message value encoding, index of value in codebook, loss
    msg, idx, cmt_loss = self.vq(self.mlp(obs))
    return msg, idx, cmt_loss

class Listener(nn.Module):
  def __init__(self, in_dim, out_dim) -> None:
    super(Listener, self).__init__()
    self.mlp = MLPNetwork(in_dim, out_dim)

  def forward(self, obs: torch.Tensor):
    return self.mlp(obs)


class MLPNetwork(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
    super(MLPNetwork, self).__init__()

    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        non_linear,
        nn.Linear(hidden_dim, hidden_dim),
        non_linear,
        nn.Linear(hidden_dim, out_dim),
    ).apply(self.init)

  @staticmethod
  def init(m):
    """init parameter of the module"""
    gain = nn.init.calculate_gain('relu')
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.01)

  def forward(self, x):
    return self.net(x)