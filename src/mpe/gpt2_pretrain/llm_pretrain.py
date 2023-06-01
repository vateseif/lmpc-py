import os
import torch
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2Model, GPT2LMHeadModel
from transformers.generation.configuration_utils import GenerationConfig

# free up GPU memory
torch.cuda.empty_cache()

# params
n_embed = 768
batch_size = 256

# RL training params
lr = 4e-3
n_episodes = 3000
n_landmarks = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# IO dimensions
speaker_in = 3
speaker_out = 5
listener_in = n_landmarks*2
listener_out = 2

# speaker generation params
num_beams = 5
early_stopping = True
no_repeat_ngram_size = 1
do_sample = False
temperature = 0.9
top_k = 50
top_p = 0.9
max_length = speaker_out


# landamrk colors
landmarks_c = sns.color_palette(n_colors=n_landmarks)
landmarks_c = torch.tensor(landmarks_c).unsqueeze(1)



class Speaker(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    # init GPT2
    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    self.transformer = GPT2LMHeadModel.from_pretrained("gpt2")

    # config file when generating text
    self.generation_config = GenerationConfig.from_model_config(self.transformer.config)

    # freeze GPT2 params
    for param in self.transformer.parameters():
      param.requires_grad = False

    # custom embedding layer for obs
    self.embed_obs = nn.Linear(speaker_in, n_embed)

    # layer norm
    self.embed_ln = nn.LayerNorm(n_embed)


  def forward(self, obs: torch.Tensor):
    ''' 
    The observation of the speaker is the color of the target landmark of the listener. 
    Given the color, the speaker returns a message to the listener.
    '''

    # dims
    batch_size = obs.shape[0]

    # compute embeddings of observation (the color)
    obs_embeddings = self.embed_ln(self.embed_obs(obs))
    attention_mask = torch.ones((batch_size, obs_embeddings.shape[1])).to(device)
    # generate text (sampling based)
    msg = self.transformer.generate(
                          inputs_embeds=obs_embeddings,
                          attention_mask=attention_mask, 
                          generation_config=self.generation_config, 
                          num_beams=num_beams,
                          early_stopping=early_stopping,
                          no_repeat_ngram_size=no_repeat_ngram_size,
                          do_sample=do_sample,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p,
                          max_length=max_length,
                          pad_token_id=self.tokenizer.eos_token_id)

    return msg


class Listener(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    # init GPT2
    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    self.transformer = GPT2Model.from_pretrained("gpt2")

    # freeze GPT2 params
    for param in self.transformer.parameters():
      param.requires_grad = False

    # GPT2 word embedding layer
    self.wte = nn.Embedding.from_pretrained(self.transformer.wte.weight)

    # custom embedding layer for processing input obs
    self.embed_obs = nn.Linear(listener_in, n_embed)

    # layer norm
    self.embed_ln = nn.LayerNorm(n_embed)

    # custom prediction layer for computing the action
    self.predict_action = nn.Linear(n_embed*(1+speaker_out), listener_out)

  def forward(self, obs:torch.Tensor, msg:torch.Tensor):
    '''
    obs: position of landmarks
    msg: message from speaker indicating which landmark to go to
    '''

    batch_size = obs.shape[0]

    # compute and stack embeddings
    word_embeddings = self.wte(msg)
    obs_embeddings = self.embed_obs(obs).unsqueeze(1)
    stacked_inputs = torch.cat((word_embeddings, obs_embeddings), 1)
    stacked_inputs = self.embed_ln(stacked_inputs)

    # compute transformer hidden states
    hidden_state = self.transformer(inputs_embeds=stacked_inputs)['last_hidden_state']

    # reshape hidden states and pass through prediction layer
    x = hidden_state.view((batch_size, -1))
    action = self.predict_action(x)

    return action

if __name__ == '__main__':
  # init models
  speaker = Speaker().to(device)
  listener = Listener().to(device)

  # init optimizer
  optimizer = torch.optim.Adam(params=list(speaker.parameters())+list(listener.parameters()), lr=lr)

  # loss function
  #lossfun = nn.L1Loss()
  lossfun = nn.MSELoss()

  loss_history = []

  for i in tqdm(range(n_episodes)):
    # random landmark position
    landmarks_p = ((torch.rand((batch_size, 2*n_landmarks)) - 0.5) * 2).to(device)
    # sample target landmark indices
    ids = torch.randint(n_landmarks, (batch_size,))
    # speaker input
    goal_landmarks = (landmarks_c.repeat(batch_size, 1, 1)[ids]).to(device)
    # messages from speaker
    msg = speaker(goal_landmarks)
    # compute action from lisener
    action = listener(landmarks_p, msg)
    # ground truth target actions
    target = torch.cat([landmarks_p[i, ix*2:(ix+1)*2].unsqueeze(0) for i, ix in enumerate(ids)], 0)
    # backprop
    optimizer.zero_grad()
    loss = lossfun(action, target)
    loss.backward()
    optimizer.step()
    
    if i%10==0:
      loss_history.append(loss.item())

  # create folder to save result
  env_dir = os.path.join(os.path.abspath(''), 'results/', "GPT2")
  if not os.path.exists(env_dir):
    os.makedirs(env_dir)
  total_files = len([file for file in os.listdir(env_dir)])
  result_dir = os.path.join(env_dir, f'{total_files + 1}')
  os.makedirs(result_dir)

  # save loss plot
  plt.plot(loss_history)
  plt.xlabel('episode')
  plt.ylabel('l1 loss')
  plt.ylim([0., 1.])
  title = f'pretraining language model GPT2'
  plt.title(title)
  plt.savefig(os.path.join(result_dir, title))

  # dict with models
  models = {
    'speaker': {
      'embed_obs':  speaker.embed_obs.state_dict(),
      'embed_ln':   speaker.embed_ln.state_dict(),
    }, 
    'listener': {
      'embed_obs':      listener.embed_obs.state_dict(),
      'embed_ln':       listener.embed_ln.state_dict(),
      'predict_action': listener.predict_action.state_dict()
    }
  }

  # store models
  torch.save(
      models,  # actor parameter
      os.path.join(result_dir, 'model.pt')
  )