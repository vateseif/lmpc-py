import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from agent import LMPCAgent, NNAgent
from pettingzoo.mpe import simple_tag_v2
from utils import evaluate_policy, SaveBestModel
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2023)

# params
n_episodes = 20000
n_frames_per_episode = 100
n_eval_episodes = 5
num_adversaries = 1
num_obstacles = 2

# tensorboard
writer = SummaryWriter(f"board/tag/{datetime.now()}")

save_best_model = SaveBestModel()

# init env
env = simple_tag_v2.parallel_env(num_adversaries=num_adversaries, num_obstacles=num_obstacles, max_cycles=n_frames_per_episode, continuous_actions=True, render_mode='rgb')
env.reset()

# init LMPC agents (adversaries) and learning-based agent
nn_agent = NNAgent(input_size=4+2*num_obstacles+2*num_adversaries, output_size=4)
adversary_agents = LMPCAgent(env.agents[:-1], "tag")


for episode in tqdm(range(n_episodes)):
  obs = env.reset()
  trajectory = []
  while env.agents:
    # LMPC agents action
    actions = adversary_agents.act(obs)
    # learning agent action
    action, log_prob = nn_agent(obs["agent_0"])
    actions["agent_0"] = np.concatenate(([1e-5], action), dtype=np.float32)
    # apply action and observe
    next_state, rewards, terminations, truncations, infos = env.step(actions)
    # cmon lets give some positive reward if not being hit
    if abs(rewards["agent_0"]) < 1: rewards["agent_0"] = 1
    trajectory.append({"state":obs["agent_0"], "action":actions["agent_0"], "reward":rewards["agent_0"], "log_prob":log_prob})
    obs = next_state

  # train agent
  policy_loss = nn_agent.train(trajectory)

  # evaluate average reward
  if episode%20==0:
    avg_reward = evaluate_policy(nn_agent, adversary_agents, env, n_eval_episodes)
    save_best_model(avg_reward, episode, nn_agent.policy, nn_agent.optimizer)

  # write to tensorboard
  writer.add_scalar("policy_loss", policy_loss, episode)
  writer.add_scalar("average reward", avg_reward, int(episode/20))
  

env.close()