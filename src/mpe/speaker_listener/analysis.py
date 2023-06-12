import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from time import process_time

from Trainer import Trainer

repeat_eval = 4   # how many times to repeat eval before averaging
min_landmarks = 3
max_landmarks = 26 # len of alphabet
landmarks_range = list(range(min_landmarks, max_landmarks+1))

s_types = ["VQ", "Gumbel", "Continuous"] # types of speaker to test
average_distances = {s:[] for s in s_types} # distances to target landmark during training
std_distances = {s:[] for s in s_types} # standard dev of distances to target landmark
compute_time = {s:[] for s in s_types}

if __name__ == "__main__":
  for n, s in product(landmarks_range, s_types):
    dist = []
    p_time = []
    for _ in range(repeat_eval):
      trainer = Trainer(num_landmarks=n, s_type=s, save_results=False)
      trainer.train()
      t0 = process_time()
      dist.append(trainer.evaluate()) # repeat eval 4 times
      p_time.append(process_time()-t0)
    average_distances[s].append(np.mean(dist)) # store avg distance
    std_distances[s].append(np.std(dist)) # store std of avg distance
    compute_time[s].append(np.mean(p_time))

  # plot avg distances for each s_type during training
  plt.figure()
  for s in s_types:
    #plt.plot(landmarks_range, average_distances[s], 'o--',label=s)
    plt.errorbar(landmarks_range, average_distances[s], std_distances[s], 
                capsize=3, fmt="--.", ecolor = "black", label=s)      
  plt.xlabel("Number of landmarks")
  plt.ylabel("Average distance")
  plt.legend(loc="best")
  plt.title("Listener's average terminal distance to target landmark")
  plt.savefig(os.path.join(trainer.env_dir, "avg_distance"))
  plt.close()

  plt.figure()
  for s in s_types:
    plt.plot(landmarks_range, compute_time[s], 'o--',label=s)   
  plt.xlabel("Number of landmarks")
  plt.ylabel("Computation time [s]")
  plt.legend(loc="best")
  plt.title("Computation time")
  plt.savefig(os.path.join(trainer.env_dir, "computation_time"))
  plt.close()
