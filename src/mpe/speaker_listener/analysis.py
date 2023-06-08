import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from Trainer import Trainer

min_landmarks = 3
max_landmarks = 26 # len of alphabet
landmarks_range = list(range(min_landmarks, max_landmarks+1))

s_types = ["VQ", "Gumbel", "Continuous"] # types of speaker to test
average_distances = {s:[] for s in s_types} # distances to target landmark during training

if __name__ == "__main__":
  for n, s in product(landmarks_range, s_types):
    dist = []
    for _ in range(4):
      trainer = Trainer(num_landmarks=n, s_type=s, save_results=False)
      trainer.train()
      dist.append(trainer.evaluate()) # repeat eval 4 times
    average_distances[s].append(np.mean(dist)) # store avg distance

  # plot avg distances for each s_type during training
  for s in s_types:
    plt.plot(landmarks_range, average_distances[s], 'o--',label=s)
  plt.xlabel("Number of landmarks")
  plt.ylabel("Average distance")
  plt.legend(loc="best")
  plt.title("Listener's average terminal distance to target landmark")
  plt.savefig(os.path.join(trainer.env_dir, "avg_distance"))
  plt.close()
