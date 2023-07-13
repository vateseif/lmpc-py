import os
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from time import process_time

from Trainer import Trainer

repeat_eval = 6   # how many times to repeat eval before averaging
num_landmarks = 10
min_decay_log = 1
max_decay_log = 6
decay_log_range = np.linspace(min_decay_log, max_decay_log, 20)
decay_log_range = 1-10**(-decay_log_range)

drift = []
drift_std = []
s_types = ["VQ"]#, "Gumbel", "Continuous"] # types of speaker to test


if __name__ == "__main__":
    for i, s in product(decay_log_range, s_types):
        dr = []
        for _ in range(repeat_eval):
            decay = i
            trainer = Trainer(num_landmarks=num_landmarks, s_type=s, save_results=False, frozen_speaker=True, alpha=10, VQ_decay=decay)
            trainer.train()
            dr.append(trainer.vocabulary_drift / num_landmarks / 10)
        drift.append(np.mean(dr))
        drift_std.append(np.std(dr))
    # plot avg distances for each s_type during training
    plt.figure()
    #plt.plot(, , '--o')
    plt.errorbar(-np.log10(1-decay_log_range), drift, drift_std, 
                capsize=3, fmt="--.", ecolor = "black")
    plt.xscale('log')
    t1 = list(range(min_decay_log, 5)) + [6]
    t2 = [1-10**(-i) for i in range(min_decay_log, 5)] + [1]
    plt.xticks(t1, t2)
    plt.xlabel("Decay rate")
    plt.ylabel("Drift")
    plt.grid(True, which="both", linestyle="--")
    plt.legend(loc="best")
    plt.title("Vocabulary drift as function of decay rate")
    plt.savefig(os.path.join(trainer.env_dir, "VQdrift"))
    plt.close()


