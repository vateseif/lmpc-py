from itertools import product

from Trainer import Trainer

min_landmarks = 3
max_landmarks = 26 # len of alphabet
landmarks_range = list(range(min_landmarks, max_landmarks+1))

s_types = ["Continuous"] # types of speaker to test

if __name__ == "__main__":
  for n, s in product(landmarks_range, s_types):
    trainer = Trainer(num_landmarks=n, s_type=s, save_results=True)
    trainer.train()
    trainer.evaluate()