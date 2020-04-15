# From the course: Bayesian Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np

# initialize constants
NUM_TRIALS = 10000
EPSILON = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
  def __init__(self, p):
    self.p = p #win rate
    self.p_estimate = 0 #estimated win rate
    self.N = 0

# draw a 1 with probability p
  def pull(self):
    return np.random.random() < self.p

# update current estimate of this bandit's mean
  def update(self, x):
    self.N = self.N+1
    self.p_estimate = ((self.N-1)*self.p_estimate+x)/self.N


def experiment():
  #initialize bandits with their respective probabilities
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

  rewards = np.zeros(NUM_TRIALS)
  num_times_explored = 0
  num_times_exploited = 0
  num_optimal = 0
  optimal_j = np.argmax([b.p for b in bandits])
  print("optimal j:", optimal_j)

# epsilon-greedy algorithm
  for i in range(NUM_TRIALS):
    # explore a random bandit
    if np.random.random() < EPSILON:
      num_times_explored += 1
      j = np.random.randint(len(bandits))

    # pick the best bandit
    else:
      num_times_exploited += 1
      j = np.argmax([b.p_estimate for b in bandits])

    # count the number of times we selected optimal bandit
    if j == optimal_j:
      num_optimal += 1

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()

    # update rewards log
    rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)

    

  # print mean estimates for each bandit
  for b in bandits:
    print("mean estimate:", b.p_estimate)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num_times_explored:", num_times_explored)
  print("num_times_exploited:", num_times_exploited)
  print("num times selected optimal bandit:", num_optimal)

  # plot the results
  cumulative_rewards = np.cumsum(rewards)
  win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
  plt.plot(win_rates)
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
  plt.show()

if __name__ == "__main__":
  experiment()
