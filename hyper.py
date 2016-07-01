import datetime as dt
import csv
import gym
import random
import numpy as np
import tensorflow as tf
import logging

from pg_agent import PGAgent

# logging.getLogger('pg_agent').setLevel(logging.WARNING)

def main():
  seed = 1
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  env = gym.make("CartPole-v0")
  f = open('/Users/tilarids/Downloads/study_774036965_trials.csv')
  reader = csv.DictReader(f)
  for experiment in reader:
    if experiment['Status'] != 'PENDING':
      continue
    agent = PGAgent(env,
                    win_step=199,
                    H=int(experiment['H']),
                    timesteps_per_batch=int(experiment['timesteps_per_batch']),
                    learning_rate=float(experiment['learning_rate']),
                    gamma=float(experiment['gamma']),
                    epochs=int(experiment['epochs']),
                    dropout=float(experiment['dropout']),
                    win_reward=float(experiment['win_reward']))
    time_before = dt.datetime.now()
    validation_mean_rewards, train_mean_rewards = agent.learn()
    elapsed_secs = (dt.datetime.now() - time_before).seconds
    print "For TrialId=%s validation result is %s and train result is %s in %s secs" % (experiment['TrialId'], validation_mean_rewards, train_mean_rewards, elapsed_secs)

if __name__ == '__main__':
  main()
