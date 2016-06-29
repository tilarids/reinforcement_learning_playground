import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import prettytensor as pt
import tempfile
import csv
import datetime as dt
import sys

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32



# hyperparameters
# H = 16 # number of hidden layer neurons
# learning_rate = 1e-3
GAMMA = 0.99 # discount factor for reward
# decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False
monitor = False
# epochs = 15


def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(xrange(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist

def discount_rewards(r, gamma = GAMMA):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


class LearningAgent(object):
  def __init__(self, env, H, timesteps_per_batch, learning_rate, gamma, epochs, dropout, win_reward):
    if not isinstance(env.observation_space, Box) or \
       not isinstance(env.action_space, Discrete):
        print("Incompatible spaces.")
        exit(-1)

    self.H = H
    self.timesteps_per_batch = timesteps_per_batch
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.epochs = epochs
    self.dropout = dropout
    self.win_reward = win_reward

    self.env = env
    self.session = tf.Session()
    self.obs = tf.placeholder(
        dtype, shape=[
            None, 2 * env.observation_space.shape[0] + env.action_space.n], name="obs")
    self.prev_obs = np.zeros(env.observation_space.shape[0])
    self.prev_action = np.zeros(env.action_space.n)
    self.action = action = tf.placeholder(dtype, shape=[None, env.action_space.n], name="action")
    self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
    self.prev_policy = action = tf.placeholder(dtype, shape=[None, env.action_space.n], name="prev_policy")

    self.policy_network, _ = (
        pt.wrap(self.obs)
            .fully_connected(H, activation_fn=tf.nn.tanh)
            .dropout(self.dropout)
            .softmax_classifier(env.action_space.n))
    self.returns = tf.placeholder(dtype, shape=[None, env.action_space.n], name="returns")

    loss = - tf.reduce_sum(tf.mul(self.action, tf.div(self.policy_network, self.prev_policy)), 1) * self.advant
    self.train = tf.train.AdamOptimizer().minimize(loss)

    self.session.run(tf.initialize_all_variables())

  def rollout(self, max_pathlength, timesteps_per_batch):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < timesteps_per_batch:

      obs, actions, rewards, action_dists, actions_one_hot = [], [], [], [], []
      ob = self.env.reset()
      self.prev_action *= 0.0
      self.prev_obs *= 0.0
      for x in xrange(max_pathlength):
        if render and 0==x % 20: env.render()
        # import pdb; pdb.set_trace()
        obs_new = np.expand_dims(np.concatenate([ob, self.prev_obs, self.prev_action], 0), 0)

        action_dist_n = self.session.run(self.policy_network, {self.obs: obs_new})

        action = int(cat_sample(action_dist_n)[0])
        self.prev_obs = ob
        self.prev_action *= 0.0
        self.prev_action[action] = 1.0

        obs.append(ob)
        actions.append(action)
        action_dists.append(action_dist_n)
        actions_one_hot.append(np.copy(self.prev_action))

        res = self.env.step(action)
        if res[2] and 199==len(rewards):
          rewards.append(self.win_reward)
        else:
          rewards.append(res[1])
        ob = res[0]

        if res[2]:
            path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                    "action_dists": np.concatenate(action_dists),
                    "rewards": np.array(rewards),
                    "actions": np.array(actions),
                    "actions_one_hot": np.array(actions_one_hot)}
            paths.append(path)
            self.prev_action *= 0.0
            self.prev_obs *= 0.0
            timesteps_sofar += len(path["rewards"])
            break
      else:
        timesteps_sofar += max_pathlength
    return paths

  def prepare_features(self, path):
    # import pdb; pdb.set_trace()
    obs = path["obs"]
    prev_obs = np.concatenate([np.zeros((1,obs.shape[1])), path["obs"][1:]], 0)
    prev_action = path['action_dists']
    return np.concatenate([obs, prev_obs, prev_action], 1)

  def predict_for_path(self, path):
    features = self.prepare_features(path)
    return self.session.run(self.policy_network, {self.obs: features})

  def learn(self):
    self.current_observation = env.reset()
#    self.prev_x = None # used in computing the difference frame

    xs,hs,dlogps,drs = [],[],[],[]

    running_reward = None
    reward_sum = 0
    episode_number = 0
    current_step = 0.0
    iteration_number = 0
    discounted_eprs = []
    mean_path_lens = []
    while True:

      paths = self.rollout(max_pathlength=10000, timesteps_per_batch=self.timesteps_per_batch)
      if not paths:
        env.monitor.start(training_dir)
        self.rollout(max_pathlength=10000, timesteps_per_batch=200*200)
        env.monitor.close()
        gym.upload(training_dir, api_key='sk_lgS7sCv1Qxq5HFMdQXR6Sw')
        sys.exit(0)

      for path in paths:
        path["prev_policy"] = self.predict_for_path(path)
        path["returns"] = discount_rewards(path["rewards"], self.gamma)
        # path["advant"] = path["returns"] - path["baseline"]
        path["advant"] = path["returns"]

      features = np.concatenate([self.prepare_features(path) for path in paths])

      advant = np.concatenate([path["advant"] for path in paths])
      advant -= advant.mean()
      advant /= (advant.std() + 1e-8)

      actions = np.concatenate([path["actions_one_hot"] for path in paths])
      prev_policy = np.concatenate([path["prev_policy"] for path in paths])

      # predictions = np.concatenate([self.predict_for_path(path) for path in paths])
      for _ in range(self.epochs):
        self.session.run(self.train, {self.obs: features,
                                      self.advant: advant,
                                      self.action: actions,
                                      self.prev_policy: prev_policy})
      iteration_number += 1

      mean_path_len = np.mean([len(path['rewards']) for path in paths])
      mean_path_lens.append(mean_path_len)
      if iteration_number > 100:
        paths = self.rollout(max_pathlength=10000, timesteps_per_batch=10000)
        return np.mean([len(path['rewards']) for path in paths]), np.mean(mean_path_lens)
      # if 0 == iteration_number % 25:
      #   print "Iteration %s finished. Mean path len: %s" % (iteration_number, mean_path_lens[-10:])

env = gym.make("CartPole-v0")
training_dir = tempfile.mkdtemp()
if monitor:
  env.monitor.start(training_dir)

# H, timesteps_per_batch, learning_rate, gamma, epochs, dropout, win_reward
#EXPERIMENTS = [(16, 1000, 1e-3, 0.99, 15, 0.75, 50)]

f = open('/Users/tilarids/Downloads/study_3013685965_trials.csv')
reader = csv.DictReader(f)
for experiment in reader:
  if experiment['Status'] != 'PENDING':
    continue
  agent = LearningAgent(env,
                        int(experiment['H']),
                        int(experiment['timesteps_per_batch']),
                        float(experiment['learning_rate']),
                        float(experiment['gamma']),
                        int(experiment['epochs']),
                        float(experiment['dropout']),
                        float(experiment['win_reward']))
  time_before = dt.datetime.now()
  validation_mean_rewards, train_mean_rewards = agent.learn()
  elapsed_secs = (dt.datetime.now() - time_before).seconds
  print "For TrialId=%s validation result is %s and train result is %s in %s secs" % (experiment['TrialId'], validation_mean_rewards, train_mean_rewards, elapsed_secs)

if monitor:
  env.monitor.close()
  gym.upload(training_dir, api_key='sk_lgS7sCv1Qxq5HFMdQXR6Sw')
