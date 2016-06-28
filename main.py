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

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32



# hyperparameters
H = 32 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = True

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

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


class LearningAgent(object):
  def __init__(self, env):
    if not isinstance(env.observation_space, Box) or \
       not isinstance(env.action_space, Discrete):
        print("Incompatible spaces.")
        exit(-1)
    self.env = env
    self.session = tf.Session()
    self.obs = tf.placeholder(
        dtype, shape=[
            None, 2 * env.observation_space.shape[0] + env.action_space.n], name="obs")
    self.prev_obs = np.zeros(env.observation_space.shape[0])
    self.prev_action = np.zeros(env.action_space.n)
    self.action = action = tf.placeholder(dtype, shape=[None, env.action_space.n], name="action")
    self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")

    self.policy_network, _ = (
        pt.wrap(self.obs)
            .fully_connected(H, activation_fn=tf.nn.tanh)
            .softmax_classifier(env.action_space.n))
    self.returns = tf.placeholder(dtype, shape=[None, env.action_space.n], name="returns")

    loss = - tf.reduce_sum(tf.mul(self.action, self.policy_network), 1) * self.advant
    self.train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

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
        ob = res[0]
        rewards.append(res[1])

        if res[2]:
            path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                    "action_dists": np.concatenate(action_dists),
                    "rewards": np.array(rewards),
                    "actions": np.array(actions),
                    "actions_one_hot": np.array(actions_one_hot)}
            paths.append(path)
            self.prev_action *= 0.0
            self.prev_obs *= 0.0
            break
      timesteps_sofar += len(path["rewards"])
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
    # import pdb; pdb.set_trace()
    discounted_eprs = []
    while True:

      paths = self.rollout(max_pathlength=10000, timesteps_per_batch=1000)
      for path in paths:
        # path["baseline"] = self.predict_for_path(path)
        path["returns"] = discount_rewards(path["rewards"])
        # path["advant"] = path["returns"] - path["baseline"]
        path["advant"] = path["returns"]

      features = np.concatenate([self.prepare_features(path) for path in paths])

      advant = np.concatenate([path["advant"] for path in paths])
      advant -= advant.mean()
      advant /= (advant.std() + 1e-8)

      actions = np.concatenate([path["actions_one_hot"] for path in paths])

      # predictions = np.concatenate([self.predict_for_path(path) for path in paths])

      # import pdb; pdb.set_trace()

      for _ in range(50):
        self.session.run(self.train, {self.obs: features,
                                      self.advant: advant,
                                      self.action: actions})
      iteration_number += 1
      mean_path_len = np.mean([len(path['rewards']) for path in paths])
      print "Iteration %s finished. Mean path len: %s" % (iteration_number, mean_path_len)

env = gym.make("CartPole-v0")
agent = LearningAgent(env)
agent.learn()
