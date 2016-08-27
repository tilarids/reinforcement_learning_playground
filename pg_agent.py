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
import sys

DTYPE = tf.float32
RENDER_EVERY = None
MONITOR = True

logger = logging.getLogger('pg_agent')
logger.setLevel(logging.INFO)

# Sample from the probability distribution.
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

def write_csv(file_name, *arrays):
  with open(file_name, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in zip(*arrays):
      writer.writerow(row)

# same as tf.gradients but returns flat tensor.
def flat_gradients(loss, var_list):
  grads = tf.gradients(loss, var_list)
  return tf.concat(0, [tf.reshape(grad, [np.prod(var_shape(v))])
                       for (v, grad) in zip(var_list, grads)])

def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

# Separate network to approximate value function. Need to init variables after
# this is instantiated.
class ValueFunction(object):
  def __init__(self, session, features_count, learning_rate, epochs, dropout):
    self.features_count = features_count
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.session = session
    self.dropout = dropout
    self.x = tf.placeholder(tf.float32, shape=[None, features_count], name="x")
    self.y = tf.placeholder(tf.float32, shape=[None], name="y")
    self.net = (pt.wrap(self.x)
                .fully_connected(64, activation_fn=tf.nn.relu)
                .dropout(self.dropout)
                .fully_connected(64, activation_fn=tf.nn.relu)
                .dropout(self.dropout)
                .fully_connected(1))
    self.net = tf.reshape(self.net, (-1, ))
    self.l2 = (self.net - self.y) * (self.net - self.y)
    self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2)

  def prepare_features(self, path):
    obs = path["obs"]
    prev_obs = np.concatenate([np.zeros((1, obs.shape[1])), path["obs"][1:]], 0)
    prev_action = path['action_dists']
    l = len(path["rewards"])
    arange = np.arange(l).reshape(-1, 1)
    ret = np.concatenate([obs, prev_obs, prev_action, arange, np.ones((l, 1))], axis=1)
    return ret

  def fit(self, paths):
    features = np.concatenate([self.prepare_features(path) for path in paths])
    returns = np.concatenate([path["returns"] for path in paths])
    for _ in range(self.epochs):
        self.session.run(self.train, {self.x: features, self.y: returns})

  def validate(self, paths):
    features = np.concatenate([self.prepare_features(path) for path in paths])
    returns = np.concatenate([path["returns"] for path in paths])
    return np.mean(self.session.run(self.l2, {self.x: features, self.y: returns}))

  def predict(self, path):
    features = self.prepare_features(path)
    return self.session.run(self.net, {self.x: features})

# Math craziness. Implements a conjugate gradient algorithm. In short, solves
# Ax = b for x having only a function x -> Ax (f_Ax) and b.
def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
  p = b.copy()
  r = b.copy()
  x = np.zeros_like(b)
  rdotr = r.dot(r)
  for i in xrange(cg_iters):
      z = f_Ax(p)
      v = rdotr / p.dot(z)
      x += v * p
      r -= v * z
      newrdotr = r.dot(r)
      mu = newrdotr / rdotr
      p = r + mu * p
      rdotr = newrdotr
      if rdotr < residual_tol:
          break
  return x

# Simple line search algorithm. That is, having objective f and initial value
# x search along the max_step vector (shrinking it exponentially) until we find
# an improvement in f.
def line_search(f, x, max_step, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * max_step
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

# Learning agent. Encapsulates training and prediction.
class PGAgent(object):
  def __init__(self, env, win_step, H, timesteps_per_batch, learning_rate, gamma, epochs, dropout, win_reward):
    if not isinstance(env.observation_space, Box) or \
       not isinstance(env.action_space, Discrete):
        logger.error("Incompatible spaces.")
        exit(-1)

    self.H = H
    self.timesteps_per_batch = timesteps_per_batch
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.epochs = epochs
    self.dropout = dropout
    self.win_reward = win_reward
    self.win_step = win_step
    self.env = env
    self.session = tf.Session()

    # Full state used for next action prediction. Contains current
    # observation, previous observation and previous action.
    self.obs = tf.placeholder(
        DTYPE,
        shape=[None, 2 * env.observation_space.shape[0] + env.action_space.n],
        name="obs")
    self.prev_obs = np.zeros(env.observation_space.shape[0])
    self.prev_action = np.zeros(env.action_space.n)

    # One hot encoding of the actual action taken.
    self.action = tf.placeholder(DTYPE,
                                 shape=[None, env.action_space.n],
                                 name="action")
    # Advantage, obviously.
    self.advant = tf.placeholder(DTYPE, shape=[None], name="advant")
    # Old policy prediction.
    self.prev_policy = tf.placeholder(DTYPE,
                                      shape=[None, env.action_space.n],
                                      name="prev_policy")

    self.policy_network, _ = (
        pt.wrap(self.obs)
            .fully_connected(H, activation_fn=tf.nn.tanh)
            .dropout(self.dropout)
            .softmax_classifier(env.action_space.n))
    self.returns = tf.placeholder(DTYPE,
                                  shape=[None, env.action_space.n],
                                  name="returns")

    loss = - tf.reduce_sum(tf.mul(self.action,
                                  tf.div(self.policy_network,
                                         self.prev_policy)), 1) * self.advant
    self.loss = loss
    self.train = tf.train.AdamOptimizer().minimize(loss)

    # Start of TRPO/Fisher/conjugate gradients util code

    # get all trainable variable names.
    var_list = tf.trainable_variables()

    # define a function to get all trainable variables in a flat form.
    def get_variables_flat_form():
      op = tf.concat(
          0, [tf.reshape(v, [np.prod(var_shape(v))]) for v in var_list])
      return op.eval(session=self.session)
    self.get_variables_flat_form = get_variables_flat_form

    # define a function to set all trainable variables from a flat tensor theta.
    def create_set_variables_from_flat_form_function():
      assigns = []
      shapes = map(var_shape, var_list)
      total_size = sum(np.prod(shape) for shape in shapes)
      theta_in = tf.placeholder(DTYPE, [total_size])
      start = 0
      assigns = []
      for (shape, v) in zip(shapes, var_list):
          size = np.prod(shape)
          assigns.append(
              tf.assign(
                  v,
                  tf.reshape(
                      theta_in[
                          start:start +
                          size],
                      shape)))
          start += size
      op = tf.group(*assigns)

      def set_variables_from_flat_form(theta):
        return self.session.run(op, feed_dict={theta_in: theta})
      return set_variables_from_flat_form

    self.set_variables_from_flat_form = create_set_variables_from_flat_form_function()

    # get operation to calculate all gradients (that is, find derivatives with
    # respect to var_list).
    self.policy_gradients_op = flat_gradients(loss, var_list)

    # get a KL divergence. Please note that we use stop_gradients here because
    # we don't care about prev_policy gradients and shouldn't update the
    # the prev_policy at all.
    kl_divergence = tf.reduce_sum(
        tf.stop_gradient(self.prev_policy) *
        tf.log((tf.stop_gradient(self.prev_policy) + 1e-8) /
               (self.policy_network + 1e-8))) / tf.cast(tf.shape(self.obs)[0], DTYPE)

    # As before, get an op to find derivatives.
    kl_divergence_gradients_op = tf.gradients(kl_divergence, var_list)

    # this is a flat representation of the variable that we are going to use in
    # our Fisher product (that is, in function y -> A*y where A is Fisher matrix,
    # flat_multiplier_tensor is our y)
    self.flat_multiplier_tensor = tf.placeholder(DTYPE, shape=[None])

    # Do the actual multiplication. Some shape shifting magic.
    start = 0
    multiplier_parts = []
    for var in var_list:
      shape = var_shape(var)
      size = np.prod(shape)
      part = tf.reshape(self.flat_multiplier_tensor[start:(start + size)], shape)
      multiplier_parts.append(part)
      start += size

    product_op_list = [tf.reduce_sum(kl_derivation * multiplier) for (kl_derivation, multiplier) in zip(kl_divergence_gradients_op, multiplier_parts)]

    # Second derivation, duh!
    self.fisher_product_op_list = flat_gradients(product_op_list, var_list)

    # End of TRPO/Fisher/conjugate gradients util code



    features_count = 2 * env.observation_space.shape[0] + env.action_space.n + 2
    self.value_function = ValueFunction(self.session,
                                        features_count,
                                        learning_rate=1e-3,
                                        epochs=50,
                                        dropout=0.5)
    self.session.run(tf.initialize_all_variables())

  def rollout(self, max_pathlength, timesteps_per_batch, render=False):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < timesteps_per_batch:

      obs, actions, rewards, action_dists, actions_one_hot = [], [], [], [], []
      ob = self.env.reset()
      self.prev_action *= 0.0
      self.prev_obs *= 0.0
      for x in xrange(max_pathlength):
        if render:
          env.render()
        obs_new = np.expand_dims(
            np.concatenate([ob, self.prev_obs, self.prev_action], 0), 0)

        action_dist_n = self.session.run(self.policy_network, {self.obs: obs_new})

        action = int(cat_sample(action_dist_n)[0])
        self.prev_obs = ob
        self.prev_action *= 0.0
        self.prev_action[action] = 1.0

        obs.append(ob)
        actions.append(action)
        action_dists.append(action_dist_n)
        actions_one_hot.append(np.copy(self.prev_action))

        res = list(self.env.step(action))
        if not self.win_step is None and self.win_step==len(rewards):
          rewards.append(self.win_reward)
          res[2] = True
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
    obs = path["obs"]
    prev_obs = np.concatenate([np.zeros((1,obs.shape[1])), path["obs"][1:]], 0)
    prev_action = path['action_dists']
    return np.concatenate([obs, prev_obs, prev_action], axis=1)

  def predict(self, path):
    features = self.prepare_features(path)
    return self.session.run(self.policy_network, {self.obs: features})

  def learn(self):
    self.current_observation = self.env.reset()

    xs,hs,dlogps,drs = [],[],[],[]

    running_reward = None
    reward_sum = 0
    episode_number = 0
    current_step = 0.0
    iteration_number = 0
    discounted_eprs = []
    mean_path_lens = []
    value_function_losses = []

    while True:
      render = not RENDER_EVERY is None and 0 == iteration_number % RENDER_EVERY
      paths = self.rollout(max_pathlength=10000,
                           timesteps_per_batch=self.timesteps_per_batch,
                           render=render)

      for path in paths:
        path["baseline"] = self.value_function.predict(path)
        path["prev_policy"] = self.predict(path)
        path["returns"] = discount_rewards(path["rewards"], self.gamma)
        path["advant"] = path["returns"] - path["baseline"]

      value_function_losses.append(self.value_function.validate(paths))
      self.value_function.fit(paths)
      features = np.concatenate([self.prepare_features(path) for path in paths])

      advant = np.concatenate([path["advant"] for path in paths])
      advant -= advant.mean()
      advant /= (advant.std() + 1e-8)

      actions = np.concatenate([path["actions_one_hot"] for path in paths])
      prev_policy = np.concatenate([path["prev_policy"] for path in paths])

      # Start of conjugate gradient magic.

      # Get current theta (weights).
      previous_parameters_flat = self.get_variables_flat_form()

      feed_dict = {self.obs: features,
                   self.advant: advant,
                   self.action: actions,
                   self.prev_policy: prev_policy}
      # This is a function that multipliers a vector by Fisher matrix. Used
      # by conjugate gradients algorithm.

      def fisher_vector_product(multiplier):
        feed_dict[self.flat_multiplier_tensor] = multiplier
        conjugate_gradients_damping = 0.1
        return self.session.run(self.fisher_product_op_list, feed_dict) + conjugate_gradients_damping * multiplier

      policy_gradients = self.session.run(self.policy_gradients_op, feed_dict)

      # Run the conjugate algorithm
      step_direction = conjugate_gradient(fisher_vector_product, -policy_gradients)

      # Calculate $s^{T}As$.
      hessian_vector_product = step_direction.dot(fisher_vector_product(step_direction))
      max_kl = 0.01

      # This is our \beta.
      max_step_length = np.sqrt(2 * max_kl / hessian_vector_product)
      max_step = max_step_length * step_direction
      neg_gradients_dot_step_direction = -policy_gradients.dot(step_direction)

      def get_loss_for(weights_flat):
        self.set_variables_from_flat_form(weights_flat)
        return self.session.run(tf.reduce_mean(self.loss), feed_dict)

      # search along the search direction.
      new_weights = line_search(get_loss_for, previous_parameters_flat, max_step, max_step_length * neg_gradients_dot_step_direction)

      self.set_variables_from_flat_form(new_weights)
      # End of conjugate gradient magic.

      iteration_number += 1

      mean_path_len = np.mean([len(path['rewards']) for path in paths])
      mean_path_lens.append(mean_path_len)
      logger.info("Iteration %s mean_path_len: %s", iteration_number, mean_path_len)
      if iteration_number > 100:
        paths = self.rollout(max_pathlength=10000, timesteps_per_batch=40000)
        ret = np.mean([len(path['rewards']) for path in paths]), np.mean(mean_path_lens)
        logger.info("Validation result: %s", ret[0])
        if not MONITOR:
          write_csv('/tmp/out.csv', mean_path_lens, value_function_losses)
        return ret

if __name__ == '__main__':
  seed = 1
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  env = gym.make("CartPole-v0")
  if MONITOR:
    training_dir = tempfile.mkdtemp()
    env.monitor.start(training_dir)

  agent = PGAgent(env,
                  win_step=199,
                  H=109,
                  timesteps_per_batch=1369,
                  learning_rate=0.028609296254614544,
                  gamma=0.9914327475117531,
                  epochs=4,
                  dropout=0.5043049954791183,
                  win_reward=1)
  agent.learn()
  if MONITOR:
    env.monitor.close()
    gym.upload(training_dir, api_key='sk_lgS7sCv1Qxq5HFMdQXR6Sw')
