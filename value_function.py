import tensorflow as tf
import prettytensor as pt
import numpy as np


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
    prev_obs = np.concatenate([np.zeros((1, obs.shape[1])), obs[1:]], 0)
    prev_action = np.concatenate([np.zeros((1, path['action_dists'].shape[1])), path['action_dists'][1:]], 0)
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
