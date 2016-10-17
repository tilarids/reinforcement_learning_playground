import random
import tempfile

import gym
import numpy as np
import tensorflow as tf

import orbitvm.p1_env

if __name__ == '__main__':
  seed = 1
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  env_name = "OrbitP1-v0"
  max_iterations = 100

  env = gym.make(env_name)
  # training_dir = tempfile.mkdtemp()
  # env.monitor.start(training_dir)

  env.reset()

  for x in xrange(1000):
    state, reward, done, _ = env.step(np.array([0,0]))
    env.render()

  # env.monitor.close()
