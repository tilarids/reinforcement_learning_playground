import random
import tempfile
import math

import gym
import numpy as np
import tensorflow as tf

import orbitvm.p1_env

def find_hohmann_impulse1(r1, r2):
    return math.sqrt(u / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)

def find_hohmann_impulse2(r1, r2):
    return math.sqrt(u / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))

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
  env.orbitvm.set_target_orbit(42164000.0)
  env.orbitvm.set_start_orbit(42164000.0 * 0.3)

  for x in xrange(100000):
    state, reward, done, _ = env.step(np.array([0,0]))
    # print env.vm_output
    if 0 == x % 100:
       env.render()

  # env.monitor.close()
