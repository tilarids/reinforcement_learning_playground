import random
import tempfile

import gym
import numpy as np
import tensorflow as tf

import trpo_agent
import caesar
import space_conversion


if __name__ == '__main__':
  seed = 1
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  env_name = "Caesar-v0"
  max_iterations = 1000

  env = gym.make(env_name)
  env = space_conversion.SpaceConversionEnv(env,
                                            gym.spaces.Box,
                                            gym.spaces.Discrete)

  training_dir = tempfile.mkdtemp()
  env.monitor.start(training_dir)

  agent = trpo_agent.TRPOAgent(
      env,
      H=309,
      timesteps_per_batch=1369,
      learning_rate=0.028609296254614544,
      gamma=0.9914327475117531,
      layers=2,
      dropout=0.5043049954791183,
      max_iterations=max_iterations)
  agent.learn()
  env.monitor.close()
