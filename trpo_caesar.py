import random
import tempfile

import gym
import numpy as np
import tensorflow as tf
from gym.envs.algorithmic.algorithmic_env import ha, AlgorithmicEnv

import trpo_agent
import caesar
import space_conversion

class PredefinedStringEnv(AlgorithmicEnv):
    def __init__(self, input_data_string, output_data_string):
        self.input_data_string = input_data_string
        self.output_data_string = output_data_string
        AlgorithmicEnv.__init__(self,
                                inp_dim=1,
                                base=26,
                                chars=True)



    def set_data(self):
        self.total_len = len(self.input_data_string)
        self.content = {}
        self.target = {}
        for i in range(self.total_len):
            self.content[ha(np.array([i]))] = ord(self.input_data_string[i])-ord('a')
            self.target[i] = ord(self.output_data_string[i]) - ord('a')
        self.total_reward = self.total_len

def use_agent_for_decoding(agent):
  training_dir = tempfile.mkdtemp()

  for line in caesar.this.s.lower().split('\n'):
    cleaned_line = ''.join(x for x in line if ord('a') <= ord(x) <= ord('z'))
    decoded_cleaned_line = ''.join(caesar.this.d[x] for x in line if ord('a') <= ord(x) <= ord('z'))

    env = PredefinedStringEnv(cleaned_line, decoded_cleaned_line)
    env = space_conversion.SpaceConversionEnv(env,
                                              gym.spaces.Box,
                                              gym.spaces.Discrete)
    env.monitor.start(training_dir, resume=True, video_callable=lambda _: True)
    agent.env = env
    agent.rollout(10000, len(cleaned_line))
    env.monitor.close()


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
      layers=1,
      dropout=0.5043049954791183,
      max_iterations=max_iterations)
  agent.learn()
  env.monitor.close()

  use_agent_for_decoding(agent)
