"""
Task is to decode text from input tape to
the output tape. http://arxiv.org/abs/1511.07275
"""
import this
import collections

import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.envs.algorithmic.algorithmic_env import ha
from gym.envs.registration import register

LEN_TO_WORD = collections.defaultdict(list)
MAX_LEN = 0

for word in this.s.split():
  encoded = ''
  decoded = ''
  for c in word.lower():
    dec = this.d.get(c, None)
    if dec:
      encoded += c
      decoded += dec
  LEN_TO_WORD[len(encoded)].append(encoded)
  MAX_LEN = max(MAX_LEN, len(encoded))

class CaesarEnv(algorithmic_env.AlgorithmicEnv):
    def __init__(self):
        algorithmic_env.AlgorithmicEnv.__init__(self,
                                                inp_dim=1,
                                                base=26,
                                                chars=True)
    def set_data(self):
        self.content = {}
        self.target = {}
        len_left = self.total_len
        i = 0
        while len_left > 0:
          rand_len = self.np_random.randint(1, min(len_left, MAX_LEN) + 1)
          if not LEN_TO_WORD[rand_len]:
            continue
          encoded = self.np_random.choice(LEN_TO_WORD[rand_len])
          for c in encoded:
            enc_val = ord(c) - ord('a')
            dec_val = ord(this.d[c]) - ord('a')
            self.content[ha(np.array([i]))] = enc_val
            self.target[i] = dec_val
            i += 1
          len_left -= rand_len

        self.total_reward = self.total_len

register(
    id='Caesar-v0',
    entry_point='caesar:CaesarEnv',
    timestep_limit=200,
    reward_threshold=25.0,
)
