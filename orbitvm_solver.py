import random
import tempfile
import math

import gym
import numpy as np
import tensorflow as tf

import orbitvm.p1_env

G = 6.67428e-11
M = 6e+24

def find_hohmann_impulse1(r1, r2):
    return math.sqrt(G * M / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)

def find_hohmann_impulse2(r1, r2):
    return math.sqrt(G * M / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))

def find_transfer_time(r1, r2):
    rr = r1 + r2
    return math.pi * math.sqrt(rr * rr * rr / (8.0 * G * M))

def hohmann_transfer(env):
    state, reward_sum, _, _ = env.step(np.array([0,0]))
    x1, y1, rtarget = state[2:5]
    rcurrent = math.sqrt(x1 * x1 + y1 * y1)
    impulse1 = find_hohmann_impulse1(rcurrent, rtarget)
    impulse2 = find_hohmann_impulse2(rcurrent, rtarget)
    transfer_time = int(find_transfer_time(rcurrent, rtarget))
    dx = - impulse1 * y1 / rcurrent
    dy = impulse1 * x1 / rcurrent

    # first pulse!
    _, reward, _, _ = env.step(np.array([dx,dy]))
    reward_sum += reward

    print "Transfer time: %s" % transfer_time
    # don't breathe!
    for x in xrange(transfer_time - 1):
        state, reward, done, _ = env.step(np.array([0,0]))
        reward_sum += reward
        if 0 == x % 100:
            env.render()

    state, reward, _, _ = env.step(np.array([0,0]))
    reward_sum += reward
    x2, y2, rtarget = state[2:5]
    rcurrent = math.sqrt(x2 * x2 + y2 * y2)

    dx = - impulse2 * y2 / rcurrent
    dy = impulse2 * x2 / rcurrent

    # second pulse!
    state, reward, _, _ = env.step(np.array([dx,dy]))
    reward_sum += reward

    for x in xrange(100000):
        state, reward, done, _ = env.step(np.array([0,0]))
        reward_sum += reward
        if 0 == x % 100:
            #print "State: %s" % state
            env.render()
        if done:
            print "It's done. breaking. "
            break
    print "Total reward: %s" % reward_sum

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

    target = 42164000.0
    start = target * 0.3

    env.orbitvm.set_target_orbit(target)
    env.orbitvm.set_start_orbit(start)

    hohmann_transfer(env)

    # env.monitor.close()
