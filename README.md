# reinforcement_learning_playground
Playground for reinforcement learning algorithms implemented in TensorFlow.

## OpenAI gym problems solved
### Vanilla Policy Gradients

Vanilla policy gradients with ValueFunction to estimate value for the specific state (I use current observation,
previous observation and previous action as a state). This same algorithm works fine without ValueFunction if you
don't stop the learning process at step 200 and continue learning after that. OpenAI Gym's monitor stops the game
at step 200 so you can't use monitor at the same time as training on more than 200 steps. (inspiration for the use of ValueFunction comes from https://github.com/wojzaremba/trpo)

[Gym evaluation - CartPole-v0](https://gym.openai.com/evaluations/eval_dWo7uqR2Ti6RX7naakndQ)

[Gym evaluation - CartPole-v1](https://gym.openai.com/evaluations/eval_eB5PuUG8QfyRSNjDK7xTA)

Reproducing:
* Consider changing the API key :)
* `python pg_agent.py`
* `python pg_agent.py CartPole-v1`

### Policy Gradients with TRPO

The same as above but use conjugate gradients + line search method described in [TRPO paper](http://arxiv.org/abs/1502.05477). Inspiration for the implementation comes from the https://github.com/wojzaremba/trpo again but I tried to make it more readable and close to the paper.

Please also note that this agent doesn't use dropout. The reason is that TRPO doesn't work well with dropout. That is, with high dropout rate the KL divergence may be very high even between exactly equal set of params. This is due to randomized nature of dropout.

[Gym evaluation - CartPole-v0](https://gym.openai.com/evaluations/eval_hVkf4zsITBaLFLxVzhbJwg)

[Gym evaluation - CartPole-v1](https://gym.openai.com/evaluations/eval_S5aAzDRtSjGFJgEycVqymw)

[Gym evaluation - Copy-v0](https://gym.openai.com/evaluations/eval_y90gKlfdR7u9w0NzEbnw)

Reproducing:
* Consider changing the API key :)
* `python trpo_agent.py`
* `python trpo_agent.py CartPole-v1`
* `python trpo_agent.py Copy-v0`

## New environments solved
# Caesar cipher

I am introducing new environment that is a fork of "Copy-v0" environment but except for copying input tape into output tape the agent needs to decode Caesar-ciphered text into output tape. The same algorithm that works with CartPole-v0 and Copy-v0 also works here. The only difference is the amount of hidden units.

[![asciicast](https://asciinema.org/a/84292.png)](https://asciinema.org/a/84292)

Reproducing: `python trpo_caesar.py`

