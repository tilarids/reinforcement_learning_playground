# reinforcement_learning_playground
Playground for reinforcement learning algorithms implemented in TensorFlow.

## OpenAI gym problems solved
### CartPole - Vanilla Policy Gradients

Vanilla policy gradients with ValueFunction to estimate value for the specific state (I use current observation,
previous observation and previous action as a state). This same algorithm works fine without ValueFunction if you
don't stop the learning process at step 200 and continue learning after that. OpenAI Gym's monitor stops the game
at step 200 so you can't use monitor at the same time as training on more than 200 steps. (inspiration for the use of ValueFunction comes from https://github.com/wojzaremba/trpo)

[Gym evaluation](https://gym.openai.com/evaluations/eval_dWo7uqR2Ti6RX7naakndQ)

### CartPole - Policy Gradients with TRPO

The same as above but use conjugate gradients + line search method described in [TRPO paper](http://arxiv.org/abs/1502.05477). Inspiration for the implementation comes from the https://github.com/wojzaremba/trpo again but I tried to make it more readable and close to the paper.

[Gym evaluation](https://gym.openai.com/evaluations/eval_hVkf4zsITBaLFLxVzhbJwg)

Reproducing:
* Consider changing the API key :)
* `python pg_agent.py`
* `python trpo_agent.py`
