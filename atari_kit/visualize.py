"""
Visualize the Atari environment (before and after preprocessing)

In order to run this file, use the following command:
python visualize.py <env-name>
"""

import gym
import sys
from preprocessing import PreprocessAtari
import matplotlib.pyplot as plt

"""
Before preprocessing
"""
# 210 x 160 x 3 channels
# env.observation_space.shape
before_env = gym.make(sys.argv[1])
before_env.reset()
before_obs, _, done, _ = before_env.step(before_env.action_space.sample())

"""
After preprocessing
"""
after_env = PreprocessAtari(before_env)
after_env.reset()
after_obs, _, done, _ = after_env.step(after_env.action_space.sample())

plt.suptitle(sys.argv[1])
plt.subplot(1, 2, 1)
plt.title("Before preprocessing")
plt.imshow(before_obs)
plt.subplot(1, 2, 2)
plt.title("After preprocessing")
plt.imshow(after_obs.reshape(84,84,1), cmap = "gray")
plt.show()

