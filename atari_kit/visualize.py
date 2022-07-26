"""
Visualize the Atari environment (before and after preprocessing)
Choice of environments:
1. "Breakout-v0"
2. "Pong-v0"
3. "BeamRider-v0"
4. "Seaquest-v0"
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


#print(before_env.observation_space.shape)
#print(after_env.observation_space.shape)

plt.suptitle(sys.argv[1])
plt.subplot(1, 2, 1)
plt.title("Before preprocessing")
plt.imshow(before_obs)
plt.subplot(1, 2, 2)
plt.title("After preprocessing")
plt.imshow(after_obs.reshape(84,84,1), cmap = "gray")
plt.show()

