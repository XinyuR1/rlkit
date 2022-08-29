"""
Preprocessing Atari Images

Directly taken from https://colab.research.google.com/github/GiannisMitr/DQN-Atari-Breakout/blob/master/dqn_atari_breakout.ipynb#scrollTo=_IA-czvUwbOn
"""

from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2

# Modify the Atari image into a gray-scale style image with a size of 84x84 with 1 channel.
class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (1, self.img_size[0], self.img_size[1]))

    def observation(self, img):
        """what happens to each observation"""

        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = img[34:-16, :, :]

        # resize image
        img = cv2.resize(img, self.img_size)
        img = img.mean(-1, keepdims=True)
        img = img.astype('float32') / 255.

        return img.reshape(1, 84, 84)