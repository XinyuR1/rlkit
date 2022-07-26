"""
Personalized CNN Networks for the Atari Experiments
-> for the file dqn-Atari.py
"""

import torch.nn.functional as F
from torch import nn as nn

"""
First CNN Model
Taken from the Pytorch Tutorial
"""
class ConvNet1(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # input channels: 1
        # output channels: 6 filters
        # kernel size: 5x5
        self.conv1 = nn.Conv2d(1, 6, 5)

        # kernel size 2x2 and stride 2
        self.pool = nn.MaxPool2d(2, 2)

        # input channels: 6 from the previous layer
        # output channels: 16 filters
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*18*18, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_actions)

    def forward(self, x):
        #print(f'Before Convolutions: {x.shape}')

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flattened tensor
        x = x.view(-1, 16*18*18)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        #print(f'After Convolutions: {x.shape}')
        return x


"""
Second CNN Model
Taken from the following paper https://arxiv.org/pdf/1312.5602.pdf  
"""
class ConvNet2(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        #print(f'Before Convolutions: {x.shape}')

        x = F.relu(self.conv1(x))
        #print(f'Convolution 1: {x.shape}')

        x = F.relu(self.conv2(x))
        #print(f'Convolution 2: {x.shape}')

        # Flattened tensor
        x = x.view(-1, 32 * 9 * 9)

        x = F.relu(self.fc1(x))
        #print(f'FC 1: {x.shape}')

        x = self.fc2(x)

        #print(f'After Convolutions: {x.shape}')
        return x

