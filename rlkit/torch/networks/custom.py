"""
Personalized CNN Networks for the Atari Experiments
-> for the file dqn-Atari.py
"""

import torch.nn.functional as F
from torch import nn as nn

"""
First CNN Model
Taken from the following paper https://arxiv.org/pdf/1509.06461.pdf
"""
class ConvNet1(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        
        #(input, output, kernel_size, stride)
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        
        x_size = x.view(-1).size(dim = 0)
        if x_size == (84 * 84):
            x = x.view(1, 1, 84, 84)
        else:
            x = x.view(32, 1, 84, 84)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

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
        

        x_size = x.view(-1).size(dim = 0)

        if x_size == (84 * 84):
            x = x.view(1, 1, 84, 84)
        else:
            x = x.view(256, 1, 84, 84)

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
        #x = x.view(-1)

        #print(f'After Convolutions: {x.shape}')
        return x

