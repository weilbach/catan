import numpy as np
from layers import *
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """
    A convolutional network.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=0)
        self.fc1 = nn.Linear(15488*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 11)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2), padding=0)

        self.init_weights()


    def init_weights(self):

        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(conv.weight, 0.0, 1e-2)
            nn.init.constant_(conv.bias, 0.0)
                
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.normal_(fc.weight, 0.0, 1e-2)
            nn.init.constant_(fc.bias, 0.0)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = self.pool(x)


        x = x.view(-1, 15488*4)

        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        # x = self.fc5(x)

        return x
        
