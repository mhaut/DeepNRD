from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from DA_HSI import LinearScheduler, NDP

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, input_width, num_classes=16, dropout_prob=0, drop_prob=0, block_size=5):
        super(SimpleCNN, self).__init__()
        if drop_prob != 0:
            self.dropblock = LinearScheduler(
                        NDP(drop_prob=drop_prob, block_size=block_size),
                        start_value=0.,
                        stop_value=drop_prob,
                        nr_steps=450
            )
        self.drop_prob = drop_prob
        self.block_size = block_size

        self.dropout_prob = dropout_prob

        self.dim = input_width
        self.conv1 = nn.Conv2d(input_channels, 50, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv3 = nn.Conv2d(100, 200, kernel_size=5)
        self.conv4 = nn.Conv2d(200, 400, kernel_size=2)
        if self.dim == 23:
            self.fc1 = nn.Linear(400, 300)
        elif self.dim == 27:
            self.fc1 = nn.Linear(2*2*400, 300)
        elif self.dim == 31:
            self.fc1 = nn.Linear(3*3*400, 300)
        else:
            print("not spatial size use, change in model simplecnn.py")
            exit()
        self.fc2 = nn.Linear(300, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.drop_prob == 0 and self.dropout_prob == 0:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3(x), 2))
        elif self.dropout_prob != 0:
            x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), p=self.dropout_prob, training=self.training), 2))
            x = F.relu(F.max_pool2d(F.dropout(self.conv3(x), p=self.dropout_prob, training=self.training), 2))
        else:
            self.dropblock.step()  # increment number of iterations
            x = F.relu(F.max_pool2d(self.dropblock(self.conv2(x)), 2))
            x = F.relu(F.max_pool2d(self.dropblock(self.conv3(x)), 2))
        #x = F.relu(F.max_pool2d(self.conv4(x), 1))
        x = F.relu(self.conv4(x))
        if self.dim == 23:
            x = x.view(-1, 400)
        elif self.dim == 27:
            x = x.view(-1, 2*2*400)
        elif self.dim == 31:
            x = x.view(-1, 3*3*400)
        else:
            print("not spatial size use, change in model simplecnn.py")
            exit()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def simplecnn(**kwargs):
    return SimpleCNN(**kwargs)
