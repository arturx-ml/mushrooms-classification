import matplotlib.pyplot as plt
import torch
import torchvision
from torch import quantization
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
DEBUG = False

class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)  # 2 - 6
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)  # 1 - 6
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.conv4 = nn.Conv2d(20, 25, 5)
        self.conv5 = nn.Conv2d(25, 30, 5)
        self.conv6 = nn.Conv2d(30, 35, 3)

        self.fc1 = nn.Linear(875, 1280)  # 2500
        self.fc2 = nn.Linear(1280, 120)  # 2500
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        if DEBUG:
            print('first_shape {}'.format(x.shape))
        x = self.pool(F.relu(self.conv1(x)))
        if DEBUG:
            print('second_shape {}'.format(x.shape))
        x = self.pool(F.relu(self.conv2(x)))
        if DEBUG:
            print('4nd_shape{}'.format(x.shape))
        x = self.pool(F.relu(self.conv3(x)))
        if DEBUG:
            print('5nd_shape{}'.format(x.shape))
        x = self.pool(F.relu(self.conv4(x)))
        if DEBUG:
            print('6nd shape before reshape{}'.format(x.shape))
        x = self.pool(F.relu(self.conv5(x)))
        if DEBUG:
            print('6nd shape before reshape{}'.format(x.shape))
        x = self.pool(F.relu(self.conv6(x)))
        if DEBUG:
            print('6nd shape before reshape{}'.format(x.shape))

        x = x.view(-1, 875)
        if DEBUG:
            print('7nd_shape after reshape{}'.format(x.shape))
        x = F.relu(self.fc1(x))
        if DEBUG:
            print('8nd_shape {}'.format(x.shape))
        x = F.relu(self.fc2(x))
        if DEBUG:
            print('9nd_shape {}'.format(x.shape))
        x = F.relu(self.fc3(x))
        if DEBUG:
            print('10nd_shape {}'.format(x.shape))
        x = self.fc4(x)
        if DEBUG:
            print('11nd_shape {}'.format(x.shape))
        return x



warnings.filterwarnings('ignore')
device = 'cuda'
model = torch.load('last_cnn.pt','cuda')
modelq = quantization.convert(model)
example = torch.rand(1, 3, 512, 512)
ex = example.to(device)
traced_script_module = torch.jit.trace(modelq, ex)
traced_script_module.save('last_jit_model_moda.pt')