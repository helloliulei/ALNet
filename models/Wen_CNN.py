"""
@author Liu Lei
"""
import torch
from torch import nn

class Wen_CNN(nn.Module):

    def __init__(self,params):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(4 * 4 * 256, 2560)
        self.output_layer = nn.Linear(2560, params.class_num)
        self.relu=nn.ReLU()

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x=x.view(-1,1,64,64)
        x=x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4 * 4 * 256)
        x = self.relu(self.fc1(x))
        x = self.output_layer(x)
        return x
    def get_type(self):
        return 'time'
    def get_len(self):
        return 4096