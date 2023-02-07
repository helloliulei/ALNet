"""
@author Liu Lei
"""
import torch
from torch import nn

class TI_CNN(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv1d(1,16,64,8,28),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(16,32,3,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(32,64,3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv4=nn.Sequential(
            nn.Conv1d(64,64,3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv5=nn.Sequential(
            nn.Conv1d(64,64,3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv6=nn.Sequential(
            nn.Conv1d(64,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc=nn.Linear(64 * 3, 100)
        self.output_layer=nn.Linear(100,params.class_num)

    def forward(self,inputs):
        inputs = inputs.type(torch.FloatTensor)
        inputs=inputs.view(-1,1,2048)
        inputs=inputs.cuda()
        out=self.conv1(inputs)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.conv5(out)
        out=self.conv6(out)
        out=out.reshape(-1,64*3)
        out=self.fc(out)
        out=self.output_layer(out)
        return out
    def get_type(self):
        return 'time'
    def get_len(self):
        return 2048
