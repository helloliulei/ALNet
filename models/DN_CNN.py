"""
@author Liu Lei
"""
import torch
from torch import nn

class DN_CNN(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.conv1=nn.Sequential(
                nn. Conv1d(1,16,49),
                nn.ReLU(),
                nn.MaxPool1d(4)
        )
        self.conv2=nn.Sequential(
                nn.Conv1d(16,16,21),
                nn.ReLU(),
                nn.MaxPool1d(4)
        )
        self.f1=nn.Sequential(
                nn.Linear(16*67,1072),
                nn.ReLU()
        )
        self.f2=nn.Sequential(
                nn.Linear(1072,100),
                nn.ReLU()
        )
        self.output_layer=nn.Linear(100,params.class_num)
    def forward(self,inputs):
        inputs = inputs.type(torch.FloatTensor)
        batch_size=inputs.size(0)
        inputs=inputs.view(batch_size,1,-1)
        inputs=inputs.cuda()
        out=self.conv1(inputs)
        out=self.conv2(out)
        out=out.view(batch_size,-1)
        out=self.f1(out)
        out=self.f2(out)
        out=self.output_layer(out)
        return out
    def get_type(self):
        return 'time'
    def get_len(self):
        return 1200