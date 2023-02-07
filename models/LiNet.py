"""
@author Liu Lei
"""

import torch
from torch import nn

class LiNet(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.a_base=nn.Sequential(
            nn.Conv1d(1,16,3,padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.a_up_base=nn.Sequential(
            nn.Conv1d(16,8,1),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.a_up_up=nn.Sequential(
            nn.Conv1d(8,16,3,padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.a_up_down=nn.Sequential(
            nn.Conv1d(8,16,1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.a_down_base=nn.Sequential(
            nn.Conv1d(16,8,1),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.a_down_up=nn.Sequential(
            nn.Conv1d(8,16,5,padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.a_down_down=nn.Sequential(
            nn.Conv1d(8,16,1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.maxpool=nn.MaxPool1d(2)

        self.b_up_base=nn.Sequential(
            nn.Conv1d(80,32,1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.b_up_up=nn.Sequential(
            nn.Conv1d(32,16,3,padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.b_up_down=nn.Sequential(
            nn.Conv1d(32,16,1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.b_down_base=nn.Sequential(
            nn.Conv1d(80,32,1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.b_down_up=nn.Sequential(
            nn.Conv1d(32,16,5,padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.b_down_down=nn.Sequential(
            nn.Conv1d(32,16,1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.fc1=nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(144,16,1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.output_layer=nn.Linear(16,params.class_num)

    def forward(self,inputs):
        inputs=inputs.view(inputs.size(0),1,-1)
        a_base_out=self.a_base(inputs)
        a_up_base_out=self.a_up_base(a_base_out)
        a_up_up_out=self.a_up_up(a_up_base_out)
        a_up_down_out=self.a_up_down(a_up_base_out)
        a_down_base_out=self.a_down_base(a_base_out)
        a_down_up_out=self.a_down_up(a_down_base_out)
        a_down_down_out=self.a_down_down(a_down_base_out)
        a_cat=torch.cat([a_base_out,a_up_up_out,a_up_down_out,a_down_up_out,a_down_down_out],dim=1)

        b_base_out=self.maxpool(a_cat)
        b_up_base_out=self.b_up_base(b_base_out)
        b_up_up_out=self.b_up_up(b_up_base_out)
        b_up_down_out=self.b_up_down(b_up_base_out)
        b_down_base_out=self.b_down_base(b_base_out)
        b_down_up_out=self.b_down_up(b_down_base_out)
        b_down_down_out=self.b_down_down(b_down_base_out)
        b_cat=torch.cat([b_base_out,b_up_up_out,b_up_down_out,b_down_up_out,b_down_down_out],dim=1)

        out=self.fc1(b_cat)
        out=out.view(inputs.size(0),-1)
        out=self.output_layer(out)
        return out
    def get_type(self):
        return 'time'
    def get_len(self):
        return 2048