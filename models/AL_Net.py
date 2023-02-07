"""
@author Liu Lei
"""
import numpy as np
from torch import nn

class AIM():
    def __init__(self,params):
        super(AIM, self).__init__()
        self.params=params
    def get_bearing_structure_params(self):
        dataset_name = self.params.dataset_name
        if 'cwru' in dataset_name:
            fs = 12000
            bd = 0.3126
            pd = 1.537
            num = 9
            b = bd / pd
            outer_ring_multiple = 3.5848
            alpha = np.arccos((1 - outer_ring_multiple / 0.5 / num) / b)
        bearing_structure_params = {
            'fs'   :fs,
            'num'  :num,
            'b'    :b,
            'alpha':alpha
        }
        return bearing_structure_params
    def get_fr(self,dataset_name,each_type_path,hp):
        if 'cwru' in dataset_name:
            hp_speed_dic = {
                0:1797,
                1:1772,
                2:1750,
                3:1730
            }
            fr = hp_speed_dic[hp]
        return fr
    def calculate_sample_len(self,params,each_type_path,hp):
        fr=self.get_fr(params.dataset_name,each_type_path,hp)
        k = params.k
        bearing_structure_params=self.get_bearing_structure_params()
        fs = bearing_structure_params['fs']
        num = bearing_structure_params['num']
        b = bearing_structure_params['b']
        alpha = bearing_structure_params['alpha']
        fr = fr / 60
        c = np.cos(alpha)
        f_in = fr * 0.5 * num * (1 + b * c)
        f_out = fr * 0.5 * num * (1 - b * c)
        f_ball = fr * 0.5 / b * (1 - (b * c) ** 2)
        f_cage = fr * 0.5 * (1 - b * c)
        max_f = np.max([f_in,f_out,f_ball,f_cage])

        if params.sample_len=='Adap' or params.sample_len=='adap':
            actual_len = int(fs * 400 / (k * max_f))
        else:
            actual_len = params.sample_len
        return actual_len
class AL_Net(nn.Module):
    def __init__(self,params):
        super().__init__()
        norm=params.norm
        fir_channels=12
        self.g=params.g
        sec_channels= fir_channels *4
        self.sec=sec_channels
        self.conv1=nn.Sequential(
            nn.Conv2d(1,fir_channels,7,),
            self.get_norm(norm_type=norm,channles=fir_channels,output_size=14),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(fir_channels,sec_channels,5,groups=fir_channels),
            self.get_norm(norm_type=norm,channles=sec_channels,output_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.output_layer=nn.Linear(sec_channels,params.class_num)
    def forward(self,inputs):
        inputs=inputs.view(-1,1,20,20)
        c1_out=self.conv1(inputs)
        c2_out=self.conv2(c1_out)
        out=c2_out.view(-1,self.sec)
        out=self.output_layer(out)
        return out
    def get_type(self):
        return 'envelope'
    def get_len(self):
        return 'Adap'
    def get_norm(self,norm_type,**kwargs):
        channles=kwargs['channles']
        output_size=kwargs['output_size']
        if norm_type=='IN':
            return nn.InstanceNorm2d(channles)
        elif norm_type=='BN':
            return nn.BatchNorm2d(channles)
        elif norm_type=='LN':
            return nn.LayerNorm((channles,output_size,output_size))
        elif norm_type=='GN':
            return nn.GroupNorm(self.g,channles)