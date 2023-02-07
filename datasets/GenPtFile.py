"""
@author Liu Lei
"""
# encoding=utf8
from datasets.ParseDatasets import CWRU

from models.AL_Net import AIM
from utils.math_util import Envelope
from utils.print_util import printc

"""
@author hello
"""
import os
import random

import numpy as np
import torch


class GenPt():
    def __init__(self,params):
        super(GenPt,self).__init__()
        self.params = params
        self.dataset_obj = self.get_dataset_obj()

    def get_dataset_obj(self):
        dataset_name = self.params.dataset_name
        if dataset_name == 'cwru':
            dataset_obj = CWRU()
        return dataset_obj

    def gen_datasets_li(self):
        each_samples = self.params.each_samples
        all_data_path = self.dataset_obj.get_data_path()
        pt_data_path = self.params.pt_data_path

        printc(f'pt_data_name={pt_data_path} is exist: {os.path.exists(pt_data_path)}','p')
        if os.path.exists(pt_data_path):
            datas_li,labels_li = torch.load(pt_data_path)
        else:
            datas_li = []
            labels_li = []

            for label in range(len(all_data_path)):
                all_type_path = all_data_path[label]
                for hp,each_type_path in enumerate(all_type_path):
                    parse_data = self.dataset_obj.parse_data_from_path(each_type_path)
                    aim=AIM(self.params)
                    actual_sample_len=aim.calculate_sample_len(self.params,each_type_path,hp)
                    starts = []
                    each_order_samples = each_samples // len(all_type_path)
                    max_start = len(parse_data) - actual_sample_len
                    while len(starts) < each_order_samples:
                        start = random.randint(0,max_start)
                        if start  in starts:
                            continue
                        starts.append(start)
                        sample_data = parse_data[start:start + actual_sample_len]
                        sample_data = np.array(sample_data)
                        if "AL_Net" in self.params.net_name:
                            sample_data = Envelope(sample_data)[:400]

                        sample_data = list(sample_data)
                        datas_li.append(sample_data)
                        labels_li.append(label)
            datas_li = torch.Tensor(datas_li)
            labels_li = torch.Tensor(labels_li)
            if self.params.save_pt:
                with open(pt_data_path,'wb') as f:
                    torch.save((datas_li,labels_li),f)
        return datas_li,labels_li