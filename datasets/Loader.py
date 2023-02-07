"""
@author Liu Lei
"""
import os

import torch
from torch.utils.data import Dataset,DataLoader

from datasets.GenPtFile import GenPt

class BaseLoader(Dataset):
    def __init__(self,params):
        self.params = params
        self.datas,self.labels= GenPt(params).gen_datasets_li()

    def __getitem__(self,index):
        data,lable = self.datas[index],int(self.labels[index])
        data = data.type(torch.FloatTensor)
        return data,lable

    def __len__(self):
        return len(self.datas)

    def gen_loader(self,shuffle=False,drop_last=False):
        debug = self.params.debug
        batch_size = self.params.batch_size
        train_rate = self.params.train_rate
        if debug:
            train_rate = 0.95
        train_len,test_len = int(train_rate * len(self)),int((1 - train_rate)  * len(self))
        train_set,test_set = torch.utils.data.random_split(self,[train_len,test_len]) # random shuffle
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)
        if debug:
            train_loader =test_loader
        return train_loader,test_loader
class Loader():
    def __init__(self,params):
        super().__init__()
        self.params=params
    def get_loaders(self,):
        pt_data_base = f"PtFile\\debug={self.params.debug}\\{self.params.data_type}"

        os.makedirs(pt_data_base,exist_ok=True)
        pt_data_path = f"{pt_data_base}\\repeat={self.params.repeat}_{self.params.each_samples}_{self.params.sample_len}_{self.params.dataset_name}.pt"

        self.params.pt_data_path = pt_data_path
        train_loader,test_loader=BaseLoader(self.params).gen_loader()
        return train_loader,test_loader