"""
@author Liu Lei
"""
import time

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

import models
from datasets.Loader import Loader
from datasets.ExperimentParams import Params
from utils.print_util import print_params

if __name__ == "__main__":
    batch_size_li=[1,2,3,4]
    net_name_li=['AL_Net','CLFormer','DN_CNN','LiNet','TI_CNN','Wen_CNN']
    for batch_size in batch_size_li:
        for net_name in net_name_li:
            start_time = time.time()
            params = Params() # experiment parameters
            py_name = __file__.replace('/','\\').split('\\')[-1].split('.')[0]
            params.py_name=py_name
            params.net_name = net_name
            params.dataset_name = 'cwru'
            params.debug = False
            params.batch_size=batch_size

            net = getattr(models,net_name)(params).cuda()

            params.sample_len = net.get_len()
            params.data_type = net.get_type()

            loader = Loader(params)
            train_loader,test_loader = loader.get_loaders()

            print_params(params)
            optimizer = optim.SGD(net.parameters(),lr=params.lr,momentum=0.9)

            for epoch in range(1,params.epochs + 1):
                params.epoch = epoch
                net.train()
                batch_id = 1
                for datas,labels in train_loader:
                    outputs=net.forward(datas.cuda()).cpu()
                    loss=CrossEntropyLoss()(outputs,labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    net.eval()
                    acc_num = 0
                    total_num = 0
                    for datas,labels in test_loader:
                        outputs = net.forward(datas.cuda())
                        predicted_label = torch.argmax(outputs,dim=1).cpu()
                        acc_num = acc_num + torch.sum(predicted_label == labels).item()
                        total_num = total_num + len(datas)
                    test_acc = round(acc_num / total_num * 100,4)
                    end_time = time.time()

                print(f"Epoch:{epoch}/{params.epochs} {net_name=} {test_acc=} {batch_size=}, total_loss={loss.item()}")