"""
@author Liu Lei
"""
from copy import copy
import warnings

import numpy as np
from prettytable import PrettyTable

warnings.filterwarnings('ignore')


def print_table(head,data):
    table = PrettyTable(head)
    dim = np.array(data).ndim
    if dim == 2:
        table.add_rows(data)
    else:
        table.add_row(data)
    print(table)

def print_dic_in_table(dic,columns=12):
    for k,v in zip(list(dic.keys()),list(dic.values())):
        if len(str(v))>20:
            dic.pop(k)
    columns=columns
    def print_dict_in_table_in(dict):
        head=dict.keys()
        data=dict.values()
        table = PrettyTable(head)
        dim = np.array(data).ndim
        if dim == 2:
            table.add_rows(data)
        else:
            table.add_row(data)
        print(table)
    if len(dic)<=columns:
        print_dict_in_table_in(dic)
    else:
        dic_temp={k:v for i,(k,v) in enumerate(dic.items()) if  i<columns}
        print_dict_in_table_in(dic_temp)
        for k,v in dic_temp.items():
            del dic[k]
        print_dic_in_table(dic)

def printl():
    print("*" * 120)

def printc(content):
    print(f"{'*' * 30} {content} {'*' * 30}")


def printc(content,color='r'):
    if color=='r':
        print(f"\033[1;31m{content}\033[0m")
    elif color=='g':
        print(f"\033[1;32m{content}\033[0m")
    elif color=='y':
        print(f"\033[1;33m{content}\033[0m")
    elif color=='p':
        print(f"\033[1;35m{content}\033[0m")
def print_params(params,columns=12):
    dic = copy(params.__dict__)
    del dic['pt_data_path']
    print_dic_in_table(dic,columns)

