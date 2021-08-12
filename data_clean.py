import sys
import os
import numpy as np
import csv
import pandas as pd
from consts import *
import shutil
import re

# def refresh_csv():
#     '''
#     add p95 into data*.csv
#     '''
#     for j in range(36):
#         filename = "res/newdata"+str(j)+".csv"
#         file = pd.read_csv(filename)
#         df = pd.DataFrame(file)
#         df.to_csv("res/data"+str(j)+".csv")

# def combine_parameters():
#     p1 = np.load("res0804/param300.npy", allow_pickle=True).item()
#     p2 = np.load("res/param300.npy", allow_pickle=True).item()
#     real_p = {}
#     for s in services:
#         real_p[s] = p1[s][:36] + p2[s][36:]
#     np.save("res/new_param300.npy", real_p)

# combine_parameters()

'''
combine results & parameter sets
'''
def merge_paras():
    para1 = np.load("res300_1/param300.npy", allow_pickle=True).item()
    para2 = np.load("res300_2/param_300.npy", allow_pickle=True).item()
    para = {}
    for s in services:
        para[s] = para1[s]+para2[s]
    np.save("res_merged/param.npy", para)

def move_csvs():
    for i in range(300):
        shutil.move("res300_1/data"+str(i)+".csv", "res_merged/data"+str(i)+".csv")
    for i in range(300):
        shutil.move("res300_2/data"+str(i)+".csv", "res_merged/data"+str(i+300)+".csv")

def move_wrk_table():
    for i in range(300):
        shutil.move("wrk_table_300_1/"+str(i), "wrk_table_merged/"+str(i))
    for i in range(300):
        shutil.move("wrk_table_300_2/"+str(i), "wrk_table_merged/"+str(i+300))

'''
check timeout & 500 errors
'''
def check_quality():
    # invalid_list = []
    cnt = 538
    para = np.load("res_clean/param.npy", allow_pickle=True).item()
    para_ori = np.load("res/param_300.npy", allow_pickle=True).item()
    # for s in services:
    #     para[s] = []
    for i in range (200):
        with open("wrk_table/"+str(i)) as f:
            text = f.read()
            timeout = re.search(r'.* (timeout \d*?)\s.*', str(text))
            response500 = re.search(r'.*Non-2xx or 3xx responses: (\d*?)\s.*', str(text))
            if response500:
                num = int(response500.group(1))
                if num > 1000:
                    continue
                    # invalid data
                    # print(i, num)
                    # invalid_list.append(i)
            # merge data
            for s in services:
                para[s].append(para_ori[s][i])
            shutil.copy("res/data"+str(i)+".csv", "res_clean/data"+str(cnt)+".csv")
            cnt += 1
    
    print(cnt)
    print(len(para["frontend"]))
    np.save("res_clean/param.npy", para)

check_quality()