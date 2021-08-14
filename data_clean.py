import sys
import os
import numpy as np
import csv
import pandas as pd
from consts import *
import shutil
import re
import heapq
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
merged+new -> clean -> remove outlier
'''


'''
check timeout & 500 errors
'''
def check_quality():
    # invalid_list = []
    cnt = 0
    para = {}
    para_ori = np.load("res_merged/param.npy", allow_pickle=True).item()
    for s in services:
        para[s] = []
    for i in range (600):
        with open("wrk_table_merged/"+str(i)) as f:
            text = f.read()
            timeout = re.search(r'.* (timeout \d*?)\s.*', str(text))
            response500 = re.search(r'.*Non-2xx or 3xx responses: (\d*?)\s.*', str(text))
            if response500:
                num = int(response500.group(1))
                if num > 300:
                    continue
                    # invalid data
                    # print(i, num)
                    # invalid_list.append(i)
            # merge data
            for s in services:
                para[s].append(para_ori[s][i])
            shutil.copy("res_merged/data"+str(i)+".csv", "res_clean/data"+str(cnt)+".csv")
            cnt += 1
    
    para_ori = np.load("res/param_300.npy", allow_pickle=True).item()
    for i in range (300):
        with open("wrk_table/"+str(i)) as f:
            text = f.read()
            timeout = re.search(r'.* (timeout \d*?)\s.*', str(text))
            response500 = re.search(r'.*Non-2xx or 3xx responses: (\d*?)\s.*', str(text))
            if response500:
                num = int(response500.group(1))
                if num > 300:
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
    return cnt


# from res_clean folder to res_rm_outlier folder
def check_outlier(length):
    p90 = []
    for i in range(length):
        data = pd.read_csv("res_clean/data"+str(i)+".csv")
        for row in range(14):
            # data.loc[row]用来取出一个service对应的行
            r = data.loc[row]
            if r["service"] == "frontend":
                p90.append(r["0.90"])
                print(r["0.90"])
    return p90

def rm_outlier(length):
    '''
    get the positions of the data points to be removed
    in this case, is [498, 198, 44, 210, 544, 599, 101, 257]
    '''
    p90 = check_outlier(length)
    sub = heapq.nlargest(int(length/100)+1, range(len(p90)), p90.__getitem__)
    print(sub)
    '''
    delete such points
    '''
    cnt = 0
    para = {}
    para_ori = np.load("res_clean/param.npy", allow_pickle=True).item()
    for s in services:
        para[s] = []
    for i in range (length):
        if i in sub:
            continue
        else:
            for s in services:
                para[s].append(para_ori[s][i])
            shutil.copy("res_clean/data"+str(i)+".csv", "res_rm_outlier/data"+str(cnt)+".csv")
            cnt += 1
            
    print(cnt)
    print(len(para["frontend"]))
    np.save("res_rm_outlier/param.npy", para)


# rm_outlier()
# check_outlier()

def run():
    length = check_quality()
    rm_outlier(length)

run()