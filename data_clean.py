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
merged+new -> clean -> remove outlier
'''



'''
combine results & parameter sets
'''
def merge_paras():
    para1 = np.load("res0813/param_50.npy", allow_pickle=True).item()
    para2 = np.load("res0814/param_100.npy", allow_pickle=True).item()
    para3 = np.load("res0816/param_500.npy", allow_pickle=True).item()
    para = {}
    for s in services:
        para[s] = para1[s]+para2[s]+para3[s]
    np.save("res_rps_merged/param.npy", para)

def move_csvs():
    for i in range(50):
        shutil.move("res0813/data"+str(i)+".csv", "res_rps_merged/data"+str(i)+".csv")
    for i in range(100):
        shutil.move("res0814/data"+str(i)+".csv", "res_rps_merged/data"+str(i+50)+".csv")
    for i in range(500):
        shutil.move("res0816/data"+str(i)+".csv", "res_rps_merged/data"+str(i+150)+".csv")

def move_wrk_table():
    for i in range(50):
        shutil.move("wrk_table0813/"+str(i), "wrk_rps_merged/"+str(i))
    for i in range(100):
        shutil.move("wrk_table0814/"+str(i), "wrk_rps_merged/"+str(i+50))
    for i in range(500):
        shutil.move("wrk_table0816/"+str(i), "wrk_rps_merged/"+str(i+150))

'''
check timeout & 500 errors
'''
def check_quality():
    # invalid_list = []
    cnt = 0
    para = {}
    para_ori = np.load("res_rps_merged/param.npy", allow_pickle=True).item()
    for s in services:
        para[s] = []
    for i in range (650):
        with open("wrk_rps_merged/"+str(i)) as f:
            text = f.read()
            timeout = re.search(r'.* (timeout \d*?)\s.*', str(text))
            response500 = re.search(r'.*Non-2xx or 3xx responses: (\d*?)\s.*', str(text))
            if response500:
                num = int(response500.group(1))
                if num > 300:
                    # invalid data
                    print(i, num)
                    # invalid_list.append(i)
                    continue
            # merge data
            for s in services:
                para[s].append(para_ori[s][i])
            shutil.copy("res_rps_merged/data"+str(i)+".csv", "res_clean_rps/data"+str(cnt)+".csv")
            cnt += 1

    print(cnt)
    print(len(para["frontend"]))
    np.save("res_clean_rps/param.npy", para)
    return cnt


# from res_clean folder to res_rm_outlier folder
def check_outlier(length):
    p90 = []
    for i in range(length):
        data = pd.read_csv("res_clean_rps/data"+str(i)+".csv")
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
    para_ori = np.load("res_clean_rps/param.npy", allow_pickle=True).item()
    for s in services:
        para[s] = []
    for i in range (length):
        if i in sub:
            continue
        else:
            for s in services:
                para[s].append(para_ori[s][i])
            shutil.copy("res_clean_rps/data"+str(i)+".csv", "res_rm_outlier_rps/data"+str(cnt)+".csv")
            cnt += 1
            
    print(cnt)
    print(len(para["frontend"]))
    np.save("res_rm_outlier_rps/param.npy", para)


# rm_outlier()
# check_outlier()

def run():
    # merge_paras()
    # move_csvs()
    # move_wrk_table()
    length = check_quality()
    rm_outlier(length)

run()