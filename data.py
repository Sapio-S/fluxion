import json
import pandas as pd
import numpy as np
import os
from consts import *

'''
将所有csv数据拼成大表读出
csvs = {"service": {"perf": [...], ...}, ...}
csv_onedic = {"service:perf":[...], ...}
'''
def combine_csv():
    csv_onedic = {}
    for f in finals:
        for p in perf:
            csv_onedic[f+":"+p] = []
    for i in range(train_size+test_size):
        data = pd.read_csv(route+"data"+str(i)+".csv")
        for row in range(14):
            # data.loc[row]用来取出一个service对应的行
            r = data.loc[row]
            if r["service"] == "redis" or r["service"] == "total":
                continue
            for p in perf:
                if p != "rps":
                    csv_onedic[r["service"]+":"+p].append(float(r[p])/1000) # change to ms
                else:
                    csv_onedic[r["service"]+":"+p].append(float(r[p]))
    return csv_onedic

'''
input for learning_assignment

input = {"service": [[para value] * 300], ...}
input_names = {"service": ["para name", "para name"], ...}
output = {"service": [...], ...}
f_input = {'Service Name': {'Performance Name': [{'Input Name': Input Val, ...}, ...] * test_size}}
f_input2 = ['Service Name': {'Performance Name': [{'Input Name': Input Val, ...}, ...] * 1}] * test_size
'''
def la_input(para, csv_onedic):
    data_dic = {}
    for s in services:
        # deal with redis and cartservice
        if s == "redis":
            data_dic["get"] = []
            data_dic["set"] = []
            for i in range(length):
                data_dic["get"].append(para[s][i].copy())
                data_dic["get"][i]["hash_max_ziplist_entries"] = para["cartservice"][i]["hash_max_ziplist_entries"]
                data_dic["get"][i]["maxmemory_samples"] = para["cartservice"][i]["maxmemory_samples"]
                data_dic["get"][i]["maxmemory"] = para["cartservice"][i]["maxmemory"]
                data_dic["set"].append(para[s][i].copy())
                data_dic["set"][i]["hash_max_ziplist_entries"] = para["cartservice"][i]["hash_max_ziplist_entries"]
                data_dic["set"][i]["maxmemory_samples"] = para["cartservice"][i]["maxmemory_samples"]
                data_dic["set"][i]["maxmemory"] = para["cartservice"][i]["maxmemory"]
        elif s == "cartservice":
            data_dic[s] = []
            for i in range(length):
                data_dic[s].append({
                    "CPU_LIMIT":para[s][i]["CPU_LIMIT"], 
                    "MEMORY_LIMIT":para[s][i]["MEMORY_LIMIT"],
                    "IPV4_RMEM":para[s][i]["IPV4_RMEM"],
                    "IPV4_WMEM":para[s][i]["IPV4_WMEM"],
                })
        else:
            data_dic[s] = []
            length = len(para[s])
            for i in range(length):
                data_dic[s].append(para[s][i]) 
            # para[s][i] = {'MAX_ADS_TO_SERVE': 3, 'CPU_LIMIT': 364, 'MEMORY_LIMIT': 414, 'IPV4_RMEM': 1376265, 'IPV4_WMEM': 3236110}  

    # generate inputs
    input = {}
    output = {}
    # f_input = {}
    f_input2 = []
    for i in range(test_size):
        f_input2.append({})
    input_names = {}
    for f in finals:
        input[f] = []
        input_names[f] = []
        # f_input[f] = {}

        for i in range(test_size):
            f_input2[i][f] = {}
        output[f] = csv_onedic[f+":0.50"][test_size:test_size+train_size] # TODO: change performance metric
        # add names
        for k, v in data_dic[f][0].items():
            input_names[f].append(k)
        input_names[f].append(f+":rps")

        # add parameter settings for input
        for p_ in data_dic[f][test_size:test_size+train_size]:
            small_instance = []
            for k, v in p_.items():
                small_instance.append(v)
            input[f].append(small_instance)
        
        # # add parameter settings for f_input
        # tmp_fluxion_instance = []
        # for p_ in data_dic[f][:test_size]:
        #     tmp_fluxion_instance.append(p_)
        # for perf in eval_metric:
        #     f_input[f][perf] = tmp_fluxion_instance

        # add parameter settings & rps values for f_input2
        for i in range(train_size):
            '''changed'''
            tmp_list = [data_dic[f][i+test_size]] # TODO: add copy()
            tmp_list[0][f+":rps"] = csv_onedic[f+":rps"][i+test_size]
            for perf in eval_metric:
                f_input2[i][f][perf] = tmp_list

        # add rps values for input [test_size, test_size+train_size]
        for i in range(train_size):
            input[f][i].append(csv_onedic[f+":rps"][test_size+i])

        # # add rps values for f_input(for prediction) [0, test_size]
        # for perf in eval_metric:
        #     sub = 0
        #     for x in f_input[f][perf]:
        #         x[f+":rps"] = csv_onedic[f+":rps"][sub]
        #         sub += 1
    # print(f_input2)
    return input, output, input_names, f_input2

# don't use. 
def fluxion_input(para): 
    data_dic = {}
    file_list = os.listdir(route)
    for s in services:
        if s == "redis":
            data_dic["get"] = {}
            data_dic["set"] = {}
            for p in perf:
                data_dic["get"][p] = []
                data_dic["set"][p] = []
                for i in range(length):
                    data_dic["get"][p].append(para[s][i])
                    data_dic["set"][p].append(para[s][i])
        for p in perf:
            data_dic[s][p] = []
            length = len(para[s])
            for i in range(length):
                # para[s][i] = {'MAX_ADS_TO_SERVE': 3, 'CPU_LIMIT': 364, 'MEMORY_LIMIT': 414, 'IPV4_RMEM': 1376265, 'IPV4_WMEM': 3236110}
                data_dic[s][p].append(para[s][i])
            
                
    # add rps as input
    for file in file_list:
        if file[:4] == "data": # dataxx.csv
            name = file.split('.')
            set = int(name[0][4:]) # 第几组参数获取的数据
            data = pd.read_csv(route+file)

            for row in range(14):
                # data.loc[row]用来取出一个service对应的行
                r = data.loc[row]
                for p in perf:
                    data_dic[r["service"]][p][set]["rps"] = r["rps"]
                    print(data_dic[r["service"]][p][set])

def read_para():
    return np.load(route+"param300.npy", allow_pickle=True).item()
    
def read_res():
    data_dic = {}
    for s in services:
        data_dic[s] = {}
    for p in perf:
        data_dic[s][p] = []
    file_list = os.listdir(route)
    # print(file_list)
    for file in file_list:
        if file[:4] == "data": # dataxx.csv
            data = pd.read_csv(route+file)
            print(data)
            break
            
def get_input():
    para = read_para()
    # print(para["adservice"][:4])
    csvs = combine_csv()
    a, b, c, d = la_input(para, csvs)
    return a, b, c, csvs, d
