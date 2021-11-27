import json
import pandas as pd
import numpy as np
import os
from consts import *
from const_dic import const_dic
import csv
import random

finals = ["adservice", "currencyservice", "emailservice", 
"paymentservice", "productcatalogservice","shippingservice", "get", 
"set", "recommendation_pod0","recommendation_pod1", "cartservice", "checkoutservice","frontend" ]

'''
将所有csv数据拼成大表读出
csvs = {"service": {"perf": [...], ...}, ...}
csv_onedic = {"service:perf":[...], ...}
'''
def combine_csv(size, route):
    csv_onedic = []
    minidic = {}
    data = pd.read_csv(route)
    for row in range(16):
        # data.loc[row]用来取出一个service对应的行
        r = data.loc[row]
        if r["service"] == "redis" or r["service"] == "total":
            continue
        for p in perf:
            minidic[r["service"]+":"+p] = float(r[p])
    for i in range(size):
        csv_onedic.append(minidic.copy()) # repeat 10000 times
    return csv_onedic

def sample_selection(num_samples, x_bounds):
    '''
    num_samples:取样区间个数
    x_bounds:array,表示每个变量的上下界, e.g.[[100, 600], [100, 600], [100, 600]]
    返回值：array, num_samples组参数，e.g. [[598, 287, 539], [339, 242, 466]]
    '''
    outputs = []

    sampled = []
    for i in range(len(x_bounds)):
        sampled.append([False] * num_samples)

    # Generate LHS random samples
    for i in range(0, num_samples):
        temp = [None] * len(x_bounds)
        for j in range(len(x_bounds)):
            idx = None
            while idx == None or sampled[j][idx] == True:
                idx = random.randint(0, num_samples - 1)  # Choose the interval to sample

            sampled[j][idx] = True  # Note that we have sampled this interval

            intervalSize = ((x_bounds[j][1] - x_bounds[j][0]) / num_samples)
            intervalLowerBound = int(x_bounds[j][0] + intervalSize * idx)
            intervalUpperBound = int(intervalLowerBound + intervalSize)
            # sample = random.uniform(intervalLowerBound, intervalUpperBound)
            sample = random.randint(intervalLowerBound, intervalUpperBound)  # Samples within the chosen interval
            temp[j] = sample
        outputs.append(temp)

    return outputs


def read_para(num_samples):
    para = {}
    for service in services:
        possible_para = const_dic[service]
        boundaries = []
        header = []
        service_list = []
        for k, v in possible_para.items():
            header.append(k) # name of parameter, e.g. adservice-MAX_ADS_TO_SERVE
            boundaries.append([v["MIN"], v["MAX"]])
        params = sample_selection(num_samples, boundaries)
        for i in range(num_samples):
            service_dic = {}
            for j in range(len(header)):
                service_dic[header[j]] = params[i][j]
            service_list.append(service_dic)
        para[service] = service_list
    
    length = num_samples
    para2 = {}
    for s in services:
        if s == "redis":
            para2["get"] = []
            para2["set"] = []
            for i in range(length):
                para2["get"].append(para[s][i].copy())
                para2["get"][i]["hash_max_ziplist_entries"] = para["cartservice"][i]["hash_max_ziplist_entries"]
                para2["get"][i]["maxmemory_samples"] = para["cartservice"][i]["maxmemory_samples"]
                para2["get"][i]["maxmemory"] = para["cartservice"][i]["maxmemory"]
                para2["set"].append(para[s][i].copy())
                para2["set"][i]["hash_max_ziplist_entries"] = para["cartservice"][i]["hash_max_ziplist_entries"]
                para2["set"][i]["maxmemory_samples"] = para["cartservice"][i]["maxmemory_samples"]
                para2["set"][i]["maxmemory"] = para["cartservice"][i]["maxmemory"]
        elif s == "cartservice":
            para2[s] = []
            for i in range(length):
                para2[s].append({
                    "CPU_LIMIT":para[s][i]["CPU_LIMIT"], 
                    "MEMORY_LIMIT":para[s][i]["MEMORY_LIMIT"],
                    "IPV4_RMEM":para[s][i]["IPV4_RMEM"],
                    "IPV4_WMEM":para[s][i]["IPV4_WMEM"],
                })
        # elif s == "checkoutservice":
        #     para2[s] = []
        #     para2["checkout_pod0"] = []
        #     para2["checkout_pod1"] = []
        #     for i in range(length):
        #         para2[s].append(para[s][i])
        #         para2["checkout_pod0"].append(para[s][i])
        #         para2["checkout_pod1"].append(para[s][i])
        elif s == "recommendationservice":
            para2[s] = []
            para2["recommendation_pod0"] = []
            para2["recommendation_pod1"] = []
            for i in range(length):
                para2[s].append(para[s][i])
                para2["recommendation_pod0"].append(para[s][i])
                para2["recommendation_pod1"].append(para[s][i])
        else:
            para2[s] = []
            for i in range(length):
                para2[s].append(para[s][i]) 

    para3 = []
    for i in range(length):
        minimap = {}
        for f in finals:
            for k in para2[f][i]:
                minimap[f+":"+k] = para2[f][i][k]
        para3.append(minimap)

    return para3

def standardize(csv, k):
    # load scalers
    dic = np.load("std_scaler_dataset_whole.npy", allow_pickle = True).item()
    if k[:19] == "recommendation_pod0" or k[:19] == "recommendation_pod1":
        k = "recommendationservice"+k[19:]
    # if k[:13] == "checkout_pod0" or k[:13] == "checkout_pod1":
    #     k = "checkoutservice"+k[13:]
    std = dic[k+":STD"]
    avg = dic[k+":AVG"]
    if std == 0:
        std = 1
    csv = [(csv[i] - avg) / std for i in range(len(csv))]
    return csv, dic

def restruct(from_route, to_name, size=10000):
    csvs = combine_csv(size, from_route)
    para = read_para(size)
    # print(para)

    for i in range(size):
        csvs[i].update(para[i])
    # print(csvs)
    with open(to_name ,"w") as f:
        f_csv = csv.DictWriter(f, csvs[0].keys())
        f_csv.writeheader()
        f_csv.writerows(csvs)
    
    data = {}
    para_dic = {}
    with open("dataset-recom2.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for name in reader.fieldnames:
            data[name] = []
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(float(row[name]))
        for name in reader.fieldnames:
            data[name], scale = standardize(data[name], name)
            para_dic.update(scale)

        with open("dataset-recom2-standardized.csv", "w") as f:
            writer = csv.DictWriter(f, reader.fieldnames)
            writer.writeheader()
            for i in range(size):
                writer.writerow({name:float(data[name][i]) for name in reader.fieldnames})
    

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    restruct("data-2recom2.csv", "dataset-recom2.csv", 100000)