import json
import pandas as pd
import numpy as np
import os
from consts import *
from const_dic import const_dic

'''
将所有csv数据拼成大表读出
csvs = {"service": {"perf": [...], ...}, ...}
csv_onedic = {"service:perf":[...], ...}
'''
def combine_csv(train_size, test_size, sub_map):
    csv_onedic = {}
    for f in finals:
        for p in perf:
            csv_onedic[f+":"+p] = []
    for i in range(train_size+test_size):
        data = pd.read_csv(route+"data"+str(sub_map[i])+".csv")
        for row in range(14):
            # data.loc[row]用来取出一个service对应的行
            r = data.loc[row]
            if r["service"] == "redis" or r["service"] == "total":
                continue
            for p in perf:
                if p != "rps":
                    # scale data. used to be ms
                    # csv_onedic[r["service"]+":"+p].append(float(r[p])/1000) 
                    csv_onedic[r["service"]+":"+p].append(float(r[p])/scale_para[r["service"]]) 
                else:
                    csv_onedic[r["service"]+":"+p].append(float(r[p]))
    return csv_onedic, {}

def combine_csv_normalize(train_size, test_size, sub_map):
    '''
    add normalize
    '''
    csv_onedic = {}
    csv_m = {}
    for f in finals:
        for p in perf:
            csv_onedic[f+":"+p] = []
            csv_m[f+":"+p+":MAX"] = 0
            csv_m[f+":"+p+":MIN"] = 10000000
    
    # record max & min value for train data only
    for i in range(test_size, test_size+train_size): 
        data = pd.read_csv(route+"data"+str(sub_map[i])+".csv")
        for row in range(14):
            # data.loc[row]用来取出一个service对应的行
            r = data.loc[row]
            if r["service"] == "redis" or r["service"] == "total":
                continue
            for p in perf:
                min_ = csv_m[r["service"]+":"+p+":MIN"]
                max_ = csv_m[r["service"]+":"+p+":MAX"]
                val = float(r[p])
                if val > max_:
                    csv_m[r["service"]+":"+p+":MAX"] = val
                if val < min_:
                    csv_m[r["service"]+":"+p+":MIN"] = val

    # normalize all data
    for i in range(train_size+test_size):
        data = pd.read_csv(route+"data"+str(sub_map[i])+".csv")
        for row in range(14):
            # data.loc[row]用来取出一个service对应的行
            r = data.loc[row]
            if r["service"] == "redis" or r["service"] == "total":
                continue
            for p in perf:
                val = 0
                try:
                    val = (float(r[p]) - csv_m[r["service"]+":"+p+":MIN"]) / ( csv_m[r["service"]+":"+p+":MAX"] - csv_m[r["service"]+":"+p+":MIN"])
                except:
                    val = 0.5
                csv_onedic[r["service"]+":"+p].append(val)
    return csv_onedic, csv_m

'''
input for learning_assignment
restructure data into desired shapes

input = {"service": [[para value] * 300], ...}
input_names = {"service": ["para name", "para name"], ...}
output = {"p90":{"service": [...], ...}, ...}
f_input = {'Service Name': {'Performance Name': [{'Input Name': Input Val, ...}, ...] * test_size}}
f_input2 = [{'Service Name': {'Performance Name': [{'Input Name': Input Val, ...}] }}] * test_size
f_input3 = [{'Service Name': {'Performance Name': [{'Input Name': Input Val, ...}] }}] * train_size
f_input4 = [{'Service Name': {'Performance Name': [{'Input Name': Input Val, ...}] }}] * valid_size
'''
def la_input(para, csv_onedic, train_size, test_size, valid_size, sub_map):
    data_dic = {}
    length = test_size+train_size+valid_size
    for s in services:
        # deal with redis and cartservice
        if s == "redis":
            data_dic["get"] = []
            data_dic["set"] = []
            for i in range(length):
                data_dic["get"].append(para[s][sub_map[i]].copy())
                data_dic["get"][i]["hash_max_ziplist_entries"] = para["cartservice"][sub_map[i]]["hash_max_ziplist_entries"]
                data_dic["get"][i]["maxmemory_samples"] = para["cartservice"][sub_map[i]]["maxmemory_samples"]
                data_dic["get"][i]["maxmemory"] = para["cartservice"][sub_map[i]]["maxmemory"]
                data_dic["set"].append(para[s][sub_map[i]].copy())
                data_dic["set"][i]["hash_max_ziplist_entries"] = para["cartservice"][sub_map[i]]["hash_max_ziplist_entries"]
                data_dic["set"][i]["maxmemory_samples"] = para["cartservice"][sub_map[i]]["maxmemory_samples"]
                data_dic["set"][i]["maxmemory"] = para["cartservice"][sub_map[i]]["maxmemory"]
        elif s == "cartservice":
            data_dic[s] = []
            for i in range(length):
                data_dic[s].append({
                    "CPU_LIMIT":para[s][sub_map[i]]["CPU_LIMIT"], 
                    "MEMORY_LIMIT":para[s][sub_map[i]]["MEMORY_LIMIT"],
                    "IPV4_RMEM":para[s][sub_map[i]]["IPV4_RMEM"],
                    "IPV4_WMEM":para[s][sub_map[i]]["IPV4_WMEM"],
                })
        else:
            data_dic[s] = []
            for i in range(length):
                data_dic[s].append(para[s][sub_map[i]]) 
            # para[s][i] = {'MAX_ADS_TO_SERVE': 3, 'CPU_LIMIT': 364, 'MEMORY_LIMIT': 414, 'IPV4_RMEM': 1376265, 'IPV4_WMEM': 3236110}  

    # generate inputs
    input = {}
    output = {}
    f_input2 = []
    f_input3 = []
    f_input4 = []
    for i in range(test_size):
        f_input2.append({})
    for i in range(train_size):
        f_input3.append({})
    for i in range(valid_size):
        f_input4.append({})
    for p in eval_metric:
        output[p] = {}
        for f in finals2:
            output[p][f] = csv_onedic[f+":"+p][test_size:test_size+train_size]
    input_names = {}
    for f in finals2:
        input[f] = []
        input_names[f] = []

        for i in range(test_size):
            f_input2[i][f] = {}
        for i in range(train_size):
            f_input3[i][f] = {}
        for i in range(valid_size):
            f_input4[i][f] = {}
        
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

        # add parameter settings & rps values for f_input2
        for i in range(test_size):
            tmp_list = [data_dic[f][i]] # TODO: add copy()
            tmp_list[0][f+":rps"] = csv_onedic[f+":rps"][i]
            for perf in eval_metric:
                f_input2[i][f][perf] = tmp_list

        # add parameter settings & rps values for f_input3
        for i in range(train_size):
            tmp_list = [data_dic[f][i+test_size+valid_size]] # TODO: add copy()
            tmp_list[0][f+":rps"] = csv_onedic[f+":rps"][i+test_size+valid_size]
            for perf in eval_metric:
                f_input3[i][f][perf] = tmp_list
        
        # add parameter settings & rps values for f_input4
        for i in range(valid_size):
            tmp_list = [data_dic[f][i+test_size]] # TODO: add copy()
            tmp_list[0][f+":rps"] = csv_onedic[f+":rps"][i+test_size]
            for perf in eval_metric:
                f_input4[i][f][perf] = tmp_list

        # add rps values for input [test_size, test_size+train_size]
        for i in range(train_size):
            input[f][i].append(csv_onedic[f+":rps"][test_size+i])

    return input, output, input_names, f_input2, f_input3, f_input4

def read_para_original():
    para = np.load(route+"param.npy", allow_pickle=True).item()
    para_m = {}
    return para, para_m

def read_para():
    para = np.load(route+"param.npy", allow_pickle=True).item()
    # scale data
    for s in services:
        for i in range(len(para[s])):
            for p in para[s][i]:
                para[s][i][p] = para[s][i][p] / const_dic[s][p]["MAX"]

    return para, {}

def get_input_original(train_size, test_size, sub_map):
    para, para_m = read_para()
    csvs, csv_m = combine_csv(train_size, test_size, sub_map)
    a, b, c, d, e = la_input(para, csvs, train_size, test_size, sub_map)
    return a, b, c, csvs, d, e

def get_input(i):
    a = np.load("tmp_data_rps/"+str(i)+"_sample_x.npy", allow_pickle=True).item()
    b = np.load("tmp_data_rps/"+str(i)+"_sample_y.npy", allow_pickle=True).item()
    c = np.load("tmp_data_rps/names.npy", allow_pickle=True).item()
    d = np.load("tmp_data_rps/"+str(i)+"_perf_data.npy", allow_pickle=True).item()
    e = np.load("tmp_data_rps/"+str(i)+"_test_data.npy", allow_pickle=True)
    f = np.load("tmp_data_rps/"+str(i)+"_train_data.npy", allow_pickle=True)
    g = np.load("tmp_data_rps/"+str(i)+"_valid_data.npy", allow_pickle=True)
    return a, b, c, d, e, f, g # samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data

'''
with validation tests
'''
def store_input_valid(sub_map, i, train_size=500, test_size=84, valid_size=84):
    para, para_m = read_para()
    csvs, csv_m = combine_csv(train_size, valid_size+test_size, sub_map)
    a, b, c, d, e, f = la_input(para, csvs, train_size, test_size, valid_size, sub_map)
    np.save("tmp_data_rps/"+str(i)+"_sample_x", a)
    np.save("tmp_data_rps/"+str(i)+"_sample_y", b)
    np.save("tmp_data_rps/names", c)
    np.save("tmp_data_rps/"+str(i)+"_perf_data", csvs)
    np.save("tmp_data_rps/"+str(i)+"_test_data", d)
    np.save("tmp_data_rps/"+str(i)+"_train_data", e)
    np.save("tmp_data_rps/"+str(i)+"_valid_data", f)
    np.save("tmp_data_rps/"+str(i)+"_csv_scale", csv_m)
    np.save("tmp_data_rps/"+str(i)+"_para_scale", para_m)

def generate_tmp_data_rps():
    print("generatring data. stored in tmp_data_rps/ folder.")
    for i in range(10):
        sub_map = np.arange(562)
        np.random.seed(i)
        np.random.shuffle(sub_map)
        store_input_valid(sub_map, i, 400, 162, 0) # no validation

if __name__ == "__main__":
    generate_tmp_data_rps()
