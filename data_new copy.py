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
def combine_csv(size, sub_map):
    csv_onedic = {}
    for f in finals:
        for p in perf:
            csv_onedic[f+":"+p] = []
    for i in range(size):
        data = pd.read_csv(route+"data"+str(sub_map[i])+".csv")
        for row in range(14):
            # data.loc[row]用来取出一个service对应的行
            r = data.loc[row]
            if r["service"] == "redis" or r["service"] == "total" or r["service"] == "checkout_pod0":
                continue
            for p in perf:
                csv_onedic[r["service"]+":"+p].append(float(r[p]))
    return csv_onedic

def normalize(csv):
    dic = {}
    for k in csv:
        mini = np.min(csv[k])
        maxi = np.max(csv[k])
        dic[k+":MAX"] = maxi
        dic[k+":MIN"] = mini
        if maxi == mini:
            csv[k] = [0.5 for i in range(len(csv[k]))]
        else:
            csv[k] = [(csv[k][i] - mini) / (maxi - mini) for i in range(len(csv[k]))]
    return csv, dic

def standardize(csv):
    dic = {}
    for k in csv:
        std = np.std(csv[k])
        avg = np.mean(csv[k])
        dic[k+":STD"] = std
        dic[k+":AVG"] = avg
        if std == 0:
            std = 1
        csv[k] = [(csv[k][i] - avg) / std for i in range(len(csv[k]))]
    return csv, dic

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
        for f in finals:
            output[p][f] = csv_onedic[f+":"+p][test_size+valid_size:test_size+valid_size+train_size]
    input_names = {}
    for f in finals:
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
        for p_ in data_dic[f][test_size+valid_size:test_size+valid_size+train_size]:
            small_instance = []
            for k, v in p_.items():
                small_instance.append(v)
            input[f].append(small_instance)

        # add parameter settings & rps values for f_input2
        for i in range(test_size):
            tmp_list = [data_dic[f][i]] # TODO: add copy()
            for down in extra_names[f]:
                tmp_list[0][down+":0.90"] = csv_onedic[down+":0.90"][i]
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
            for down in extra_names[f]:
                tmp_list[0][down+":0.90"] = csv_onedic[down+":0.90"][i]
            tmp_list[0][f+":rps"] = csv_onedic[f+":rps"][i+test_size]
            for perf in eval_metric:
                f_input4[i][f][perf] = tmp_list

        # add rps values for input [test_size, test_size+train_size]
        for i in range(train_size):
            input[f][i].append(csv_onedic[f+":rps"][i+test_size+valid_size])

    return input, output, input_names, f_input2, f_input3, f_input4

def read_para():
    para = np.load(route+"param.npy", allow_pickle=True).item()
    # scale data
    for s in services:
        for i in range(len(para[s])):
            for p in para[s][i]:
                para[s][i][p] = (para[s][i][p] - const_dic[s][p]["MIN"]) / (const_dic[s][p]["MAX"] - const_dic[s][p]["MIN"])

    return para

def get_input(i):
    a = np.load("tmp_data_scale/"+str(i)+"_sample_x.npy", allow_pickle=True).item()
    b = np.load("tmp_data_scale/"+str(i)+"_sample_y.npy", allow_pickle=True).item()
    c = np.load("tmp_data_scale/names.npy", allow_pickle=True).item()
    d = np.load("tmp_data_scale/"+str(i)+"_perf_data.npy", allow_pickle=True).item()
    e = np.load("tmp_data_scale/"+str(i)+"_test_data.npy", allow_pickle=True)
    f = np.load("tmp_data_scale/"+str(i)+"_train_data.npy", allow_pickle=True)
    g = np.load("tmp_data_scale/"+str(i)+"_valid_data.npy", allow_pickle=True)
    return a, b, c, d, e, f, g # samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data

def get_input_norm(i):
    a = np.load("tmp_data_norm0910/"+str(i)+"_sample_x.npy", allow_pickle=True).item()
    b = np.load("tmp_data_norm0910/"+str(i)+"_sample_y.npy", allow_pickle=True).item()
    c = np.load("tmp_data_norm0910/names.npy", allow_pickle=True).item()
    d = np.load("tmp_data_norm0910/"+str(i)+"_perf_data.npy", allow_pickle=True).item()
    e = np.load("tmp_data_norm0910/"+str(i)+"_test_data.npy", allow_pickle=True)
    f = np.load("tmp_data_norm0910/"+str(i)+"_train_data.npy", allow_pickle=True)
    g = np.load("tmp_data_norm0910/"+str(i)+"_valid_data.npy", allow_pickle=True)
    h = np.load("tmp_data_norm0910/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
    return a, b, c, d, e, f, g, h # samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data, scale

def get_input_std(i):
    a = np.load("tmp_data_0929valid/"+str(i)+"_sample_x.npy", allow_pickle=True).item()
    b = np.load("tmp_data_0929valid/"+str(i)+"_sample_y.npy", allow_pickle=True).item()
    c = np.load("tmp_data_0929valid/names.npy", allow_pickle=True).item()
    d = np.load("tmp_data_0929valid/"+str(i)+"_perf_data.npy", allow_pickle=True).item()
    e = np.load("tmp_data_0929valid/"+str(i)+"_test_data.npy", allow_pickle=True)
    f = np.load("tmp_data_0929valid/"+str(i)+"_train_data.npy", allow_pickle=True)
    g = np.load("tmp_data_0929valid/"+str(i)+"_valid_data.npy", allow_pickle=True)
    h = np.load("tmp_data_0929valid/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
    return a, b, c, d, e, f, g, h # samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data, scale

'''
with validation tests
'''
def store_input_scale(sub_map, i, train_size=500, test_size=84, valid_size=84):
    para = read_para()
    csvs = combine_csv(train_size+valid_size+test_size, sub_map)
    a, b, c, d, e, f = la_input(para, csvs, train_size, test_size, valid_size, sub_map)
    np.save("tmp_data_scale/"+str(i)+"_sample_x", a)
    np.save("tmp_data_scale/"+str(i)+"_sample_y", b)
    np.save("tmp_data_scale/names", c)
    np.save("tmp_data_scale/"+str(i)+"_perf_data", csvs)
    np.save("tmp_data_scale/"+str(i)+"_test_data", d)
    np.save("tmp_data_scale/"+str(i)+"_train_data", e)
    np.save("tmp_data_scale/"+str(i)+"_valid_data", f)

def store_input_std(sub_map, i, train_size=500, test_size=84, valid_size=84):
    para = read_para()
    csvs = combine_csv(train_size+valid_size+test_size, sub_map)
    csvs, csv_m = standardize(csvs)
    a, b, c, d, e, f = la_input(para, csvs, train_size, test_size, valid_size, sub_map)
    np.save("tmp_data_0929valid/"+str(i)+"_sample_x", a)
    np.save("tmp_data_0929valid/"+str(i)+"_sample_y", b)
    np.save("tmp_data_0929valid/names", c)
    np.save("tmp_data_0929valid/"+str(i)+"_perf_data", csvs)
    np.save("tmp_data_0929valid/"+str(i)+"_test_data", d)
    np.save("tmp_data_0929valid/"+str(i)+"_train_data", e)
    np.save("tmp_data_0929valid/"+str(i)+"_valid_data", f)
    np.save("tmp_data_0929valid/"+str(i)+"_csv_scale", csv_m)

def store_input_norm(sub_map, i, train_size=500, test_size=84, valid_size=84):
    para = read_para()
    csvs = combine_csv(train_size+valid_size+test_size, sub_map)
    csvs, csv_m = normalize(csvs)
    a, b, c, d, e, f = la_input(para, csvs, train_size, test_size, valid_size, sub_map)
    np.save("tmp_data_norm0910/"+str(i)+"_sample_x", a)
    np.save("tmp_data_norm0910/"+str(i)+"_sample_y", b)
    np.save("tmp_data_norm0910/names", c)
    np.save("tmp_data_norm0910/"+str(i)+"_perf_data", csvs)
    np.save("tmp_data_norm0910/"+str(i)+"_test_data", d)
    np.save("tmp_data_norm0910/"+str(i)+"_train_data", e)
    np.save("tmp_data_norm0910/"+str(i)+"_valid_data", f)
    np.save("tmp_data_norm0910/"+str(i)+"_csv_scale", csv_m)

def generate_tmp_data_scale():
    print("generatring data. stored in tmp_data_scale/ folder.")
    for i in range(10):
        sub_map = np.arange(983)
        np.random.seed(i)
        np.random.shuffle(sub_map)
        store_input_scale(sub_map, i, 800,133,50) # no validation

def generate_tmp_data_std():
    print("generatring data. stored in tmp_data_0929valid/ folder.")
    for i in range(10):
        sub_map = np.arange(983)
        np.random.seed(i)
        np.random.shuffle(sub_map)
        store_input_std(sub_map, i, 800,133,50) # no validation

def generate_tmp_data_norm():
    print("generatring data. stored in tmp_data_norm0910/ folder.")
    for i in range(10):
        sub_map = np.arange(983)
        np.random.seed(i)
        np.random.shuffle(sub_map)
        store_input_norm(sub_map, i, 800,133,50) # no validation

def norm_scaler(y, mini, maxi):
    if type(y) is np.float64 or type(y) is float:
        return y * (maxi - mini) + mini
    else:
        return [x * (maxi - mini) + mini for x in y]

def std_scaler(y, avg, std):
    if type(y) is np.float64 or type(y) is float:
        return y * std + avg
    else:
        return [x * std + avg for x in y]

def test():
    l = {"1":np.arange(100)}
    l,a = normalize(l)
    print(norm_scaler(l["1"], a["1:MIN"], a["1:MAX"]))
    l = {"1":np.arange(100)}
    l,a = standardize(l)
    print(std_scaler(l["1"], a["1:AVG"], a["1:STD"]))

if __name__ == "__main__":
    # generate_tmp_data_norm()
    generate_tmp_data_std()
    # test()

