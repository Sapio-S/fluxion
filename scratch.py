import numpy as np
from consts import *

dics = {}
dics["frontend:0.90:MAX"] = 0
dics["frontend:0.90:MIN"] = 0

for i in range(10):
    dic = np.load("tmp_data_rps/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
    dics["frontend:0.90:MAX"] += dic["frontend:0.90:MAX"]
    dics["frontend:0.90:MIN"] += dic["frontend:0.90:MIN"]
print(dics)