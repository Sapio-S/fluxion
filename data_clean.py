import sys
import os
import numpy as np
import csv
import pandas as pd
from consts import *

def refresh_csv():
    '''
    add p95 into data*.csv
    '''
    for j in range(36):
        filename = "res/newdata"+str(j)+".csv"
        file = pd.read_csv(filename)
        df = pd.DataFrame(file)
        df.to_csv("res/data"+str(j)+".csv")

def combine_parameters():
    p1 = np.load("res0804/param300.npy", allow_pickle=True).item()
    p2 = np.load("res/param300.npy", allow_pickle=True).item()
    real_p = {}
    for s in services:
        real_p[s] = p1[s][:36] + p2[s][36:]
    np.save("res/new_param300.npy", real_p)

combine_parameters()

