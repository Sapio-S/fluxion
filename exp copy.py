# import numpy as np

# def func(x):
#     return [i*10 for i in x]

# dx = np.random.normal(1, 0.1, 100)
# dy = np.random.normal(2, 0.1, 100)

# x = np.arange(100,200,1)
# np.random.shuffle(x)
# y = func(x) # groundtruth

# realx = x+dx
# realy = func(realx) + dy

# # get expected error
# errs = []
# for i in range(100):
#     errs.append(realy[i] - y[i])
# print(np.mean(errs))
# import itertools
# sample_size = 5
# num = 3
# permutation = [0]
# for k in range(num):
#     permutation = list(itertools.product( permutation,range(sample_size)))
#     p2 = []
#     for t in permutation:
#         print(t)
#         p2.append(t[0], t[1])
#     permutation = p2
# print(permutation)


# sample_size = 2
# permutation = []
# for i1 in range(sample_size):
#     for i2 in range(sample_size):
#         for i3 in range(sample_size):
#             for i4 in range(sample_size):
#                 for i5 in range(sample_size):
#                     for i6 in range(sample_size):
#                         for i7 in range(sample_size):
#                             for i8 in range(sample_size):
#                                 permutation.append([i1,i2,i3,i4,i5,i6,i7,i8])
# print(permutation)



import os, sys
import random
sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne
import itertools
from consts import *
from data_new import get_input, get_input_norm, get_input_std, norm_scaler, std_scaler
import numpy as np
import lib_data
import csv
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/2checkout.csv"
samples_x, samples_y, checkout, err_msg = lib_data.readCSVFile([dataset_filename], ["frontend:0.90"], "frontend:0.90")
print(np.mean(checkout), np.std(checkout))
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/single.csv"
samples_x, samples_y, checkout, err_msg = lib_data.readCSVFile([dataset_filename], ["frontend:0.90"], "frontend:0.90")
print(np.mean(checkout), np.std(checkout))
# samples_x, samples_y, checkout0, err_msg = lib_data.readCSVFile([dataset_filename], ["checkout_pod0:0.90"], "checkout_pod0:0.90")
# samples_x, samples_y, checkout1, err_msg = lib_data.readCSVFile([dataset_filename], ["checkout_pod1:0.90"], "checkout_pod1:0.90")

# for i in checkout1:
#     print(i)

# samples_x, samples_y, checkout, err_msg = lib_data.readCSVFile([dataset_filename], ["checkoutservice:0.90"], "checkoutservice:0.90")
# print(np.mean(checkout), np.std(checkout))

# samples_x, samples_y, checkout, err_msg = lib_data.readCSVFile([dataset_filename], ["frontend:0.90"], "frontend:0.90")
# print(np.mean(checkout), np.std(checkout))