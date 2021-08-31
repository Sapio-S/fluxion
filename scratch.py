import numpy as np
from consts import *

# dics = {}
# dics["frontend:0.100:MAX"] = 0
# dics["frontend:0.100:MIN"] = 0

# for i in range(10):
#     dic = np.load("tmp_data_rps/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
#     dics["frontend:0.100:MAX"] += dic["frontend:0.100:MAX"]
#     dics["frontend:0.100:MIN"] += dic["frontend:0.100:MIN"]
# print(dics)

# python3 demo_fluxion.py

import os, random, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from IngestionEngine_CSV import ingestionEngine

def transpose(grid):
    grid2 = [[float(row[i]) for row in grid] for i in range(len(grid[0]))]
    return grid2

if __name__ == "__main__":

    # data1 = np.arange(0,1,0.05)
    # data2 = np.arange(1,2,0.05)
    # data3 = np.arange(2,3,0.05)
    # data4 = np.arange(3,4,0.05)
    # data5 = np.arange(4,5,0.05)
    # data6 = np.arange(5,6,0.05)
    # data7 = np.arange(6,7,0.05)

    data1 = np.arange(0,2,0.05)
    data2 = np.arange(1,3,0.05)
    data3 = np.arange(2,4,0.05)
    data4 = np.arange(3,5,0.05)
    data5 = np.arange(4,6,0.05)
    data6 = np.arange(5,7,0.05)
    data7 = np.arange(6,8,0.05)
    np.random.seed(0)
    s = list(zip(data1, data2, data3, data4, data5, data6, data7))
    np.random.shuffle(s)
    data1, data2, data3, data4, data5, data6, data7 = zip(*s)

    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    print("========== Create Learning Assignments ==========")
    sample_x1_names = ["1", "2", "3"]
    samples_x1 = transpose([data1[:30], data2[:30], data3[:30]])
    la1 = LearningAssignment(zoo, sample_x1_names)
    la1.create_and_add_model(samples_x1, list(data4[:30]), GaussianProcess)
    sample_x2_names = ["4", "5", "6"]
    samples_x2 = transpose([data4[:30], data5[:30], data6[:30]])
    la2 = LearningAssignment(zoo, sample_x2_names)
    la2.create_and_add_model(samples_x2, list(data7[:30]), GaussianProcess)

    print("========== Add Services ==========")
    fluxion.add_service("la1", "4-0", la1, [None, None, None], [None, None, None])
    fluxion.add_service("la2", "7", la2, ["la1", None, None], ["4-0", None, None])
    fluxion.add_service("la3", "7", la2, [None, None, None], [None, None, None])
    
    print("========== Predict ==========")
    errs = []
    data4_2 = []
    for i in range(30, 40):
        fluxion_input = {"la1": {"4-0": [{"1": data1[i], "2": data2[i], "3": data3[i]}]}}
        prediction = fluxion.predict("la1", "4-0", fluxion_input)
        print(prediction["la1"]["4-0"]["val"], data4[i])
        data4_2.append(prediction["la1"]["4-0"]["val"])
        errs.append(abs(prediction["la1"]["4-0"]["val"]- data4[i]))
    print(np.mean(errs))

    # multimodel
    errs = []
    for i in range(30, 40):
        fluxion_input = {"la1": {"4-0": [{"1": data1[i], "2": data2[i], "3": data3[i]}]},
                        "la2": {"7": [{"5": data5[i], "6": data6[i]}]}
                        }
        prediction = fluxion.predict("la2", "7", fluxion_input)
        print(prediction["la2"]["7"]["val"], data7[i])
        errs.append(abs(prediction["la2"]["7"]["val"]-data7[i]))
    print(np.mean(errs))

    # baseline(real data input)
    errs = []
    for i in range(30, 40):
        fluxion_input = {
                        "la3": {"7": [{"4": data4[i], "5": data5[i], "6": data6[i]}]}
                        }
        prediction = fluxion.predict("la3", "7", fluxion_input)
        print(prediction["la3"]["7"]["val"], data7[i])
        errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
    print(np.mean(errs))

    # la1 output as input
    errs = []
    for i in range(30, 40):
        fluxion_input = {
                        "la3": {"7": [{"4": data4_2[i-30], "5": data5[i], "6": data6[i]}]}
                        }
        prediction = fluxion.predict("la3", "7", fluxion_input)
        print(prediction["la3"]["7"]["val"], data7[i])
        errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
    print(np.mean(errs))