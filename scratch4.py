import numpy as np

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
    induced_err = 0.001
    data1 = np.arange(0,2,0.05)+np.random.normal(0,induced_err,40)
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
    samples_x1 = transpose([data1[:20], data2[:20], data3[:20]])
    la1 = LearningAssignment(zoo, sample_x1_names)
    la1.create_and_add_model(samples_x1, list(data4[:20]), GaussianProcess)
    sample_x2_names = ["4", "5", "6"]
    samples_x2 = transpose([data4[:20], data5[:20], data6[:20]])
    la2 = LearningAssignment(zoo, sample_x2_names)
    la2.create_and_add_model(samples_x2, list(data7[:20]), GaussianProcess)

    print("========== Add Services ==========")
    fluxion.add_service("la1", "4-0", la1, [None, None, None], [None, None, None])
    fluxion.add_service("la2", "7", la2, [None, None, None], [None, None, None])
    
    print("========== Predict valid error ==========")
    valid_err = []
    # data4_2 = []
    for i in range(20, 30):
        fluxion_input = {"la1": {"4-0": [{"1": data1[i], "2": data2[i], "3": data3[i]}]}}
        prediction = fluxion.predict("la1", "4-0", fluxion_input)
        valid_err.append(abs(prediction["la1"]["4-0"]["val"]- data4[i]))
    # for i in range(10):
    #     print(valid_err[i])
    MAE = np.mean(valid_err)

    print("========== Predict test error ==========")
    errs = []
    data4_2 = []
    for i in range(30,40):
        fluxion_input = {"la1": {"4-0": [{"1": data1[i], "2": data2[i], "3": data3[i]}]}}
        prediction = fluxion.predict("la1", "4-0", fluxion_input)
        # print(prediction["la1"]["4-0"]["val"], data4[i])
        data4_2.append(prediction["la1"]["4-0"]["val"])
        # errs.append(prediction["la1"]["4-0"]["val"]- data4[i])
    # print(errs)

    # multimodel
    errs = []
    for i in range(30, 40):
        val = []
        sample = np.random.rand(5)

        # sample points
        for k in range(5):
            input_point = data4_2[i-30] + sample[k]*MAE
            fluxion_input = {
                                "la2": {"7": [{"4": input_point, "5": data5[i], "6": data6[i]}]}
                            }
            prediction = fluxion.predict("la2", "7", fluxion_input)
            val.append(prediction["la2"]["7"]["val"])
        for k in range(5):
            input_point = data4_2[i-30] - sample[k]*MAE
            fluxion_input = {
                                "la2": {"7": [{"4": input_point, "5": data5[i], "6": data6[i]}]}
                            }
            prediction = fluxion.predict("la2", "7", fluxion_input)
            val.append(prediction["la2"]["7"]["val"])

        # original value
        input_point = data4_2[i-30]
        fluxion_input = {
                            "la2": {"7": [{"4": input_point, "5": data5[i], "6": data6[i]}]}
                        }
        prediction = fluxion.predict("la2", "7", fluxion_input)
        val.append(prediction["la2"]["7"]["val"])

        errs.append(abs(np.mean(val)-data7[i]))

    print(np.mean(errs))

    


