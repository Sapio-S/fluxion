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
    
    for i in range(4):
        print(i)
        induced_err = (0.1) ** (2*i)
        print("dataset err", induced_err)
        data1 = np.arange(0,2,0.05)+np.random.normal(0,induced_err,40)
        data2 = np.arange(1,3,0.05)+np.random.normal(0,induced_err,40)
        data3 = np.arange(2,4,0.05)+np.random.normal(0,induced_err,40)
        data4 = np.arange(3,5,0.05)+np.random.normal(0,induced_err,40)
        data5 = np.arange(4,6,0.05)+np.random.normal(0,induced_err,40)
        data6 = np.arange(5,7,0.05)+np.random.normal(0,induced_err,40)
        data7 = np.arange(6,8,0.05)+np.random.normal(0,induced_err,40)
        np.random.seed(0)
        s = list(zip(data1, data2, data3, data4, data5, data6, data7))
        np.random.shuffle(s)
        data1, data2, data3, data4, data5, data6, data7 = zip(*s)


        train_size = 30
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)

        print("========== Create Learning Assignments ==========")
        
        sample_x2_names = ["4", "5", "6"]
        samples_x2 = transpose([data4[:train_size], data5[:train_size], data6[:train_size]])
        la2 = LearningAssignment(zoo, sample_x2_names)
        la2.create_and_add_model(samples_x2, list(data7[:train_size]), GaussianProcess)

        print("========== Add Services ==========")
        fluxion.add_service("la3", "7", la2, [None, None, None], [None, None, None])
        
        print("========== Predict ==========")
        for j in range(4):
            induced_err = (0.1) ** (2*j)
            print("input err", induced_err)
            data_err4 = np.random.normal(0,induced_err,10)
            feed_data4 = data4[30:] + data_err4
            errs = []
            for i in range(30, 40):
                fluxion_input = {
                                "la3": {"7": [{"4": feed_data4[i-30], "5": data5[i], "6": data6[i]}]}
                                }
                prediction = fluxion.predict("la3", "7", fluxion_input)
                # print(prediction["la3"]["7"]["val"], data7[i])
                errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
            print(np.mean(errs))

    # induced_err = 1e-02
    # data_err4 = np.random.normal(0,induced_err,10)
    # # feed_data = [data4[30+i] * (1+data_err[i]) for i in range(10)]
    # feed_data4 = data4[30:] + data_err4
    # # print(feed_data)
    # errs = []
    # for i in range(30, 40):
    #     fluxion_input = {
    #                     "la3": {"7": [{"4": feed_data4[i-30], "5": data5[i], "6": data6[i]}]}
    #                     }
    #     prediction = fluxion.predict("la3", "7", fluxion_input)
    #     # print(prediction["la3"]["7"]["val"], data7[i])
    #     errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
    # print(np.mean(errs))

    # data_err5 = np.random.normal(0,induced_err,10)
    # # feed_data = [data4[30+i] * (1+data_err[i]) for i in range(10)]
    # feed_data5 = data5[30:] + data_err5
    # # print(feed_data)
    # errs = []
    # for i in range(30, 40):
    #     fluxion_input = {
    #                     "la3": {"7": [{"4": feed_data4[i-30], "5": feed_data5[i-30], "6": data6[i]}]}
    #                     }
    #     prediction = fluxion.predict("la3", "7", fluxion_input)
    #     # print(prediction["la3"]["7"]["val"], data7[i])
    #     errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
    # print(np.mean(errs))

    # data_err6 = np.random.normal(0,induced_err,10)
    # # feed_data = [data4[30+i] * (1+data_err[i]) for i in range(10)]
    # feed_data6 = data6[30:] + data_err6
    # # print(feed_data)
    # errs = []
    # for i in range(30, 40):
    #     fluxion_input = {
    #                     "la3": {"7": [{"4":feed_data4[i-30], "5":feed_data5[i-30], "6":feed_data6[i-30]}]}
    #                     }
    #     prediction = fluxion.predict("la3", "7", fluxion_input)
    #     # print(prediction["la3"]["7"]["val"], data7[i])
    #     errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
    # print(np.mean(errs))

    # errs = []
    # for i in range(30, 40):
    #     fluxion_input = {
    #                     "la3": {"7": [{"4":data4[i], "5":feed_data5[i-30], "6":data6[i]}]}
    #                     }
    #     prediction = fluxion.predict("la3", "7", fluxion_input)
    #     # print(prediction["la3"]["7"]["val"], data7[i])
    #     errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
    # print(np.mean(errs))


    # errs = []
    # for i in range(30, 40):
    #     fluxion_input = {
    #                     "la3": {"7": [{"4":data4[i], "5":data5[i], "6":feed_data6[i-30]}]}
    #                     }
    #     prediction = fluxion.predict("la3", "7", fluxion_input)
    #     # print(prediction["la3"]["7"]["val"], data7[i])
    #     errs.append(abs(prediction["la3"]["7"]["val"]-data7[i]))
    # print(np.mean(errs))

