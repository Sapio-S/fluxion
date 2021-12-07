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
    data = {}
    np.random.seed(0)
    data[10] = np.arange(6,8,0.05)
    train_size = 30
    for number in range(2,10):
        print("number of input", number)
        for i in range(number):
            data[i] = np.arange(i,i+2,0.05)

        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)

        # print("========== Create Learning Assignments ==========")
        
        sample_x2_names = [str(i) for i in range(number)]
        samples_x2 = transpose([data[i][:train_size] for i in range(number)])
        la2 = LearningAssignment(zoo, sample_x2_names)
        la2.create_and_add_model(samples_x2, list(data[10][:train_size]), GaussianProcess)

        # print("========== Add Services ==========")
        fluxion.add_service("la3", "7", la2, [None]*number, [None]*number)
        
        # print("========== Predict ==========")
        for j in range(1):
            feed_data = {}
            induced_err = 0.001
            # print("input err", induced_err)
            for i in range(number):
                data_err = np.random.normal(0,induced_err,10)
                feed_data[i] = data[i][30:] + data_err


            # data_err = np.random.normal(0,induced_err,10)
            # feed_data[0] = feed_data[0] + data_err

            errs = []
            for i in range(10):
                fluxion_input = {
                                "la3": {"7": [{ str(k): feed_data[k][i] for k in range(number)}]
                                    }
                                }
                prediction = fluxion.predict("la3", "7", fluxion_input)
                # print(prediction["la3"]["7"]["val"], data7[i])
                errs.append(abs(prediction["la3"]["7"]["val"]-data[10][i+30]))
            print(np.mean(errs))
