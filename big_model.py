import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo

from consts import *
from data_new import get_input, get_input_norm, get_input_std, norm_scaler, std_scaler
import numpy as np


def singlemodel(samples_x, samples_y, x_names, perf_data, test_data, train_data):
    paras = []
    for name, value in x_names.items():
        for perf in value:
            paras.append(name+":"+perf)
    
    new_x = []
    for i in range(train_size):
        tmp = []
        for k, v in samples_x.items():
            tmp += v[i]
        new_x.append(tmp)
    
    train_errs = []
    test_errs = []
    for i in range(1):
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)
        e2e = LearningAssignment(zoo, paras)
        e2e.create_and_add_model(
            new_x, 
            perf_data["frontend:0.90"][test_size:test_size+train_size], 
            GaussianProcess
            ) 
        fluxion.add_service("e2e", "0.90", e2e, [None] * len(paras), [None] * len(paras))

        # get train error
        errs = []
        pred = []
        for i in range(train_size):
            minimap = {}
            for f in finals:
                for k, v in train_data[i][f]["0.90"][0].items():
                    minimap[f+":"+k] = v
            prediction = fluxion.predict("e2e", "0.90", {"e2e":{"0.90":[minimap]}})
            v1 = prediction["e2e"]["0.90"]["val"]
            v2 = perf_data["frontend:0.90"][i+test_size]
            v1 = norm_scaler(v1, scale["frontend:0.90:MIN"], scale["frontend:0.90:MAX"])
            v2 = norm_scaler(v2, scale["frontend:0.90:MIN"], scale["frontend:0.90:MAX"])
            errs.append(abs(v1-v2))
            pred.append(v1)
        train_errs.append(np.mean(errs))

        # get test error
        errs = []
        pred = []
        for i in range(test_size):
            minimap = {}
            for f in finals:
                for k, v in test_data[i][f]["0.90"][0].items():
                    minimap[f+":"+k] = v
            prediction = fluxion.predict("e2e", "0.90", {"e2e":{"0.90":[minimap]}})
            v1 = prediction["e2e"]["0.90"]["val"]
            v2 = perf_data["frontend:0.90"][i]
            v1 = norm_scaler(v1, scale["frontend:0.90:MIN"], scale["frontend:0.90:MAX"])
            v2 = norm_scaler(v2, scale["frontend:0.90:MIN"], scale["frontend:0.90:MAX"])
            errs.append(abs(v1-v2))
        test_errs.append(np.mean(errs))
    return np.mean(train_errs),np.mean(test_errs)


if __name__ == "__main__":
    train_list = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]
    for i in range(10):
        f = open('log/0824std/big_model'+str(i),"w")
        sys.stdout = f

        train_size = train_list[i]
        test_size = 162
        print("train size is", train_size)
        print("test size is", test_size)
        train_errs = []
        test_errs = []
        for i in range(10):
            samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data, scale = get_input_std(i)
            t1, t3 = singlemodel(samples_x, samples_y, x_names, perf_data, test_data, train_data)
            train_errs.append(t1)
            test_errs.append(t3)

        print("avg training error",np.mean(train_errs))
        print("avg test error",np.mean(test_errs))
        print("")
