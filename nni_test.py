
import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
from GraphEngine.ModelZoo.model_zoo import Model_Zoo

from consts import *
from data import get_input
import numpy as np
import nni

def singlemodel(samples_x, samples_y, x_names, perf_data, train_data, test_data, valid_data, params):
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
    valid_errs = []
    
    for i in range(1):
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)
        e2e = LearningAssignment(zoo, paras)
        e2e.create_and_add_model(
            new_x, 
            perf_data["frontend:0.90"][test_size+valid_size:test_size+valid_size+train_size], 
            MultiLayerPerceptron, 
            model_class_args=[
                (params['searchSpace']["hidden_size1"],params['searchSpace']["hidden_size2"],params['searchSpace']["hidden_size3"]), 
                "tanh", # tanh or logistic
                "sgd", 
                1e-4, # alpha
                params['searchSpace']["lr"], # learning_rate_init
                "adaptive", 
                # max_iter
                ]
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
            v2 = perf_data["frontend:0.90"][i+test_size+valid_size]
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
            errs.append(abs(v1-v2))
        test_errs.append(np.mean(errs))

        # get valid error
        errs = []
        pred = []
        for i in range(valid_size):
            minimap = {}
            for f in finals:
                for k, v in valid_data[i][f]["0.90"][0].items():
                    minimap[f+":"+k] = v
            prediction = fluxion.predict("e2e", "0.90", {"e2e":{"0.90":[minimap]}})
            v1 = prediction["e2e"]["0.90"]["val"]
            v2 = perf_data["frontend:0.90"][i]
            errs.append(abs(v1-v2))
        valid_errs.append(np.mean(errs))
    return np.mean(train_errs),np.mean(test_errs), np.mean(valid_errs)


if __name__ == "__main__":
    train_list = [10, 25, 50, 100, 150, 200, 300, 400, 500]
    train_size = train_list[1]
    test_size = 84
    valid_size = 84
    print("train size is", train_size)
    print("valid size is", valid_size)
    print("test size is", test_size)
    train_errs = []
    valid_errs = []
    test_errs = []

    params = nni.get_next_parameter()
    for i in range(10): # TODO
        samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data = get_input(i)
        t1, t2, t3 = singlemodel(samples_x, samples_y, x_names, perf_data, train_data, test_data, valid_data, params)
        train_errs.append(t1)
        test_errs.append(t2)
        valid_errs.append(t3)
    print("avg training error",np.mean(train_errs))
    print("avg test error",np.mean(test_errs))
    print("avg valid error",np.mean(valid_errs))
    nni.report_final_result(np.mean(valid_errs))
    print("")