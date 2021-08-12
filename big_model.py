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
    train_errs_percent = []
    test_errs = []
    test_errs_percent = []
    for i in range(5):
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)
        e2e = LearningAssignment(zoo, paras)
        e2e.create_and_add_model(
            new_x, 
            perf_data["frontend:0.90"][test_size:test_size+train_size], 
            MultiLayerPerceptron, 
            model_class_args=[
                (50,50), 
                "tanh", # tanh or logistic
                "sgd", 
                1e-4, # alpha
                1e-3, # learning_rate_init
                "adaptive", 
                1000
                ]
            ) 
        fluxion.add_service("e2e", "0.90", e2e, [None] * len(paras), [None] * len(paras))

        # get train error
        errs = []
        errs_percent = []
        pred = []
        for i in range(train_size):
            minimap = {}
            for f in finals:
                for k, v in train_data[i][f]["0.90"][0].items():
                    minimap[f+":"+k] = v
            prediction = fluxion.predict("e2e", "0.90", {"e2e":{"0.90":[minimap]}})
            v1 = prediction["e2e"]["0.90"]["val"]
            v2 = perf_data["frontend:0.90"][i+test_size]
            errs.append(abs(v1-v2))
            errs_percent.append(abs(v1-v2)/v2)
            pred.append(v1)
        train_errs.append(np.mean(errs))
        train_errs_percent.append(np.mean(errs_percent))
        print("train std:", np.std(pred))

        # get test error
        errs = []
        errs_percent = []
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
            errs_percent.append(abs(v1-v2)/v2)
            pred.append(v1)
        test_errs.append(np.mean(errs))
        print("test std:", np.std(pred))
        test_errs_percent.append(np.mean(errs_percent))
    return np.mean(train_errs), np.mean(train_errs_percent), np.mean(test_errs), np.mean(test_errs_percent)


if __name__ == "__main__":
    f = open('log/big_model_log_(50,50)',"w")
    sys.stdout = f
    train_list = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    for i in range(11):
        train_size = train_list[i]
        test_size = 120
        print("train size is", train_size)
        print("test size is", test_size)
        sub_map = np.arange(600)
        train_errs = []
        train_errs_percent = []
        test_errs = []
        test_errs_percent = []
        for i in range(5):
            np.random.seed(i)
            np.random.shuffle(sub_map)
            samples_x, samples_y, x_names, perf_data, test_data, train_data = get_input(train_size, test_size, sub_map)
            t1, t2, t3, t4 = singlemodel(samples_x, samples_y, x_names, perf_data, test_data, train_data)
            train_errs.append(t1)
            train_errs_percent.append(t2)
            test_errs.append(t3)
            test_errs_percent.append(t4)
        print("avg training error for big model",np.mean(train_errs))
        print("avg training error percentage for big model",np.mean(train_errs_percent))
        print("avg test error for single model",np.mean(test_errs))
        print("avg test error percentage for big model",np.mean(test_errs_percent))
