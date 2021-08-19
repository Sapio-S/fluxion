
import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne

from consts import *
from data import get_input
import numpy as np

def combine_list(list1, list2):
    for i in range(train_size):
        list1[i].append(list2[i])

def combine_data(extra_names, perf_data, x_slice):
    for name in extra_names:
        for perf in eval_metric:
            combine_list(x_slice, perf_data[name+":"+perf])
    return x_slice

def multimodel(sample_x, sample_y, x_names, perf_data, test_data, train_data, train_size, test_size):
    train_errs = {}
    test_errs = {}

    for f in finals2:
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)
        names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
        f_input = combine_data(extra_names[f], perf_data, sample_x[f])
        for p in eval_metric:
            la = LearningAssignment(zoo, x_names[f]+names)
            la.create_and_add_model(f_input[:][:train_size], sample_y[p][f][:train_size], GaussianProcess)
        fluxion.add_service(f, p, la, [None]*(len(x_names[f])+len(names)), [None]*(len(x_names[f])+len(extra_names[f])))

    # get whole train error
        errs = []
        for i in range(train_size):
            minimap = {}
            for n in x_names[f]:
                minimap[n] = train_data[i][f]["0.90"][0][n]
            for n in names:
                minimap[n] = perf_data[n][i]
            prediction = fluxion.predict(f, "0.90", {f:{"0.90":[minimap]}})
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i+test_size]
            errs.append(abs(v1-v2))
        train_errs[f] = np.mean(errs)

    # get test error
        errs = []
        for i in range(test_size):
            minimap = {}
            for n in x_names[f]:
                minimap[n] = train_data[i][f]["0.90"][0][n]
            for n in names:
                minimap[n] = perf_data[n][i]
            prediction = fluxion.predict(f, "0.90", {f:{"0.90":[minimap]}})
            v1 = prediction[f]["0.90"]["val"]
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i]
            errs.append(abs(v1-v2))
        test_errs[f] = np.mean(errs) # calculate MAE for every service

    return train_errs, test_errs

if __name__ == "__main__":
    train_list = [10, 25, 50, 100, 150, 200, 250, 350, 450, 550]
    
    for train_sub in range(10):
        f = open("log/log0819/train_size_"+str(train_sub)+'',"w")
        sys.stdout = f
        
        train_errs = {}
        test_errs = {}
        for f in finals2:
            test_errs[f] = 0.0
            train_errs[f] = 0.0

        train_size = train_list[train_sub]
        test_size = 118
        print("train size is", train_size)
        print("test size is", test_size)
        
        for i in range(10):
            samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data = get_input(i)
            train_err, test_err = multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data, train_size, test_size)
            for f in finals2:
                train_errs[f] += train_err[f]
                test_errs[f] += test_err[f]

        print(train_errs)
        print(test_errs)
        print("")