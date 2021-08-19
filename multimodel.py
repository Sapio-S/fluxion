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
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    la_map = {} # {"service":{"perf":la, ...}, ...}
    for f in finals2:
        la_map[f] = {}
        names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
        f_input = combine_data(extra_names[f], perf_data, sample_x[f])
        for p in eval_metric:
            la = LearningAssignment(zoo, x_names[f]+names)
            la.create_and_add_model(f_input[:][:train_size], sample_y[p][f][:train_size], GaussianProcess)
            la_map[f][p] = la

    add_list = []
    for f in finals2:
        names = [n for n in extra_names[f] for i in range(len(eval_metric))]
        try:
            for p in eval_metric:
                fluxion.add_service(f, p, la_map[f][p], [None]*len(x_names[f])+names, [None]*len(x_names[f])+eval_metric*len(extra_names[f]))
        except:
            add_list.append(f)
            continue
    while(len(add_list) > 0):
        new_add_list = []
        for f in add_list:
            names = [n for n in extra_names[f] for i in range(len(eval_metric))]
            try:
                for p in eval_metric:
                    fluxion.add_service(f, p, la_map[f][p], [None]*len(x_names[f])+names, [None]*len(x_names[f])+eval_metric*len(extra_names[f]))
            except:
                new_add_list.append(f)
                continue
        add_list = new_add_list

    # get whole train error
    for f in finals2:
        errs = []
        for i in range(train_size):
            prediction = fluxion.predict(f, "0.90", train_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i+test_size]
            errs.append(abs(v1-v2))
        train_err = np.mean(errs)

    # get test error
    test_err = {}
    for f in finals2:
        errs = []
        for i in range(test_size):
            prediction = fluxion.predict(f, "0.90", test_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i]
            errs.append(abs(v1-v2))
        test_err[f] = np.mean(errs) # calculate MAE for every service

    return train_err, test_err

if __name__ == "__main__":
    train_list = [10, 25, 50, 100, 150, 200, 250, 350, 450, 550]
    f = open("log/graph0814/multi_"+str(len(eval_metric))+'_log4(7-9)',"w")
    sys.stdout = f

    for train_sub in range(7, 10):
        train_errs = []
        test_errs = {}
        for f in finals2:
            test_errs[f] = []

        train_size = train_list[train_sub]
        test_size = 118
        print("train size is", train_size)
        print("test size is", test_size)
        
        for i in range(10):
            samples_x, samples_y, x_names, perf_data, test_data, train_data = get_input(i)
            train_err, test_err = multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data, train_size, test_size)
            train_errs.append(train_err)
            for f in finals2:
                test_errs[f].append(test_err[f])

        print("avg train err for 10 times", np.mean(train_errs))
        print("avg test err for 10 times", np.mean(test_errs["frontend"]))
        for f in finals2:
            print(f, np.mean(test_errs[f]))
        print("")