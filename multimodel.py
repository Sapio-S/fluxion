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
    for i in range(train_size): # TODO: change to 300
        list1[i].append(list2[i])

def combine_data(extra_names, perf_data, x_slice):
    for name in extra_names:
        for perf in eval_metric:
            combine_list(x_slice, perf_data[name+":"+perf])
    return x_slice

def multimodel(sample_x, sample_y, x_names, perf_data, test_data, train_data):
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    
    # print("========== Create Learning Assignments ==========")

    la_map = {} # {"service":{"perf":la, ...}, ...}
    for f in finals2:
        la_map[f] = {}
        names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
        f_input = combine_data(extra_names[f], perf_data, sample_x[f])
        for p in eval_metric:
            la = LearningAssignment(zoo, x_names[f]+names)
            la.create_and_add_model(f_input, sample_y[p][f], GaussianProcess)
            la_map[f][p] = la
            # # prediction
            # errs = []
            # for i, s in enumerate(f_input):
            #     pre = la.predict(s)["val"]
            #     real = sample_y[p][f][i]
            #     errs.append(abs(pre-real))
            # # print(f, ": la error of percentile", p, np.mean(errs))

    # print("========== Add Services ==========") 
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
            # print(v1, v2)
            errs.append(abs(v1-v2))
        train_err = np.mean(errs)
        # print(f, "avg training error for multimodel", np.mean(errs))

    # get test error
    test_err = 0
    for f in finals2:
        errs = []
        # errs_percent = []
        for i in range(test_size):
            prediction = fluxion.predict(f, "0.90", test_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i]
            errs.append(abs(v1-v2))
            # errs_percent.append(abs(v1-v2)/(v2+0.000000001))
            # if f == collect:
            #     print(v1, v2)
        if f == collect:
            test_err = np.mean(errs)
            # print(f, "avg test error for multimodel", np.mean(errs))

    # visualize graph
    # fluxion.visualize_graph_engine_diagrams("frontend", "0.90", output_filename="frontend-multi"+str(len(eval_metric)))
    return train_err, test_err

def generate_all_shuffle():
    map_list = []
    original = np.arange(test_size+train_size)
    for i in range(10):
        map_list.append(original.copy())
        np.random.shuffle(original)
    return map_list

if __name__ == "__main__":
    train_list = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    f = open("log/graph0812/multi_"+str(len(eval_metric))+'_log',"w")
    sys.stdout = f

    for train_sub in range(14):
        train_errs = []
        test_errs = []
        train_size = train_list[train_sub]
        test_size = 125
        print("train size is", train_size)
        print("test size is", test_size)
        for i in range(10):
            sub_map = np.arange(725)
            np.random.seed(i)
            np.random.shuffle(sub_map)
            samples_x, samples_y, x_names, perf_data, test_data, train_data = get_input(train_size, test_size, sub_map)
            train_err, test_err = multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data)
            train_errs.append(train_err)
            test_errs.append(test_err)
        print("______________________________________________")
        print("avg train err for 10 times", np.mean(train_errs))
        print("avg test err for 10 times", np.mean(test_errs))
        print(train_errs)
        print(test_errs)
        print("")