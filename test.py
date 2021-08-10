import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
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

def multimodel():
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    
    print("========== Create Learning Assignments ==========")
    sample_x, sample_y, x_names, perf_data, test_data, train_data = get_input() # 现在的sample_y只有p50

    la_map = {} # {"perf":{"service":la, ...}, ...}
    for p in eval_metric:
        la_map[p] = {}
        for f in finals2:
            names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
            f_input = combine_data(extra_names[f], perf_data, sample_x[f])
            la = LearningAssignment(zoo, x_names[f]+names)
            la.create_and_add_model(f_input, sample_y[p][f], GaussianProcess)
            la_map[p][f] = la
            # # prediction
            # errs = []
            # for i, s in enumerate(f_input):
            #     pre = la.predict(s)["val"]
            #     real = sample_y[p][f][i]
            #     errs.append(abs(pre-real))
            # print("la error for", f, "of percentile", p, np.mean(errs))
    
    print("========== Add Services ==========") 
    add_list = []
    for f in finals2:
        names = [n for n in extra_names[f] for i in range(len(eval_metric))]
        try:
            for p in eval_metric:
                fluxion.add_service(f, p, la_map[p][f], [None]*len(x_names[f])+names, [None]*len(x_names[f])+eval_metric*len(extra_names[f]))
        except:
            add_list.append(f)
            continue
    while(len(add_list) > 0):
        # print(add_list)
        new_add_list = []
        for f in add_list:
            names = [n for n in extra_names[f] for i in range(len(eval_metric))]
            try:
                for p in eval_metric:
                    fluxion.add_service(f, p, la_map[p][f], [None]*len(x_names[f])+names, [None]*len(x_names[f])+eval_metric*len(extra_names[f]))
            except:
                new_add_list.append(f)
                continue
        add_list = new_add_list

    # get whole train error
    errs = []
    f = "frontend"
    for i in range(train_size):
        prediction = fluxion.predict(f, "0.90", train_data[i])
        v1 = prediction[f]["0.90"]["val"]
        v2 = perf_data[f+":0.90"][i+test_size]
        errs.append(abs(v1-v2))
    train_err = np.mean(errs)
    # print("avg training error for multimodel", np.mean(errs))

    # get test error
    test_err = 0
    for f in finals2:
        errs = []
        errs_percent = []
        for i in range(test_size):
            prediction = fluxion.predict(f, "0.90", test_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i]
            errs.append(abs(v1-v2))
            errs_percent.append(abs(v1-v2)/(v2+0.000000001))
        print(f, "avg test error for multimodel", np.mean(errs))
        print(f, "avg test error in percentage is", np.mean(errs_percent))
        if f == "frontend":
            test_err = np.mean(errs)

    # visualize graph
    # fluxion.visualize_graph_engine_diagrams("frontend", "0.90", output_filename="frontend-multi"+str(len(eval_metric)))
    return train_err, test_err
    
def singlemodel():
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    samples_x, samples_y, x_names, perf_data, test_data, train_data = get_input() # 现在的sample_y只有p50

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
    
    e2e = LearningAssignment(zoo, paras)
    e2e.create_and_add_model(new_x, perf_data["frontend:0.90"][test_size:test_size+train_size], MultiLayerPerceptron, model_class_args=[]) 
    fluxion.add_service("e2e", "0.90", e2e, [None] * len(paras), [None] * len(paras))

    fluxion.visualize_graph_engine_diagrams("e2e", "0.90", output_filename="e2e-single")

    # get train error
    errs = []
    for i in range(train_size):
        minimap = {}
        for f in finals:
            for k, v in train_data[i][f]["0.90"][0].items():
                minimap[f+":"+k] = v
        prediction = fluxion.predict("e2e", "0.90", {"e2e":{"0.90":[minimap]}})
        v1 = prediction["e2e"]["0.90"]["val"]
        v2 = perf_data["frontend:0.90"][i+test_size]
        errs.append(abs(v1-v2))
    print("avg training error for single model",np.mean(errs))

    # get test error
    errs = []
    for i in range(test_size):
        minimap = {}
        for f in finals:
            for k, v in test_data[i][f]["0.90"][0].items():
                minimap[f+":"+k] = v
        prediction = fluxion.predict("e2e", "0.90", {"e2e":{"0.90":[minimap]}})
        v1 = prediction["e2e"]["0.90"]["val"]
        v2 = perf_data["frontend:0.90"][i]
        # print(v1, v2)
        errs.append(abs(v1-v2))
    print("avg test error for single model",np.mean(errs))

def generate_all_shuffle():
    map_list = []
    original = np.arange(test_size+train_size)
    for i in range(10):
        map_list.append(original.copy())
        np.random.shuffle(original)
    return map_list

if __name__ == "__main__":
    print("train size is", train_size)
    print("test size is", test_size)
    train_errs = []
    test_errs = []
    for i in range(10):
        change_data_order(sub_map, i)
        train_err, test_err = multimodel()
        train_errs.append(train_err)
        test_errs.append(test_err)
    print("avg train err for 10 times", np.mean(train_errs))
    print("avg test err for 10 times", np.mean(test_errs))
    print(train_errs)
    print(test_errs)
    # singlemodel()