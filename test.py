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

def combine_data(extra_names, perf, perf_data, x_slice):
    # new_name_list = []
    for name in extra_names:
        # new_name_list.append(name+":"+perf)
        combine_list(x_slice, perf_data[name+":"+perf])
    return x_slice

def multimodel():
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    
    print("========== Create Learning Assignments ==========")
    sample_x, sample_y, x_names, perf_data, test_data, train_data = get_input() # 现在的sample_y只有p50

    extra_names = {
        "adservice":[],
        "cartservice":["get", "set"], 
        "checkoutservice":["emailservice", "paymentservice", "shippingservice", "currencyservice", "productcatalogservice", "cartservice"], 
        "currencyservice":[], 
        "emailservice":[], 
        "frontend":["adservice", "checkoutservice", "shippingservice", "currencyservice", "productcatalogservice", "recommendationservice", "cartservice"], 
        "paymentservice":[], 
        "productcatalogservice":[], 
        "recommendationservice":["productcatalogservice"], 
        "shippingservice":[], 
        "get":[], 
        "set":[]
    }

    la_map = {}

    for f in finals:
        f_input = combine_data(extra_names[f], "0.50", perf_data, sample_x[f])
        la = LearningAssignment(zoo, x_names[f]+extra_names[f])
        la.create_and_add_model(f_input, sample_y[f], GaussianProcess)
        la_map[f] = la
        # prediction
        errs = []
        for i, s in enumerate(f_input):
            pre = la.predict(s)["val"]
            real = sample_y[f][i]
            errs.append(abs(pre-real))
        print("la error for ", f, np.mean(errs))
    
    print("========== Add Services ==========")
    add_list = []
    for f in finals:
        try:
            fluxion.add_service(f, "0.50", la_map[f], [None]*len(x_names[f])+extra_names[f], [None]*len(x_names[f])+["0.50"]*len(extra_names[f]))
        except:
            add_list.append(f)
            continue
    while(len(add_list) > 0):
        new_add_list = []
        for f in add_list:
            try:
                fluxion.add_service(f, "0.50", la_map[f], [None]*len(x_names[f])+extra_names[f], [None]*len(x_names[f])+["0.50"]*len(extra_names[f]))
            except:
                new_add_list.append(f)
                continue
        add_list = new_add_list

    # get whole train error
    errs = []
    f = "frontend"
    for i in range(train_size):
        prediction = fluxion.predict(f, "0.50", train_data[i])
        v1 = prediction[f]["0.50"]["val"]
        v2 = perf_data[f+":0.50"][i+test_size]
        errs.append(abs(v1-v2))
    print("avg training error for multimodel", np.mean(errs))

    # get test error
    errs = []
    for i in range(test_size):
        prediction = fluxion.predict(f, "0.50", test_data[i])
        v1 = prediction[f]["0.50"]["val"]
        v2 = perf_data[f+":0.50"][i]
        errs.append(abs(v1-v2))
    print("avg test error for multimodel", np.mean(errs))

    # visualize graph
    # fluxion.visualize_graph_engine_diagrams("frontend", "0.50", output_filename="frontend-multi")
    
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
    e2e.create_and_add_model(new_x, perf_data["frontend:0.50"][test_size:test_size+train_size], GaussianProcess) 
    fluxion.add_service("e2e", "0.50", e2e, [None] * len(paras), [None] * len(paras))

    # fluxion.visualize_graph_engine_diagrams("e2e", "0.50", output_filename="e2e-single")

    # get train error
    errs = []
    for i in range(train_size):
        minimap = {}
        for f in finals:
            for k, v in train_data[i][f]["0.50"][0].items():
                minimap[f+":"+k] = v
        prediction = fluxion.predict("e2e", "0.50", {"e2e":{"0.50":[minimap]}})
        v1 = prediction["e2e"]["0.50"]["val"]
        v2 = perf_data["frontend:0.50"][i+test_size]
        errs.append(abs(v1-v2))
    print("avg training error for single model",np.mean(errs))

    # get test error
    errs = []
    for i in range(test_size):
        minimap = {}
        for f in finals:
            for k, v in test_data[i][f]["0.50"][0].items():
                minimap[f+":"+k] = v
        prediction = fluxion.predict("e2e", "0.50", {"e2e":{"0.50":[minimap]}})
        v1 = prediction["e2e"]["0.50"]["val"]
        v2 = perf_data["frontend:0.50"][i]
        # print(v1, v2)
        errs.append(abs(v1-v2))
    print("avg test error for single model",np.mean(errs))

if __name__ == "__main__":
    print("train size is", train_size)
    print("test size is", test_size)

    multimodel()
    singlemodel()