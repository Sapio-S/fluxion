import os, sys
import random
sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne

from consts import *
from data_new import get_input, get_input_norm, get_input_std, norm_scaler, std_scaler
import numpy as np

valid_size = 50
test_size = 133
bin_num = 10
sample_size = 100
def combine_list(list1, list2):
    for i in range(train_size):
        list1[i].append(list2[i+test_size+valid_size])

def combine_data(extra_names, perf_data, x_slice):
    for name in extra_names:
        for perf in eval_metric:
            combine_list(x_slice, perf_data[name+":"+perf])
    return x_slice

def sample_error(num, points):
    sub_seq = np.random.randint(0,valid_size)
    start = bin_num
    cnt = 0
    for sub in range(bin_num):
        if cnt < sub_seq:
            cnt += num[sub]
        else:
            start = sub
            break
    return np.random.uniform(points[start-1], points[start], 1)
    # return 0

def multimodel(sample_x, sample_y, x_names, perf_data, test_data, train_data, train_size, test_size):
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    # create learning assignments
    la_map = {} # {"service":{"perf":la, ...}, ...}
    for f in finals:
        la_map[f] = {}
        names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
        f_input = combine_data(extra_names[f], perf_data, sample_x[f])
        for p in eval_metric:
            la = LearningAssignment(zoo, x_names[f]+names)
            la.create_and_add_model(f_input[:train_size], sample_y[p][f][:train_size], GaussianProcess)
            la_map[f][p] = la

    # add all models into fluxion (no edges)
    for f in finals:
        names = [n for n in extra_names[f] for i in range(len(eval_metric))]
        for p in eval_metric:
            fluxion.add_service(f, p, la_map[f][p], [None]*len(x_names[f]+names), [None]*len(x_names[f]+eval_metric*len(extra_names[f])))

    # compute model error distributions
    errors = {}
    hist_num = {}
    hist_val = {}
    for f in finals:
        errors[f] = []
        hist_num[f] = []
        hist_val[f] = []
        for i in range(valid_size):
            prediction = fluxion.predict(f, "0.90", valid_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i+test_size]
            errors[f].append(v1-v2)
        hist_num[f], hist_val[f] = np.histogram(errors[f],bins=bin_num)

    # compute test error
    prediction = {}
    test_err = {}
    for f in finals:
        errs = []
        prediction[f] = []
        for i in range(test_size):
            pred = []
            # generate a set of input points
            for k in range(sample_size):
                data_piece = test_data[i].copy()
                for down in extra_names[f]:
                    data_piece[f]["0.90"][0][down+":0.90"] = prediction[down][i]-sample_error(hist_num[down], hist_val[down])
                pred.append(fluxion.predict(f, "0.90", data_piece)[f]["0.90"]["val"])
            
            v1 = np.mean(pred)
            v2 = perf_data[f+":0.90"][i]
            prediction[f].append(v1)
            v1 = std_scaler(v1, scale[f+":0.90:AVG"], scale[f+":0.90:STD"])
            v2 = std_scaler(v2, scale[f+":0.90:AVG"], scale[f+":0.90:STD"])
            errs.append(abs(v1-v2))
            
        test_err[f] = np.mean(errs) # calculate MAE for every service
    
    return test_err

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    train_list = [10, 25, 50, 100, 150, 200, 300, 400, 550, 700, 850]
    for train_sub in range(9):
        f = open("log/0929scale/valid_"+str(train_list[train_sub]),"w")
        sys.stdout = f
        train_errs = []
        test_errs = {}
        for f in finals:
            test_errs[f] = []

        train_size = train_list[train_sub]
        
        print("train size is", train_size)
        # print("test size is", test_size)
        for i in range(10):
            samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data, scale = get_input_std(i)
            test_err = multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data, train_size, test_size)
            for f in finals:
                test_errs[f].append(test_err[f])

        print(test_errs["frontend"])
        print("avg test err for 10 times", np.mean(test_errs["frontend"]))
        for f in finals:
            print(np.mean(test_errs[f]))