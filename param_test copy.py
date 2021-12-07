
import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne

from consts import *
from data_new import get_input, get_input_norm, get_input_std, norm_scaler, std_scaler
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
    for f in finals:
        print(f)
        standard = 0.0
        # normal model
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)
        names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
        f_input = combine_data(extra_names[f], perf_data, sample_x[f])
        p = "0.90"
        la = LearningAssignment(zoo, x_names[f]+names)
        la.create_and_add_model(f_input[:train_size][:], sample_y[p][f][:train_size], GaussianProcess, max_inputs_selected=len(x_names[f]+names)-1)
        fluxion.add_service(f, p, la, [None]*len(x_names[f]+names), [None]*len(x_names[f]+names))


if __name__ == "__main__":

    train_size = 100
    test_size = 162
    print("train size is", train_size)
    print("test size is", test_size)
    
    for i in range(10):
        samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data, scale = get_input_norm(i)
        multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data, train_size, test_size)
            