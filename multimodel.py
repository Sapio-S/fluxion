import os, sys
import random
sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment_normal import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne

from consts import *
from data_new import get_input, get_input_norm, get_input_std, norm_scaler, std_scaler
import numpy as np

valid_size= 50
test_size = 133

def combine_list(list1, list2):
    for i in range(train_size+50):
        list1[i].append(list2[i+test_size+valid_size])

def combine_data(extra_names, perf_data, x_slice):
    for name in extra_names:
        for perf in eval_metric:
            combine_list(x_slice, perf_data[name+":"+perf])
    return x_slice

def multimodel(sample_x, sample_y, x_names, perf_data, test_data, train_data, train_size, test_size):
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    la_map = {} # {"service":{"perf":la, ...}, ...}
    for f in finals:
        la_map[f] = {}
        names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
        f_input = combine_data(extra_names[f], perf_data, sample_x[f])
        for p in eval_metric:
            la = LearningAssignment(zoo, x_names[f]+names)
            la.create_and_add_model(f_input[:train_size], sample_y[p][f][:train_size], GaussianProcess)
            la.set_err_dist(f_input[train_size:train_size+50], sample_y[p][f][train_size:train_size+50])
            la_map[f][p] = la

    for f in finals:
        names = [n for n in extra_names[f] for i in range(len(eval_metric))]
        for p in eval_metric:
            fluxion.add_service(f, p, la_map[f][p], [None]*len(x_names[f])+names, [None]*len(x_names[f])+eval_metric*len(extra_names[f]))

    # fluxion.visualize_graph_engine_diagrams("frontend", "0.90", output_filename="frontend_mine", is_draw_edges=True)
    # get whole train error
    for f in finals:
        errs = []
        for i in range(train_size):
            prediction = fluxion.predict(f, "0.90", train_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i+test_size+valid_size]
            v1 = std_scaler(v1, scale[f+":0.90:AVG"], scale[f+":0.90:STD"])
            v2 = std_scaler(v2, scale[f+":0.90:AVG"], scale[f+":0.90:STD"])
            errs.append(abs(v1-v2))
        train_err = np.mean(errs)

    # get test error
    test_err = {}
    for f in finals:
        errs = []
        for i in range(test_size):
            prediction = fluxion.predict(f, "0.90", test_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i]
            v1 = std_scaler(v1, scale[f+":0.90:AVG"], scale[f+":0.90:STD"])
            v2 = std_scaler(v2, scale[f+":0.90:AVG"], scale[f+":0.90:STD"])
            errs.append(abs(v1-v2))
        test_err[f] = np.mean(errs) # cculate MAE for every service
        
    return train_err, test_err

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    train_list = [10, 25, 50, 100, 150, 200, 300, 400, 550, 700, 850]
    for train_sub in range(9):
        f = open("log/1012scale/error_"+str(train_list[train_sub]),"w")
        sys.stdout = f
        train_errs = []
        test_errs = {}
        for f in finals:
            test_errs[f] = []

        train_size = train_list[train_sub]
        
        print("train size is", train_size)
        print("test size is", test_size)
        for i in range(10):
            # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)
            # training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            # training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            # testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
            # testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
            samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data, scale = get_input_std(i)
            train_err, test_err = multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data, train_size, test_size)
            train_errs.append(train_err)
            for f in finals:
                test_errs[f].append(test_err[f])

        print(test_errs["frontend"])
        print("avg train err for 10 times", np.mean(train_errs))
        print("avg test err for 10 times", np.mean(test_errs["frontend"]))
        for f in finals:
            print(np.mean(test_errs[f]))
        print("")