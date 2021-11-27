
import os, sys
import json, statistics
sys.path.insert(1, "../")
sys.path.append("/home/yuqingxie/autosys/code")
sys.path.append("/home/yuqingxie/autosys/code/fluxion")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
from GraphEngine.ModelZoo.model_zoo import Model_Zoo

from consts import *
from data import get_input
import numpy as np
import nni
import lib_data
import random
from param import *
target_deployment_name = "boutique_p95_p95"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10

num_training_data = 50
num_training_data = 133
small_models_preds = []
small_models_abs_errs = []
small_models_raw_errs = []
fluxion_abs_errs = []
big_gp_abs_errs = []
experiment_ids_completed = []

def singlemodel(params, i):
        
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    all_lrn_asgmts = {}
    selected_training_idxs = None
    selected_training_idxs = None
    
    small_models_preds.append({})
    small_models_abs_errs.append({})
    small_models_raw_errs.append({})
    fluxion_abs_errs.append([])
    big_gp_abs_errs.append([])
    # ========== Compute Big models' errors ==========
    # STEP 1: Prepare target services' input names
    expanded_sample_x_names = expand_sample_x_name(target_service_name)
    expanded_sample_x_names = list(set(expanded_sample_x_names))
    print("Big-* models have", len(expanded_sample_x_names), "inputs")
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], expanded_sample_x_names, target_service_name)
    
    # STEP 2: Determine training and training indexes
    print(dataset_filename, "has", len(samples_x), "data points")
    selected_training_idxs = random.sample(range(0, len(samples_x)), k=num_training_data)
    selected_training_idxs = set(range(0, len(samples_x))) - set(selected_training_idxs)
    selected_training_idxs = random.sample(selected_training_idxs, k=num_training_data)
    
    # STEP 3: Split dataset into training and training
    training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
    training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
    training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
    training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
    
    # STEP 4: Compute Big-GP's training MAE
    all_lrn_asgmts['big_gp_model'] = LearningAssignment(zoo, expanded_sample_x_names)
    created_model_name = all_lrn_asgmts['big_gp_model'].create_and_add_model(
        training_samples_x, 
        training_samples_y_aggregation,  
        MultiLayerPerceptron, 
        model_class_args=[
            (params['searchSpace']["hidden_size1"],params['searchSpace']["hidden_size2"],params['searchSpace']["hidden_size3"],params['searchSpace']["hidden_size4"],params['searchSpace']["hidden_size5"]), 
            "tanh", # tanh or logistic
            "sgd", 
            1e-4, # alpha
            params['searchSpace']["lr"], # learning_rate_init
            "adaptive", 
            # max_iter
            ]
        ) 
    #print(zoo.dump_model_info(created_model_name))
    for training_sample_x, training_sample_y_aggregation in zip(training_samples_x, training_samples_y_aggregation):
        pred = all_lrn_asgmts['big_gp_model'].predict(training_sample_x)['val']
        big_gp_abs_errs[-1].append(abs(pred - training_sample_y_aggregation))
    print([round(statistics.mean(errs), 8) for errs in big_gp_abs_errs])
    return np.mean([round(statistics.mean(errs), 8) for errs in big_gp_abs_errs])



if __name__ == "__main__":
    train_list = [50]
    train_size = train_list[0]
    # test_size = 84
    # valid_size = 84
    print("train size is", train_size)
    # print("valid size is", valid_size)
    # print("test size is", test_size)
    train_errs = []
    valid_errs = []
    test_errs = []

    params = nni.get_next_parameter()
    for i in range(1): # TODO
        # samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data = get_input(i)
        t1 = singlemodel(params, i)
        train_errs.append(t1)
        # test_errs.append(t2)
        # valid_errs.append(t3)
    print("avg training error",np.mean(train_errs))
    # print("avg test error",np.mean(test_errs))
    # print("avg valid error",np.mean(valid_errs))
    nni.report_final_result(np.mean(train_errs))
    print("")