
import os, sys
import json, statistics
sys.path.insert(1, "../")
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
target_deployment_name = "boutique_p95_p95"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10

num_training_data = 50
# num_training_data = 133
small_models_preds = []
small_models_abs_errs = []
small_models_raw_errs = []
fluxion_abs_errs = []
big_gp_abs_errs = []
experiment_ids_completed = []

dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2scale-standardized.csv"
all_sample_x_names={}
all_sample_x_names['adservice:0.90'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
all_sample_x_names['productcatalogservice:0.90'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
all_sample_x_names['recommendation_pod0:0.90'] = ["recommendation_pod0:CPU_LIMIT", "recommendation_pod0:MEMORY_LIMIT", "recommendation_pod0:MAX_WORKERS", "recommendation_pod0:MAX_RESPONSE", "recommendation_pod0:IPV4_RMEM", "recommendation_pod0:IPV4_WMEM", "recommendation_pod0:rps",
                                                    "productcatalogservice:0.90"]
all_sample_x_names['recommendation_pod1:0.90'] = ["recommendation_pod1:CPU_LIMIT", "recommendation_pod1:MEMORY_LIMIT", "recommendation_pod1:MAX_WORKERS", "recommendation_pod1:MAX_RESPONSE", "recommendation_pod1:IPV4_RMEM", "recommendation_pod1:IPV4_WMEM", "recommendation_pod1:rps",
                                                    "productcatalogservice:0.90"]
all_sample_x_names['emailservice:0.90'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
all_sample_x_names['paymentservice:0.90'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
all_sample_x_names['shippingservice:0.90'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
all_sample_x_names['currencyservice:0.90'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
all_sample_x_names['get:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
all_sample_x_names['set:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
all_sample_x_names['cartservice:0.90'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                            "get:0.90", "set:0.90"]
all_sample_x_names['checkout_pod0:0.90'] = ["checkout_pod0:CPU_LIMIT", "checkout_pod0:MEMORY_LIMIT", "checkout_pod0:IPV4_RMEM", "checkout_pod0:IPV4_WMEM", "checkout_pod0:rps",
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names['checkout_pod1:0.90'] = ["checkout_pod1:CPU_LIMIT", "checkout_pod1:MEMORY_LIMIT", "checkout_pod1:IPV4_RMEM", "checkout_pod1:IPV4_WMEM", "checkout_pod1:rps",
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
# all_sample_x_names['checkoutservice:0.90'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
#                                                 "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names['frontend:0.90'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                        "adservice:0.90", "checkoutservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "recommendationservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]

def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        if sample_x_name == "recommendationservice:0.90":
            tmp_sample_x_names += expand_sample_x_name("recommendation_pod0:0.90")
            tmp_sample_x_names += expand_sample_x_name("recommendation_pod1:0.90")
        if sample_x_name == "checkoutservice:0.90":
            tmp_sample_x_names += expand_sample_x_name("checkout_pod0:0.90")
            tmp_sample_x_names += expand_sample_x_name("checkout_pod1:0.90")
        elif sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    return tmp_sample_x_names


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
    
    # STEP 2: Determine training and testing indexes
    print(dataset_filename, "has", len(samples_x), "data points")
    selected_training_idxs = random.sample(range(0, len(samples_x)), k=num_training_data)
    selected_training_idxs = set(range(0, len(samples_x))) - set(selected_training_idxs)
    selected_training_idxs = random.sample(selected_training_idxs, k=num_training_data)
    
    # STEP 3: Split dataset into training and testing
    training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
    training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
    training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
    training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
    
    # STEP 4: Compute Big-GP's testing MAE
    all_lrn_asgmts['big_gp_model'] = LearningAssignment(zoo, expanded_sample_x_names)
    created_model_name = all_lrn_asgmts['big_gp_model'].create_and_add_model(
        training_samples_x, 
        training_samples_y_aggregation,  
        MultiLayerPerceptron, 
        model_class_args=[
            (params['searchSpace']["hidden_size1"],params['searchSpace']["hidden_size2"],params['searchSpace']["hidden_size3"]), 
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