# python3 fluxion_vs_monolith.py

import json, numpy, sys, random, statistics

sys.path.insert(1, "../")
# sys.path.insert(1, "../../Dem/o")
from fluxion import Fluxion
import lib_data
from GraphEngine.learning_assignment import LearningAssignment
import GraphEngine.lib_learning_assignment as lib_learning_assignment
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron

# from GraphEngine.learning_assignment_normal import LearningAssignment
import numpy as np
import time
from consts import *
num_training_data = 25
num_testing_data = 154
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 1
dump_base_directory = "demo_model_zoo"
all_sample_x_names = {}
# dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-scale-standardized.csv"
# dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/dataset-whole-standardized.csv"
all_sample_x_names['adservice:0.90'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
all_sample_x_names['productcatalogservice:0.90'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
all_sample_x_names['recommendationservice:0.90'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice:rps",
                                                    "productcatalogservice:0.90"]
all_sample_x_names['emailservice:0.90'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
all_sample_x_names['paymentservice:0.90'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
all_sample_x_names['shippingservice:0.90'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
all_sample_x_names['currencyservice:0.90'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
all_sample_x_names['get:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
all_sample_x_names['set:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
all_sample_x_names['cartservice:0.90'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                            "get:0.90", "set:0.90"]
all_sample_x_names['checkoutservice:0.90'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names['frontend:0.90'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                        "adservice:0.90", "checkoutservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "recommendationservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
valid_size = 50
test_size = 133
bin_num = 10
sample_size = 100
model_name = {
    'adservice:0.90':"66f9864e3ec24396b9c9b2f8cafe68d6",
    'productcatalogservice:0.90':"812158ffbda740ebab6bfbed9c62653f",
    'recommendationservice:0.90':"4326848094a5414684a0883e75d1b45a",
    'emailservice:0.90':"5625918688c54f059fcd1ba68e0991e6",
    'paymentservice:0.90':"412efe185ba342db863bfcf462bcb139",
    'shippingservice:0.90':"ed0ac78de159432ba9d48c6bdb545199",
    'currencyservice:0.90':"e50ef3be7ce24ff191216550ae3d32cc",
    'get:0.90':"29cae95190fd4552b51cee9de0c57998",
    'set:0.90':"a7f00f6b6ae546caa6d1786c259c93be",
    'cartservice:0.90':"e33a2a58f5574dd4ac387531447567eb",
    'checkoutservice:0.90':"553c3c3ac6294967857a44ae40b331ad",
    'frontend:0.90':"9025b65872174a1f8f349029a3cb5287",
}
def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        if sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    
    return tmp_sample_x_names

small_models_preds = []
small_models_abs_errs = []
small_models_raw_errs = []
fluxion_abs_errs = []
big_gp_abs_errs = []
experiment_ids_completed = []

def get_permutation():
    permutation = []
    for i1 in range(sample_size):
        for i2 in range(sample_size):
            for i3 in range(sample_size):
                for i4 in range(sample_size):
                    for i5 in range(sample_size):
                        for i6 in range(sample_size):
                            for i7 in range(sample_size):
                                for i8 in range(sample_size):
                                    permutation.append([i8,i7,i6,i5,i4,i3,i2,i1])
    return permutation

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

for dataset in range(1,5):
    testfile = "data_for_error/data_"+str(dataset)+".csv"
    trainfile = "/home/yuqingxie/autosys/code/fluxion/data_for_error/test.csv"
    random.seed(0)
    np.random.seed(0)

    f = open("log/1013/my_error_"+str(dataset),"w")
    sys.stdout = f

    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    all_lrn_asgmts = {}
    selected_training_idxs = None
    selected_testing_idxs = None

    each_model_errs = {}
    error_dist = {}
    hist_num = {}
    hist_val = {}
    for t in model_name.keys():
        each_model_errs[t] = [[]]
        error_dist[t] = []
        hist_num[t] = []
        hist_val[t] = []
    
    # ========== Compute small models' errors ==========
    for sample_y_name in all_sample_x_names.keys():
        
        sample_x_names = all_sample_x_names[sample_y_name]
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([trainfile], sample_x_names, sample_y_name)

        print("training dataset has", len(samples_x), "data points")
        selected_testing_idxs = random.sample(range(0, len(samples_x)), k=num_testing_data)
        selected_training_idxs = set(range(0, len(samples_x)))
        selected_training_idxs = random.sample(selected_training_idxs, k=num_training_data)

        # STEP 1: Split dataset into training and testing
        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        
        # STEP 2: Train
        all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
        created_model_name = all_lrn_asgmts[sample_y_name].create_and_add_model(training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])

    # ========== Compute Fluxion's errors ==========
    # STEP 1: Prepare (1) Fluxion and (2) a list of input names that we will need to read from CSV
    def _build_fluxion(tmp_service_name, visited_services=[]):
        if tmp_service_name in visited_services:
            return []
        visited_services.append(tmp_service_name)
        
        service_dependencies_name = []
        ret_inputs_name = []
        
        for sample_x_name in all_sample_x_names[tmp_service_name]:
            if sample_x_name in all_sample_x_names.keys():
                ret_inputs_name += _build_fluxion(sample_x_name, visited_services)
                service_dependencies_name.append(sample_x_name)
            else:
                service_dependencies_name.append(None)
                ret_inputs_name.append(sample_x_name)
        
        fluxion.add_service(tmp_service_name, tmp_service_name, all_lrn_asgmts[tmp_service_name], [None]*len(service_dependencies_name), [None]*len(service_dependencies_name))
        
        return ret_inputs_name
    
    inputs_name = _build_fluxion(target_service_name) + list(model_name.keys())

    # calculate error distribution
    for t in model_name.keys():
        target_service_name = t
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([testfile], inputs_name, target_service_name)
        for sample_idx, sample_x, sample_y_aggregation in zip(range(len(samples_x)), samples_x, samples_y_aggregation):
            fluxion_input = {}
            for val, input_name in zip(sample_x, inputs_name):
                service_name = None
                for tmp_name in all_sample_x_names.keys():
                    if input_name in all_sample_x_names[tmp_name]:
                        service_name = tmp_name
                        
                        if service_name not in fluxion_input.keys():
                            fluxion_input[service_name] = {}
                        if service_name not in fluxion_input[service_name].keys():
                            fluxion_input[service_name][service_name] = [{}]
                        if input_name not in fluxion_input[service_name].keys():
                            fluxion_input[service_name][service_name][0][input_name] = val

            pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
            error_dist[t].append((pred - sample_y_aggregation))
            hist_num[t], hist_val[t] = np.histogram(error_dist[t][-1],bins=10)


    # Compute Fluxion's testing MAE
    predictions = {}
    for t in model_name.keys():
        predictions[t] = {}
        target_service_name = t
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([trainfile], inputs_name, target_service_name)
        for sample_idx, sample_x, sample_y_aggregation in zip(range(len(samples_x)), samples_x, samples_y_aggregation):
            if sample_idx not in selected_testing_idxs:
                continue

            preds = []
            fluxion_input = {}
            for val, input_name in zip(sample_x, inputs_name):
                service_name = None
                tmp_name = t
                if input_name in all_sample_x_names[tmp_name]: 
                    service_name = tmp_name
                    
                    if service_name not in fluxion_input.keys():
                        fluxion_input[service_name] = {}
                    if service_name not in fluxion_input[service_name].keys():
                        fluxion_input[service_name][service_name] = [{}]
                    if input_name not in fluxion_input[service_name].keys():
                        fluxion_input[service_name][service_name][0][input_name] = val
            
            for k in range(sample_size):
                data_piece = fluxion_input.copy()
                for val, input_name in zip(sample_x, inputs_name):
                    tmp_name = t
                    if input_name in all_sample_x_names[tmp_name]: 
                        service_name = tmp_name
                        if input_name in all_sample_x_names.keys():
                            data_piece[service_name][service_name][0][input_name] = predictions[input_name][sample_idx] - sample_error(hist_num[input_name], hist_val[input_name])
                preds.append(fluxion.predict(target_service_name, target_service_name, data_piece)[target_service_name][target_service_name]['val'])
            
            pred = np.mean(preds)
            each_model_errs[t][-1].append(abs(pred - sample_y_aggregation))
            predictions[t][sample_idx] = pred
    
    print("______res______")
    for t in model_name.keys():
        print(t)
        for errs in each_model_errs[t]:
            print(round(statistics.mean(errs), 8))
            print("avg:", np.mean(errs))
            print("std:", np.std(errs))