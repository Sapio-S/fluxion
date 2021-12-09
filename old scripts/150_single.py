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
import numpy as np
# num_training_data = 983
total_data = 582
num_testing_data = 150
scaler = np.load("/home/yuqingxie/autosys/code/PlayGround/yuqingxie/150-single-whole-std-scaler.npy", allow_pickle = True).item()
target_deployment_name = "boutique_p95_p95"  # "boutique_p95_p95", "boutique_p95_p95", "hotel_p95_p95", "hotel_p95_p95", "hotel_p95_p50p85p95p95"
target_service_name = "frontend:0.95"  # "frontend:0.95", "frontend:0.95", "wrk|frontend|overall|lat-95", "wrk|frontend|overall|lat-95"
num_experiments = 10
# dump_base_directory = "demo_model_zoo"
all_sample_x_names = {}
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-150-single-whole-standardized.csv"
all_sample_x_names['adservice:0.95'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
all_sample_x_names['productcatalogservice:0.95'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
all_sample_x_names['recommendationservice:0.95'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice:rps",
                                                    "productcatalogservice:0.95"]
all_sample_x_names['emailservice:0.95'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
all_sample_x_names['paymentservice:0.95'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
all_sample_x_names['shippingservice:0.95'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
all_sample_x_names['currencyservice:0.95'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
all_sample_x_names['get:0.95'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
all_sample_x_names['set:0.95'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
all_sample_x_names['cartservice:0.95'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                            "get:0.95", "set:0.95"]
all_sample_x_names['checkoutservice:0.95'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
                                                "emailservice:0.95", "paymentservice:0.95", "shippingservice:0.95", "currencyservice:0.95", "cartservice:0.95", "productcatalogservice:0.95"]
all_sample_x_names['frontend:0.95'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                        "adservice:0.95", "checkoutservice:0.95", "shippingservice:0.95", "currencyservice:0.95", "recommendationservice:0.95", "cartservice:0.95", "productcatalogservice:0.95"]
# scaler = np.load("/home/yuqingxie/autosys/code/PlayGround/yuqingxie/150-std-scaler.npy", allow_pickle = True).item()

def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        if sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    
    return tmp_sample_x_names

inputs_name=['frontend:CPU_LIMIT', 'frontend:MEMORY_LIMIT', 'frontend:IPV4_RMEM', 'frontend:IPV4_WMEM', 'frontend:rps', 'adservice:MAX_ADS_TO_SERVE', 'adservice:CPU_LIMIT', 'adservice:MEMORY_LIMIT', 'adservice:IPV4_RMEM', 'adservice:IPV4_WMEM', 'adservice:rps', 'checkoutservice:CPU_LIMIT', 'checkoutservice:MEMORY_LIMIT', 'checkoutservice:IPV4_RMEM', 'checkoutservice:IPV4_WMEM', 'checkoutservice:rps', 'emailservice:CPU_LIMIT', 'emailservice:MEMORY_LIMIT', 'emailservice:MAX_WORKERS', 'emailservice:IPV4_RMEM', 'emailservice:IPV4_WMEM', 'emailservice:rps', 'paymentservice:CPU_LIMIT', 'paymentservice:MEMORY_LIMIT', 'paymentservice:IPV4_RMEM', 'paymentservice:IPV4_WMEM', 'paymentservice:rps', 'shippingservice:CPU_LIMIT', 'shippingservice:MEMORY_LIMIT', 'shippingservice:IPV4_RMEM', 'shippingservice:IPV4_WMEM', 'shippingservice:rps', 'currencyservice:CPU_LIMIT', 'currencyservice:MEMORY_LIMIT', 'currencyservice:IPV4_RMEM', 'currencyservice:IPV4_WMEM', 'currencyservice:rps', 'cartservice:CPU_LIMIT', 'cartservice:MEMORY_LIMIT', 'cartservice:IPV4_RMEM', 'cartservice:IPV4_WMEM', 'cartservice:rps', 'get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps', 'get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps', 'productcatalogservice:CPU_LIMIT', 'productcatalogservice:MEMORY_LIMIT', 'productcatalogservice:IPV4_RMEM', 'productcatalogservice:IPV4_WMEM', 'productcatalogservice:rps', 'recommendationservice:CPU_LIMIT', 'recommendationservice:MEMORY_LIMIT', 'recommendationservice:MAX_WORKERS', 'recommendationservice:MAX_RESPONSE', 'recommendationservice:IPV4_RMEM', 'recommendationservice:IPV4_WMEM', 'recommendationservice:rps']
dataset_filename2 = "dataset-p95-standardized.csv" 
samples_x0, samples_y0, samples_y_aggregation0, err_msg = lib_data.readCSVFile([dataset_filename2], inputs_name, target_service_name)

# train_size = [10, 25, 50, 100, 150, 200, 300, 450, 550, 650, 800]
train_size = [10, 25, 50, 100, 200, 300, 400]
for num_training_data in train_size:
    small_models_preds = []
    small_models_abs_errs = []
    small_models_raw_errs = []
    fluxion_abs_errs = []
    
    all_errs=[]
    experiment_ids_completed = []

    for num_experiments_so_far in range(num_experiments):
        f = open("log/1204_3/fluxion_single_"+str(num_training_data)+"_"+str(num_experiments_so_far),"w")
        sys.stdout = f
        print("========== Experiments finished so far:", num_experiments_so_far, "==========")
        experiment_ids_completed.append(num_experiments_so_far)
        random.seed(42 + num_experiments_so_far)
        numpy.random.seed(42 + num_experiments_so_far)
        
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)
        all_lrn_asgmts = {}
        selected_training_idxs = None
        selected_testing_idxs = None
        errs = []
        small_models_preds.append({})
        small_models_abs_errs.append({})
        small_models_raw_errs.append({})
        fluxion_abs_errs.append([])

        selected_testing_idxs = random.sample(range(0, total_data), k=num_testing_data)
        selected_training_idxs = set(range(0, total_data)) - set(selected_testing_idxs)
        selected_training_idxs = random.sample(selected_training_idxs, k=num_training_data)

        # ========== Compute small models' errors ==========
        for sample_y_name in all_sample_x_names.keys():
            sample_x_names = all_sample_x_names[sample_y_name]
            samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)
            
            # STEP 1: Split dataset into training and testing
            training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            # testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
            # testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
            
            # STEP 2: Train
            all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
            created_model_name = all_lrn_asgmts[sample_y_name].create_and_add_model(training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
            
        # # ========== Compute Fluxion's errors ==========
        # # STEP 1: Prepare (1) Fluxion and (2) a list of input names that we will need to read from CSV
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
            
            fluxion.add_service(tmp_service_name, tmp_service_name, all_lrn_asgmts[tmp_service_name], service_dependencies_name, service_dependencies_name)
            
            return ret_inputs_name
        
        inputs_name = _build_fluxion(target_service_name)

        # STEP 2: Compute Fluxion's testing MAE
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], inputs_name, target_service_name)
        for sample_idx, sample_x, sample_y_aggregation in zip(range(len(samples_x)), samples_x, samples_y_aggregation):
            if sample_idx not in selected_testing_idxs:
                continue
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
            errs.append(abs(pred - sample_y_aggregation))
        
        # output the test MAE
        print("test MAE is", np.mean(errs))
        all_errs.append(np.mean(errs))


        preds = []
        # print(samples_x, samples_y_aggregation)
        for sample_x, sample_y_aggregation in zip(samples_x0, samples_y_aggregation0):
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
            preds.append(pred)
        # output the best result & its parameter setting
        best = min(preds)
        best_index = preds.index(best)
        print(len(preds))
        print(best_index)
        print(best)
        print(samples_x0[best_index])

        # use scaler to convert it back to original form
        print("")
        cnt = 0
        real_para = {}
        for k in inputs_name:
            # real_para[k] = samples_x_real[best_index]
            std = scaler[k+":STD"]
            avg = scaler[k+":AVG"]
            if std == 0:
                std = 1
            real_para[k] = samples_x0[best_index][cnt] * std + avg
            cnt += 1
        print(real_para)
        np.save("log/1204_3/fluxion_single_"+str(num_training_data)+"_"+str(num_experiments_so_far), real_para)
        print(best * scaler["frontend:0.95:STD"] + scaler["frontend:0.95:AVG"])
        print(all_errs)
        print("overall MAE is", np.mean(all_errs))


