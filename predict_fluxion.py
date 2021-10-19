import numpy as np
import json, numpy, sys, random, statistics

sys.path.insert(1, "../")
sys.path.insert(1, "../Demo")
from fluxion import Fluxion
import lib_data
from GraphEngine.learning_assignment import LearningAssignment
import GraphEngine.lib_learning_assignment as lib_learning_assignment
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
import pandas as pd

train_size = [10, 25, 50, 100, 150, 200, 300, 450, 550, 650, 800]
pretrain_size = 10
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10

dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-whole-standardized.csv"
dataset_filename2 = "dataset-standardized.csv"
all_sample_x_names={}
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
# scale_sample_x_names={}
# scale_sample_x_names["recommendationservice:0.90"] = ['recommendation_pod0:0.90','recommendation_pod1:0.90']
# scale_sample_x_names["checkoutservice:0.90"] = ['checkout_pod0:0.90','checkout_pod1:0.90']
scaler = np.load("std_scaler_dataset_whole.npy", allow_pickle = True).item()

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

    return ret_inputs_name

def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        # if sample_x_name == "recommendationservice:0.90":
        #     tmp_sample_x_names += expand_sample_x_name("recommendation_pod0:0.90")
        #     tmp_sample_x_names += expand_sample_x_name("recommendation_pod1:0.90")
        # if sample_x_name == "checkoutservice:0.90":
        #     tmp_sample_x_names += expand_sample_x_name("checkout_pod0:0.90")
        #     tmp_sample_x_names += expand_sample_x_name("checkout_pod1:0.90")
        if sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    return tmp_sample_x_names

inputs_name = _build_fluxion(target_service_name)
samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], inputs_name, target_service_name)
# samples_x_real, samples_y_real, samples_y_aggregation_real, err_msg = lib_data.readCSVFile(["dataset.csv"], inputs_name, target_service_name)

for pretrain_size in train_size:
    for num in range(num_experiments):
        f = open("log/1020single/fluxion_"+str(pretrain_size)+"_"+str(num),"w")
        sys.stdout = f
        path = "saved_model/dic"+str(pretrain_size)+"_"+str(num)
        name_list = np.load(path+".npy", allow_pickle = True)

        zoo = Model_Zoo()
        zoo.load("saved_model/"+str(pretrain_size)+"_"+str(num))
        fluxion = Fluxion(zoo)
        all_lrn_asgmts = {}
        # selected_training_idxs = None
        # selected_testing_idxs = None

        for sample_y_name in all_sample_x_names.keys():
            sample_x_names = all_sample_x_names[sample_y_name]
            samples_x0, samples_y0, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)

            selected_training_idxs = range(pretrain_size)

            # STEP 1: Split dataset into training and testing
            training_samples_x = [samples_x0[idx] for idx in selected_training_idxs]
            training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            
            # STEP 2: Load models
            all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
            for name in name_list:
                try:
                    created_model_name = all_lrn_asgmts[sample_y_name].add_model(name)
                    name_list.remove(name)
                except:
                    continue
            
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
            
            fluxion.add_service(tmp_service_name, tmp_service_name, all_lrn_asgmts[tmp_service_name], service_dependencies_name, service_dependencies_name)
            
            return ret_inputs_name

        # add services
        inputs_name = _build_fluxion(target_service_name)

        # scale service
        # fluxion.scale_service_horizontal("recommendationservice:0.90", 2)
        # fluxion.scale_service_horizontal("checkoutservice:0.90",2)

            # # train scaled service
            # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['recommendation_pod0:0.90','recommendation_pod1:0.90'], "recommendationservice:0.90")
            # selected_testing_idxs = random.sample(range(0, len(samples_x)), k=test_size)
            # selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
            # selected_training_idxs = random.sample(selected_training_idxs, k=train_size)

            # training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            # training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            # fluxion.train_service("recommendationservice:0.90", "recommendationservice:0.90", training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
            
            # # train scaled service
            # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['checkout_pod0:0.90','checkout_pod1:0.90'], "checkoutservice:0.90")
            # selected_testing_idxs = random.sample(range(0, len(samples_x)), k=test_size)
            # selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
            # selected_training_idxs = random.sample(selected_training_idxs, k=train_size)

            # training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            # training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            # fluxion.train_service("checkoutservice:0.90", "checkoutservice:0.90", training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])

            # STEP 2: Predict the output of this set of parameters
        preds = []
        # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], inputs_name, target_service_name)
        # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], inputs_name+["recommendation_pod0:rps", "recommendation_pod1:rps", "checkout_pod0:rps", "checkout_pod1:rps"], target_service_name)
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

            # # special for scaled out services
            # fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"].append(fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"][0].copy())
            # fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"][0]["recommendationservice:rps"] = sample_x[-2] # checkout_pod0
            # fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"][1]["recommendationservice:rps"] = sample_x[-1] # checkout_pod0
            # fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"].append(fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0].copy())
            # fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0]["checkoutservice:rps"] = sample_x[-2] # checkout_pod0
            # fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][1]["checkoutservice:rps"] = sample_x[-1] # checkout_pod0
            pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
            preds.append(abs(pred))
        
        # output the best result & its parameter setting
        best = min(preds)
        best_index = preds.index(best)
        print(best_index)
        print(best)
        print(samples_x[best_index])

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
            real_para[k] = samples_x[best_index][cnt] * std + avg
            cnt += 1
        print(real_para)
        np.save("log/1020single/res_fluxion_"+str(pretrain_size)+"_"+str(num), real_para)
        print(best * scaler["frontend:0.90:STD"] + scaler["frontend:0.90:AVG"])

