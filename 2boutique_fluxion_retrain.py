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
retrain_list = []
total_data = 283
num_testing_data = 83
test_size=83
pretrain_size=900
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10
# dump_base_directory = "demo_model_zoo"

new_dataset = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2-standardized.csv"
pretrain_dataset = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-whole-standardized.csv"
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
all_sample_x_names2={}

all_sample_x_names2['adservice_pod0:0.90'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice_pod0:rps"]
all_sample_x_names2['productcatalogservice_pod0:0.90'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice_pod0:rps"]
all_sample_x_names2['recommendationservice_pod0:0.90'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice_pod0:rps",
                                                    "productcatalogservice:0.90"]
all_sample_x_names2['emailservice_pod0:0.90'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice_pod0:rps"]
all_sample_x_names2['paymentservice_pod0:0.90'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice_pod0:rps"]
all_sample_x_names2['shippingservice_pod0:0.90'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice_pod0:rps"]
all_sample_x_names2['currencyservice_pod0:0.90'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice_pod0:rps"]
all_sample_x_names2['get_pod0:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get_pod0:rps']
all_sample_x_names2['set_pod0:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set_pod0:rps']
all_sample_x_names2['cartservice_pod0:0.90'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                            "get:0.90", "set:0.90"]
all_sample_x_names2['checkoutservice_pod0:0.90'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice_pod0:rps",
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names2['frontend_pod0:0.90'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend_pod0:rps",
                                        "adservice:0.90", "checkoutservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "recommendationservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]

# retrain_list = ["productcatalogservice:0.90", "recommendationservice:0.90",'checkoutservice:0.90', "frontend:0.90"]
retrain_list = ["adservice:0.90", "recommendationservice:0.90",'emailservice:0.90', "frontend:0.90"]

x_names = all_sample_x_names.keys()
train_name = {}
scale_podname = []
for service in x_names:
    podname = service[:-5]
    # pods = [podname+"_pod0",podname+"_pod1"]
    train_name[service] = [podname+"_pod0:0.90",podname+"_pod1:0.90"]
    scale_podname.append(podname+"_pod0:rps")
    scale_podname.append(podname+"_pod1:rps")
# print(train_name)
'''
{'adservice:0.90': ['adservice_pod0:0.90', 'adservice_pod1:0.90'], 'productcatalogservice:0.90': ['productcatalogservice_pod0:0.90', 'productcatalogservice_pod1:0.90'], 'recommendationservice:0.90': ['recommendationservice_pod0:0.90', 'recommendationservice_pod1:0.90'], 'emailservice:0.90': ['emailservice_pod0:0.90', 'emailservice_pod1:0.90'], 'paymentservice:0.90': ['paymentservice_pod0:0.90', 'paymentservice_pod1:0.90'], 'shippingservice:0.90': ['shippingservice_pod0:0.90', 'shippingservice_pod1:0.90'], 'currencyservice:0.90': ['currencyservice_pod0:0.90', 'currencyservice_pod1:0.90'], 'get:0.90': ['get_pod0:0.90', 'get_pod1:0.90'], 'set:0.90': ['set_pod0:0.90', 'set_pod1:0.90'], 'cartservice:0.90': ['cartservice_pod0:0.90', 'cartservice_pod1:0.90'], 'checkoutservice:0.90': ['checkoutservice_pod0:0.90', 'checkoutservice_pod1:0.90'], 'frontend:0.90': ['frontend_pod0:0.90', 'frontend_pod1:0.90']}
'''



# dump_directory = "model_150"
# model_name = {
#     'adservice:0.90':"957814e6f7ef4e0081412c153bd3b2e3",
#     'productcatalogservice:0.90':"f7a3e65b6cdf445da9715524f84abec7",
#     'recommendationservice:0.90':"a7fde617b7a1426e9332b5d477ccce54",
#     'emailservice:0.90':"5c0edd6bbd484602ac4ad6cae83bdfeb",
#     'paymentservice:0.90':"5ca977e68ac94224a575fe858de1b497",
#     'shippingservice:0.90':"a6e959c59ad04f348c4e0b962b0a87ff",
#     'currencyservice:0.90':"147e847c67f54d34a3312dc15f5a98b4",
#     'get:0.90':"67cb457260464cb89fdd7f0f8414ab62",
#     'set:0.90':"18613adf032f4bad9c52f2d9d46b6090",
#     'cartservice:0.90':"bf30e47eb2154155b4b9d9de62126dca",
#     'checkoutservice:0.90':"c17f0956daf5482bb40a0051092c9e80",
#     'frontend:0.90':"6154251e64484a4696cc660edafca0b8",
# }
# dump_directory = "model_single"
# model_name = {
#     'adservice:0.90':"cf9fe0901b4a460a813707b6d00f7b83",
#     'productcatalogservice:0.90':"1becb3643a2848bcafbb38e567203086",
#     'recommendationservice:0.90':"4de76bc450e944a798757ecc1b9ff07f",
#     'emailservice:0.90':"6d3eb0ba189747a19dd578169507a04e",
#     'paymentservice:0.90':"ddffcdcb6d1541199f1de72dd99123d8",
#     'shippingservice:0.90':"f2e4551f3a634523812cc2b38439c04f",
#     'currencyservice:0.90':"135c06f9afff4b06bf85e74a1fa03a52",
#     'get:0.90':"4e1bcd1913874fecac2e827b9229fd92",
#     'set:0.90':"addd9f2656bf41ce8ec982fd398063e6",
#     'cartservice:0.90':"c2178c851c7248269f5b03eccc41afa6",
#     'checkoutservice:0.90':"4f3cf87acf694c69821a7c778c961287",
#     'frontend:0.90':"036582b44ac34f5cbf356c88a5698819",
# }
dump_directory = "demo_model_zoo"
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


all_lrn_asgmts = {}
selected_training_idxs = None
selected_testing_idxs = None

small_models_preds = []
small_models_abs_errs = []
small_models_raw_errs = []

train_sizes = [10,25,50,100,150,200]
# train_sizes=[5]
for train_size in train_sizes:
    all_errs = []
    f = open("log/1208/fluxion_2_retrain_part_"+str(train_size),"w")
    sys.stdout = f
    small_models_preds.append({})
    small_models_abs_errs.append({})
    small_models_raw_errs.append({})
    for num_experiments_so_far in range(num_experiments):
        zoo = Model_Zoo()
        zoo.load(dump_directory)
        fluxion = Fluxion(zoo)
        print("========== Experiments finished so far:", num_experiments_so_far, "==========")
        random.seed(42 + num_experiments_so_far)
        np.random.seed(42 + num_experiments_so_far)
        selected_testing_idxs = random.sample(range(0, total_data), k=test_size)
        selected_training_idxs = set(range(0, total_data)) - set(selected_testing_idxs)
        selected_training_idxs = random.sample(selected_training_idxs, k=train_size)
        errs = []

        # ========== Compute small models' errors ==========
        for sample_y_name in all_sample_x_names2.keys():
            sample_x_names = all_sample_x_names2[sample_y_name]
            samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([new_dataset], sample_x_names, sample_y_name)
            
            # STEP 1: Split dataset into training and testing
            training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
            testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
            
            # STEP 2: Train
            sample_y_name = sample_y_name[:-10]+":0.90"
            sample_x_names = all_sample_x_names[sample_y_name]
            all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
            if sample_y_name in retrain_list:
                created_model_name = all_lrn_asgmts[sample_y_name].create_and_add_model(training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
            else:
                created_model_name = all_lrn_asgmts[sample_y_name].add_model(model_name[sample_y_name])
            
            # # STEP 3: Compute MAE with testing dataset
            small_models_preds[-1][sample_y_name] = []
            for testing_sample_x in testing_samples_x:
                small_models_preds[-1][sample_y_name].append(all_lrn_asgmts[sample_y_name].predict(testing_sample_x)['val'])
            small_models_raw_errs[-1][sample_y_name] = [t - p for p, t in zip(small_models_preds[-1][sample_y_name], testing_samples_y_aggregation)]
            small_models_abs_errs[-1][sample_y_name] = [abs(err) for err in small_models_raw_errs[-1][sample_y_name]]
        
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

        # scale services
        inputs_name = _build_fluxion(target_service_name)
        for service in x_names:
            fluxion.scale_service_horizontal(service, 2)

        # train scaled service
        for service in x_names:
            samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([new_dataset], train_name[service], service)
            training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            fluxion.train_service(service, service, training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
        
        # STEP 2: Compute Fluxion's test MAE
        total_name = inputs_name+scale_podname
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([new_dataset], total_name, "frontend:0.90")
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

            for service in train_name.keys():
                service_name = service[:-5]
                fluxion_input[service][service].append(fluxion_input[service][service][0].copy())
                pod0_sub = total_name.index(train_name[service][0][:-5]+":rps")
                pod1_sub = total_name.index(train_name[service][1][:-5]+":rps")
                fluxion_input[service][service][0][service_name+":rps"] = sample_x[pod0_sub]
                fluxion_input[service][service][1][service_name+":rps"] = sample_x[pod1_sub]
            # print(fluxion_input)
            # exit(0)
            pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
            errs.append(abs(pred - sample_y_aggregation))
        print("test MAE",np.mean(errs))
        all_errs.append(np.mean(errs))
    print(all_errs)
    print(np.mean(all_errs))

    print("| small_models_abs_errs:")
    for sample_x_name in all_sample_x_names:
        print(sample_x_name, [round(statistics.mean(small_models_abs_errs[idx][sample_x_name]), 8) for idx in range(len(small_models_abs_errs))])
 

