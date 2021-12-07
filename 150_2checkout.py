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
total_data = 544
num_testing_data = 144
test_size=144
pretrain_size=582
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10
# dump_base_directory = "demo_model_zoo"

dataset_filename2 = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-scale-standardized.csv"
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-150-single-whole-standardized.csv"
all_sample_x_names0={}
all_sample_x_names0['adservice:0.90'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
all_sample_x_names0['productcatalogservice:0.90'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
all_sample_x_names0['recommendationservice:0.90'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice:rps",
                                                    "productcatalogservice:0.90"]
all_sample_x_names0['emailservice:0.90'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
all_sample_x_names0['paymentservice:0.90'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
all_sample_x_names0['shippingservice:0.90'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
all_sample_x_names0['currencyservice:0.90'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
all_sample_x_names0['get:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
all_sample_x_names0['set:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
all_sample_x_names0['cartservice:0.90'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                            "get:0.90", "set:0.90"]
all_sample_x_names0['checkoutservice:0.90'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names0['frontend:0.90'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                        "adservice:0.90", "checkoutservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "recommendationservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
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

zoo = Model_Zoo()
zoo.load(dump_directory)
fluxion = Fluxion(zoo)
all_lrn_asgmts = {}
selected_training_idxs = None
selected_testing_idxs = None

# ========== Compute small models' errors ==========
for sample_y_name in all_sample_x_names0.keys():
    sample_x_names = all_sample_x_names0[sample_y_name]
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)
    # print(len(samples_x))
    selected_training_idxs = range(pretrain_size)

    # STEP 1: Split dataset into training and testing
    training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
    training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
    
    # STEP 2: Train
    all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
    created_model_name = all_lrn_asgmts[sample_y_name].add_model(model_name[sample_y_name])

# ========== Compute Fluxion's errors ==========
# STEP 1: Prepare (1) Fluxion and (2) a list of input names that we will need to read from CSV
def _build_fluxion(tmp_service_name, visited_services=[]):
    if tmp_service_name in visited_services:
        return []
    visited_services.append(tmp_service_name)
    
    service_dependencies_name = []
    ret_inputs_name = []
    
    for sample_x_name in all_sample_x_names0[tmp_service_name]:
        if sample_x_name in all_sample_x_names0.keys():
            ret_inputs_name += _build_fluxion(sample_x_name, visited_services)
            service_dependencies_name.append(sample_x_name)
        else:
            service_dependencies_name.append(None)
            ret_inputs_name.append(sample_x_name)
    
    fluxion.add_service(tmp_service_name, tmp_service_name, all_lrn_asgmts[tmp_service_name], service_dependencies_name, service_dependencies_name)
    
    return ret_inputs_name

# add services
inputs_name = _build_fluxion(target_service_name)
fluxion.scale_service_horizontal("checkoutservice:0.90", 2)
dataset_filename3 = "dataset-test-standardized.csv"
samples_x0, samples_y0, samples_y_aggregation0, err_msg0 = lib_data.readCSVFile([dataset_filename3], inputs_name+['checkout_pod0:rps','checkout_pod1:rps'], "frontend:0.90")
scaler = np.load("/home/yuqingxie/autosys/code/PlayGround/yuqingxie/std_scaler_dataset_whole.npy", allow_pickle = True).item()
train_sizes = [25,50,100,200,400]
# train_size = [10]
for train_size in train_sizes:
    small_models_preds = []
    small_models_abs_errs = []
    small_models_raw_errs = []
    fluxion_abs_errs = []
    big_gp_abs_errs = []
    experiment_ids_completed = []
    each_model_errs = {}
    all_errs = []

    for t in model_name.keys():
        each_model_errs[t] = []

    for num_experiments_so_far in range(num_experiments):
        f = open("log/1204_3/fluxion_2checkout_"+str(train_size)+"_"+str(num_experiments_so_far),"w")
        sys.stdout = f
        print("========== Experiments finished so far:", num_experiments_so_far, "==========")
        experiment_ids_completed.append(num_experiments_so_far)
        random.seed(42 + num_experiments_so_far)
        np.random.seed(42 + num_experiments_so_far)
        
        small_models_preds.append({})
        small_models_abs_errs.append({})
        small_models_raw_errs.append({})
        fluxion_abs_errs.append([])
        big_gp_abs_errs.append([])
        errs = []

        for t in model_name.keys():
            each_model_errs[t].append([])


        # train scaled service
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['checkout_pod0:0.90','checkout_pod1:0.90'], "checkoutservice:0.90")
        # print(len(samples_x))
        selected_testing_idxs = random.sample(range(0, len(samples_x)), k=test_size)
        selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
        selected_training_idxs = random.sample(selected_training_idxs, k=train_size)

        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        fluxion.train_service("checkoutservice:0.90", "checkoutservice:0.90", training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])

        # prediction
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
            fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"].append(fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0].copy())
            fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0]["checkoutservice:rps"] = sample_x[-2] # checkout_pod0
            fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][1]["checkoutservice:rps"] = sample_x[-1] # checkout_pod0
            # print(fluxion_input["checkoutservice:0.90"])
            # fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"].append(fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0].copy())
            # fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0]["checkoutservice:rps"] = sample_x[-2] # checkout_pod0
            # fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][1]["checkoutservice:rps"] = sample_x[-1] # checkout_pod0
            # print(fluxion_input)
            # pred = all_lrn_asgmts['big_dnn_model'].predict(testing_sample_x)['val']
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
            # if k[:13] == "checkout_pod0" or k[:13] == "checkout_pod1":
            #     k = "checkoutservice"+k[13:]
            std = scaler[k+":STD"]
            avg = scaler[k+":AVG"]
            if std == 0:
                std = 1
            real_para[k] = samples_x0[best_index][cnt] * std + avg
            cnt += 1
        print(real_para)
        np.save("log/1204_3/fluxion_2checkout_"+str(train_size)+"_"+str(num_experiments_so_far), real_para)
        print(best * scaler["frontend:0.90:STD"] + scaler["frontend:0.90:AVG"])

        # STEP 2: Compute Fluxion's testing MAE
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], inputs_name+['checkout_pod0:rps','checkout_pod1:rps'], "frontend:0.90")
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
            fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"].append(fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0].copy())
            fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0]["checkoutservice:rps"] = sample_x[-2] # checkout_pod0
            fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][1]["checkoutservice:rps"] = sample_x[-1] # checkout_pod0
            pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
            fluxion_abs_errs[-1].append(abs(pred - sample_y_aggregation))
            errs.append(abs(pred - sample_y_aggregation))
        print("test MAE")
        all_errs.append(np.mean(errs))
        print(all_errs)
        print(np.mean(all_errs))
    #     errs.append(np.mean([round(statistics.mean(errs), 8) for errs in fluxion_abs_errs]))
    # print(np.mean(errs))

