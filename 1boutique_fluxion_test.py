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
total_data = 600
num_testing_data = 150
test_size=150
pretrain_size=10
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 1
# dump_base_directory = "demo_model_zoo"

new_dataset = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-100-standardized.csv"
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

x_names = all_sample_x_names.keys()

# print(train_name)
'''
{'adservice:0.90': ['adservice_pod0:0.90', 'adservice_pod1:0.90'], 'productcatalogservice:0.90': ['productcatalogservice_pod0:0.90', 'productcatalogservice_pod1:0.90'], 'recommendationservice:0.90': ['recommendationservice_pod0:0.90', 'recommendationservice_pod1:0.90'], 'emailservice:0.90': ['emailservice_pod0:0.90', 'emailservice_pod1:0.90'], 'paymentservice:0.90': ['paymentservice_pod0:0.90', 'paymentservice_pod1:0.90'], 'shippingservice:0.90': ['shippingservice_pod0:0.90', 'shippingservice_pod1:0.90'], 'currencyservice:0.90': ['currencyservice_pod0:0.90', 'currencyservice_pod1:0.90'], 'get:0.90': ['get_pod0:0.90', 'get_pod1:0.90'], 'set:0.90': ['set_pod0:0.90', 'set_pod1:0.90'], 'cartservice:0.90': ['cartservice_pod0:0.90', 'cartservice_pod1:0.90'], 'checkoutservice:0.90': ['checkoutservice_pod0:0.90', 'checkoutservice_pod1:0.90'], 'frontend:0.90': ['frontend_pod0:0.90', 'frontend_pod1:0.90']}
'''
dump_directory = "single_model_0.90"
model_name = {
    'adservice:0.90':"912f5029bc8149d682d8da7ccb26cbb2",
    'productcatalogservice:0.90':"8f6b32209d7746f28f000ce015eef6f4",
    'recommendationservice:0.90':"ffe5395fc57e4f0f817dd61f2789d4ac",
    'emailservice:0.90':"580cc4ab534d4972b73392c87d32ca42",
    'paymentservice:0.90':"897fc4da2c3342eaab4696b3e50297e0",
    'shippingservice:0.90':"6e3dca117ec541e2a9194e70ea8d399e",
    'currencyservice:0.90':"a13081619354461aa4ed0cabf36ef0c0",
    'get:0.90':"4cbdfb1ab0d54826bd12579ed6a7d47e",
    'set:0.90':"88321e1d78e44dfdb8c3da7fdc50ba75",
    'cartservice:0.90':"0eadb0f5f79b4e14b8aec00f6c8c771e",
    'checkoutservice:0.90':"bfc77cb1d93b42d985256f34920e38d3",
    'frontend:0.90':"d196dbc5f9754fb492bbcfeb846d5ff9",
}
def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
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
for sample_y_name in all_sample_x_names.keys():
    sample_x_names = all_sample_x_names[sample_y_name]
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([pretrain_dataset], sample_x_names, sample_y_name)
    # print(len(samples_x))
    selected_training_idxs = range(len(samples_x))

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
# for service in x_names:
#     fluxion.scale_service_horizontal(service, 2)

small_models_preds = []
small_models_abs_errs = []
small_models_raw_errs = []

train_sizes = [50]
# train_sizes=[5]
for train_size in train_sizes:
    all_errs = []
    # f = open("log/1208/fluxion_2_"+str(train_size),"w")
    # sys.stdout = f
        
    small_models_preds.append({})
    small_models_abs_errs.append({})
    small_models_raw_errs.append({})
    for num_experiments_so_far in range(num_experiments):

        print("========== Experiments finished so far:", num_experiments_so_far, "==========")
        random.seed(42 + num_experiments_so_far)
        np.random.seed(42 + num_experiments_so_far)
        selected_testing_idxs = random.sample(range(0, total_data), k=test_size)
        selected_training_idxs = set(range(0, total_data)) - set(selected_testing_idxs)
        selected_training_idxs = random.sample(selected_training_idxs, k=train_size)
        errs = []

        # # train scaled service
        # for service in x_names:
        #     samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([new_dataset], train_name[service], service)
        #     training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        #     training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        #     fluxion.train_service(service, service, training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])

    for sample_y_name in all_sample_x_names.keys():
        sample_x_names=all_sample_x_names[sample_y_name]
        # sample_x_names = all_sample_x_names2[train_name[sample_y_name][0]]+all_sample_x_names2[train_name[sample_y_name][1]]
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([new_dataset], sample_x_names, sample_y_name)
        
        # STEP 1: Split dataset into training and testing
        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
        testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
        
        # STEP 3: Compute MAE with testing dataset
        # sample_y_name=sample_y_name[:-10]+":0.90"
        small_models_preds[-1][sample_y_name] = []
        for testing_sample_x in testing_samples_x:
            small_models_preds[-1][sample_y_name].append(all_lrn_asgmts[sample_y_name].predict(testing_sample_x)['val'])
        small_models_raw_errs[-1][sample_y_name] = [t - p for p, t in zip(small_models_preds[-1][sample_y_name], testing_samples_y_aggregation)]
        small_models_abs_errs[-1][sample_y_name] = [abs(err) for err in small_models_raw_errs[-1][sample_y_name]]
    
    print("| small_models_abs_errs:")
    for sample_x_name in all_sample_x_names:
        print(sample_x_name, [round(statistics.mean(small_models_abs_errs[idx][sample_x_name]), 8) for idx in range(len(small_models_abs_errs))])
    
