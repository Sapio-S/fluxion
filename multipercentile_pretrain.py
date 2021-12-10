# python3 fluxion_vs_monolith.py

import json, sys, random, statistics

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
total_data = 600
# num_testing_data = 150
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10
# dump_base_directory = "demo_model_zoo"
all_sample_x_names = {}
# perf = ["0.50", "0.90"]
perf = ["0.50", "0.90", "0.95", "0.85"]
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-100-85-standardized.csv"
# all_sample_x_names['productcatalogservice:'] = ["productcatalogservice:CPU_LIMIT"]
# all_sample_x_names['recommendationservice:'] = ["recommendationservice:CPU_LIMIT","productcatalogservice:"]
all_sample_x_names['adservice:'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
all_sample_x_names['productcatalogservice:'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
all_sample_x_names['recommendationservice:'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice:rps",
                                                    "productcatalogservice:"]
all_sample_x_names['emailservice:'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
all_sample_x_names['paymentservice:'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
all_sample_x_names['shippingservice:'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
all_sample_x_names['currencyservice:'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
all_sample_x_names['get:'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
all_sample_x_names['set:'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
all_sample_x_names['cartservice:'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                            "get:", "set:"]
all_sample_x_names['checkoutservice:'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
                                                "emailservice:", "paymentservice:", "shippingservice:", "currencyservice:", "cartservice:", "productcatalogservice:"]
all_sample_x_names['frontend:'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                        "adservice:", "checkoutservice:", "shippingservice:", "currencyservice:", "recommendationservice:", "cartservice:", "productcatalogservice:"]
dump_base_directory = "multi_model"
def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        if sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    
    return tmp_sample_x_names

train_size = [600]
for num_training_data in train_size:
    small_models_preds = []

    all_errs = []
    experiment_ids_completed = []

    errs = []
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    all_lrn_asgmts = {}
    selected_training_idxs = None
    selected_testing_idxs = None

    selected_training_idxs = range(0, total_data)

    # ========== Compute small models' errors ==========
    for p in perf:
        for sample_y_name_ in all_sample_x_names.keys():
            sample_x_names_ = all_sample_x_names[sample_y_name_]
            sample_x_names = []
            for x in sample_x_names_:
                if x[-1] == ":":
                    for p2 in perf:
                        sample_x_names.append(x+p2)
                else:
                    sample_x_names.append(x)
            sample_y_name = sample_y_name_+p
            samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)

            # STEP 1: Split dataset into training and testing
            training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]

            # STEP 2: Train
            all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
            created_model_name = all_lrn_asgmts[sample_y_name].create_and_add_model(training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])

dump_directory = dump_base_directory
zoo.dump(dump_directory)

def _build_fluxion(tmp_service_name, visited_services=[]):
    if tmp_service_name in visited_services:
        return []
    visited_services.append(tmp_service_name)
    
    service_dependencies_name = []
    service_dependencies_name2 = []
    ret_inputs_name = []
    
    for sample_x_name in all_sample_x_names[tmp_service_name[:-4]]:
        if sample_x_name in all_sample_x_names.keys():
            for p in perf:
                ret_inputs_name += _build_fluxion(sample_x_name+p, visited_services)
                service_dependencies_name.append(sample_x_name)
                service_dependencies_name2.append(sample_x_name+p)
                # ret_inputs_name.append(sample_x_name+p)
        else:
            service_dependencies_name.append(None)
            service_dependencies_name2.append(None)
            ret_inputs_name.append(sample_x_name)
    fluxion.add_service(tmp_service_name[:-4], tmp_service_name, all_lrn_asgmts[tmp_service_name], service_dependencies_name, service_dependencies_name2)
    
    return ret_inputs_name

inputs_name = _build_fluxion(target_service_name)
dump_directory = dump_base_directory+"_with_merge"
zoo.dump(dump_directory)
fluxion.visualize_graph_engine_diagrams(target_service_name[:-4], target_service_name, output_filename="multipercentile", is_draw_edges=True)