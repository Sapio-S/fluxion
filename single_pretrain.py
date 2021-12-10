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

num_training_data = 600
num_testing_data = 0
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.50"  # "frontend:0.50", "frontend:0.50", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 1
dump_base_directory = "single_model_0.50"
all_sample_x_names = {}
# dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-scale-standardized.csv"
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-100-85-standardized.csv"
all_sample_x_names['adservice:0.50'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
all_sample_x_names['productcatalogservice:0.50'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
all_sample_x_names['recommendationservice:0.50'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice:rps",
                                                    "productcatalogservice:0.50"]
all_sample_x_names['emailservice:0.50'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
all_sample_x_names['paymentservice:0.50'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
all_sample_x_names['shippingservice:0.50'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
all_sample_x_names['currencyservice:0.50'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
all_sample_x_names['get:0.50'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
all_sample_x_names['set:0.50'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
all_sample_x_names['cartservice:0.50'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                            "get:0.50", "set:0.50"]
all_sample_x_names['checkoutservice:0.50'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
                                                "emailservice:0.50", "paymentservice:0.50", "shippingservice:0.50", "currencyservice:0.50", "cartservice:0.50", "productcatalogservice:0.50"]
all_sample_x_names['frontend:0.50'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                        "adservice:0.50", "checkoutservice:0.50", "shippingservice:0.50", "currencyservice:0.50", "recommendationservice:0.50", "cartservice:0.50", "productcatalogservice:0.50"]

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

for num_experiments_so_far in range(num_experiments):
    print("========== Experiments finished so far:", num_experiments_so_far, "==========")
    experiment_ids_completed.append(num_experiments_so_far)
    # random.seed(42 + num_experiments_so_far)
    # numpy.random.seed(42 + num_experiments_so_far)
    
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    all_lrn_asgmts = {}
    selected_training_idxs = range(600)
    # selected_testing_idxs = None
    
    # ========== Compute small models' errors ==========
    for sample_y_name in all_sample_x_names.keys():
        sample_x_names = all_sample_x_names[sample_y_name]
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)

        print(dataset_filename, "has", len(samples_x), "data points")
        # selected_testing_idxs = random.sample(range(0, len(samples_x)), k=num_testing_data)
        # selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
        # selected_training_idxs = random.sample(selected_training_idxs, k=num_training_data)

        # STEP 1: Split dataset into training and testing
        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        # testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
        # testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
        
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
        
        fluxion.add_service(tmp_service_name, tmp_service_name, all_lrn_asgmts[tmp_service_name], service_dependencies_name, service_dependencies_name)
        
        return ret_inputs_name
    
    inputs_name = _build_fluxion(target_service_name)

    dump_directory = dump_base_directory
    zoo.dump(dump_directory)
    
    print("==================================================")
    print("| num_training_data:", num_training_data)
    print("| num_testing_data:", num_testing_data)
    print("| target_deployment_name:", target_deployment_name)
    print("| target_service_name:", target_service_name)
    print("| num_experiments:", num_experiments)
    print("| experiment_ids_completed:", experiment_ids_completed)
    print("| dataset_filename:", dataset_filename)
