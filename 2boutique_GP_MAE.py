# python3 fluxion_vs_monolith.py

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
import numpy as np

num_testing_data = 120
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10

all_sample_x_names={}
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2-screen-standardized.csv"
all_sample_x_names['adservice:0.90'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM"]
all_sample_x_names['productcatalogservice:0.90'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM"]
all_sample_x_names['recommendationservice:0.90'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM",
                                                    "productcatalogservice:0.90"]
all_sample_x_names['emailservice:0.90'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM"]
all_sample_x_names['paymentservice:0.90'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM"]
all_sample_x_names['shippingservice:0.90'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM"]
all_sample_x_names['currencyservice:0.90'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM"]
all_sample_x_names['get:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory']
all_sample_x_names['set:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory']
all_sample_x_names['cartservice:0.90'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", 
                                            "get:0.90", "set:0.90"]
all_sample_x_names['checkoutservice:0.90'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", 
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names['frontend:0.90'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM",
                                        "adservice:0.90", "checkoutservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "recommendationservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]

def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        if sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    return tmp_sample_x_names

# train_size = [10, 25, 50, 100, 150, 200, 300, 450, 550, 650, 800]
train_size=[10, 25, 50, 100, 150, 200,300]
# train_size=[5]
expanded_sample_x_names = expand_sample_x_name(target_service_name)
expanded_sample_x_names = list(set(expanded_sample_x_names))*2
for service in all_sample_x_names.keys():
    expanded_sample_x_names.append(service[:-5]+"_pod0:rps")
    expanded_sample_x_names.append(service[:-5]+"_pod1:rps")
print(len(expanded_sample_x_names))
# dataset_filename2 = "dataset-checkout2-standardized.csv"
# inputs_name = expanded_sample_x_names
# samples_x0, samples_y0, samples_y_aggregation0, err_msg0 = lib_data.readCSVFile([dataset_filename2], expanded_sample_x_names, target_service_name)
for num_training_data in train_size:
    f = open("log/1211/GP_2_screen_"+str(num_training_data),"w")
    sys.stdout = f 

    all_errs = []
    for num_experiments_so_far in range(10):
        print("========== Experiments finished so far:", num_experiments_so_far, "==========")
        random.seed(42 + num_experiments_so_far)
        numpy.random.seed(42 + num_experiments_so_far)
        
        zoo = Model_Zoo()
        fluxion = Fluxion(zoo)
        all_lrn_asgmts = {}
        selected_training_idxs = None
        selected_testing_idxs = None

        errs = []
        # ========== Compute Big models' errors ==========
        # STEP 1: Prepare target services' input names
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], expanded_sample_x_names, target_service_name)
        
        # STEP 2: Determine training and testing indexes
        print(dataset_filename, "has", len(samples_x), "data points")
        selected_testing_idxs = random.sample(range(0, len(samples_x)), k=num_testing_data)
        selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
        selected_training_idxs = random.sample(selected_training_idxs, k=num_training_data)
        
        # STEP 3: Split dataset into training and testing
        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
        testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
        
        # STEP 4: Compute Big-GP's testing MAE
        all_lrn_asgmts['big_gp_model'] = LearningAssignment(zoo, expanded_sample_x_names)
        created_model_name = all_lrn_asgmts['big_gp_model'].create_and_add_model(training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
        for testing_sample_x, testing_sample_y_aggregation in zip(testing_samples_x, testing_samples_y_aggregation):
            pred = all_lrn_asgmts['big_gp_model'].predict(testing_sample_x)['val']
            errs.append(abs(pred - testing_sample_y_aggregation))
        print("test MAE is", np.mean(errs))
        all_errs.append(np.mean(errs))
    print(all_errs)
    print(np.mean(all_errs))
