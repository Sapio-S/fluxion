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

num_testing_data = 100
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10

# dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2scale-standardized.csv"
all_sample_x_names={}
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-scale-standardized.csv"
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
all_sample_x_names['checkout_pod0:0.90'] = ["checkout_pod0:CPU_LIMIT", "checkout_pod0:MEMORY_LIMIT", "checkout_pod0:IPV4_RMEM", "checkout_pod0:IPV4_WMEM", "checkout_pod0:rps",
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names['checkout_pod1:0.90'] = ["checkout_pod1:CPU_LIMIT", "checkout_pod1:MEMORY_LIMIT", "checkout_pod1:IPV4_RMEM", "checkout_pod1:IPV4_WMEM", "checkout_pod1:rps",
                                                "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
# all_sample_x_names['checkoutservice:0.90'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
#                                                 "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
all_sample_x_names['frontend:0.90'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                        "adservice:0.90", "checkoutservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "recommendationservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
scaler = np.load("std_scaler_dataset_whole.npy", allow_pickle = True).item()

def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        # if sample_x_name == "recommendationservice:0.90":
        #     tmp_sample_x_names += expand_sample_x_name("recommendation_pod0:0.90")
        #     tmp_sample_x_names += expand_sample_x_name("recommendation_pod1:0.90")
        if sample_x_name == "checkoutservice:0.90":
            tmp_sample_x_names += expand_sample_x_name("checkout_pod0:0.90")
            tmp_sample_x_names += expand_sample_x_name("checkout_pod1:0.90")
        if sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    return tmp_sample_x_names

# train_size = [10, 25, 50, 100, 150, 200, 300, 450, 550, 650, 800]
train_size=[10, 25, 50, 100, 150, 200, 300, 400]
expanded_sample_x_names = expand_sample_x_name(target_service_name)
expanded_sample_x_names = list(set(expanded_sample_x_names))
dataset_filename2 = "dataset-checkout2-standardized.csv"
inputs_name = expanded_sample_x_names
samples_x0, samples_y0, samples_y_aggregation0, err_msg0 = lib_data.readCSVFile([dataset_filename2], expanded_sample_x_names, target_service_name)

for num_training_data in train_size:
    
    big_DNN_abs_errs = []
    experiment_ids_completed = []

    for num_experiments_so_far in range(num_experiments):
        f = open("log/1116/DNN_2checkout_"+str(num_training_data)+"_"+str(num_experiments_so_far),"w")
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
        
        # STEP 4: Compute Big-DNN's testing MAE
        all_lrn_asgmts['big_DNN_model'] = LearningAssignment(zoo, expanded_sample_x_names)
        created_model_name = all_lrn_asgmts['big_DNN_model'].create_and_add_model(
            training_samples_x, 
            training_samples_y_aggregation, 
            MultiLayerPerceptron, 
            model_class_args=[
                (497,442,320,27,9), 
                "tanh", # tanh or logistic
                "sgd", 
                1e-4, # alpha
                0.01, # learning_rate_init
                "adaptive", 
                # max_iter
                ]
        )
                # prediction
        preds = []
        # print(samples_x, samples_y_aggregation)
        for testing_sample_x, testing_sample_y_aggregation in zip(samples_x0, samples_y_aggregation0):
            # print(testing_sample_x)
            pred = all_lrn_asgmts['big_DNN_model'].predict(testing_sample_x)['val']
            # pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
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
            if k[:13] == "checkout_pod0" or k[:13] == "checkout_pod1":
                k = "checkoutservice"+k[13:]
            std = scaler[k+":STD"]
            avg = scaler[k+":AVG"]
            if std == 0:
                std = 1
            real_para[k] = samples_x0[best_index][cnt] * std + avg
            cnt += 1
        print(real_para)
        np.save("log/1116/DNN_2checkout_"+str(num_training_data)+"_"+str(num_experiments_so_far), real_para)
        print(best * scaler["frontend:0.90:STD"] + scaler["frontend:0.90:AVG"])
    #     for testing_sample_x, testing_sample_y_aggregation in zip(testing_samples_x, testing_samples_y_aggregation):
    #         pred = all_lrn_asgmts['big_DNN_model'].predict(testing_sample_x)['val']
    #         errs.append(abs(pred - testing_sample_y_aggregation))
    #     print("test MAE is", np.mean(errs))
    #     big_DNN_abs_errs.append(np.mean(errs))
    # print(big_DNN_abs_errs)
    # for kkk in big_DNN_abs_errs:
    #     print(kkk)
