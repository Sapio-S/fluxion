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
total_data = 523
num_testing_data = 150
test_size=150
pretrain_size=10
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10
# dump_base_directory = "demo_model_zoo"

new_dataset = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-3-90-standardized.csv"
single_dataset = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-100-standardized.csv"
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
retrain_list = ["currencyservice_pod0:0.90", "get_pod0:0.90",'set_pod0:0.90', "cartservice_pod0:0.90"]

x_names = all_sample_x_names.keys()
train_name = {}
scale_podname = []
for service in x_names:
    podname = service[:-5]
    train_name[service] = [podname+"_pod0:0.90",podname+"_pod1:0.90",podname+"_pod2:0.90"]
    scale_podname.append(podname+"_pod0:rps")
    scale_podname.append(podname+"_pod1:rps")
    scale_podname.append(podname+"_pod2:rps")

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


all_lrn_asgmts = {}
selected_training_idxs = None
selected_testing_idxs = None

small_models_preds = []
small_models_abs_errs = []
small_models_raw_errs = []

train_sizes = [100]
# train_sizes=[5]
for train_size in train_sizes:
    all_errs = []
    f = open("log/1213/fluxion_3_retrain_"+str(train_size),"w")
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
        resampled_training_idxs = random.sample(range(0,train_size*2), k=train_size)
        errs = []

        # ========== Compute small models' errors ==========
        for sample_y_name in all_sample_x_names2.keys():
            # STEP 2: Train
            if sample_y_name in retrain_list:            
                sample_x_names = all_sample_x_names2[sample_y_name]
                samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([new_dataset], sample_x_names, sample_y_name)
                
                # STEP 1: Split dataset into training and testing
                training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
                training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
                
                sample_y_name = sample_y_name[:-10]+":0.90"
                sample_x_names = all_sample_x_names[sample_y_name]
                samples_x2, samples_y2, samples_y_aggregation2, err_msg = lib_data.readCSVFile([single_dataset], sample_x_names, sample_y_name)
                
                training_samples_x2 = [samples_x2[idx] for idx in selected_training_idxs]
                training_samples_y_aggregation2 = [samples_y_aggregation2[idx] for idx in selected_training_idxs]
                
                total_training_samples_x = training_samples_x+training_samples_x2
                final_training_samples_x = [total_training_samples_x[idx] for idx in resampled_training_idxs]
                
                total_training_samples_y_aggregation = training_samples_y_aggregation+training_samples_y_aggregation2
                final_training_samples_y_aggregation = [total_training_samples_y_aggregation[idx] for idx in resampled_training_idxs]
                
                all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
                created_model_name = all_lrn_asgmts[sample_y_name].create_and_add_model(final_training_samples_x, final_training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
            else:
                sample_y_name = sample_y_name[:-10]+":0.90"
                sample_x_names = all_sample_x_names[sample_y_name]
                all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
                created_model_name = all_lrn_asgmts[sample_y_name].add_model(model_name[sample_y_name])
            
            # # # STEP 3: Compute MAE with testing dataset
            # small_models_preds[-1][sample_y_name] = []
            # for testing_sample_x in testing_samples_x:
            #     small_models_preds[-1][sample_y_name].append(all_lrn_asgmts[sample_y_name].predict(testing_sample_x)['val'])
            # small_models_raw_errs[-1][sample_y_name] = [t - p for p, t in zip(small_models_preds[-1][sample_y_name], testing_samples_y_aggregation)]
            # small_models_abs_errs[-1][sample_y_name] = [abs(err) for err in small_models_raw_errs[-1][sample_y_name]]
        
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
            fluxion.scale_service_horizontal(service, 3)

        # train scaled service
        for service in x_names:
            samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([new_dataset], train_name[service], service)
            training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
            training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
            fluxion.train_service(service, service, training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False], is_delete_prev_models=True)
        
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
                fluxion_input[service][service].append(fluxion_input[service][service][0].copy())
                pod0_sub = total_name.index(train_name[service][0][:-5]+":rps")
                pod1_sub = total_name.index(train_name[service][1][:-5]+":rps")
                pod2_sub = total_name.index(train_name[service][2][:-5]+":rps")
                fluxion_input[service][service][0][service_name+":rps"] = sample_x[pod0_sub]
                fluxion_input[service][service][1][service_name+":rps"] = sample_x[pod1_sub]
                fluxion_input[service][service][2][service_name+":rps"] = sample_x[pod2_sub]
            pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
            errs.append(abs(pred - sample_y_aggregation))
        print("test MAE",np.mean(errs))
        all_errs.append(np.mean(errs))
    print(all_errs)
    print(np.mean(all_errs))

