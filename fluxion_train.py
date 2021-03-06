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

pretrain_size = 983
test_size = 128
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10


dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/single-standardized.csv"
dataset_filename2 = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2scale-standardized.csv"
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
scale_sample_x_names={}
scale_sample_x_names["recommendationservice:0.90"] = ['recommendation_pod0:0.90','recommendation_pod1:0.90']
scale_sample_x_names["checkoutservice:0.90"] = ['checkout_pod0:0.90','checkout_pod1:0.90']
# real version
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

# # for test only
# dump_directory = "demo_model_zoo_0"
# model_name = {
#     'adservice:0.90':"ce2388c65ade4dd6be623d34f283eb23",
#     'productcatalogservice:0.90':"ebe0255609f446c5bb11acc3dc7ae7c7",
#     'recommendationservice:0.90':"31a395d02150429ca3620926e600a640",
#     'emailservice:0.90':"6d109ded546640eba30b00c763dc52b8",
#     'paymentservice:0.90':"0e22bc6af0004b2199f6d33a6828b750",
#     'shippingservice:0.90':"bbaba457e23c49a8bb00cf5e37611eee",
#     'currencyservice:0.90':"f4e983b0c287459f9248e4249914fec8",
#     'get:0.90':"cef58693b6fa451fb03baaf2f6a6416a",
#     'set:0.90':"dac157312bf24122a960990c2b10688d",
#     'cartservice:0.90':"a6dde3bd9d8448fa90577bb6c11b88c6",
#     'checkoutservice:0.90':"802e8bdd8bf74ee2a683e2520f19c842",
#     'frontend:0.90':"039fec524587444f84d095b37cdf25cb",
# }


def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        if sample_x_name == "recommendationservice:0.90":
            tmp_sample_x_names += expand_sample_x_name("recommendation_pod0:0.90")
            tmp_sample_x_names += expand_sample_x_name("recommendation_pod1:0.90")
        if sample_x_name == "checkoutservice:0.90":
            tmp_sample_x_names += expand_sample_x_name("checkout_pod0:0.90")
            tmp_sample_x_names += expand_sample_x_name("checkout_pod1:0.90")
        elif sample_x_name in all_sample_x_names.keys():
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
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)

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
fluxion.scale_service_horizontal("recommendationservice:0.90", 2)
fluxion.scale_service_horizontal("checkoutservice:0.90",2)

train_sizes = [150]
for train_size in train_sizes:
    small_models_preds = []
    small_models_abs_errs = []
    small_models_raw_errs = []
    fluxion_abs_errs = []
    big_gp_abs_errs = []
    experiment_ids_completed = []
    each_model_errs = {}

    for t in model_name.keys():
        each_model_errs[t] = []

    f = open("log/1019/fluxion_2scale_"+str(train_size),"w")
    sys.stdout = f
    for num_experiments_so_far in range(num_experiments):
        print("========== Experiments finished so far:", num_experiments_so_far, "==========")
        experiment_ids_completed.append(num_experiments_so_far)
        random.seed(42 + num_experiments_so_far)
        numpy.random.seed(42 + num_experiments_so_far)
        
        small_models_preds.append({})
        small_models_abs_errs.append({})
        small_models_raw_errs.append({})
        fluxion_abs_errs.append([])
        big_gp_abs_errs.append([])

        for t in model_name.keys():
            each_model_errs[t].append([])

        # train scaled service
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['recommendation_pod0:0.90','recommendation_pod1:0.90'], "recommendationservice:0.90")
        selected_testing_idxs = random.sample(range(0, len(samples_x)), k=test_size)
        selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
        selected_training_idxs = random.sample(selected_training_idxs, k=train_size)

        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        fluxion.train_service("recommendationservice:0.90", "recommendationservice:0.90", training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
        
        # train scaled service
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['checkout_pod0:0.90','checkout_pod1:0.90'], "checkoutservice:0.90")
        selected_testing_idxs = random.sample(range(0, len(samples_x)), k=test_size)
        selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
        selected_training_idxs = random.sample(selected_training_idxs, k=train_size)

        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        fluxion.train_service("checkoutservice:0.90", "checkoutservice:0.90", training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])

        # STEP 2: Compute Fluxion's testing MAE
        for t in model_name.keys():
            target_service_name = t
            samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], inputs_name+["recommendation_pod0:rps", "recommendation_pod1:rps", "checkout_pod0:rps", "checkout_pod1:rps"], target_service_name)
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
                
                # special for scaled out services
                fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"].append(fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"][0].copy())
                fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"][0]["recommendationservice:rps"] = sample_x[-2] # checkout_pod0
                fluxion_input["recommendationservice:0.90"]["recommendationservice:0.90"][1]["recommendationservice:rps"] = sample_x[-1] # checkout_pod0
                fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"].append(fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0].copy())
                fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][0]["checkoutservice:rps"] = sample_x[-2] # checkout_pod0
                fluxion_input["checkoutservice:0.90"]["checkoutservice:0.90"][1]["checkoutservice:rps"] = sample_x[-1] # checkout_pod0
                pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
                each_model_errs[t][-1].append(abs(pred - sample_y_aggregation))

        # print("==================================================")
        # print("| num_training_data:", train_size)
        # print("| num_testing_data:", test_size)
        # print("| target_deployment_name:", target_deployment_name)
        # print("| target_service_name:", target_service_name)
        # print("| num_experiments:", num_experiments)
        # print("| experiment_ids_completed:", experiment_ids_completed)
        # print("| dataset_filename:", dataset_filename)

        # print("==========")
        # print("| fluxion_abs_errs:")
        
        # print([round(statistics.mean(errs), 8) for errs in fluxion_abs_errs])

    # for data collection
    for t in model_name.keys():
        print(t)
        for errs in each_model_errs[t]:
            print(round(statistics.mean(errs), 8))
    # for errs in fluxion_abs_errs:
    #     print(round(statistics.mean(errs), 8))