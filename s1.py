import json, sys, random, statistics

sys.path.insert(1, "../")
# sys.path.insert(1, "../../Dem/o")
from fluxion import Fluxion
import lib_data
import numpy as np
from GraphEngine.learning_assignment import LearningAssignment
import GraphEngine.lib_learning_assignment as lib_learning_assignment
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
finals = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice", "get", "set"]

# # for i in range(10):
# #     dic = np.load("/home/yuqingxie/autosys/code/fluxion/tmp_data_norm0910/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
# #     print(dic["frontend:0.90:MAX"], dic["frontend:0.90:MIN"])
dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-100.csv"
dataset_filename2 = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2-whole.csv"
# dataset_filename3="/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2checkout-150.csv"
# dataset_filename4="/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-150-checkout1.csv"
# dataset_filename3 = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2recom.csv"
samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], ['frontend:0.90'], "frontend:0.90")
print(dataset_filename)
print(np.max(samples_y_aggregation))
print(np.min(samples_y_aggregation))
print( np.mean(samples_y_aggregation))
print( np.std(samples_y_aggregation))

samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], ['frontend:rps'], "frontend:rps")
print(dataset_filename)
print( np.mean(samples_y_aggregation))

samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['frontend:0.90'], "frontend:0.90")
print(dataset_filename2)
print(np.max(samples_y_aggregation))
print(np.min(samples_y_aggregation))
print( np.mean(samples_y_aggregation))
print( np.std(samples_y_aggregation))

samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['frontend:rps'], "frontend:rps")
print(dataset_filename2)
print( np.mean(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename3], ['frontend:0.90'], "frontend:0.90")
# print(dataset_filename3)
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename3], ['frontend:rps'], "frontend:rps")
# print(dataset_filename3)
# print( np.mean(samples_y_aggregation))
# dataset_filename4 = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2-standardized.csv"
# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename4], ['frontend:0.90'], "frontend:0.90")
# print(dataset_filename4)
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename4], ['frontend:rps'], "frontend:rps")
# print(dataset_filename4)
# print( np.mean(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename4], ['frontend:0.95'], "frontend:0.95")
# print(dataset_filename4)
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# from consts import *
# for f in finals:
#     name = f +"_pod0:rps"
#     samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename4], [name], name)
#     print(f,np.mean(samples_y_aggregation))




# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], ['recommendationservice:0.90'], "recommendationservice:0.90")
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename3], ['recommendationservice:0.90'], "recommendationservice:0.90")
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], ['recommendationservice:0.90'], "recommendationservice:rps")
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename3], ['recommendationservice:0.90'], "recommendationservice:rps")
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# import json, sys, random, statistics

# sys.path.insert(1, "../")
# # sys.path.insert(1, "../../Dem/o")
# from fluxion import Fluxion
# import lib_data
# import numpy as np
# from GraphEngine.learning_assignment import LearningAssignment
# import GraphEngine.lib_learning_assignment as lib_learning_assignment
# from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
# from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron


# for i in range(1):
#     dic = np.load("/home/yuqingxie/autosys/code/fluxion/tmp_data_0929valid/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
#     print(dic["frontend:0.90:AVG"], dic["frontend:0.90:STD"])
# # dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/single.csv"
# # dataset_filename2 = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/2checkout.csv"
# # dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-whole.csv"
# # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], ['checkoutservice:rps'], "frontend:0.90")
# # # print(np.max(samples_y_aggregation), np.min(samples_y_aggregation))
# # print(np.std(samples_y_aggregation))
# # # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['checkoutservice:rps'], "checkoutservice:rps")
# # print(np.max(samples_y_aggregation), np.min(samples_y_aggregation))
# print('adservice:0.90'[:-5])
# l = np.load("saved_model/dic10_0.npy", allow_pickle = True)
# print(l)



# random.seed(42 + 0)
# np.random.seed(42 + 0)
# selected_testing_idxs = random.sample(range(0, 10), k=5)
# selected_training_idxs = set(range(0, 10)) - set(selected_testing_idxs)
# selected_training_idxs = random.sample(selected_training_idxs, k=1)
# print(selected_training_idxs)




# x = np.load("res/wrk_para.npy", allow_pickle = True)
# print(x)