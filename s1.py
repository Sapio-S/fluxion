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
# # for i in range(10):
# #     dic = np.load("/home/yuqingxie/autosys/code/fluxion/tmp_data_norm0910/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
# #     print(dic["frontend:0.90:MAX"], dic["frontend:0.90:MIN"])
# dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/single.csv"
# dataset_filename2 = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/2checkout.csv"
# dataset_filename3 = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-2recom.csv"
# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], ['frontend:0.90'], "frontend:0.90")
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['frontend:0.90'], "frontend:0.90")
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename3], ['frontend:0.90'], "frontend:0.90")
# print(np.max(samples_y_aggregation))
# print(np.min(samples_y_aggregation))
# print( np.mean(samples_y_aggregation))
# print( np.std(samples_y_aggregation))

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
for i in range(1):
    dic = np.load("/home/yuqingxie/autosys/code/fluxion/tmp_data_0929valid/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
    print(dic["frontend:0.90:AVG"], dic["frontend:0.90:STD"])
# dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/single.csv"
# dataset_filename2 = "/home/yuqingxie/autosys/code/PlayGround/OSDI22/GoogleBoutique/2checkout.csv"
# dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-whole.csv"
# samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], ['checkoutservice:rps'], "frontend:0.90")
# # print(np.max(samples_y_aggregation), np.min(samples_y_aggregation))
# print(np.std(samples_y_aggregation))
# # samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename2], ['checkoutservice:rps'], "checkoutservice:rps")
# print(np.max(samples_y_aggregation), np.min(samples_y_aggregation))
print('adservice:0.90'[:-5])