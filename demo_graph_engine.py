# python3 demo_graph_engine.py

import sys, random

import lib_data

sys.path.insert(1, "../")
from GraphEngine.graph_engine import GraphEngine
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess

zoo = Model_Zoo()
ge = GraphEngine()

print("========== Create Learning Assignments ==========")
sample_x_names = ["CacheType", "BPS", "InsertThreshold"]
samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile(["datasamples_mlft.csv"], sample_x_names, ["Latency_99"], [1])
la_0 = LearningAssignment(zoo, sample_x_names)
la_0.create_and_add_model(samples_x, samples_y_aggregation, GaussianProcess)
la_1 = LearningAssignment(zoo, sample_x_names)
la_1.create_and_add_model(samples_x, samples_y_aggregation, GaussianProcess)

print("========== Add Nodes ==========")
ge.add_node("la_src", la_0)
ge.add_node("la_dst", la_0)

print("========== Add Edges ==========")
ge.add_edge("la_src", "la_dst", "CacheType")
ge.add_edge("la_src", "la_dst", "BPS")
ge.add_edge("la_src", "la_dst", "InsertThreshold")
try:
    ge.add_edge("la_dst", "la_src", "InsertThreshold")
except RuntimeError as e:
    print(e)
    print("Successfully avoid adding a loop in the graph!")

print("========== Add Edges ==========")
print(ge.visualize_raw("la_dst"))

print("========== Predict ==========")
graph_input = {'la_src': {'CacheType':10, 'BPS':10, 'InsertThreshold':10}}
print(ge.predict("la_dst", graph_input, is_recursive=True, return_std=True))
