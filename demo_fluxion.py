# python3 demo_fluxion.py

import os, sys

import lib_data

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from IngestionEngine_CSV import ingestionEngine

if __name__ == "__main__":
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    
    print("========== Create Learning Assignments ==========")
    sample_x_names = ["CacheType", "BPS", "InsertThreshold"]
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile(["datasamples_mlft.csv"], sample_x_names, ["Latency_99"], [1])
    print(samples_x)
    print(samples_y_aggregation)
    la1 = LearningAssignment(zoo, sample_x_names)
    la1.create_and_add_model(samples_x, samples_y_aggregation, GaussianProcess)
    la2 = LearningAssignment(zoo, ["CacheType", "BPS", "InsertThreshold"])
    la2.create_and_add_model(samples_x, samples_y_aggregation, GaussianProcess)
    la3 = LearningAssignment(zoo, ["CacheType", "BPS", "InsertThreshold"])
    la3.create_and_add_model(samples_x, samples_y_aggregation, GaussianProcess)
    
    print("========== Add Services ==========")
    fluxion.add_service("la1", "Latency_99", la1, [None, None, None], [None, None, None])
    fluxion.add_service("la2", "Latency_99", la2, ["la1", None, None], ["Latency_99", None, None])
    fluxion.add_service("la2", "Latency_50", la2, ["la1", None, None], ["Latency_99", None, None])
    fluxion.add_service("la3", "Latency_99", la2, ["la1", "la2", "la2"], ["Latency_99", "Latency_50", "Latency_99"])
    
    print("========== Scale Services ==========")
    fluxion.visualize_graph_engine_diagrams("la3", "Latency_99", output_filename="before_scaling")
    fluxion.scale_service_horizontal("la2", 2)
    fluxion.visualize_graph_engine_diagrams("la3", "Latency_99", output_filename="after_scaling")
    
    print("========== Test \"visualize_graph_engine_raw\" ==========")
    print("Print la1:")
    print(fluxion.visualize_graph_engine_raw("la1", "Latency_99"))
    print("Print la2:")
    print(fluxion.visualize_graph_engine_raw("la2", "Latency_99"))
    
    print("========== Test \"get_graph_inputs_name\" ==========")
    print(fluxion.get_graph_inputs_name("la1", "Latency_99"))
    print(fluxion.get_graph_inputs_name("la2", "Latency_99"))
