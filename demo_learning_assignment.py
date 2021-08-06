# python3 demo_learning_assignment.py

import sys, random

import lib_data

sys.path.insert(1, "../")
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess

zoo = Model_Zoo()

print("========== Create Learning Assignments ==========")
sample_x_names = ["CacheType", "BPS", "InsertThreshold"]
samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile(["datasamples_mlft.csv"], sample_x_names, ["Latency_99"], [1])
learning_assignment = LearningAssignment(zoo, sample_x_names)

print("Create and add a model (with model selection)...")
learning_assignment.create_and_add_model(samples_x, samples_y_aggregation, GaussianProcess, max_inputs_selected=2)
print("Create and add a model (without model selection)...")
learning_assignment.create_and_add_model(samples_x, samples_y_aggregation, GaussianProcess)
print("Learning assignment has the following models:", learning_assignment.get_models_name())
print("Zoo has the following models:", zoo.get_models_name())

print("========== Fit ensemble ==========")
weights = learning_assignment.fit_ensemble(samples_x, samples_y_aggregation, [[0, 20], [0, 20]])
print("Model weights:", weights)

print("========== Predict ==========")
print("Predict with only the most recent model...")
print(learning_assignment.predict([1, 2, 3]))
print("Predict with weighted ensemble...")
print(learning_assignment.predict_ensemble([1, 2, 3]))
