# python3 demo_gaussian_process.py

import os, sys

import lib_data

sys.path.insert(1, "../")
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess as Sklearn_GP
from GraphEngine.Model.framework_pytorch.gaussian_process import GaussianProcess as GPyTorch_GP

sample_x_names = ["CacheType", "BPS", "InsertThreshold"]
samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile(["datasamples_mlft.csv"], sample_x_names, ["Latency_99"], [1])

train_inputs = samples_x[:20]
train_outputs = samples_y_aggregation[:20]
pred_inputs = samples_x[20:]
pred_outputs = samples_y_aggregation[20:]

print("===== Train Sklearn GP =====")
sklearn_model = Sklearn_GP(input_names=sample_x_names, num_restarts_optimizer=50)
sklearn_model.train(model_inputs=train_inputs, model_outputs=train_outputs)
print("===== Train GPyTorch GP =====")
gpytorch_model = GPyTorch_GP(input_names=sample_x_names, num_restarts_optimizer=50)
gpytorch_model.train(model_inputs=train_inputs, model_outputs=train_outputs)

print("===== Predict =====")
for pred_input, pred_output in zip(pred_inputs, pred_outputs):
    sklearn_output = sklearn_model.predict(pred_input)
    gpytorch_output = gpytorch_model.predict(pred_input)
    print("Sklearn: %f, GPyTorch: %f, Real: %f" % (sklearn_output['val'], gpytorch_output['val'], pred_output))
