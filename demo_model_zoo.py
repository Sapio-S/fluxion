# python3 demo_model_zoo.py

import os, sys

import lib_data

sys.path.insert(1, "../")
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
from GraphEngine.Model.framework_sklearn.random_forest_classifier import RandomForestClassifer
from GraphEngine.Model.framework_sklearn.bayesian_gaussian_mixture import BayesianGaussianMixture
from GraphEngine.Model.framework_rule.merge import Merge
from GraphEngine.ModelZoo.model_zoo import Model_Zoo

num_models_per_experiment = 2
dump_base_directory = "demo_model_zoo"

for model_type in ["GaussianProcess", "MultiLayerPerceptron", "RandomForestClassifer", "BayesianGaussianMixture", "Merge"]:
    print("========== Test \"{}\" ==========".format(model_type))
    dump_directory = os.path.join(dump_base_directory, model_type)
    
    print("[{}] Initialize a zoo".format(os.path.basename(__file__)))
    zoo = Model_Zoo()
    
    sample_x_names = ["CacheType", "BPS", "InsertThreshold"]
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile(["datasamples_mlft.csv"], sample_x_names, ["Latency_99"], [1])
    
    models_training_inputs = []
    models_training_outputs = []
    models_name = []
    for i in range(num_models_per_experiment):
        print("[{}] Initialize and add model \"{}\" to zoo".format(os.path.basename(__file__), i))
        
        if model_type == "GaussianProcess":
            model = GaussianProcess(input_names=sample_x_names)
        elif model_type == "MultiLayerPerceptron":
            model = MultiLayerPerceptron(input_names=sample_x_names, hidden_layer_size=[10,10,10])
        elif model_type == "RandomForestClassifer":
            model = RandomForestClassifer(input_names=sample_x_names)
        elif model_type == "BayesianGaussianMixture":
            model = BayesianGaussianMixture(input_names=sample_x_names)
        elif model_type == "Merge":
            model = Merge(input_names=sample_x_names, operation="avg")
        
        models_name.append(str(i))
        zoo.add(models_name[-1], model)
        models_training_inputs.append(samples_x)
        models_training_outputs.append(samples_y_aggregation)

    for model_inputs, model_outputs, model_name in zip(models_training_inputs, models_training_outputs, models_name):
        print("[{}] Train model \"{}\" (model_inputs={}. model_outputs={})".format(os.path.basename(__file__), model_name, model_inputs, model_outputs))
        zoo.train(model_inputs=model_inputs, model_outputs=model_outputs, model_name=model_name)

    for model_name in models_name:
        model_input = [2.0, 175.0, 450.0]
        output = zoo.predict(model_input, model_name=model_name, return_std=True)
        print("[{}] Predict model \"{}\" (x = {}, y = {})".format(os.path.basename(__file__), model_name, model_input, output))

    # Dump to old zoo
    print("[{}] Dump zoo to \"{}\" directory".format(os.path.basename(__file__), dump_directory))
    zoo.dump(dump_directory)

    # Load to new zoo
    print("[{}] Initialize a new zoo".format(os.path.basename(__file__)))
    new_zoo = Model_Zoo()
    print("[{}] Load previously dumped models to the new zoo".format(os.path.basename(__file__)))
    new_zoo.load(dump_directory)

    # Test new zoo's predict
    for model_name in models_name:
        model_input = [2.0, 175.0, 450.0]
        output = new_zoo.predict(model_input, model_name=model_name, return_std=True)
        print("[{}] Predict model \"{}\" of new zoo (x = {}, y = {})".format(os.path.basename(__file__), model_name, model_input, output))

    # Remove
    print("[{}] Zoo has the following models: {}".format(os.path.basename(__file__), new_zoo.get_models_name()))
    print("[{}] Remove model \"{}\" from the new zoo".format(os.path.basename(__file__), "1"))
    new_zoo.remove("0")
    print("[{}] Zoo has the following models: {}".format(os.path.basename(__file__), new_zoo.get_models_name()))
