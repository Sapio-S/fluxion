
import numpy as np
import os, random, sys
sys.path.insert(1, "../")
from fluxion import Fluxion
import lib_data
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from IngestionEngine_CSV import ingestionEngine

def get_output(inputs):
    index = []
    for i in range(len(inputs)):
        index.append(random.random())
    total = np.sum(index)
    index2 = []
    for i in index:
        index2.append(i/total)
    print("the index of the function is", index2)
    output = [np.sum([index2[k]*inputs[k][i] for k in range(len(inputs))]) for i in range(len(inputs[0]))]
    return output

def transpose(grid):
    grid2 = [[float(row[i]) for row in grid] for i in range(len(grid[0]))]
    return grid2

inputs = []
real_inputs = []
f = open("log/1105/error2_no_noise","w")
sys.stdout = f
for type in range(10):
    print(type)
    random.seed(type)
    np.random.seed(type)
    
    # simulate inputs
    inputs.append(np.random.normal(0,1,200))
    # generate ground truth
    output = get_output(inputs)
    # print("ground truth",np.std(output), np.mean(output))
    print(np.std(output))
    print(np.mean(output))
    # input noise
    n = np.random.normal(0,0.1,200)
    # real_inputs.append(n+inputs[type])
    real_inputs.append(inputs[type])
    # n_output = np.random.normal(0,0.2,200)
    # output += n_output
    # # print("added output noise",np.std(output), np.mean(output))
    # print(np.std(output))
    # print(np.mean(output))

    # train
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    train_size = 50
    # print("========== Create Learning Assignments ==========")
            
    sample_x2_names = [str(t) for t in range(type+1)]
    samples_x2 = transpose([real_inputs[i][:train_size] for i in range(type+1)])
    la2 = LearningAssignment(zoo, sample_x2_names)
    la2.create_and_add_model(samples_x2, list(output[:train_size]), GaussianProcess)

    # print("========== Add Services ==========")
    fluxion.add_service("la3", "11", la2, [None]*(type+1), [None]*(type+1))

    # print("========== Predict ==========")

    errs = []
    preds = []
    for i in range(train_size, 200):
        dic = {}
        for k in range(type+1):
            dic[str(k)] = real_inputs[k][i]
        fluxion_input = {
                        "la3": {"11": [dic]}
                        }
        prediction = fluxion.predict("la3", "11", fluxion_input)
        # print(prediction["la3"]["11"]["val"], data11[i])
        errs.append(abs(prediction["la3"]["11"]["val"]-output[i]))
        preds.append(prediction["la3"]["11"]["val"])
    # print("prediction error",np.mean(errs))
    print(np.mean(errs))
    # print("prediction data",np.std(preds), np.mean(preds))
    print(np.std(preds))
    print(np.mean(preds))
    print()