
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

def get_output(input1, input2, input3, type):
    output = []
    if type == 1:
        for i in range(len(input1)):
            output.append(0) # constant output
    elif type == 2:
        for i in range(len(input1)):
            output.append((input1[i]+input2[i]+input3[i])/3) # linear combination
    elif type == 3:
        for i in range(len(input1)):
            output.append(0.7*input1[i]+0.2*input2[i]+0.1*input3[i]) # linear combination
    elif type == 4:
        for i in range(len(input1)):
            output.append(10*input1[i]+0.3*input2[i]+0.1*input3[i]) # linear combination
    elif type == 5:
        for i in range(len(input1)):
            output.append(input1[i]**2+input2[i]**2+input3[i]**2) # linear combination
    return output

def get_output_2(input1, input2, input3):
    output = []
    index1 = random.random()
    index2 = random.random()
    index3 = random.random()
    zoom = random.random() * 10
    total = index1+index2+index3
    i1 = index1/total * zoom
    i2 = index2/total * zoom
    i3 = index3/total * zoom
    print("the function is",i1,i2,i3)
    for i in range(len(input1)):
        output.append(i1*input1[i]+i2*input2[i]+i3*input3[i]) # linear combination
    return output

def transpose(grid):
    grid2 = [[float(row[i]) for row in grid] for i in range(len(grid[0]))]
    return grid2

f = open("log/1105/error1_output_noise","w")
sys.stdout = f
for type in range(10):
    print(type)
    random.seed(type)
    np.random.seed(type)
    # simulate inputs
    input1 = np.random.normal(0,1,200)
    input2 = np.random.normal(0,1,200)
    input3 = np.random.normal(0,1,200)

    # generate ground truth
    # output = get_output(input1, input2, input3, type)
    output = get_output_2(input1, input2, input3)
    print(np.std(output))
    print(np.mean(output))

    # input noise
    n1 = np.random.normal(0,0.1,200)
    n2 = np.random.normal(0,0.1,200)
    n3 = np.random.normal(0,0.1,200)
    input1 += n1
    input2 += n2
    input3 += n3

    n_output = np.random.normal(0,0.2,200)
    output += n_output
    print(np.std(output))
    print(np.mean(output))

    # train
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    train_size = 50
    # print("========== Create Learning Assignments ==========")
            
    sample_x2_names = ["4", "5", "6"]
    samples_x2 = transpose([input1[:train_size], input2[:train_size], input3[:train_size]])
    la2 = LearningAssignment(zoo, sample_x2_names)
    la2.create_and_add_model(samples_x2, list(output[:train_size]), GaussianProcess)

    # print("========== Add Services ==========")
    fluxion.add_service("la3", "7", la2, [None, None, None], [None, None, None])

    # print("========== Predict ==========")

    errs = []
    preds = []
    for i in range(train_size, 200):
        fluxion_input = {
                        "la3": {"7": [{"4": input1[i], "5": input2[i], "6": input3[i]}]}
                        }
        prediction = fluxion.predict("la3", "7", fluxion_input)
        # print(prediction["la3"]["7"]["val"], data7[i])
        errs.append(abs(prediction["la3"]["7"]["val"]-output[i]))
        preds.append(prediction["la3"]["7"]["val"])
    # print("prediction error",np.mean(errs))
    print(np.mean(errs))
    # print("prediction data",np.std(preds), np.mean(preds))
    print(np.std(preds))
    print(np.mean(preds))
    print()