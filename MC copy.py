
import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne

import numpy as np

if __name__ == "__main__":
    train_size = 50
    test_size = 50

    # prepare data
    x = np.arange(100,110,0.1)
    x_out = np.arange(110,100,-0.1)
    y = np.arange(100,110,0.1)
    z = [x[i]*y[i] for i in range(100)]
    np.random.seed(0)
    np.random.shuffle(x)
    np.random.seed(0)
    np.random.shuffle(x_out)
    np.random.seed(0)
    np.random.shuffle(y)
    np.random.seed(0)
    np.random.shuffle(z)

    x += np.random.normal(0,0.5,100)
    x_out += np.random.normal(0,0.5,100)
    y += np.random.normal(0,0.5,100)
    z += np.random.normal(0,0.5,100)

    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    # train model_1: x -> x_out, f(x) = -x+5
    m1_la = LearningAssignment(zoo, ["x"])
    m1_la.create_and_add_model([[i] for i in x[:train_size]], list(x_out[:train_size]), GaussianProcess)
    # m1_la.set_err_dist([[i] for i in x[train_size:]], list(x_out[train_size:]))
    fluxion.add_service("f", "x_out", m1_la, [None], [None])

    # train model_2: x_out, y -> z, g(x_out, y) = x_out * y
    m2_la = LearningAssignment(zoo, ["x_out", "y"])
    m2_la.create_and_add_model([[x_out[i], y[i]] for i in range(train_size)], list(z[:train_size]), GaussianProcess)
    # m2_la.set_err_dist([[x_out[i], y[i]] for i in range(train_size,100)], list(z[train_size:]))
    # fluxion.add_service("g", "z", m2_la, ["f", None], ["x_out", None])
    fluxion.add_service("g", "z", m2_la, [None, None], [None, None])
    
    test_err = []
    real_err = []


    m1_err = []
    f1_output = []
    for i in range(test_size):
        prediction = fluxion.predict("f", "x_out", {"f":{"x_out":[{"x":x[i+train_size]}]}})
        v1 = prediction["f"]["x_out"]["val"]
        m1_err.append(v1-x_out[i+train_size])
        f1_output.append(v1)

                # # histogram
    sample_size = 50
    bin_num = 10
    for i in range(test_size):
        sample_points = np.random.rand(sample_size)
        num, points = np.histogram(m1_err, bins=bin_num)
        # pointsss = []
        this_preds = []
        for k in range(sample_size):
            sub_seq = int(sample_points[k]*50)
            start = bin_num
            cnt = 0
            for sub in range(bin_num):
                if cnt < sub_seq:
                    cnt += num[sub]
                else:
                    start = sub
                    break
            val = f1_output[i] - np.random.uniform(points[start-1], points[start], 1) # sample a point

            # pointsss.append(val)
            prediction = fluxion.predict("g", "z", {"g":{"z":[{"y":y[i+train_size], "x_out":val}]}})
            v1 = prediction["g"]["z"]["val"]
            this_preds.append(v1)
        v1 = np.mean(this_preds)
        v2 = z[i+train_size]
        v3 = x[i+train_size] * y[i+train_size]
        test_err.append(abs(v1-v2))
        real_err.append(abs(v1-v3))

    # # test result

    # for i in range(test_size):
    #     x_ = x[i+train_size]
    #     y_ = y[i+train_size]
    #     prediction = fluxion.predict("g", "z", {"f":{"x_out":[{"x":x_}]},"g":{"z":[{"y":y_}]}})
    #     v1 = prediction["g"]["z"]["val"]
    #     # print(prediction["g"]["z"])
    #     v2 = z[i+train_size]
    #     v3 = x[i+train_size] * y_
    #     test_err.append(abs(v1-v2))
    #     real_err.append(abs(v1-v3))
    #     # print(prediction["g"]["z"]["val"], v2)
    
    print(np.mean(test_err))
    print(np.mean(real_err))

    
