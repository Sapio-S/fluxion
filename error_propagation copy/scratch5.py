import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne

from consts import *
from data_new import get_input, get_input_norm, get_input_std, norm_scaler, std_scaler
import numpy as np

def combine_list(list1, list2):
    for i in range(train_size):
        list1[i].append(list2[i])

def combine_data(extra_names, perf_data, x_slice):
    for name in extra_names:
        for perf in eval_metric:
            combine_list(x_slice, perf_data[name+":"+perf])
    return x_slice


def multimodel(sample_x, sample_y, x_names, perf_data, test_data, train_data, train_size, test_size):
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    f2 = "checkoutservice"
    f1 = "paymentservice"
    la_map = {} # {"service":{"perf":la, ...}, ...}
    for f in [f1,f2]:
        la_map[f] = {}
        names = [s1+":"+s2 for s1 in extra_names[f] for s2 in eval_metric]
        # names = []
        f_input = combine_data(extra_names[f], perf_data, sample_x[f])
        for p in eval_metric:
            la = LearningAssignment(zoo, x_names[f]+names)
            la.create_and_add_model(f_input[:][:train_size], sample_y[p][f][:train_size], GaussianProcess)
            la_map[f][p] = la

    names = [n for n in extra_names[f2] for i in range(len(eval_metric))]
    # names = []
    for p in eval_metric:
        fluxion.add_service(f1, p, la_map[f1][p], [None]*len(x_names[f1]), [None]*len(x_names[f1]))
    for p in eval_metric:
        fluxion.add_service(f2, p, la_map[f2][p], [None]*len(x_names[f2]+names), [None]*len(x_names[f2]+names))

    # get test error
    test_err = {}
    low1 = {}
    high1 = {}
    low12 = {}
    high12 = {}
    abs_errs = []
    errs = []
    f1_output = []

    for i in range(test_size):
        f1_input = {}
        f1_input[f1] = test_data[i][f1]
        prediction = fluxion.predict(f1, "0.90", f1_input)
        v1 = prediction[f1]["0.90"]["val"]
        v2 = perf_data[f1+":0.90"][i]
        abs_errs.append(abs(v1-v2))
        errs.append(v1-v2)
        f1_output.append(v1)
        # print(v1, v2)
        # v1 = std_scaler(v1, scale[f1+":0.90:AVG"], scale[f1+":0.90:STD"])
        # v2 = std_scaler(v2, scale[f1+":0.90:AVG"], scale[f1+":0.90:STD"])
    # test_err[f1] = np.mean(abs_errs) # calculate MAE for every service
    mae = np.mean(abs_errs)
    # print("f1 MAE", mae)

    f2_errs = []
    for k in range(test_size):
        f2_input = {}
        f2_input[f2] = test_data[i][f2]
        f2_input[f2]["0.90"][0][f1+":0.90"] = f1_output[i]
        prediction = fluxion.predict(f2, "0.90", f2_input)
        v1 = prediction[f2]["0.90"]["val"]
        v2 = perf_data[f2+":0.90"][i]
        v1 = std_scaler(v1, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
        v2 = std_scaler(v2, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
        f2_errs.append(abs(v1-v2))
    f2_mae = np.mean(f2_errs)

    abs_errs = []
    
    worse_cnt = 0
    better_cnt = 0
    for sample_size in range(2, 50, 2):
        inside = 0
        low1_ = []
        low12_ = []
        high1_ = []
        high12_ = []
        for i in range(test_size):
            f2_input = {}
            f2_input[f2] = test_data[i][f2]
            preds = []
            # # median
            # f2_input[f2]["0.90"][0][f1+":0.90"] = f1_output[i]+np.median(errs)
            # prediction = fluxion.predict(f2, "0.90", f2_input)
            # v1 = prediction[f2]["0.90"]["val"]
            # v2 = perf_data[f2+":0.90"][i]
            # v1 = std_scaler(v1, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            # v2 = std_scaler(v2, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            # preds.append(v1)

            # percentile
            # sample_size = 1
            sample_points = np.random.rand(sample_size)
            for k in range(sample_size):
                val = f1_output[i] - np.percentile(errs, int(sample_points[k] * 10))
                f2_input[f2]["0.90"][0][f1+":0.90"] = val
                prediction = fluxion.predict(f2, "0.90", f2_input)
                v1 = prediction[f2]["0.90"]["val"]
                v2 = perf_data[f2+":0.90"][i]
                v1 = std_scaler(v1, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
                v2 = std_scaler(v2, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
                preds.append(v1)

            # # histogram
            # bin_num = 10
            # sample_points = np.random.rand(sample_size)
            # num, points = np.histogram(errs, bins=bin_num)
            # for k in range(sample_size):
            #     sub_seq = int(sample_points[k]*162)
            #     start = bin_num
            #     cnt = 0
            #     for sub in range(bin_num):
            #         if cnt < sub_seq:
            #             cnt += num[sub]
            #         else:
            #             start = sub
            #             break
            #     val = f1_output[i] - np.random.uniform(points[start-1], points[start], 1)
            #     f2_input[f2]["0.90"][0][f1+":0.90"] = val
            #     prediction = fluxion.predict(f2, "0.90", f2_input)
            #     v1 = prediction[f2]["0.90"]["val"]
            #     v2 = perf_data[f2+":0.90"][i]
            #     v1 = std_scaler(v1, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            #     v2 = std_scaler(v2, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            #     preds.append(v1)


            # # random sample
            # # sample_size = 5
            # sample_points = np.random.normal(0,1,sample_size)
            # for k in range(sample_size):
            #     val = f1_output[i] - mae*sample_points[k]
            #     f2_input[f2]["0.90"][0][f1+":0.90"] = val
            #     prediction = fluxion.predict(f2, "0.90", f2_input)
            #     v1 = prediction[f2]["0.90"]["val"]
            #     v2 = perf_data[f2+":0.90"][i]
            #     v1 = std_scaler(v1, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            #     v2 = std_scaler(v2, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            #     preds.append(v1)

            
            # range
            # sample_points = np.random.uniform(np.min(errs), np.max(errs), sample_size)
            # for k in range(sample_size):
            #     f2_input[f2]["0.90"][0][f1+":0.90"] = f1_output[i]+sample_points[k]
            #     prediction = fluxion.predict(f2, "0.90", f2_input)
            #     v1 = prediction[f2]["0.90"]["val"]
            #     v2 = perf_data[f2+":0.90"][i]
            #     v1 = std_scaler(v1, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            #     v2 = std_scaler(v2, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            #     preds.append(v1)
            # # print(preds)
            # range_low = np.min(preds)
            # range_high = np.max(preds)

            # # original prediction
            # f2_input[f2]["0.90"][0][f1+":0.90"] = f1_output[i]
            # prediction = fluxion.predict(f2, "0.90", f2_input)
            # v1 = prediction[f2]["0.90"]["val"]
            # v2 = perf_data[f2+":0.90"][i]
            # v1 = std_scaler(v1, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            # v2 = std_scaler(v2, scale[f2+":0.90:AVG"], scale[f2+":0.90:STD"])
            # preds.append(v1)
            # ori = v1

            # if range_low - f2_mae <= v2 and v2 <= range_high + f2_mae:
            #     inside += 1
            # low1_.append(range_low)
            # high1_.append(range_high)
            # low12_.append(range_low - f2_mae)
            # high12_.append(range_high + f2_mae)
            # if abs(ori-v2) > abs(np.mean(preds)-v2):
            #     worse_cnt += 1
            # else:
            #     better_cnt += 1
        #     abs_errs.append(abs(np.mean(preds)-v2))
        test_err[sample_size] = np.mean(abs_errs) # calculate MAE for every service
        # test_err[sample_size] = inside
        # low1[sample_size] = np.mean(low1_)
        # high1[sample_size] = np.mean(high1_)
        # low12[sample_size] = np.mean(low12_)
        # high12[sample_size] = np.mean(high12_)
        # print(inside)
        # print(np.mean(abs_errs))
    # print(worse_cnt)
    # print(better_cnt)
    # print(inside)
    return test_err,low1,high1,low12,high12

if __name__ == "__main__":
    train_list = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400]
    # f = open("log/scratch/range-real-m1+m2-bound","w")
    # sys.stdout = f
    for train_sub in range(2,3):
        # f = open("log/scratch/multi_"+str(len(eval_metric))+'_log'+str(train_list[train_sub]),"w")
        # sys.stdout = f
        train_errs = []
        test_errs = {}
        low1s={}
        high1s={}
        low12s={}
        high12s={}
        for f in range(0, 50, 2):
            test_errs[f] = []
            low1s[f] = []
            high1s[f] = []
            low12s[f] = []
            high12s[f] = []

        train_size = train_list[train_sub]
        test_size = 162
        print("train size is", train_size)
        print("test size is", test_size)
        for i in range(10):
            samples_x, samples_y, x_names, perf_data, test_data, train_data, valid_data, scale = get_input_std(i)
            test_err,low1,high1,low12,high12 = multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data, train_size, test_size)
            # train_errs.append(train_err)
            for f in test_err:
                test_errs[f].append(test_err[f])
                # low1s[f].append(low1[f])
                # high1s[f].append(high1[f])
                # low12s[f].append(low12[f])
                # high12s[f].append(high12[f])


        # print("avg train err for 10 times", np.mean(train_errs))
        # print("avg test err for 10 times", np.mean(test_errs["checkoutservice"]))
        for f in test_errs:
            print(np.mean(test_errs[f]))
        print("")
        # for f in test_errs:
        #     print(np.mean(high1s[f]))
        # print("")
        # for f in test_errs:
        #     print(np.mean(low12s[f]))
        # print("")
        # for f in test_errs:
        #     print(np.mean(high12s[f]))
        # print("")