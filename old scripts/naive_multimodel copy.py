import os, sys

sys.path.insert(1, "../")
from fluxion import Fluxion
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
# from IngestionEngine_CSV import ingestionEngipythonne

from consts import *
from data import get_input
import numpy as np

def combine_list(list1, list2):
    for i in range(train_size): # TODO: change to 300
        list1[i].append(list2[i])


def combine_data(extra_names, perf, perf_data, x_slice):
    # new_name_list = []
    for name in extra_names:
        # new_name_list.append(name+":"+perf)
        combine_list(x_slice, perf_data[name+":"+perf])
    return x_slice

def multimodel(sample_x, sample_y, x_names, perf_data, test_data, train_data):
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    sample_y = sample_y["0.90"]
    recommendation_extra_names = ["productcatalogservice"]
    cart_extra_names = ["get", "set"]
    checkout_extra_names = ["emailservice", "paymentservice", "shippingservice", "currencyservice", "productcatalogservice", "cartservice"]
    frontend_extra_names = ["adservice", "checkoutservice", "shippingservice", "currencyservice", "productcatalogservice", "recommendationservice", "cartservice"]
    
    cart_input = combine_data(cart_extra_names, "0.90", perf_data, sample_x["cartservice"])
    cart = LearningAssignment(zoo, x_names["cartservice"]+cart_extra_names)
    cart.create_and_add_model(cart_input, sample_y["cartservice"], GaussianProcess)

    recommendation_input = combine_data(recommendation_extra_names, "0.90", perf_data, sample_x["recommendationservice"])
    recommendation = LearningAssignment(zoo, x_names["recommendationservice"]+recommendation_extra_names)
    recommendation.create_and_add_model(recommendation_input, sample_y["recommendationservice"], GaussianProcess)
    
    frontend_input = combine_data(frontend_extra_names, "0.90", perf_data, sample_x["frontend"])
    frontend = LearningAssignment(zoo, x_names["frontend"]+frontend_extra_names)
    frontend.create_and_add_model(frontend_input, sample_y["frontend"], GaussianProcess)

    checkout_input = combine_data(checkout_extra_names, "0.90", perf_data, sample_x["checkoutservice"])
    checkout = LearningAssignment(zoo, x_names["checkoutservice"]+checkout_extra_names)
    checkout.create_and_add_model(checkout_input, sample_y["checkoutservice"], GaussianProcess)

    ad = LearningAssignment(zoo, x_names["adservice"])
    ad.create_and_add_model(sample_x["adservice"], sample_y["adservice"], GaussianProcess)
    email = LearningAssignment(zoo, x_names["emailservice"])
    email.create_and_add_model(sample_x["emailservice"], sample_y["emailservice"], GaussianProcess)
    payment = LearningAssignment(zoo, x_names["paymentservice"])
    payment.create_and_add_model(sample_x["paymentservice"], sample_y["paymentservice"], GaussianProcess)
    shipping = LearningAssignment(zoo, x_names["shippingservice"])
    shipping.create_and_add_model(sample_x["shippingservice"], sample_y["shippingservice"], GaussianProcess)
    currency = LearningAssignment(zoo, x_names["currencyservice"])
    currency.create_and_add_model(sample_x["currencyservice"], sample_y["currencyservice"], GaussianProcess)
    catalog = LearningAssignment(zoo, x_names["productcatalogservice"])
    catalog.create_and_add_model(sample_x["productcatalogservice"], sample_y["productcatalogservice"], GaussianProcess)
    get = LearningAssignment(zoo, x_names["get"])
    get.create_and_add_model(sample_x["get"], sample_y["get"], GaussianProcess)
    set_ = LearningAssignment(zoo, x_names["set"])
    set_.create_and_add_model(sample_x["set"], sample_y["set"], GaussianProcess)

    fluxion.add_service("emailservice", "0.90", email, [None]*len(x_names["emailservice"]), [None]*len(x_names["emailservice"]))
    fluxion.add_service("paymentservice", "0.90", payment, [None]*len(x_names["paymentservice"]), [None]*len(x_names["paymentservice"]))
    fluxion.add_service("shippingservice", "0.90", shipping, [None]*len(x_names["shippingservice"]), [None]*len(x_names["shippingservice"]))
    fluxion.add_service("currencyservice", "0.90", currency, [None]*len(x_names["currencyservice"]), [None]*len(x_names["currencyservice"]))
    fluxion.add_service("productcatalogservice", "0.90", catalog, [None]*len(x_names["productcatalogservice"]), [None]*len(x_names["productcatalogservice"]))
    fluxion.add_service("get", "0.90", get, [None]*len(x_names["get"]), [None]*len(x_names["get"]))
    fluxion.add_service("set", "0.90", set_, [None]*len(x_names["set"]), [None]*len(x_names["set"]))
    fluxion.add_service("adservice", "0.90", ad, [None]*len(x_names["adservice"]), [None]*len(x_names["adservice"]))
    fluxion.add_service("cartservice", "0.90", cart, [None]*len(x_names["cartservice"])+["get", "set"], [None]*len(x_names["cartservice"])+["0.90", "0.90"])
    fluxion.add_service("checkoutservice", "0.90", checkout, [None]*len(x_names["checkoutservice"])+["emailservice", "paymentservice", "shippingservice", "currencyservice", "productcatalogservice", "cartservice"], [None]*len(x_names["checkoutservice"])+["0.90", "0.90", "0.90", "0.90", "0.90", "0.90"])
    fluxion.add_service("recommendationservice", "0.90", recommendation, [None]*len(x_names["recommendationservice"])+["productcatalogservice"], [None]*len(x_names["recommendationservice"])+["0.90"])
    fluxion.add_service("frontend", "0.90", frontend, [None, None, None, None, None, "adservice", "checkoutservice", "shippingservice", "currencyservice", "productcatalogservice", "recommendationservice", "cartservice"], [None, None, None, None, None, "0.90", "0.90", "0.90", "0.90", "0.90", "0.90", "0.90"])


    # get whole train error
    for f in finals2:
        errs = []
        for i in range(train_size):
            prediction = fluxion.predict(f, "0.90", train_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i+test_size]
            errs.append(abs(v1-v2))
        train_err = np.mean(errs)

    # get test error
    test_err = {}
    for f in finals2:
        errs = []
        for i in range(test_size):
            prediction = fluxion.predict(f, "0.90", test_data[i])
            v1 = prediction[f]["0.90"]["val"]
            v2 = perf_data[f+":0.90"][i]
            errs.append(abs(v1-v2))
        test_err[f] = np.mean(errs) # calculate MAE for every service

    return train_err, test_err

if __name__ == "__main__":
    train_list = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # f = open("log/graph0814/multi_"+str(len(eval_metric))+'_log',"w")
    # sys.stdout = f

    for train_sub in range(1):
        train_errs = []
        test_errs = {}
        for f in finals2:
            test_errs[f] = []

        train_size = 25
        test_size = 118
        print("train size is", train_size)
        print("test size is", test_size)
        for i in range(1):
            samples_x, samples_y, x_names, perf_data, test_data, train_data = get_input(i)
            train_err, test_err = multimodel(samples_x, samples_y, x_names, perf_data, test_data, train_data)
            train_errs.append(train_err)
            for f in finals2:
                test_errs[f].append(test_err[f])

        print("avg train err for 10 times", np.mean(train_errs))
        print("avg test err for 10 times", np.mean(test_errs["frontend"]))
        # print(train_errs)
        # print(test_errs)
        for f in finals2:
            print(f, np.mean(test_errs[f]))
        print("")