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

def multimodel():
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    
    print("========== Create Learning Assignments ==========")
    sample_x, sample_y, x_names, perf_data, test_data, train_data = get_input() # 现在的sample_y只有p50

    extra_names = {
        "adservice":[],
        "cartservice":["get", "set"], 
        "checkoutservice":["emailservice", "paymentservice", "shippingservice", "currencyservice", "productcatalogservice", "cartservice"], 
        "currencyservice":[], 
        "emailservice":[], 
        "frontend":["adservice", "checkoutservice", "shippingservice", "currencyservice", "productcatalogservice", "recommendationservice", "cartservice"], 
        "paymentservice":[], 
        "productcatalogservice":[], 
        "recommendationservice":["productcatalogservice"], 
        "shippingservice":[], 
        "get":[], 
        "set":[]
    }
    
    recommendation_extra_names = ["productcatalogservice"]
    cart_extra_names = ["get", "set"]
    checkout_extra_names = ["emailservice", "paymentservice", "shippingservice", "currencyservice", "productcatalogservice", "cartservice"]
    frontend_extra_names = ["adservice", "checkoutservice", "shippingservice", "currencyservice", "productcatalogservice", "recommendationservice", "cartservice"]
    
    f = "cartservice"
    cart_input = combine_data(cart_extra_names, "0.50", perf_data, sample_x["cartservice"])
    cart = LearningAssignment(zoo, x_names["cartservice"]+cart_extra_names)
    cart.create_and_add_model(cart_input, sample_y["cartservice"], GaussianProcess)
    errs = []
    for i, s in enumerate(cart_input):
        pre = cart.predict(s)["val"]
        real = sample_y["cartservice"][i]
        errs.append(abs(pre-real))
    print("la error for ", f, np.mean(errs))


    recommendation_input = combine_data(recommendation_extra_names, "0.50", perf_data, sample_x["recommendationservice"])
    recommendation = LearningAssignment(zoo, x_names["recommendationservice"]+recommendation_extra_names)
    recommendation.create_and_add_model(recommendation_input, sample_y["recommendationservice"], GaussianProcess)
    
    frontend_input = combine_data(frontend_extra_names, "0.50", perf_data, sample_x["frontend"])
    frontend = LearningAssignment(zoo, x_names["frontend"]+frontend_extra_names)
    frontend.create_and_add_model(frontend_input, sample_y["frontend"], GaussianProcess)

    checkout_input = combine_data(checkout_extra_names, "0.50", perf_data, sample_x["checkoutservice"])
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
    set = LearningAssignment(zoo, x_names["set"])
    set.create_and_add_model(sample_x["set"], sample_y["set"], GaussianProcess)

    print("========== Add Services ==========")
    fluxion.add_service("emailservice", "0.50", email, [None]*len(x_names["emailservice"]), [None]*len(x_names["emailservice"]))
    fluxion.add_service("paymentservice", "0.50", payment, [None]*len(x_names["paymentservice"]), [None]*len(x_names["paymentservice"]))
    fluxion.add_service("shippingservice", "0.50", shipping, [None]*len(x_names["shippingservice"]), [None]*len(x_names["shippingservice"]))
    fluxion.add_service("currencyservice", "0.50", currency, [None]*len(x_names["currencyservice"]), [None]*len(x_names["currencyservice"]))
    fluxion.add_service("productcatalogservice", "0.50", catalog, [None]*len(x_names["productcatalogservice"]), [None]*len(x_names["productcatalogservice"]))
    fluxion.add_service("get", "0.50", get, [None]*len(x_names["get"]), [None]*len(x_names["get"]))
    fluxion.add_service("set", "0.50", set, [None]*len(x_names["set"]), [None]*len(x_names["set"]))
    fluxion.add_service("adservice", "0.50", ad, [None]*len(x_names["adservice"]), [None]*len(x_names["adservice"]))
    fluxion.add_service("cartservice", "0.50", cart, [None]*len(x_names["cartservice"])+["get", "set"], [None]*len(x_names["cartservice"])+["0.50", "0.50"])
    fluxion.add_service("checkoutservice", "0.50", checkout, [None]*len(x_names["checkoutservice"])+["emailservice", "paymentservice", "shippingservice", "currencyservice", "productcatalogservice", "cartservice"], [None]*len(x_names["checkoutservice"])+["0.50", "0.50", "0.50", "0.50", "0.50", "0.50"])
    fluxion.add_service("recommendationservice", "0.50", recommendation, [None]*len(x_names["recommendationservice"])+["productcatalogservice"], [None]*len(x_names["recommendationservice"])+["0.50"])
    fluxion.add_service("frontend", "0.50", frontend, [None, None, None, None, None, "adservice", "checkoutservice", "shippingservice", "currencyservice", "productcatalogservice", "recommendationservice", "cartservice"], [None, None, None, None, None, "0.50", "0.50", "0.50", "0.50", "0.50", "0.50", "0.50"])

    # for f in finals:
    #     errs = []
    #     for i in range(train_size):
    #         prediction = fluxion.predict(f, "0.50", train_data[i])
    #         v1 = prediction[f]["0.50"]["val"]
    #         v2 = perf_data[f+":0.50"][i+test_size]
    #         errs.append(abs(v1-v2))
    #     print(f, "avg error for multimodel", np.mean(errs))
    
    # get train error
    errs = []
    f = "frontend"
    for i in range(train_size):
        prediction = fluxion.predict(f, "0.50", train_data[i])
        v1 = prediction[f]["0.50"]["val"]
        v2 = perf_data[f+":0.50"][i+test_size]
        errs.append(abs(v1-v2))
    print("avg training error for multimodel", np.mean(errs))

    # get test error
    errs = []
    for i in range(test_size):
        prediction = fluxion.predict(f, "0.50", test_data[i])
        v1 = prediction[f]["0.50"]["val"]
        v2 = perf_data[f+":0.50"][i]
        errs.append(abs(v1-v2))
    print("avg test error for multimodel", np.mean(errs))

    # visualize graph
    # fluxion.visualize_graph_engine_diagrams("frontend", "0.50", output_filename="frontend-multi")
    
def singlemodel():
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)

    samples_x, samples_y, x_names, perf_data, test_data, train_data = get_input() # 现在的sample_y只有p50

    paras = []
    for name, value in x_names.items():
        for perf in value:
            paras.append(name+":"+perf)
    
    new_x = []
    for i in range(train_size):
        tmp = []
        for k, v in samples_x.items():
            tmp += v[i]
        new_x.append(tmp)
    
    e2e = LearningAssignment(zoo, paras)
    e2e.create_and_add_model(new_x, perf_data["frontend:0.50"][test_size:test_size+train_size], GaussianProcess) 
    fluxion.add_service("e2e", "0.50", e2e, [None] * len(paras), [None] * len(paras))

    # fluxion.visualize_graph_engine_diagrams("e2e", "0.50", output_filename="e2e-single")

    # get train error
    errs = []
    for i in range(train_size):
        minimap = {}
        for f in finals:
            for k, v in train_data[i][f]["0.50"][0].items():
                minimap[f+":"+k] = v
        prediction = fluxion.predict("e2e", "0.50", {"e2e":{"0.50":[minimap]}})
        v1 = prediction["e2e"]["0.50"]["val"]
        v2 = perf_data["frontend:0.50"][i+test_size]
        errs.append(abs(v1-v2))
    print("avg training error for single model",np.mean(errs))

    # get test error
    errs = []
    for i in range(test_size):
        minimap = {}
        for f in finals:
            for k, v in test_data[i][f]["0.50"][0].items():
                minimap[f+":"+k] = v
        prediction = fluxion.predict("e2e", "0.50", {"e2e":{"0.50":[minimap]}})
        v1 = prediction["e2e"]["0.50"]["val"]
        v2 = perf_data["frontend:0.50"][i]
        print(v1, v2)
        errs.append(abs(v1-v2))
    print("avg test error for single model",np.mean(errs))

if __name__ == "__main__":
    print("train size is", train_size)
    print("test size is", test_size)

    multimodel()
    singlemodel()