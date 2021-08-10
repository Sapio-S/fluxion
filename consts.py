import numpy as np

route = "res300_1/"
train_size = 150
test_size = 130
services = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice", "redis"]
quantile = ["0.50", '0.90', '0.95', '0.99']
headers = ["service", "rps","avg", "0.50", '0.90', '0.95', '0.99']
perf = ["rps", "avg", "0.50", '0.90', '0.95', '0.99']

# record service dependencies
extra_names = {
    "adservice":[],
    "cartservice":["get", "set"], 
    "checkoutservice": [],
    # "checkoutservice":["emailservice", "paymentservice", "shippingservice", "currencyservice", "productcatalogservice", "cartservice"], 
    "currencyservice":[], 
    "emailservice":[], 
    "frontend":["checkoutservice"],
    # "frontend":["adservice", "checkoutservice", "shippingservice", "currencyservice", "productcatalogservice", "recommendationservice", "cartservice"], 
    "paymentservice":[], 
    "productcatalogservice":[], 
    "recommendationservice":["productcatalogservice"], 
    "shippingservice":[], 
    "get":[], 
    "set":[]
}

finals = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice", "get", "set"]
finals2 = ["checkoutservice", "frontend"]
eval_metric = ["0.90", "0.50", "0.95", "0.99"] # TODO: change eval metric, used in generating prediction

sub_map = np.arange(test_size+train_size) # 支持下标随机
def change_data_order(last_map, i):
    np.random.seed(i)
    global sub_map
    np.random.shuffle(last_map)