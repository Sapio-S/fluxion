route = "res/"
train_size = 10
test_size = 130
services = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice", "redis"]
quantile = ["0.50", '0.90', '0.95', '0.99']
headers = ["service", "rps","avg", "0.50", '0.90', '0.95', '0.99']
perf = ["rps", "avg", "0.50", '0.90', '0.95', '0.99']

finals = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice", "get", "set"]

eval_metric = ["0.50"] # TODO: change eval metric, used in generating prediction