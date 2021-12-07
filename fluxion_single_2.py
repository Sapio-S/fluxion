# python3 fluxion_vs_monolith.py

import json, numpy, sys, random, statistics

sys.path.insert(1, "../")
sys.path.insert(1, "../Demo")
from fluxion import Fluxion
import lib_data
from GraphEngine.learning_assignment import LearningAssignment
import GraphEngine.lib_learning_assignment as lib_learning_assignment
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron

num_training_data = 10
num_testing_data = 86
target_deployment_name = "boutique_p90_p90"  # "boutique_p90_p90", "boutique_p95_p95", "hotel_p90_p90", "hotel_p95_p95", "hotel_p90_p50p85p90p95"
target_service_name = "frontend:0.90"  # "frontend:0.90", "frontend:0.95", "wrk|frontend|overall|lat-90", "wrk|frontend|overall|lat-95"
num_experiments = 10

all_sample_x_names = {}
if target_deployment_name == "boutique_p90_p90":
    #dataset_filename = "../OSDI22/GoogleBoutique/single-normalized.csv"
    dataset_filename = "/home/yuqingxie/autosys/code/PlayGround/yuqingxie/dataset-150-standardized.csv"
    
    all_sample_x_names['adservice:0.90'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
    all_sample_x_names['productcatalogservice:0.90'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
    all_sample_x_names['recommendationservice:0.90'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice:rps",
                                                        "productcatalogservice:0.90"]
    all_sample_x_names['emailservice:0.90'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
    all_sample_x_names['paymentservice:0.90'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
    all_sample_x_names['shippingservice:0.90'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
    all_sample_x_names['currencyservice:0.90'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
    all_sample_x_names['get:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
    all_sample_x_names['set:0.90'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
    all_sample_x_names['cartservice:0.90'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                              "get:0.90", "set:0.90"]
    all_sample_x_names['checkoutservice:0.90'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
                                                  "emailservice:0.90", "paymentservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]
    all_sample_x_names['frontend:0.90'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                           "adservice:0.90", "checkoutservice:0.90", "shippingservice:0.90", "currencyservice:0.90", "recommendationservice:0.90", "cartservice:0.90", "productcatalogservice:0.90"]

elif target_deployment_name == "boutique_p95_p95":
    #dataset_filename = "../OSDI22/GoogleBoutique/single-normalized.csv"
    dataset_filename = "../OSDI22/GoogleBoutique/single-standardized.csv"
    
    all_sample_x_names['adservice:0.95'] = ["adservice:MAX_ADS_TO_SERVE", "adservice:CPU_LIMIT", "adservice:MEMORY_LIMIT", "adservice:IPV4_RMEM", "adservice:IPV4_WMEM", "adservice:rps"]
    all_sample_x_names['productcatalogservice:0.95'] = ["productcatalogservice:CPU_LIMIT", "productcatalogservice:MEMORY_LIMIT", "productcatalogservice:IPV4_RMEM", "productcatalogservice:IPV4_WMEM", "productcatalogservice:rps"]
    all_sample_x_names['recommendationservice:0.95'] = ["recommendationservice:CPU_LIMIT", "recommendationservice:MEMORY_LIMIT", "recommendationservice:MAX_WORKERS", "recommendationservice:MAX_RESPONSE", "recommendationservice:IPV4_RMEM", "recommendationservice:IPV4_WMEM", "recommendationservice:rps",
                                                        "productcatalogservice:0.95"]
    all_sample_x_names['emailservice:0.95'] = ["emailservice:CPU_LIMIT", "emailservice:MEMORY_LIMIT", "emailservice:MAX_WORKERS", "emailservice:IPV4_RMEM", "emailservice:IPV4_WMEM", "emailservice:rps"]
    all_sample_x_names['paymentservice:0.95'] = ["paymentservice:CPU_LIMIT", "paymentservice:MEMORY_LIMIT", "paymentservice:IPV4_RMEM", "paymentservice:IPV4_WMEM", "paymentservice:rps"]
    all_sample_x_names['shippingservice:0.95'] = ["shippingservice:CPU_LIMIT", "shippingservice:MEMORY_LIMIT", "shippingservice:IPV4_RMEM", "shippingservice:IPV4_WMEM", "shippingservice:rps"]
    all_sample_x_names['currencyservice:0.95'] = ["currencyservice:CPU_LIMIT", "currencyservice:MEMORY_LIMIT", "currencyservice:IPV4_RMEM", "currencyservice:IPV4_WMEM", "currencyservice:rps"]
    all_sample_x_names['get:0.95'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'get:rps']
    all_sample_x_names['set:0.95'] = ['get:CPU_LIMIT', 'get:MEMORY_LIMIT', 'get:IPV4_RMEM', 'get:IPV4_WMEM', 'get:hash_max_ziplist_entries', 'get:maxmemory_samples', 'get:maxmemory', 'set:rps']
    all_sample_x_names['cartservice:0.95'] = ["cartservice:CPU_LIMIT", "cartservice:MEMORY_LIMIT", "cartservice:IPV4_RMEM", "cartservice:IPV4_WMEM", "cartservice:rps",
                                              "get:0.95", "set:0.95"]
    all_sample_x_names['checkoutservice:0.95'] = ["checkoutservice:CPU_LIMIT", "checkoutservice:MEMORY_LIMIT", "checkoutservice:IPV4_RMEM", "checkoutservice:IPV4_WMEM", "checkoutservice:rps",
                                                  "emailservice:0.95", "paymentservice:0.95", "shippingservice:0.95", "currencyservice:0.95", "cartservice:0.95", "productcatalogservice:0.95"]
    all_sample_x_names['frontend:0.95'] = ["frontend:CPU_LIMIT", "frontend:MEMORY_LIMIT", "frontend:IPV4_RMEM", "frontend:IPV4_WMEM", "frontend:rps",
                                           "adservice:0.95", "checkoutservice:0.95", "shippingservice:0.95", "currencyservice:0.95", "recommendationservice:0.95", "cartservice:0.95", "productcatalogservice:0.95"]

elif target_deployment_name == "hotel_p90_p90":
    dataset_filename = "../OSDI22/DeathStarBench/HotelReservation/v3/single-rps_50_300-standardized/data.csv"
    
    # Reservation
    all_sample_x_names['reservation|mgo|scan|lat-90'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                         "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                         "reservation|mgo|scan|rps", "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|insert|lat-90'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                           "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                           "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|memc|read|lat-90'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                          "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|insert|lat-90'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|update|lat-90'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names["frontend|reservation|overall|lat-90"] = ["k:reservation:net.ipv4.tcp_rmem", "k:reservation:net.ipv4.tcp_wmem", "frontend|reservation|overall|rps",
                                                                 'reservation|mgo|scan|lat-90', 'reservation|mgo|insert|lat-90',
                                                                 'reservation|memc|read|lat-90', 'reservation|memc|insert|lat-90', 'reservation|memc|update|lat-90']
    # Geo
    all_sample_x_names["search|geo|overall|lat-90"] = ["k:geo:net.ipv4.tcp_rmem", "k:geo:net.ipv4.tcp_wmem",
                                                       "search|geo|overall|rps"]
    # Profile
    all_sample_x_names['profile|mgo|read|lat-90'] = ["c:mongodb-profile:cache", "c:mongodb-profile:eviction_dirty_target", "c:mongodb-profile:eviction_dirty_trigger",                                                                                      "k:mongodb-profile:net.ipv4.tcp_rmem", "k:mongodb-profile:net.ipv4.tcp_wmem",
                                                     "profile|mgo|read|rps"]
    all_sample_x_names["profile|memc|read|lat-90"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                      "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|insert|lat-90"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                        "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["frontend|profile|overall|lat-90"] = ["k:profile:net.ipv4.tcp_rmem", "k:profile:net.ipv4.tcp_wmem",
                                                             "frontend|profile|overall|rps",
                                                             "profile|mgo|read|lat-90", "profile|memc|read|lat-90", "profile|memc|insert|lat-90"]
    # Rate
    all_sample_x_names["rate|mgo|scan|lat-90"] = ["c:mongodb-rate:cache", "c:mongodb-rate:eviction_dirty_target", "c:mongodb-rate:eviction_dirty_trigger",
                                                  "k:mongodb-rate:net.ipv4.tcp_rmem", "k:mongodb-rate:net.ipv4.tcp_wmem",
                                                  "rate|mgo|scan|rps"]
    all_sample_x_names["rate|memc|read|lat-90"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                   "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|insert|lat-90"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                     "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["search|rate|overall|lat-90"] = ["k:rate:net.ipv4.tcp_rmem", "k:rate:net.ipv4.tcp_wmem",
                                                        "search|rate|overall|rps",
                                                        "rate|mgo|scan|lat-90", "rate|memc|read|lat-90", "rate|memc|insert|lat-90"]
    # Search
    all_sample_x_names["frontend|search|overall|lat-90"] = ["k:search:net.ipv4.tcp_rmem", "k:search:net.ipv4.tcp_wmem",
                                                            "frontend|search|overall|rps",
                                                            "search|rate|overall|lat-90", "search|geo|overall|lat-90"]
    # User
    all_sample_x_names["user|mgo|read|lat-90"] = ["c:mongodb-user:cache", "c:mongodb-user:eviction_dirty_target", "c:mongodb-user:eviction_dirty_trigger",
                                                  "k:mongodb-user:net.ipv4.tcp_rmem", "k:mongodb-user:net.ipv4.tcp_wmem",
                                                  "user|mgo|read|rps"]
    all_sample_x_names["frontend|user|overall|lat-90"] = ["k:user:net.ipv4.tcp_rmem", "k:user:net.ipv4.tcp_wmem",
                                                          "frontend|user|overall|rps",
                                                          "user|mgo|read|lat-90"]
    # Recommendation
    all_sample_x_names["frontend|recommendation|overall|lat-90"] = ["k:recommendation:net.ipv4.tcp_rmem", "k:recommendation:net.ipv4.tcp_wmem",
                                                                    "frontend|recommendation|overall|rps"]
    # Frontend
    all_sample_x_names["wrk|frontend|overall|lat-90"] = ["wrk|frontend|overall|rps",
                                                         "frontend|recommendation|overall|lat-90", "frontend|reservation|overall|lat-90", "frontend|profile|overall|lat-90", "frontend|search|overall|lat-90", "frontend|user|overall|lat-90"]

elif target_deployment_name == "hotel_p95_p95":
    dataset_filename = "../OSDI22/DeathStarBench/HotelReservation/v3/single-rps_50_300-standardized/data.csv"
    
    # Reservation
    all_sample_x_names['reservation|mgo|scan|lat-95'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                         "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                         "reservation|mgo|scan|rps", "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|insert|lat-95'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                           "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                           "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|memc|read|lat-95'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                          "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|insert|lat-95'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|update|lat-95'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names["frontend|reservation|overall|lat-95"] = ["k:reservation:net.ipv4.tcp_rmem", "k:reservation:net.ipv4.tcp_wmem", "frontend|reservation|overall|rps",
                                                                 'reservation|mgo|scan|lat-95', 'reservation|mgo|insert|lat-95',
                                                                 'reservation|memc|read|lat-95', 'reservation|memc|insert|lat-95', 'reservation|memc|update|lat-95']
    # Geo
    all_sample_x_names["search|geo|overall|lat-95"] = ["k:geo:net.ipv4.tcp_rmem", "k:geo:net.ipv4.tcp_wmem",
                                                       "search|geo|overall|rps"]
    # Profile
    all_sample_x_names['profile|mgo|read|lat-95'] = ["c:mongodb-profile:cache", "c:mongodb-profile:eviction_dirty_target", "c:mongodb-profile:eviction_dirty_trigger",                                                                                      "k:mongodb-profile:net.ipv4.tcp_rmem", "k:mongodb-profile:net.ipv4.tcp_wmem",
                                                     "profile|mgo|read|rps"]
    all_sample_x_names["profile|memc|read|lat-95"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                      "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|insert|lat-95"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                        "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["frontend|profile|overall|lat-95"] = ["k:profile:net.ipv4.tcp_rmem", "k:profile:net.ipv4.tcp_wmem",
                                                             "frontend|profile|overall|rps",
                                                             "profile|mgo|read|lat-95", "profile|memc|read|lat-95", "profile|memc|insert|lat-95"]
    # Rate
    all_sample_x_names["rate|mgo|scan|lat-95"] = ["c:mongodb-rate:cache", "c:mongodb-rate:eviction_dirty_target", "c:mongodb-rate:eviction_dirty_trigger",
                                                  "k:mongodb-rate:net.ipv4.tcp_rmem", "k:mongodb-rate:net.ipv4.tcp_wmem",
                                                  "rate|mgo|scan|rps"]
    all_sample_x_names["rate|memc|read|lat-95"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                   "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|insert|lat-95"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                     "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["search|rate|overall|lat-95"] = ["k:rate:net.ipv4.tcp_rmem", "k:rate:net.ipv4.tcp_wmem",
                                                        "search|rate|overall|rps",
                                                        "rate|mgo|scan|lat-95", "rate|memc|read|lat-95", "rate|memc|insert|lat-95"]
    # Search
    all_sample_x_names["frontend|search|overall|lat-95"] = ["k:search:net.ipv4.tcp_rmem", "k:search:net.ipv4.tcp_wmem",
                                                            "frontend|search|overall|rps",
                                                            "search|rate|overall|lat-95", "search|geo|overall|lat-95"]
    # User
    all_sample_x_names["user|mgo|read|lat-95"] = ["c:mongodb-user:cache", "c:mongodb-user:eviction_dirty_target", "c:mongodb-user:eviction_dirty_trigger",
                                                  "k:mongodb-user:net.ipv4.tcp_rmem", "k:mongodb-user:net.ipv4.tcp_wmem",
                                                  "user|mgo|read|rps"]
    all_sample_x_names["frontend|user|overall|lat-95"] = ["k:user:net.ipv4.tcp_rmem", "k:user:net.ipv4.tcp_wmem",
                                                          "frontend|user|overall|rps",
                                                          "user|mgo|read|lat-95"]
    # Recommendation
    all_sample_x_names["frontend|recommendation|overall|lat-95"] = ["k:recommendation:net.ipv4.tcp_rmem", "k:recommendation:net.ipv4.tcp_wmem",
                                                                    "frontend|recommendation|overall|rps"]
    # Frontend
    all_sample_x_names["wrk|frontend|overall|lat-95"] = ["wrk|frontend|overall|rps",
                                                         "frontend|recommendation|overall|lat-95", "frontend|reservation|overall|lat-95", "frontend|profile|overall|lat-95", "frontend|search|overall|lat-95", "frontend|user|overall|lat-95"]

elif target_deployment_name == "hotel_p90_p50p85p90p95":
    dataset_filename = "../OSDI22/DeathStarBench/HotelReservation/v3/single-rps_50_300-standardized/data.csv"
    
    # Reservation
    all_sample_x_names['reservation|mgo|scan|lat-50'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                         "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                         "reservation|mgo|scan|rps", "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|scan|lat-90'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                         "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                         "reservation|mgo|scan|rps", "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|scan|lat-95'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                         "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                         "reservation|mgo|scan|rps", "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|scan|lat-85'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                         "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                         "reservation|mgo|scan|rps", "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|insert|lat-50'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                           "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                           "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|insert|lat-90'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                           "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                           "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|insert|lat-95'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                           "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                           "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|mgo|insert|lat-85'] = ["c:mongodb-reservation:cache", "c:mongodb-reservation:eviction_dirty_target", "c:mongodb-reservation:eviction_dirty_trigger",
                                                           "k:mongodb-reservation:net.ipv4.tcp_rmem", "k:mongodb-reservation:net.ipv4.tcp_wmem",
                                                           "reservation|mgo|insert|rps"]
    all_sample_x_names['reservation|memc|read|lat-50'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                          "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|read|lat-90'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                          "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|read|lat-95'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                          "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|read|lat-85'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                          "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|insert|lat-50'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|insert|lat-90'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|insert|lat-95'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|insert|lat-85'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|update|lat-50'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|update|lat-90'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|update|lat-95'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names['reservation|memc|update|lat-85'] = ["c:memcached-reserve:memory-limit", "c:memcached-reserve:threads", "c:memcached-reserve:slab-growth-factor",
                                                            "reservation|memc|insert|rps", "reservation|memc|update|rps", "reservation|memc|read|rps"]
    all_sample_x_names["frontend|reservation|overall|lat-50"] = ["k:reservation:net.ipv4.tcp_rmem", "k:reservation:net.ipv4.tcp_wmem", "frontend|reservation|overall|rps",
                                                                 'reservation|mgo|scan|lat-50', 'reservation|mgo|scan|lat-90', 'reservation|mgo|scan|lat-95', 'reservation|mgo|scan|lat-85',
                                                                 'reservation|mgo|insert|lat-50', 'reservation|mgo|insert|lat-90', 'reservation|mgo|insert|lat-95', 'reservation|mgo|insert|lat-85',
                                                                 'reservation|memc|read|lat-50', 'reservation|memc|read|lat-90', 'reservation|memc|read|lat-95', 'reservation|memc|read|lat-85',
                                                                 'reservation|memc|insert|lat-50', 'reservation|memc|insert|lat-90', 'reservation|memc|insert|lat-95', 'reservation|memc|insert|lat-85',
                                                                 'reservation|memc|update|lat-50', 'reservation|memc|update|lat-90', 'reservation|memc|update|lat-95', 'reservation|memc|update|lat-85']
    all_sample_x_names["frontend|reservation|overall|lat-90"] = ["k:reservation:net.ipv4.tcp_rmem", "k:reservation:net.ipv4.tcp_wmem", "frontend|reservation|overall|rps",
                                                                 'reservation|mgo|scan|lat-50', 'reservation|mgo|scan|lat-90', 'reservation|mgo|scan|lat-95', 'reservation|mgo|scan|lat-85',
                                                                 'reservation|mgo|insert|lat-50', 'reservation|mgo|insert|lat-90', 'reservation|mgo|insert|lat-95', 'reservation|mgo|insert|lat-85',
                                                                 'reservation|memc|read|lat-50', 'reservation|memc|read|lat-90', 'reservation|memc|read|lat-95', 'reservation|memc|read|lat-85',
                                                                 'reservation|memc|insert|lat-50', 'reservation|memc|insert|lat-90', 'reservation|memc|insert|lat-95', 'reservation|memc|insert|lat-85',
                                                                 'reservation|memc|update|lat-50', 'reservation|memc|update|lat-90', 'reservation|memc|update|lat-95', 'reservation|memc|update|lat-85']
    all_sample_x_names["frontend|reservation|overall|lat-95"] = ["k:reservation:net.ipv4.tcp_rmem", "k:reservation:net.ipv4.tcp_wmem", "frontend|reservation|overall|rps",
                                                                 'reservation|mgo|scan|lat-50', 'reservation|mgo|scan|lat-90', 'reservation|mgo|scan|lat-95', 'reservation|mgo|scan|lat-85',
                                                                 'reservation|mgo|insert|lat-50', 'reservation|mgo|insert|lat-90', 'reservation|mgo|insert|lat-95', 'reservation|mgo|insert|lat-85',
                                                                 'reservation|memc|read|lat-50', 'reservation|memc|read|lat-90', 'reservation|memc|read|lat-95', 'reservation|memc|read|lat-85',
                                                                 'reservation|memc|insert|lat-50', 'reservation|memc|insert|lat-90', 'reservation|memc|insert|lat-95', 'reservation|memc|insert|lat-85',
                                                                 'reservation|memc|update|lat-50', 'reservation|memc|update|lat-90', 'reservation|memc|update|lat-95', 'reservation|memc|update|lat-85']
    all_sample_x_names["frontend|reservation|overall|lat-85"] = ["k:reservation:net.ipv4.tcp_rmem", "k:reservation:net.ipv4.tcp_wmem", "frontend|reservation|overall|rps",
                                                                 'reservation|mgo|scan|lat-50', 'reservation|mgo|scan|lat-90', 'reservation|mgo|scan|lat-95', 'reservation|mgo|scan|lat-85',
                                                                 'reservation|mgo|insert|lat-50', 'reservation|mgo|insert|lat-90', 'reservation|mgo|insert|lat-95', 'reservation|mgo|insert|lat-85',
                                                                 'reservation|memc|read|lat-50', 'reservation|memc|read|lat-90', 'reservation|memc|read|lat-95', 'reservation|memc|read|lat-85',
                                                                 'reservation|memc|insert|lat-50', 'reservation|memc|insert|lat-90', 'reservation|memc|insert|lat-95', 'reservation|memc|insert|lat-85',
                                                                 'reservation|memc|update|lat-50', 'reservation|memc|update|lat-90', 'reservation|memc|update|lat-95', 'reservation|memc|update|lat-85']
    
    # Geo
    all_sample_x_names["search|geo|overall|lat-50"] = ["k:geo:net.ipv4.tcp_rmem", "k:geo:net.ipv4.tcp_wmem",
                                                       "search|geo|overall|rps"]
    all_sample_x_names["search|geo|overall|lat-90"] = ["k:geo:net.ipv4.tcp_rmem", "k:geo:net.ipv4.tcp_wmem",
                                                       "search|geo|overall|rps"]
    all_sample_x_names["search|geo|overall|lat-95"] = ["k:geo:net.ipv4.tcp_rmem", "k:geo:net.ipv4.tcp_wmem",
                                                       "search|geo|overall|rps"]
    all_sample_x_names["search|geo|overall|lat-85"] = ["k:geo:net.ipv4.tcp_rmem", "k:geo:net.ipv4.tcp_wmem",
                                                       "search|geo|overall|rps"]
    
    # Profile
    all_sample_x_names['profile|mgo|read|lat-50'] = ["c:mongodb-profile:cache", "c:mongodb-profile:eviction_dirty_target", "c:mongodb-profile:eviction_dirty_trigger",                                                                                      "k:mongodb-profile:net.ipv4.tcp_rmem", "k:mongodb-profile:net.ipv4.tcp_wmem",
                                                     "profile|mgo|read|rps"]
    all_sample_x_names['profile|mgo|read|lat-90'] = ["c:mongodb-profile:cache", "c:mongodb-profile:eviction_dirty_target", "c:mongodb-profile:eviction_dirty_trigger",                                                                                      "k:mongodb-profile:net.ipv4.tcp_rmem", "k:mongodb-profile:net.ipv4.tcp_wmem",
                                                     "profile|mgo|read|rps"]
    all_sample_x_names['profile|mgo|read|lat-95'] = ["c:mongodb-profile:cache", "c:mongodb-profile:eviction_dirty_target", "c:mongodb-profile:eviction_dirty_trigger",                                                                                      "k:mongodb-profile:net.ipv4.tcp_rmem", "k:mongodb-profile:net.ipv4.tcp_wmem",
                                                     "profile|mgo|read|rps"]
    all_sample_x_names['profile|mgo|read|lat-85'] = ["c:mongodb-profile:cache", "c:mongodb-profile:eviction_dirty_target", "c:mongodb-profile:eviction_dirty_trigger",                                                                                      "k:mongodb-profile:net.ipv4.tcp_rmem", "k:mongodb-profile:net.ipv4.tcp_wmem",
                                                     "profile|mgo|read|rps"]
    all_sample_x_names["profile|memc|read|lat-50"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                      "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|read|lat-90"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                      "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|read|lat-95"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                      "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|read|lat-85"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                      "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|insert|lat-50"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                        "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|insert|lat-90"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                        "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|insert|lat-95"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                        "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["profile|memc|insert|lat-85"] = ["c:memcached-profile:memory-limit", "c:memcached-profile:threads", "c:memcached-profile:slab-growth-factor",
                                                        "profile|memc|read|rps", "profile|memc|insert|rps"]
    all_sample_x_names["frontend|profile|overall|lat-50"] = ["k:profile:net.ipv4.tcp_rmem", "k:profile:net.ipv4.tcp_wmem",
                                                             "frontend|profile|overall|rps",
                                                             "profile|mgo|read|lat-50", "profile|mgo|read|lat-90", "profile|mgo|read|lat-95", "profile|mgo|read|lat-85",
                                                             "profile|memc|read|lat-50", "profile|memc|read|lat-90", "profile|memc|read|lat-95", "profile|memc|read|lat-85",
                                                             "profile|memc|insert|lat-50", "profile|memc|insert|lat-90", "profile|memc|insert|lat-95", "profile|memc|insert|lat-85"]
    all_sample_x_names["frontend|profile|overall|lat-90"] = ["k:profile:net.ipv4.tcp_rmem", "k:profile:net.ipv4.tcp_wmem",
                                                             "frontend|profile|overall|rps",
                                                             "profile|mgo|read|lat-50", "profile|mgo|read|lat-90", "profile|mgo|read|lat-95", "profile|mgo|read|lat-85",
                                                             "profile|memc|read|lat-50", "profile|memc|read|lat-90", "profile|memc|read|lat-95", "profile|memc|read|lat-85",
                                                             "profile|memc|insert|lat-50", "profile|memc|insert|lat-90", "profile|memc|insert|lat-95", "profile|memc|insert|lat-85"]
    all_sample_x_names["frontend|profile|overall|lat-95"] = ["k:profile:net.ipv4.tcp_rmem", "k:profile:net.ipv4.tcp_wmem",
                                                             "frontend|profile|overall|rps",
                                                             "profile|mgo|read|lat-50", "profile|mgo|read|lat-90", "profile|mgo|read|lat-95", "profile|mgo|read|lat-85",
                                                             "profile|memc|read|lat-50", "profile|memc|read|lat-90", "profile|memc|read|lat-95", "profile|memc|read|lat-85",
                                                             "profile|memc|insert|lat-50", "profile|memc|insert|lat-90", "profile|memc|insert|lat-95", "profile|memc|insert|lat-85"]
    all_sample_x_names["frontend|profile|overall|lat-85"] = ["k:profile:net.ipv4.tcp_rmem", "k:profile:net.ipv4.tcp_wmem",
                                                             "frontend|profile|overall|rps",
                                                             "profile|mgo|read|lat-50", "profile|mgo|read|lat-90", "profile|mgo|read|lat-95", "profile|mgo|read|lat-85",
                                                             "profile|memc|read|lat-50", "profile|memc|read|lat-90", "profile|memc|read|lat-95", "profile|memc|read|lat-85",
                                                             "profile|memc|insert|lat-50", "profile|memc|insert|lat-90", "profile|memc|insert|lat-95", "profile|memc|insert|lat-85"]
    
    # Rate
    all_sample_x_names["rate|mgo|scan|lat-50"] = ["c:mongodb-rate:cache", "c:mongodb-rate:eviction_dirty_target", "c:mongodb-rate:eviction_dirty_trigger",
                                                  "k:mongodb-rate:net.ipv4.tcp_rmem", "k:mongodb-rate:net.ipv4.tcp_wmem",
                                                  "rate|mgo|scan|rps"]
    all_sample_x_names["rate|mgo|scan|lat-90"] = ["c:mongodb-rate:cache", "c:mongodb-rate:eviction_dirty_target", "c:mongodb-rate:eviction_dirty_trigger",
                                                  "k:mongodb-rate:net.ipv4.tcp_rmem", "k:mongodb-rate:net.ipv4.tcp_wmem",
                                                  "rate|mgo|scan|rps"]
    all_sample_x_names["rate|mgo|scan|lat-95"] = ["c:mongodb-rate:cache", "c:mongodb-rate:eviction_dirty_target", "c:mongodb-rate:eviction_dirty_trigger",
                                                  "k:mongodb-rate:net.ipv4.tcp_rmem", "k:mongodb-rate:net.ipv4.tcp_wmem",
                                                  "rate|mgo|scan|rps"]
    all_sample_x_names["rate|mgo|scan|lat-85"] = ["c:mongodb-rate:cache", "c:mongodb-rate:eviction_dirty_target", "c:mongodb-rate:eviction_dirty_trigger",
                                                  "k:mongodb-rate:net.ipv4.tcp_rmem", "k:mongodb-rate:net.ipv4.tcp_wmem",
                                                  "rate|mgo|scan|rps"]
    all_sample_x_names["rate|memc|read|lat-50"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                   "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|read|lat-90"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                   "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|read|lat-95"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                   "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|read|lat-85"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                   "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|insert|lat-50"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                     "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|insert|lat-90"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                     "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|insert|lat-95"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                     "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["rate|memc|insert|lat-85"] = ["c:memcached-rate:memory-limit", "c:memcached-rate:threads", "c:memcached-rate:slab-growth-factor",
                                                     "rate|memc|read|rps", "rate|memc|insert|rps"]
    all_sample_x_names["search|rate|overall|lat-50"] = ["k:rate:net.ipv4.tcp_rmem", "k:rate:net.ipv4.tcp_wmem",
                                                        "search|rate|overall|rps",
                                                        "rate|mgo|scan|lat-50", "rate|mgo|scan|lat-90", "rate|mgo|scan|lat-95", "rate|mgo|scan|lat-85",
                                                        "rate|memc|read|lat-50", "rate|memc|read|lat-90", "rate|memc|read|lat-95", "rate|memc|read|lat-85",
                                                        "rate|memc|insert|lat-50", "rate|memc|insert|lat-90", "rate|memc|insert|lat-95", "rate|memc|insert|lat-85"]
    all_sample_x_names["search|rate|overall|lat-90"] = ["k:rate:net.ipv4.tcp_rmem", "k:rate:net.ipv4.tcp_wmem",
                                                        "search|rate|overall|rps",
                                                        "rate|mgo|scan|lat-50", "rate|mgo|scan|lat-90", "rate|mgo|scan|lat-95", "rate|mgo|scan|lat-85",
                                                        "rate|memc|read|lat-50", "rate|memc|read|lat-90", "rate|memc|read|lat-95", "rate|memc|read|lat-85",
                                                        "rate|memc|insert|lat-50", "rate|memc|insert|lat-90", "rate|memc|insert|lat-95", "rate|memc|insert|lat-85"]
    all_sample_x_names["search|rate|overall|lat-95"] = ["k:rate:net.ipv4.tcp_rmem", "k:rate:net.ipv4.tcp_wmem",
                                                        "search|rate|overall|rps",
                                                        "rate|mgo|scan|lat-50", "rate|mgo|scan|lat-90", "rate|mgo|scan|lat-95", "rate|mgo|scan|lat-85",
                                                        "rate|memc|read|lat-50", "rate|memc|read|lat-90", "rate|memc|read|lat-95", "rate|memc|read|lat-85",
                                                        "rate|memc|insert|lat-50", "rate|memc|insert|lat-90", "rate|memc|insert|lat-95", "rate|memc|insert|lat-85"]
    all_sample_x_names["search|rate|overall|lat-85"] = ["k:rate:net.ipv4.tcp_rmem", "k:rate:net.ipv4.tcp_wmem",
                                                        "search|rate|overall|rps",
                                                        "rate|mgo|scan|lat-50", "rate|mgo|scan|lat-90", "rate|mgo|scan|lat-95", "rate|mgo|scan|lat-85",
                                                        "rate|memc|read|lat-50", "rate|memc|read|lat-90", "rate|memc|read|lat-95", "rate|memc|read|lat-85",
                                                        "rate|memc|insert|lat-50", "rate|memc|insert|lat-90", "rate|memc|insert|lat-95", "rate|memc|insert|lat-85"]
    
    # Search
    all_sample_x_names["frontend|search|overall|lat-50"] = ["k:search:net.ipv4.tcp_rmem", "k:search:net.ipv4.tcp_wmem",
                                                            "frontend|search|overall|rps",
                                                            "search|rate|overall|lat-50", "search|rate|overall|lat-90", "search|rate|overall|lat-95", "search|rate|overall|lat-85",
                                                            "search|geo|overall|lat-50", "search|geo|overall|lat-90", "search|geo|overall|lat-95", "search|geo|overall|lat-85"]
    all_sample_x_names["frontend|search|overall|lat-90"] = ["k:search:net.ipv4.tcp_rmem", "k:search:net.ipv4.tcp_wmem",
                                                            "frontend|search|overall|rps",
                                                            "search|rate|overall|lat-50", "search|rate|overall|lat-90", "search|rate|overall|lat-95", "search|rate|overall|lat-85",
                                                            "search|geo|overall|lat-50", "search|geo|overall|lat-90", "search|geo|overall|lat-95", "search|geo|overall|lat-85"]
    all_sample_x_names["frontend|search|overall|lat-95"] = ["k:search:net.ipv4.tcp_rmem", "k:search:net.ipv4.tcp_wmem",
                                                            "frontend|search|overall|rps",
                                                            "search|rate|overall|lat-50", "search|rate|overall|lat-90", "search|rate|overall|lat-95", "search|rate|overall|lat-85",
                                                            "search|geo|overall|lat-50", "search|geo|overall|lat-90", "search|geo|overall|lat-95", "search|geo|overall|lat-85"]
    all_sample_x_names["frontend|search|overall|lat-85"] = ["k:search:net.ipv4.tcp_rmem", "k:search:net.ipv4.tcp_wmem",
                                                            "frontend|search|overall|rps",
                                                            "search|rate|overall|lat-50", "search|rate|overall|lat-90", "search|rate|overall|lat-95", "search|rate|overall|lat-85",
                                                            "search|geo|overall|lat-50", "search|geo|overall|lat-90", "search|geo|overall|lat-95", "search|geo|overall|lat-85"]
    
    # User
    all_sample_x_names["user|mgo|read|lat-50"] = ["c:mongodb-user:cache", "c:mongodb-user:eviction_dirty_target", "c:mongodb-user:eviction_dirty_trigger",
                                                  "k:mongodb-user:net.ipv4.tcp_rmem", "k:mongodb-user:net.ipv4.tcp_wmem",
                                                  "user|mgo|read|rps"]
    all_sample_x_names["user|mgo|read|lat-90"] = ["c:mongodb-user:cache", "c:mongodb-user:eviction_dirty_target", "c:mongodb-user:eviction_dirty_trigger",
                                                  "k:mongodb-user:net.ipv4.tcp_rmem", "k:mongodb-user:net.ipv4.tcp_wmem",
                                                  "user|mgo|read|rps"]
    all_sample_x_names["user|mgo|read|lat-95"] = ["c:mongodb-user:cache", "c:mongodb-user:eviction_dirty_target", "c:mongodb-user:eviction_dirty_trigger",
                                                  "k:mongodb-user:net.ipv4.tcp_rmem", "k:mongodb-user:net.ipv4.tcp_wmem",
                                                  "user|mgo|read|rps"]
    all_sample_x_names["user|mgo|read|lat-85"] = ["c:mongodb-user:cache", "c:mongodb-user:eviction_dirty_target", "c:mongodb-user:eviction_dirty_trigger",
                                                  "k:mongodb-user:net.ipv4.tcp_rmem", "k:mongodb-user:net.ipv4.tcp_wmem",
                                                  "user|mgo|read|rps"]
    all_sample_x_names["frontend|user|overall|lat-50"] = ["k:user:net.ipv4.tcp_rmem", "k:user:net.ipv4.tcp_wmem",
                                                          "frontend|user|overall|rps",
                                                          "user|mgo|read|lat-50", "user|mgo|read|lat-90", "user|mgo|read|lat-95", "user|mgo|read|lat-85"]
    all_sample_x_names["frontend|user|overall|lat-90"] = ["k:user:net.ipv4.tcp_rmem", "k:user:net.ipv4.tcp_wmem",
                                                          "frontend|user|overall|rps",
                                                          "user|mgo|read|lat-50", "user|mgo|read|lat-90", "user|mgo|read|lat-95", "user|mgo|read|lat-85"]
    all_sample_x_names["frontend|user|overall|lat-95"] = ["k:user:net.ipv4.tcp_rmem", "k:user:net.ipv4.tcp_wmem",
                                                          "frontend|user|overall|rps",
                                                          "user|mgo|read|lat-50", "user|mgo|read|lat-90", "user|mgo|read|lat-95", "user|mgo|read|lat-85"]
    all_sample_x_names["frontend|user|overall|lat-85"] = ["k:user:net.ipv4.tcp_rmem", "k:user:net.ipv4.tcp_wmem",
                                                          "frontend|user|overall|rps",
                                                          "user|mgo|read|lat-50", "user|mgo|read|lat-90", "user|mgo|read|lat-95", "user|mgo|read|lat-85"]
    
    # Recommendation
    all_sample_x_names["frontend|recommendation|overall|lat-50"] = ["k:recommendation:net.ipv4.tcp_rmem", "k:recommendation:net.ipv4.tcp_wmem",
                                                                    "frontend|recommendation|overall|rps"]
    all_sample_x_names["frontend|recommendation|overall|lat-90"] = ["k:recommendation:net.ipv4.tcp_rmem", "k:recommendation:net.ipv4.tcp_wmem",
                                                                    "frontend|recommendation|overall|rps"]
    all_sample_x_names["frontend|recommendation|overall|lat-95"] = ["k:recommendation:net.ipv4.tcp_rmem", "k:recommendation:net.ipv4.tcp_wmem",
                                                                    "frontend|recommendation|overall|rps"]
    all_sample_x_names["frontend|recommendation|overall|lat-85"] = ["k:recommendation:net.ipv4.tcp_rmem", "k:recommendation:net.ipv4.tcp_wmem",
                                                                    "frontend|recommendation|overall|rps"]
    
    # Frontend
    all_sample_x_names["wrk|frontend|overall|lat-90"] = ["wrk|frontend|overall|rps",
                                                         "frontend|recommendation|overall|lat-50", "frontend|recommendation|overall|lat-90", "frontend|recommendation|overall|lat-95", "frontend|recommendation|overall|lat-85",
                                                         "frontend|reservation|overall|lat-50", "frontend|reservation|overall|lat-90", "frontend|reservation|overall|lat-95", "frontend|reservation|overall|lat-85",
                                                         "frontend|profile|overall|lat-50", "frontend|profile|overall|lat-90", "frontend|profile|overall|lat-95", "frontend|profile|overall|lat-85",
                                                         "frontend|search|overall|lat-50", "frontend|search|overall|lat-90", "frontend|search|overall|lat-95", "frontend|search|overall|lat-85",
                                                         "frontend|user|overall|lat-50", "frontend|user|overall|lat-90", "frontend|user|overall|lat-95", "frontend|user|overall|lat-85"]

def expand_sample_x_name(service_name):
    tmp_sample_x_names = []
    for sample_x_name_idx, sample_x_name in zip(range(len(all_sample_x_names[service_name])), all_sample_x_names[service_name]):
        if sample_x_name in all_sample_x_names.keys():
            tmp_sample_x_names += expand_sample_x_name(sample_x_name)
        else:
            tmp_sample_x_names.append(sample_x_name)
    
    return tmp_sample_x_names

small_models_preds = []
small_models_abs_errs = []
small_models_raw_errs = []
fluxion_abs_errs = []
big_gp_abs_errs = []
experiment_ids_completed = []

for num_experiments_so_far in range(num_experiments):
    print("========== Experiments finished so far:", num_experiments_so_far, "==========")
    experiment_ids_completed.append(num_experiments_so_far)
    random.seed(42 + num_experiments_so_far)
    numpy.random.seed(42 + num_experiments_so_far)
    
    zoo = Model_Zoo()
    fluxion = Fluxion(zoo)
    all_lrn_asgmts = {}
    selected_training_idxs = None
    selected_testing_idxs = None
    
    small_models_preds.append({})
    small_models_abs_errs.append({})
    small_models_raw_errs.append({})
    fluxion_abs_errs.append([])
    big_gp_abs_errs.append([])
    
    # ========== Compute Big models' errors ==========
    # STEP 1: Prepare target services' input names
    expanded_sample_x_names = expand_sample_x_name(target_service_name)
    expanded_sample_x_names = list(set(expanded_sample_x_names))
    print("Big-* models have", len(expanded_sample_x_names), "inputs")
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], expanded_sample_x_names, target_service_name)
    
    # STEP 2: Determine training and testing indexes
    print(dataset_filename, "has", len(samples_x), "data points")
    selected_testing_idxs = random.sample(range(0, len(samples_x)), k=num_testing_data)
    selected_training_idxs = set(range(0, len(samples_x))) - set(selected_testing_idxs)
    selected_training_idxs = random.sample(selected_training_idxs, k=num_training_data)
    
    # STEP 3: Split dataset into training and testing
    training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
    training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
    testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
    testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
    
    # STEP 4: Compute Big-GP's testing MAE
    all_lrn_asgmts['big_gp_model'] = LearningAssignment(zoo, expanded_sample_x_names)
    created_model_name = all_lrn_asgmts['big_gp_model'].create_and_add_model(training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
    #print(zoo.dump_model_info(created_model_name))
    for testing_sample_x, testing_sample_y_aggregation in zip(testing_samples_x, testing_samples_y_aggregation):
        pred = all_lrn_asgmts['big_gp_model'].predict(testing_sample_x)['val']
        big_gp_abs_errs[-1].append(abs(pred - testing_sample_y_aggregation))
    
    # ========== Compute small models' errors ==========
    for sample_y_name in all_sample_x_names.keys():
        sample_x_names = all_sample_x_names[sample_y_name]
        samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], sample_x_names, sample_y_name)
        
        # STEP 1: Split dataset into training and testing
        training_samples_x = [samples_x[idx] for idx in selected_training_idxs]
        training_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_training_idxs]
        testing_samples_x = [samples_x[idx] for idx in selected_testing_idxs]
        testing_samples_y_aggregation = [samples_y_aggregation[idx] for idx in selected_testing_idxs]
        
        # STEP 2: Train
        all_lrn_asgmts[sample_y_name] = LearningAssignment(zoo, sample_x_names)
        created_model_name = all_lrn_asgmts[sample_y_name].create_and_add_model(training_samples_x, training_samples_y_aggregation, GaussianProcess, model_class_args=[True, 250, False])
        
        # STEP 3: Compute MAE with testing dataset
        small_models_preds[-1][sample_y_name] = []
        for testing_sample_x in testing_samples_x:
            small_models_preds[-1][sample_y_name].append(all_lrn_asgmts[sample_y_name].predict(testing_sample_x)['val'])
        small_models_raw_errs[-1][sample_y_name] = [t - p for p, t in zip(small_models_preds[-1][sample_y_name], testing_samples_y_aggregation)]
        small_models_abs_errs[-1][sample_y_name] = [abs(err) for err in small_models_raw_errs[-1][sample_y_name]]
    
    # ========== Compute Fluxion's errors ==========
    # STEP 1: Prepare (1) Fluxion and (2) a list of input names that we will need to read from CSV
    def _build_fluxion(tmp_service_name, visited_services=[]):
        if tmp_service_name in visited_services:
            return []
        visited_services.append(tmp_service_name)
        
        service_dependencies_name = []
        ret_inputs_name = []
        
        for sample_x_name in all_sample_x_names[tmp_service_name]:
            if sample_x_name in all_sample_x_names.keys():
                ret_inputs_name += _build_fluxion(sample_x_name, visited_services)
                service_dependencies_name.append(sample_x_name)
            else:
                service_dependencies_name.append(None)
                ret_inputs_name.append(sample_x_name)
        
        fluxion.add_service(tmp_service_name, tmp_service_name, all_lrn_asgmts[tmp_service_name], service_dependencies_name, service_dependencies_name)
        
        return ret_inputs_name
    
    inputs_name = _build_fluxion(target_service_name)
    samples_x, samples_y, samples_y_aggregation, err_msg = lib_data.readCSVFile([dataset_filename], inputs_name, target_service_name)
    
    # STEP 2: Compute Fluxion's testing MAE
    for sample_idx, sample_x, sample_y_aggregation in zip(range(len(samples_x)), samples_x, samples_y_aggregation):
        if sample_idx not in selected_testing_idxs:
            continue
        
        fluxion_input = {}
        for val, input_name in zip(sample_x, inputs_name):
            service_name = None
            for tmp_name in all_sample_x_names.keys():
                if input_name in all_sample_x_names[tmp_name]:
                    service_name = tmp_name
                    
                    if service_name not in fluxion_input.keys():
                        fluxion_input[service_name] = {}
                    if service_name not in fluxion_input[service_name].keys():
                        fluxion_input[service_name][service_name] = [{}]
                    if input_name not in fluxion_input[service_name].keys():
                        fluxion_input[service_name][service_name][0][input_name] = val
        
        pred = fluxion.predict(target_service_name, target_service_name, fluxion_input)[target_service_name][target_service_name]['val']
        fluxion_abs_errs[-1].append(abs(pred - sample_y_aggregation))
    
    print("==================================================")
    print("| num_training_data:", num_training_data)
    print("| num_testing_data:", num_testing_data)
    print("| target_deployment_name:", target_deployment_name)
    print("| target_service_name:", target_service_name)
    print("| num_experiments:", num_experiments)
    print("| experiment_ids_completed:", experiment_ids_completed)
    print("| dataset_filename:", dataset_filename)
    
    print("==========")
    print("| small_models_abs_errs:")
    for sample_x_name in all_sample_x_names:
        print(sample_x_name, [round(statistics.mean(small_models_abs_errs[idx][sample_x_name]), 8) for idx in range(len(small_models_abs_errs))])
    
    print("==========")
    print("| fluxion_abs_errs:")
    print([round(statistics.mean(errs), 8) for errs in fluxion_abs_errs])
    
    print("==========")
    print("| big_gp_abs_errs:")
    print([round(statistics.mean(errs), 8) for errs in big_gp_abs_errs])
