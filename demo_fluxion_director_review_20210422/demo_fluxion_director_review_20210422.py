# python3 demo_fluxion_director_review_20210422.py

import os, random, statistics, sys, time

sys.path.insert(1, "../")
from GraphEngine.learning_assignment import LearningAssignment
from GraphEngine.graph_engine import GraphEngine
from GraphEngine.Model.framework_sklearn.gaussian_process import GaussianProcess
from GraphEngine.Model.framework_sklearn.multi_layer_perceptron import MultiLayerPerceptron
from GraphEngine.Model.framework_sklearn.random_forest_classifier import RandomForestClassifer
from GraphEngine.Model.framework_rule.merge import Merge
from GraphEngine.ModelZoo.model_zoo import Model_Zoo
from IngestionEngine_CSV import ingestionEngine
from Model.lib_model import Model

def _get_perf_name():
    return "written_throughput:benchmark"

def _get_configknobs_name(sys_component_name):
    return ["write_buffer_size:tune", "level0_stop_writes_trigger:tune", "max_write_buffer_number:tune"]

def _construct_graph(zoo, ge, query_client, learning_assignments):
    # STEP 1: Create learning assignments for all containers
    sys.stderr.write("[{}] Creating learning assignments for all containers...\n".format(os.path.basename(__file__)))
    containers_name = query_client.get("container_name", remove_duplicates=True)
    containers_name = containers_name[0]['val']
    for container_name in containers_name:
        if container_name not in ge.get_nodes_name():
            sample_x_names = _get_configknobs_name("N/A")
            #la = LearningAssignment(zoo, container_name, sample_x_names)
            la = learning_assignments[container_name]
            ge.add_node(container_name, la)
    
    # STEP 2: Create learning assignments for all nodes
    sys.stderr.write("[{}] Creating learning assignments for all nodes...\n".format(os.path.basename(__file__)))
    nodes_name = query_client.get("node_name", remove_duplicates=True)
    nodes_name = nodes_name[0]['val']
    print("nodes_name:", nodes_name)
    for node_name in nodes_name:
        if node_name not in ge.get_nodes_name():
            containers_name = query_client.get("container_name", remove_duplicates=True, conditions={'node_name': node_name})
            if len(containers_name) > 0:
                sample_x_names = ["container_" + container_name for container_name in containers_name[0]['val']]
                
                la = LearningAssignment(zoo, sample_x_names)
                #la.add_model([], [], Merge, model_class_args = ["avg"])
                ge.add_node(node_name, la)
                
                # STEP 2-1: Add edges
                for container_name in containers_name[0]['val']:
                    ge.add_edge(container_name, node_name, "container_" + container_name)
    
    # STEP 3: Create learning assignment for the cluster
    sys.stderr.write("[{}] Creating learning assignments for the cluster...\n".format(os.path.basename(__file__)))
    nodes_name = query_client.get("node_name", remove_duplicates=True)
    nodes_name = nodes_name[0]['val']
    print("nodes_name:", nodes_name)
    sample_x_names = ["node_" + node_name for node_name in nodes_name]
    la = LearningAssignment(zoo, sample_x_names)
    #la.add_model([], [], Merge, model_class_args = ["avg"])
    ge.add_node("cluster", la)
    
    # STEP 3-1: Add edges
    sys.stderr.write("[{}] Adding edges\n".format(os.path.basename(__file__)))
    for node_name in nodes_name:
        containers_name = query_client.get("container_name", remove_duplicates=True, conditions={'node_name': node_name})
        if len(containers_name) > 0:
            ge.add_edge(node_name, "cluster", "node_" + node_name)

def _get_data(sys_component_name, sample_x_names, sample_y_name, query_client, is_samples_x_in_dict=False):
    samples_x = []
    samples_y = []
    
    data = {}
    for samples_x_name in sample_x_names:
        data[samples_x_name] = query_client.get(samples_x_name, stime=0, conditions={'container_name': sys_component_name})
    data[sample_y_name] = query_client.get(sample_y_name, stime=0, conditions={'container_name': sys_component_name})
    
    for idx in range(len(data[sample_y_name])):
        if is_samples_x_in_dict == True:
            samples_x.append({})
            for samples_x_name in sample_x_names:
                samples_x[idx][samples_x_name] = float(data[samples_x_name][idx]['val'][0])
        else:
            samples_x.append([])
            for samples_x_name in sample_x_names:
                samples_x[idx].append(float(data[samples_x_name][idx]['val'][0]))
        
        samples_y.append(None)
        samples_y[idx] = float(data[sample_y_name][idx]['val'][0])
    
    return samples_x, samples_y

if __name__ == "__main__":
    zoo = Model_Zoo()
    learning_assignments = {}  # Keeps track of assignments created, for reuse
    
    # ===== STEP 1: Create and train models for RocksDBs =====
    ge = GraphEngine()
    for deployment_name in ["fillrandom92", "fillrandom192", "readrandomwriterandom92"]:
        query_client = ingestionEngine.IngestionClient(deployment_name)
        containers_name = query_client.get("container_name")
        containers_name = containers_name[0]['val']
        
        for container_name in containers_name:
            sample_x_names = _get_configknobs_name(container_name)
            sample_y_name = _get_perf_name()
            samples_x, samples_y = _get_data(container_name, sample_x_names, sample_y_name, query_client, is_samples_x_in_dict=False)
            
            # [MIKE] Artificially reduce the number of samples to speed up the demo
            tmp_idxs = random.choices(range(len(samples_x)), k=30)
            samples_x = [samples_x[tmp_idx] for tmp_idx in tmp_idxs]
            samples_y = [samples_y[tmp_idx] for tmp_idx in tmp_idxs]
            
            la = LearningAssignment(zoo, sample_x_names)
            learning_assignments[container_name] = la
            learning_assignments[container_name + "_n1"] = la
            learning_assignments[container_name + "_n2"] = la
            ge.add_node(container_name, la)
            
            ge.get_node(container_name)['state'] = "training"
            #ge.visualize_nodes_diagrams(output_filename="graph_" + "%.0f" % (time.time() * 1000) + "_training_node_" + container_name)
            ge.visualize_nodes_diagrams(output_filename="graph")
            
            la.add_model(samples_x, samples_y, GaussianProcess)
            
            ge.get_node(container_name)['state'] = "None"
            #ge.visualize_nodes_diagrams(output_filename="graph_" + "%.0f" % (time.time() * 1000) + "_after_training_node_" + container_name)
            ge.visualize_nodes_diagrams(output_filename="graph")
    
    # ===== STEP 2: Construct the graph =====
    #for deployment_name in ["fillrandom92_fillrandom192_fillrandom92_fillrandom192_readrandomwriterandom92"]:
    for deployment_name in ["fillrandom92_fillrandom192", "fillrandom92_fillrandom192_readrandomwriterandom92", "fillrandom92_fillrandom192_fillrandom92_fillrandom192_readrandomwriterandom92"]:
        print("===== Detected Kubernetes cluster changes...")
        #time.sleep(3)
        
        ge = GraphEngine()
        query_client = ingestionEngine.IngestionClient(deployment_name)
        _construct_graph(zoo, ge, query_client, learning_assignments)
        #ge.visualize_diagrams("cluster", output_filename="graph_" + "%.0f" % (time.time() * 1000) + "_after_construction")
        ge.visualize_diagrams("cluster", output_filename="graph")
        
        # STEP 2-1: Train node models
        graph_inputs = []
        graph_outputs = []
        # STEP 2-1-1: Prepare graph inputs and outputs
        containers_name = query_client.get("container_name", remove_duplicates=True)
        containers_name = containers_name[0]['val']
        for container_name in containers_name:
            sample_x_names = _get_configknobs_name(container_name)
            sample_y_name = _get_perf_name()
            samples_x, samples_y = _get_data(container_name, sample_x_names, sample_y_name, query_client)
            
            # Merge all samples_x into graph inputs, all samples_y into graph outputs
            for idx in range(len(samples_y)):
                if len(graph_inputs) <= idx:
                    graph_inputs.append({})
                    graph_outputs.append({})
                
                graph_inputs[idx][container_name] = {}
                for samples_x_name, sample_x in zip(sample_x_names, samples_x[idx]):
                    graph_inputs[idx][container_name][samples_x_name] = sample_x
                
                graph_outputs[idx][container_name] = samples_y[idx]
        
        for num_samples in range(20, 60, 20):
            # STEP 2-1-2: Train and evaluate node models one by one
            nodes_name = query_client.get("node_name", remove_duplicates=True)
            nodes_name = nodes_name[0]['val']
            for node_name in nodes_name:
                if node_name in ge.get_nodes_name():
                    containers_name = query_client.get("container_name", remove_duplicates=True, conditions={'node_name': node_name})
                    containers_name = containers_name[0]['val']
                    
                    # Prepare node training outputs
                    tmp_graph_outputs = []
                    for idx in range(len(graph_outputs)):
                        tmp_graph_output = []
                        for container_name in containers_name:
                            tmp_graph_output.append(graph_outputs[idx][container_name])
                        tmp_graph_output = statistics.mean(tmp_graph_output)
                        tmp_graph_outputs.append(tmp_graph_output)
                    
                    # Train the node
                    ge.get_node(node_name)['state'] = "training"
                    ge.get_node(node_name)['pred_loss'] = None
                    #ge.visualize_diagrams("cluster", output_filename="graph_" + "%.0f" % (time.time() * 1000) + "_training_" + node_name + "_" + str(num_samples))
                    ge.visualize_diagrams("cluster", output_filename="graph")
                    ge.train_node(node_name, graph_inputs[0:num_samples], tmp_graph_outputs[0:num_samples], GaussianProcess, is_delete_prev_models=True)
                    ge.get_node(node_name)['state'] = None
                    
                    # Evaluate the current graph
                    losses = []
                    for graph_input, tmp_graph_output in zip(graph_inputs, tmp_graph_outputs):
                        prediction = ge.predict(node_name, graph_input)
                        loss = abs(prediction[node_name]['val'] - tmp_graph_output)
                        losses.append(loss)
                    ge.get_node(node_name)['pred_loss'] = sum(losses)
                    
                    #ge.visualize_diagrams("cluster", output_filename="graph_" + "%.0f" % (time.time() * 1000) + "_after_training_" + node_name + "_" + str(num_samples))
                    ge.visualize_diagrams("cluster", output_filename="graph")

            # Prepare cluster training outputs
            containers_name = query_client.get("container_name", remove_duplicates=True)
            containers_name = containers_name[0]['val']
            
            tmp_graph_outputs = []
            for idx in range(len(graph_outputs)):
                tmp_graph_output = []
                for container_name in containers_name:
                    tmp_graph_output.append(graph_outputs[idx][container_name])
                tmp_graph_output = statistics.mean(tmp_graph_output)
                tmp_graph_outputs.append(tmp_graph_output)
            
            # Train the node
            ge.get_node("cluster")['state'] = "training"
            ge.get_node("cluster")['pred_loss'] = None
            #ge.visualize_diagrams("cluster", output_filename="graph_" + "%.0f" % (time.time() * 1000) + "_training_cluster")
            ge.visualize_diagrams("cluster", output_filename="graph")
            ge.train_node("cluster", graph_inputs[0:num_samples], tmp_graph_outputs[0:num_samples], GaussianProcess, is_delete_prev_models=True)
            ge.get_node("cluster")['state'] = None
            
            # Evaluate the current graph
            losses = []
            for graph_input, tmp_graph_output in zip(graph_inputs, tmp_graph_outputs):
                prediction = ge.predict("cluster", graph_input)
                loss = abs(prediction['cluster']['val'] - tmp_graph_output)
                losses.append(loss)
            ge.get_node("cluster")['pred_loss'] = sum(losses)
            
            #ge.visualize_diagrams("cluster", output_filename="graph_" + "%.0f" % (time.time() * 1000) + "_after_training_cluster")
            ge.visualize_diagrams("cluster", output_filename="graph")
            